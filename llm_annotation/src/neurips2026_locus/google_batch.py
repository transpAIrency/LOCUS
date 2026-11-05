"""Gemini Developer API file batches; SDK imported only for network commands.

API contract: https://ai.google.dev/gemini-api/docs/batch-api
"""

from __future__ import annotations

from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from .contract import decode_json

THINKING_LEVEL = "LOW"


class GoogleBatchStatusError(ValueError):
    """A local status check whose operational message is safe to display."""

    def __init__(self, message: str):
        self.safe_message = message
        self.details = {"error": {"code": "invalid_batch_status"}}
        super().__init__(message)


def build_request(
    custom_id: str,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str | list[str],
    schema: dict[str, Any],
    max_output_tokens: int = 4096,
) -> dict[str, Any]:
    """The model is set once on the batch, not on its individual requests."""
    user_messages = [user_prompt] if isinstance(user_prompt, str) else user_prompt
    return {
        "key": custom_id,
        "request": {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": text}]} for text in user_messages
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": max_output_tokens,
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
            },
        },
    }


def parse_result(row: dict[str, Any]) -> tuple[str, dict | None, dict, str | None]:
    """Reject blocked/truncated responses; do not treat thought text as output."""
    custom_id = row.get("key")
    if not isinstance(custom_id, str) or not custom_id:
        raise ValueError("Google result is missing its key")
    metadata: dict[str, Any] = {"provider": "google"}
    if row.get("error"):
        return custom_id, None, metadata, json.dumps(row["error"], ensure_ascii=False)
    response = row.get("response")
    if not isinstance(response, dict):
        return custom_id, None, metadata, "Missing response"
    usage = response.get("usageMetadata")
    usage = usage if isinstance(usage, dict) else {}
    metadata.update(
        {
            "response_model": response.get("modelVersion"),
            "usage": usage,
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "reasoning_tokens": usage.get("thoughtsTokenCount"),
        }
    )
    feedback = response.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        return custom_id, None, metadata, "Prompt was blocked"
    candidates = response.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], dict)
    ):
        return custom_id, None, metadata, "Expected exactly one response candidate"
    candidate = candidates[0]
    metadata["finish_reason"] = candidate.get("finishReason")
    if candidate.get("finishReason") != "STOP":
        return custom_id, None, metadata, "Classification did not finish normally"
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
        return custom_id, None, metadata, "Missing response content"
    if any(
        not part.get("thought") and not isinstance(part.get("text"), str)
        for part in parts
    ):
        return custom_id, None, metadata, "Unsupported non-text response content"
    text_parts = [
        part["text"]
        for part in parts
        if not part.get("thought") and isinstance(part.get("text"), str)
    ]
    try:
        classification = decode_json("".join(text_parts))
    except ValueError:
        classification = None
    if not isinstance(classification, dict):
        return custom_id, None, metadata, "Classification is not a JSON object"
    return custom_id, classification, metadata, None


class GoogleBatchClient:
    """Use the official SDK without automatically retrying job creation."""

    def __init__(self, api_key: str, timeout: float = 60):
        """Use Gemini Developer API credentials and disable automatic submit retries."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "Install the Google extra: pip install '.[google]'"
            ) from None
        self.client = genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                base_url="https://generativelanguage.googleapis.com",
                api_version="v1beta",
                timeout=int(timeout * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def upload(self, path: Path) -> str:
        """Upload a bounded JSONL shard using the API's required file MIME type."""
        return self.client.files.upload(
            file=str(path), config={"mime_type": "jsonl", "display_name": path.name}
        ).name

    def delete_file(self, file_id: str) -> None:
        """Delete one exact file resource; a missing file is already cleaned up."""
        if not isinstance(file_id, str) or not re.fullmatch(
            r"files/[A-Za-z0-9_-]+", file_id
        ):
            raise ValueError("Google cleanup requires an exact files/<id> resource name")
        try:
            self.client.files.delete(name=file_id)
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if status != 404:
                raise

    def submit(self, file_id: str, model: str, display_name: str) -> str:
        """Create one batch; the caller journals an uncertain response for recovery."""
        return self.client.batches.create(
            model=model, src=file_id, config={"display_name": display_name}
        ).name

    def status(self, job_id: str) -> dict[str, Any]:
        """Treat partial success as terminal so successful rows can be collected."""
        job = self.client.batches.get(name=job_id)

        def field(value, name):
            if isinstance(value, Mapping):
                return value.get(name)
            return getattr(value, name, None)

        # The SDK translates most MLDev BATCH_STATE values to JOB_STATE;
        # normalize both prefixes for terminal values it has not translated.
        raw_state = field(job, "state")
        state = raw_state if isinstance(raw_state, str) else field(raw_state, "name")
        if not isinstance(state, str):
            state = field(raw_state, "value")
        state = (
            state.strip()
            .upper()
            .removeprefix("JOB_STATE_")
            .removeprefix("BATCH_STATE_")
            if isinstance(state, str)
            else ""
        )
        if state not in {
            "QUEUED",
            "PENDING",
            "RUNNING",
            "PAUSED",
            "UPDATING",
            "CANCELLING",
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "EXPIRED",
            "PARTIALLY_SUCCEEDED",
        }:
            reported = repr(state[:120]) if state else "missing"
            raise GoogleBatchStatusError(
                f"Google batch {job_id} returned {reported} state; "
                "check its provider status before resuming"
            )
        failed = state in {
            "FAILED",
            "CANCELLED",
            "EXPIRED",
            "PARTIALLY_SUCCEEDED",
        }
        output_file = field(field(job, "dest"), "file_name")
        input_file = field(field(job, "src"), "file_name")
        for value in (output_file, input_file):
            if value is not None and (not isinstance(value, str) or not value):
                raise GoogleBatchStatusError(
                    f"Google batch {job_id} returned an invalid file reference; "
                    "check its provider status before resuming"
                )
        return {
            "id": field(job, "name") or job_id,
            # MLDev currently does not populate src; do not infer it from a name.
            "input_file_id": input_file,
            "model": field(job, "model"),
            "display_name": field(job, "display_name"),
            "status": ("completed" if state == "SUCCEEDED" else state.lower()),
            "done": state == "SUCCEEDED" or failed,
            "failed": failed,
            "output_files": [output_file] if output_file else [],
            # Google includes per-request errors in the same output JSONL file.
            "error_files": [],
        }

    def iter_results(self, file_id: str) -> Iterator[dict[str, Any]]:
        """Parse a bounded SDK byte download without a second decoded line list."""
        # The SDK returns bytes, not a stream. The harness therefore bounds shard
        # sizes; TextIOWrapper avoids also materializing a list of decoded lines.
        with TextIOWrapper(
            BytesIO(self.client.files.download(file=file_id)), encoding="utf-8"
        ) as lines:
            for line in lines:
                if line.strip():
                    row = decode_json(line)
                    if not isinstance(row, dict):
                        raise ValueError("Google result line must be a JSON object")
                    yield row

    def close(self) -> None:
        """Release SDK HTTP resources after the operation completes."""
        self.client.close()
