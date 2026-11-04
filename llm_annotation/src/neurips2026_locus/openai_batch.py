"""OpenAI Responses batch transport; no SDK import until submission.

API contract: https://developers.openai.com/api/docs/guides/batch
The supplied municipal-law bundle defines the request policy and schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .contract import decode_json

REASONING_EFFORT = "low"


def build_request(
    custom_id: str,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str | list[str],
    schema: dict[str, Any],
    max_output_tokens: int = 4096,
) -> dict[str, Any]:
    """Build one strict-schema request without contacting OpenAI."""
    user_messages = [user_prompt] if isinstance(user_prompt, str) else user_prompt
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "developer", "content": system_prompt},
                *({"role": "user", "content": text} for text in user_messages),
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "municipal_law_annotation",
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": REASONING_EFFORT},
        },
    }


def parse_result(row: dict[str, Any]) -> tuple[str, dict | None, dict, str | None]:
    """Keep provider metadata; never repair or invent a classification."""
    custom_id = row.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise ValueError("OpenAI result is missing its custom_id")
    metadata: dict[str, Any] = {"provider": "openai"}
    if row.get("error"):
        return custom_id, None, metadata, json.dumps(row["error"], ensure_ascii=False)
    response = row.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        return custom_id, None, metadata, "Missing response or non-200 response status"
    body = response.get("body")
    if not isinstance(body, dict):
        return custom_id, None, metadata, "Missing response body"
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    metadata.update(
        {
            "response_model": body.get("model"),
            "request_id": response.get("request_id"),
            "usage": usage,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
    )
    metadata["status"] = body.get("status")
    if (
        body.get("status") != "completed"
        or body.get("error")
        or body.get("incomplete_details")
    ):
        return custom_id, None, metadata, "Classification did not finish normally"
    output = body.get("output")
    if not isinstance(output, list):
        return custom_id, None, metadata, "Missing response output"
    messages = [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
    ]
    if len(messages) != 1 or messages[0].get("role") != "assistant":
        return custom_id, None, metadata, "Expected exactly one assistant message"
    if messages[0].get("status") not in (None, "completed"):
        return custom_id, None, metadata, "Assistant message did not finish normally"
    content = messages[0].get("content")
    if not isinstance(content, list) or not content:
        return custom_id, None, metadata, "Missing assistant content"
    if any(
        not isinstance(part, dict)
        or part.get("type") != "output_text"
        or not isinstance(part.get("text"), str)
        for part in content
    ):
        return custom_id, None, metadata, "Refusal or unsupported assistant content"
    try:
        classification = decode_json("".join(part["text"] for part in content))
    except ValueError:
        classification = None
    if not isinstance(classification, dict):
        return custom_id, None, metadata, "Classification is not a valid JSON object"
    return custom_id, classification, metadata, None


class OpenAIBatchClient:
    """One upload/submit/status call at a time; submission is never retried."""

    def __init__(self, api_key: str, timeout: float = 60):
        """Fix the official endpoint and disable automatic paid-job creation retries."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "Install the OpenAI extra: pip install '.[openai]'"
            ) from None
        # Retrying an uncertain create response could submit a duplicate paid job.
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            timeout=timeout,
            max_retries=0,
        )

    def upload(self, path: Path) -> str:
        """Upload one temporary shard; close its local file as soon as upload returns."""
        with path.open("rb") as handle:
            return self.client.files.create(file=handle, purpose="batch").id

    def submit(self, file_id: str, model: str, display_name: str) -> str:
        """Create a 24-hour Responses batch with a recoverable unique description."""
        # The model is already embedded in every JSONL request.
        return self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": display_name},
        ).id

    def status(self, job_id: str) -> dict[str, Any]:
        """Normalize terminal states and both result/error file references."""
        job = self.client.batches.retrieve(job_id)
        state = job.status
        failed = state in {"failed", "expired", "cancelled"}
        return {
            "id": job.id,
            "input_file_id": job.input_file_id,
            "model": None,  # Batch objects do not report the model in the input file.
            "display_name": (job.metadata or {}).get("description"),
            "status": state,
            "done": state == "completed" or failed,
            "failed": failed,
            "output_files": [job.output_file_id] if job.output_file_id else [],
            "error_files": [job.error_file_id] if job.error_file_id else [],
        }

    def iter_results(self, file_id: str) -> Iterator[dict[str, Any]]:
        """Stream strict JSON objects, preserving provider IDs for the ledger join."""
        with self.client.files.with_streaming_response.content(file_id) as response:
            for line in response.iter_lines():
                if line.strip():
                    row = decode_json(line)
                    if not isinstance(row, dict):
                        raise ValueError("OpenAI result line must be a JSON object")
                    yield row

    def close(self) -> None:
        """Release SDK HTTP resources after the operation completes."""
        self.client.close()
