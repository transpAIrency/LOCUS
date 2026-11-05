"""Resumable file batches, bounded in flight, with byte-indexed source access.

Only ledgers and final exports persist locally. Request shards are temporary.
Never retry an uncertain job creation automatically: recover its remote ID.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Any, Callable
from uuid import uuid4

from . import google_batch, openai_batch
from .contract import (
    ContractError,
    compose_input_text,
    decode_json,
    is_header_only_record,
)
from .generative import (
    DEFAULT_OPENAI_MODEL,
    POLICY_VERSION,
    build_messages,
    contract_hash,
    parse_classification,
)
from .jev import (
    DEFAULT_ENDPOINT,
    JevError,
    RetryDelayExceeded,
    load_dotenv,
    retry_delay,
)
from .files import file_stamp
from .pipeline import (
    AnnotationLedger,
    DEFAULT_WORKERS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_INFLIGHT,
    ROW_ERRORS_EXIT_CODE,
    InputError,
    acquire_annotation_lock,
    check_request_builders,
    check_cancelled,
    canonical_json,
    distinct_paths,
    export_annotations_jsonl,
    indexed_jsonl,
    iter_jsonl,
    ledger_metadata,
    sha256_file,
    source_identity,
    source_locator,
    utc_now,
)

MAX_SHARD_BYTES = 50 * 1024 * 1024
MAX_BATCH_ROWS = 50_000
RETRYABLE_OPERATION_EXIT_CODE = 4
DEFAULT_GOOGLE_MODEL = "gemini-3.8-flash"


def shard_byte_limit(provider: str) -> int:
    """Bound uploads below each API's file limit; permit a small offline test budget."""
    if MAX_SHARD_BYTES != 50 * 1024 * 1024:
        return MAX_SHARD_BYTES
    return {"openai": 160 * 1024 * 1024, "google": 1024 * 1024 * 1024}[provider]


def _error_payload(error) -> dict:
    """Read structured provider errors, never their request or exception repr."""
    body = error if isinstance(error, dict) else getattr(error, "body", None)
    if not isinstance(body, dict):
        body = getattr(error, "response_json", None)
    if not isinstance(body, dict):
        body = getattr(error, "details", None)
    if not isinstance(body, dict):
        return {}
    return body.get("error", body) if isinstance(body.get("error", body), dict) else {}


def _http_status(error) -> int | None:
    response = getattr(error, "response", None)
    payload = _error_payload(error)
    return next(
        (
            value
            for value in (
                getattr(error, "status_code", None),
                getattr(error, "code", None),
                getattr(response, "status_code", None),
                payload.get("status_code"),
                payload.get("code"),
            )
            if type(value) is int
        ),
        None,
    )


def _queue_limit(error) -> bool:
    """Recognize a provider's explicit queued-token quota rejection."""
    payload = _error_payload(error)
    text = " ".join(
        str(payload.get(key, "")) for key in ("code", "status", "message", "details")
    )
    return bool(
        re.search(
            r"enqueued|queue|batch.{0,40}token|token.{0,40}(?:limit|quota)", text, re.I
        )
    )


def _sanitized_message(value) -> str | None:
    """Keep the provider's guidance while removing credentials and body dumps."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    value = re.sub(
        r"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?[^\s,;\"']+",
        "Authorization: [redacted]",
        value,
    )
    value = re.sub(r"(?i)\bbearer\s+[^\s,;\"']+", "Bearer [redacted]", value)
    value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+)\b", "[redacted]", value)
    value = re.sub(
        r"(?i)\b(?:api[ _-]*key|access[ _-]*token|refresh[ _-]*token|token|secret)"
        r"[\"']?\s*[:=]\s*[\"']?[^\s,;\"'}]+",
        "credential: [redacted]",
        value,
    )
    value = re.sub(
        r"(?i)([?&](?:key|api_key|token|access_token)=)[^&\s]+",
        r"\1[redacted]",
        value,
    )
    value = re.split(
        r"(?i)\b(?:request(?:\s+body)?|payload)[\"']?\s*[:=]\s*[\[{]",
        value,
        maxsplit=1,
    )[0]
    return " ".join(value.split())[:600]


def _transient_operation_error(error) -> bool:
    """Classify control-plane failures; never use this to replay creation."""
    status = _http_status(error)
    if status is not None:
        return status in {408, 429} or 500 <= status <= 599
    import httpx

    cause = error
    for _ in range(6):
        if isinstance(cause, (httpx.TransportError, ConnectionError, TimeoutError)):
            return True
        cause = cause.__cause__
        if cause is None:
            break
    return False


def provider_error_details(error) -> dict:
    """Keep actionable operational facts without echoing provider-supplied text."""
    payload = _error_payload(error)
    status = _http_status(error)
    code = payload.get("code")
    if not isinstance(code, str):
        code = payload.get("status") or payload.get("type")
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", code):
        code = None
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    request_id_value = (
        getattr(error, "request_id", None)
        or payload.get("request_id")
        or headers.get("x-request-id")
        or headers.get("x-goog-request-id")
    )
    if not isinstance(request_id_value, str) or not re.fullmatch(
        r"[A-Za-z0-9_.:-]{1,200}", request_id_value
    ):
        request_id_value = None
    if status == 429:
        message = (
            "Queued-token quota reached; waiting and reducing oversized shards"
            if _queue_limit(error)
            else "Provider quota or rate limit reached; waiting before retrying"
        )
    elif status == 401:
        message = "Provider rejected the API credentials"
    elif status == 403:
        message = "Provider denied access to this model or project"
    elif status == 413:
        message = "Provider rejected the request size; reduce the batch size"
    elif status in {400, 404, 405, 406, 411, 415, 422}:
        message = (
            "Provider rejected the request; check model, parameters, and project access"
        )
    elif status == 402:
        message = "Provider rejected billing; check account credit and spending limits"
    elif status is not None and status >= 500:
        message = "Provider server failed; batch creation may have succeeded"
    else:
        message = "Provider operation failed; inspect provider status before retrying"
    message = (
        _sanitized_message(payload.get("message"))
        or _sanitized_message(getattr(error, "message", None))
        or message
    )
    if isinstance(error, getattr(google_batch, "GoogleBatchStatusError", ())):
        message = error.safe_message
    return {
        "http_status": status,
        "code": code,
        "message": message,
        "request_id": request_id_value,
    }


def _quota_delay(error, attempts: int) -> float:
    """Honor Retry-After, including long waits, without blocking the CLI."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            delay = deadline.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            delay = math.nan
    if math.isfinite(delay):
        return max(1.0, delay)
    return min(60.0 * 2 ** min(attempts - 1, 6), 3600.0)


def _retry_at_utc(value: float) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    )


def _submit_job(db, config, client, number, file_id, name, rows, stop_event) -> bool:
    """Retry only a definite rejection; uncertain creation always needs recovery."""
    check_cancelled(stop_event)
    with db:
        db.execute("UPDATE batch_jobs SET state='submitting' WHERE number=?", (number,))
    if stop_event is not None and stop_event.is_set():
        with db:
            db.execute(
                "UPDATE batch_jobs SET state='preparing' WHERE number=?", (number,)
            )
        raise CancelledError()
    try:
        remote_id = client.submit(file_id, config["model"], name)
    except Exception as exc:
        details = provider_error_details(exc)
        status = details["http_status"]
        with db:
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                (f"batch_error:{number}", canonical_json(details)),
            )
        if status == 429:
            previous = db.execute(
                "SELECT value FROM metadata WHERE key='batch_quota_attempts'"
            ).fetchone()
            attempts = int(previous[0]) + 1 if previous else 1
            retry_at = time.time() + _quota_delay(exc, attempts)
            schedule = {
                "retry_at": retry_at,
                "attempts": attempts,
                "error": details,
                "shrink": _queue_limit(exc),
            }
            with db:
                db.execute(
                    "UPDATE batch_jobs SET state='quota_wait' WHERE number=?", (number,)
                )
                db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                    (f"batch_quota:{number}", canonical_json(schedule)),
                )
                db.execute(
                    "INSERT OR REPLACE INTO metadata "
                    "VALUES ('batch_quota_attempts', ?)",
                    (str(attempts),),
                )
            print(
                json.dumps(
                    {
                        "provider": config["provider"],
                        "batch_number": number,
                        "phase": "quota_wait",
                        "retry_at_utc": _retry_at_utc(retry_at),
                        "error": details,
                    }
                ),
                flush=True,
            )
            return False
        if status in {400, 401, 402, 403, 404, 405, 406, 411, 413, 415, 422}:
            with db:
                db.execute(
                    "UPDATE batch_jobs SET state='preparing' WHERE number=?", (number,)
                )
            raise InputError(
                f"{config['provider']} rejected batch {number}: "
                f"{canonical_json(details)}; correct the cause and resume"
            ) from None
        raise InputError(
            f"{config['provider']} batch operation failed for batch {number}: "
            f"{canonical_json(details)}; "
            "submission is uncertain; attach its remote ID before continuing"
        ) from None
    if not isinstance(remote_id, str) or not remote_id:
        raise InputError(
            f"Batch {number} returned no job ID; submission may have succeeded"
        )
    with db:
        db.execute(
            "UPDATE batch_jobs SET remote_id=?, state='submitted' WHERE number=?",
            (remote_id, number),
        )
        db.execute("DELETE FROM metadata WHERE key=?", (f"batch_quota:{number}",))
        db.execute("DELETE FROM metadata WHERE key='batch_quota_attempts'")
    print(
        json.dumps({"submitted": remote_id, "batch_number": number, "rows": rows}),
        flush=True,
    )
    check_cancelled(stop_event)
    return True


def retry_provider_read(
    operation: Callable[[], Any], *, provider: str, stop_event=None
) -> Any:
    """Retry transient GET/collection failures with at most three attempts.

    Never wrap uploads or batch creation. A collection attempt commits whole
    shards or rolls them back; repeating it skips already collected shards
    and cannot duplicate annotations or submit new model work. Invalid data
    and authentication errors remain immediate failures.
    """
    for attempt in range(1, 4):
        check_cancelled(stop_event)
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = _http_status(exc)
            if not _transient_operation_error(exc) or attempt == 3:
                raise
            check_cancelled(stop_event)
            delay = retry_delay(getattr(response, "headers", None) or {}, attempt)
            print(
                json.dumps(
                    {
                        "provider": provider,
                        "phase": "retrying_read",
                        "attempt": attempt,
                        "retry_in_seconds": delay,
                        "http_status": status,
                        "error_type": type(exc).__name__,
                    }
                ),
                flush=True,
            )
            if stop_event is None:
                time.sleep(delay)
            else:
                stop_event.wait(delay)
    raise AssertionError("read retry loop must return or raise")


def configuration(args, *, hash_input: bool = True) -> dict[str, Any]:
    """Bind semantic settings to the ledger; avoid hashing unchanged files on polls."""
    if args.provider not in {"openai", "google"}:
        raise InputError("Native batches require provider openai or google")
    openai = args.provider == "openai"
    input_hash = None
    if hash_input:
        if args.ledger.exists():
            with closing(AnnotationLedger.open_readonly(args.ledger)) as ledger:
                stored = ledger_metadata(ledger.connection)
                stamp = ledger.connection.execute(
                    "SELECT value FROM metadata WHERE key='input_stamp'"
                ).fetchone()
            if args.batch_action in {
                "status",
                "attach",
                "collect",
                "reset-uncertain",
            } or (stamp and json.loads(stamp[0]) == file_stamp(args.input)):
                input_hash = stored["input_sha256"]
        if input_hash is None:
            input_hash = sha256_file(args.input)
    return {
        "ledger_schema_version": 1,
        "input_sha256": input_hash,
        "provider": args.provider,
        "request_submission": "batch",
        "calls_per_record": 1,
        "prompt_version": POLICY_VERSION,
        "contract_sha256": contract_hash(),
        "model": args.model
        or (DEFAULT_OPENAI_MODEL if openai else DEFAULT_GOOGLE_MODEL),
        "max_output_tokens": args.max_output_tokens or 4096,
        "reasoning_effort": (
            openai_batch.REASONING_EFFORT
            if openai
            else google_batch.THINKING_LEVEL.lower()
        ),
        "header_only_policy": args.header_only_policy,
        "max_text_chars": args.max_text_chars,
        "max_records": args.max_records,
    }


def adapter(provider: str):
    """Choose an explicit native transport; unknown names are never a fallback."""
    if provider not in {"openai", "google"}:
        raise InputError("Native batches require provider openai or google")
    return {"openai": openai_batch, "google": google_batch}[provider]


def api_key(provider: str, override: str | None) -> str:
    """Read provider credentials without echoing them."""
    if provider not in {"openai", "google"}:
        raise InputError("Native batches require provider openai or google")
    if override is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", override):
        raise InputError(
            "API-key variable name must be a valid environment variable name"
        )
    names = (
        [override]
        if override
        else (
            ["OPENAI_API_KEY"]
            if provider == "openai"
            else ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        )
    )
    for name in names:
        key = os.environ.get(name, "").strip()
        if key:
            if any(ord(char) < 33 or ord(char) > 126 for char in key):
                raise InputError("API key contains invalid characters")
            return key
    if override:
        raise InputError("Set the environment variable named by --api-key-env locally")
    raise InputError(f"Set {' or '.join(names)} locally to use {provider}")


def request_id(source_id: str) -> str:
    """Produce a fixed-length ASCII key accepted by both batch APIs."""
    return sha256(source_id.encode("utf-8")).hexdigest()


def request_for(record: dict, source_id: str, config: dict) -> dict:
    """Render shared policy/text and wrap it in the selected API's envelope."""
    system, messages, schema = build_messages(
        record, max_text_chars=config["max_text_chars"]
    )
    return adapter(config["provider"]).build_request(
        request_id(source_id),
        model=config["model"],
        system_prompt=system,
        user_prompt=messages,
        schema=schema,
        max_output_tokens=config["max_output_tokens"],
    )


def request_digest(request: dict) -> str:
    """Bind a prepared shard row to the exact canonical request contents."""
    return sha256(canonical_json(request).encode("utf-8")).hexdigest()


def ensure_pending_index(db: sqlite3.Connection) -> None:
    """Keep first-N pending selection ordered without a corpus-wide sort.

    Including job_number before source_index lets SQLite use both the NULL
    equality and the requested order. The former source_index-only partial
    index could lose to the general job index, causing a full pending scan on
    every submission. Index migration does not change prepared requests.
    """
    db.execute("""CREATE INDEX IF NOT EXISTS batch_request_pending_ordered
                  ON batch_requests(job_number, source_index)
                  WHERE job_number IS NULL AND result_json IS NULL AND error IS NULL""")
    db.execute("DROP INDEX IF EXISTS batch_request_pending")


def initialize(
    ledger: AnnotationLedger, input_path: Path, config: dict, *, stop_event=None
) -> None:
    """Validate and index input in one transaction before any paid submission."""
    db = ledger.connection
    check_request_builders(db)
    with db:
        db.execute("""CREATE TABLE IF NOT EXISTS batch_jobs (
            number INTEGER PRIMARY KEY, file_id TEXT, remote_id TEXT UNIQUE,
            state TEXT NOT NULL, display_name TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS batch_requests (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_index INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL, source_offset INTEGER NOT NULL,
            request_sha256 TEXT NOT NULL, source_info TEXT NOT NULL,
            job_number INTEGER, result_json TEXT, error TEXT)""")
        db.execute(
            "CREATE INDEX IF NOT EXISTS batch_request_job ON batch_requests(job_number)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS batch_request_result ON batch_requests(id) WHERE result_json IS NOT NULL"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS batch_request_error ON batch_requests(id) WHERE error IS NOT NULL"
        )
        ensure_pending_index(db)
        count = 0
        last_progress = time.monotonic()
        with closing(indexed_jsonl(input_path)) as source:
            while config["max_records"] is None or count < config["max_records"]:
                check_cancelled(stop_event)
                try:
                    index, offset, record = next(source)
                except StopIteration:
                    break
                now = time.monotonic()
                if now - last_progress >= 15:
                    print(
                        json.dumps(
                            {
                                "provider": config["provider"],
                                "phase": "preparing",
                                "selected_rows": count,
                                "scanned_lines": index + 1,
                            }
                        ),
                        flush=True,
                    )
                    last_progress = now
                source_id, digest = source_identity(record, index)
                info = {
                    "locator": source_locator(record, index),
                    "text_characters": len(compose_input_text(record)),
                }
                if (
                    is_header_only_record(record)
                    and config["header_only_policy"] == "exclude"
                ):
                    db.execute(
                        "INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            source_id,
                            index,
                            digest,
                            "header_only_excluded",
                            canonical_json(
                                {
                                    "status": "header_only_excluded",
                                    "prompt_version": POLICY_VERSION,
                                    "source": {
                                        "id": source_id,
                                        "sha256": digest,
                                        "locator": info["locator"],
                                    },
                                }
                            ),
                            utc_now(),
                        ),
                    )
                    continue
                request = request_for(record, source_id, config)
                if (
                    len(canonical_json(request).encode("utf-8")) + 1
                    > shard_byte_limit(config["provider"])
                ):
                    raise InputError(
                        f"Source row {index} exceeds the batch shard byte limit"
                    )
                db.execute(
                    "INSERT INTO batch_requests VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
                    (
                        request_id(source_id),
                        source_id,
                        index,
                        digest,
                        offset,
                        request_digest(request),
                        canonical_json(info),
                    ),
                )
                count += 1
        if count == 0:
            raise InputError("No eligible records selected for batch annotation")
        db.execute("INSERT INTO metadata VALUES ('batch_initialized', '1')")
        db.execute("INSERT INTO metadata VALUES ('batch_total', ?)", (str(count),))


def _row_retry_passes(db: sqlite3.Connection) -> int:
    row = db.execute(
        "SELECT value FROM metadata WHERE key='row_retry_passes'"
    ).fetchone()
    return int(row[0]) if row else 0


def summary(db: sqlite3.Connection) -> dict:
    """Report bounded progress counts and actionable unresolved submissions."""
    deferred = []
    for number, encoded in db.execute(
        "SELECT number, value FROM batch_jobs JOIN metadata ON key='batch_quota:' || number "
        "WHERE state='quota_wait' ORDER BY number"
    ):
        schedule = json.loads(encoded)
        deferred.append(
            {
                "batch_number": number,
                "retry_at": schedule["retry_at"],
                "retry_at_utc": _retry_at_utc(schedule["retry_at"]),
                "error": schedule["error"],
            }
        )
    next_retry = min((item["retry_at"] for item in deferred), default=None)
    operation_row = db.execute(
        "SELECT value FROM metadata WHERE key='batch_operation_retry'"
    ).fetchone()
    operation_retry = json.loads(operation_row[0])["retry_at"] if operation_row else None
    pending_cleanup = sum(
        len(json.loads(encoded)["pending"])
        for (encoded,) in db.execute(
            "SELECT value FROM metadata WHERE key GLOB 'batch_cleanup:*'"
        )
    )
    return {
        "requests": int(
            db.execute("SELECT value FROM metadata WHERE key='batch_total'").fetchone()[
                0
            ]
        ),
        "responses": db.execute(
            "SELECT COUNT(*) FROM batch_requests WHERE result_json IS NOT NULL"
        ).fetchone()[0],
        "errors": db.execute(
            "SELECT COUNT(*) FROM batch_requests WHERE error IS NOT NULL"
        ).fetchone()[0],
        "annotated": db.execute(
            "SELECT COUNT(*) FROM annotations WHERE status='annotated'"
        ).fetchone()[0],
        "retry_passes": _row_retry_passes(db),
        "batch_states": dict(
            db.execute("SELECT state, COUNT(*) FROM batch_jobs GROUP BY state")
        ),
        "deferred_jobs": deferred,
        "next_retry_at": next_retry,
        "next_retry_at_utc": (
            _retry_at_utc(next_retry) if next_retry is not None else None
        ),
        "operation_retry_at": operation_retry,
        "operation_retry_at_utc": (
            _retry_at_utc(operation_retry) if operation_retry is not None else None
        ),
        "pending_cleanup_files": pending_cleanup,
        "uncertain_submissions": [
            dict(zip(("batch_number", "display_name", "input_file_id"), row))
            for row in db.execute(
                "SELECT number, display_name, file_id FROM batch_jobs WHERE state='submitting' ORDER BY number"
            )
        ],
    }


def submit_ready(
    ledger: AnnotationLedger,
    input_path: Path,
    config: dict,
    client,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_inflight: int = DEFAULT_MAX_INFLIGHT,
    stop_event=None,
) -> None:
    """Seek pending rows and submit only enough shards to fill available slots."""
    check_cancelled(stop_event)
    db = ledger.connection
    unresolved = db.execute(
        "SELECT number FROM batch_jobs WHERE state='submitting'"
    ).fetchone()
    check_request_builders(db)
    if unresolved:
        raise InputError(
            f"Batch {unresolved[0]} has an uncertain submission; attach its remote ID before continuing"
        )
    with db:
        # Prepared ledgers from earlier versions receive the same index fix.
        ensure_pending_index(db)
        # Upload failures can be retried: batch creation was never attempted.
        db.execute(
            "UPDATE batch_requests SET job_number=NULL WHERE job_number IN (SELECT number FROM batch_jobs WHERE state='preparing')"
        )
        db.execute("DELETE FROM batch_jobs WHERE state='preparing'")
    active = db.execute(
        "SELECT COUNT(*) FROM batch_jobs WHERE state='submitted'"
    ).fetchone()[0]
    waiting = db.execute(
        "SELECT number, file_id, display_name FROM batch_jobs WHERE state='quota_wait' ORDER BY number"
    ).fetchall()
    for number, file_id, name in waiting:
        encoded = db.execute(
            "SELECT value FROM metadata WHERE key=?", (f"batch_quota:{number}",)
        ).fetchone()
        if encoded is None:
            raise InputError(f"Deferred batch {number} has no retry schedule")
        schedule = json.loads(encoded[0])
        if time.time() < schedule["retry_at"] or active >= max_inflight:
            return
        rows = db.execute(
            "SELECT COUNT(*) FROM batch_requests WHERE job_number=?", (number,)
        ).fetchone()[0]
        if schedule["shrink"] and active == 0 and rows > 1:
            smaller = max(1, rows // 2)
            with db:
                db.execute(
                    "UPDATE batch_requests SET job_number=NULL WHERE job_number=?",
                    (number,),
                )
                db.execute(
                    "UPDATE batch_jobs SET state='quota_split' WHERE number=?",
                    (number,),
                )
                db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('batch_size_limit', ?)",
                    (str(smaller),),
                )
            print(
                json.dumps(
                    {
                        "phase": "reducing_batch_size",
                        "batch_number": number,
                        "previous_rows": rows,
                        "batch_size_limit": smaller,
                    }
                ),
                flush=True,
            )
            continue
        if not _submit_job(
            db, config, client, number, file_id, name, rows, stop_event
        ):
            return
        active += 1
    limit = db.execute(
        "SELECT value FROM metadata WHERE key='batch_size_limit'"
    ).fetchone()
    if limit is not None:
        batch_size = min(batch_size, int(limit[0]))
        if batch_size < 1:
            raise InputError("Ledger contains an invalid adaptive batch size")
    opener = gzip.open if input_path.suffix == ".gz" else open
    with (
        opener(input_path, "rb") as source,
        tempfile.TemporaryDirectory(prefix="locus-batch-") as directory,
    ):
        path = Path(directory) / "requests.jsonl"
        for _ in range(max(0, max_inflight - active)):
            check_cancelled(stop_event)
            pending = db.execute(
                """SELECT id, source_id, source_index, source_sha256, source_offset, request_sha256
                FROM batch_requests WHERE job_number IS NULL AND result_json IS NULL AND error IS NULL
                ORDER BY source_index LIMIT ?""",
                (batch_size,),
            ).fetchall()
            if not pending:
                break
            ids, size = [], 0
            with path.open("wb") as handle:
                for key, source_id, index, digest, offset, prepared_hash in pending:
                    check_cancelled(stop_event)
                    source.seek(offset)
                    record = decode_json(source.readline())
                    if source_identity(record, index) != (source_id, digest):
                        raise InputError(
                            "Source row differs from its prepared identity"
                        )
                    request = request_for(record, source_id, config)
                    if request_digest(request) != prepared_hash:
                        raise InputError(
                            "Reconstructed request differs from its prepared hash"
                        )
                    payload = (canonical_json(request) + "\n").encode("utf-8")
                    if len(payload) > shard_byte_limit(config["provider"]):
                        raise InputError(
                            f"Source row {index} exceeds the batch shard byte limit"
                        )
                    if ids and size + len(payload) > shard_byte_limit(
                        config["provider"]
                    ):
                        break
                    handle.write(payload)
                    ids.append(key)
                    size += len(payload)
            number = db.execute(
                "SELECT COALESCE(MAX(number), 0)+1 FROM batch_jobs"
            ).fetchone()[0]
            # A unique suffix distinguishes identical requests submitted in
            # separate runs, especially when Google's status omits input_file_id.
            name = f"locus-{sha256_file(path)}-{uuid4().hex}"
            with db:
                db.execute(
                    "INSERT INTO batch_jobs VALUES (?, NULL, NULL, 'preparing', ?)",
                    (number, name),
                )
                db.executemany(
                    "UPDATE batch_requests SET job_number=? WHERE id=?",
                    ((number, key) for key in ids),
                )
            check_cancelled(stop_event)
            file_id = client.upload(path)
            # Uploading does not create an inference job. Keep this state safely
            # retryable if Ctrl-C arrived while the file was being transferred.
            with db:
                db.execute(
                    "UPDATE batch_jobs SET file_id=? WHERE number=?",
                    (file_id, number),
                )
            check_cancelled(stop_event)
            if not _submit_job(
                db, config, client, number, file_id, name, len(ids), stop_event
            ):
                return


def _write_annotation(db, source_id, index, digest, status, annotation) -> None:
    """Replace only a failed checkpoint, never a completed paid annotation."""
    cursor = db.execute(
        """INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            source_index=excluded.source_index, status=excluded.status,
            annotation_json=excluded.annotation_json, created_at_utc=excluded.created_at_utc
        WHERE annotations.status='failed' AND annotations.source_sha256=excluded.source_sha256""",
        (source_id, index, digest, status, canonical_json(annotation), utc_now()),
    )
    if cursor.rowcount != 1:
        raise InputError("Refusing to replace a completed annotation or changed source")


def _record_batch_failures(db, config, number, remote_id) -> None:
    """Make native failures visible to the same strict exporter as Jev failures."""
    for source_id, index, digest, encoded_info, error in db.execute(
        "SELECT source_id, source_index, source_sha256, source_info, error "
        "FROM batch_requests WHERE job_number=? AND result_json IS NULL AND error IS NOT NULL",
        (number,),
    ):
        try:
            details = decode_json(error)
        except ValueError:
            details = {"message": error}
        if not isinstance(details, dict):
            details = {"message": "Native batch request failed"}
        annotation = {
            "prompt_version": POLICY_VERSION,
            "status": "failed",
            "source": {
                "id": source_id,
                "sha256": digest,
                "locator": json.loads(encoded_info)["locator"],
            },
            "error": {"type": "ProviderBatchError", **details},
            "run_details": {
                "provider": config["provider"],
                "request_submission": "batch",
                "calls_per_record": 1,
                "requested_model": config["model"],
                "batch_id": remote_id,
            },
        }
        _write_annotation(db, source_id, index, digest, "failed", annotation)


def collect_results(
    ledger: AnnotationLedger, config: dict, client, *, stop_event=None, cleanup=False
) -> None:
    """Atomically collect each shard; join by ID, never by returned order."""
    db = ledger.connection
    if cleanup:
        cleanup_remote_files(ledger, config, client, stop_event=stop_event)
    builder = db.execute(
        "SELECT value FROM metadata WHERE key='request_builder_sha256'"
    ).fetchone()
    jobs = db.execute(
        "SELECT number, remote_id FROM batch_jobs WHERE state='submitted'"
    ).fetchall()
    for number, remote_id in jobs:
        check_cancelled(stop_event)
        status = client.status(remote_id)
        print(
            json.dumps(
                {"batch_number": number, "id": remote_id, "status": status["status"]}
            ),
            flush=True,
        )
        if not status["done"]:
            continue
        with db, ExitStack() as streams:
            seen: set[str] = set()  # Bounded by one shard, never by corpus size.
            for file_id in dict.fromkeys(
                status["output_files"] + status["error_files"]
            ):
                # Explicitly close the HTTP/file generator on parse or SQL failure.
                results = streams.enter_context(closing(client.iter_results(file_id)))
                for raw in results:
                    key, value, metadata, error = adapter(
                        config["provider"]
                    ).parse_result(raw)
                    expected = db.execute(
                        """SELECT source_id, source_index, source_sha256, request_sha256, source_info
                        FROM batch_requests WHERE id=? AND job_number=?""",
                        (key, number),
                    ).fetchone()
                    if expected is None or key in seen:
                        raise InputError(
                            "Batch results contain an unknown or duplicate request ID"
                        )
                    seen.add(key)
                    result = None
                    if error is not None:
                        if isinstance(raw.get("error"), dict):
                            error = canonical_json(provider_error_details(raw["error"]))
                        else:
                            error = canonical_json(
                                {
                                    "message": "Provider returned an incomplete or invalid classification response"
                                }
                            )
                    if error is None:
                        try:
                            classification = parse_classification(value)
                            if (
                                not isinstance(metadata.get("response_model"), str)
                                or not metadata["response_model"].strip()
                            ):
                                raise ContractError(
                                    "Provider response omitted its model identifier"
                                )
                            metadata.update(
                                {
                                    "request_sha256": expected[3],
                                    "response_sha256": request_digest(raw),
                                    "batch_id": remote_id,
                                }
                            )
                            result = canonical_json(
                                {"classification": classification, "metadata": metadata}
                            )
                        except ContractError:
                            error = canonical_json(
                                {
                                    "message": "Provider classification does not satisfy the annotation schema"
                                }
                            )
                    db.execute(
                        "UPDATE batch_requests SET result_json=?, error=? WHERE id=?",
                        (result, error, key),
                    )
                    if result is not None:
                        source_id, index, digest, _, encoded_info = expected
                        info = json.loads(encoded_info)
                        length = info["text_characters"]
                        sent = (
                            min(length, config["max_text_chars"])
                            if config["max_text_chars"]
                            else length
                        )
                        annotation = {
                            "prompt_version": POLICY_VERSION,
                            "source": {
                                "id": source_id,
                                "sha256": digest,
                                "locator": info["locator"],
                            },
                            "final": classification,
                            "final_metadata": metadata,
                            "run_details": {
                                "provider": config["provider"],
                                "request_submission": "batch",
                                "calls_per_record": 1,
                                "requested_model": config["model"],
                                "reasoning_effort": config["reasoning_effort"],
                                "contract_sha256": config["contract_sha256"],
                                "request_builder_sha256": (
                                    builder[0] if builder else None
                                ),
                                "annotated_at_utc": utc_now(),
                                "text_truncated": sent != length,
                                "input_text_characters": length,
                                "sent_text_characters": sent,
                            },
                        }
                        _write_annotation(
                            db, source_id, index, digest, "annotated", annotation
                        )
            db.execute(
                "UPDATE batch_requests SET error=? WHERE job_number=? AND result_json IS NULL AND error IS NULL",
                (f"No result returned; batch status {status['status']}", number),
            )
            _record_batch_failures(db, config, number, remote_id)
            if config["provider"] == "google":
                input_file = db.execute(
                    "SELECT file_id FROM batch_jobs WHERE number=?", (number,)
                ).fetchone()[0]
                owned_files = list(
                    dict.fromkeys(
                        file_id
                        for file_id in (
                            [input_file] + status["output_files"] + status["error_files"]
                        )
                        if isinstance(file_id, str) and file_id
                    )
                )
                db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                    (
                        f"batch_cleanup:{number}",
                        canonical_json({"pending": owned_files, "deleted": []}),
                    ),
                )
            db.execute(
                "UPDATE batch_jobs SET state='collected' WHERE number=?", (number,)
            )
        if cleanup:
            cleanup_remote_files(ledger, config, client, stop_event=stop_event)


def cleanup_remote_files(ledger, config, client, *, stop_event=None) -> None:
    """Delete only this ledger's durably collected Google files, one checkpoint at a time."""
    if config["provider"] != "google":
        raise InputError("Remote file cleanup is available only for Google")
    db = ledger.connection
    schedules = db.execute(
        "SELECT key, value FROM metadata JOIN batch_jobs "
        "ON key='batch_cleanup:' || number WHERE state='collected' ORDER BY number"
    ).fetchall()
    for key, encoded in schedules:
        schedule = json.loads(encoded)
        while schedule["pending"]:
            check_cancelled(stop_event)
            file_id = schedule["pending"][0]
            deleted = client.delete_file(file_id)
            schedule["pending"].pop(0)
            outcome = "skipped" if deleted is False else "deleted"
            schedule.setdefault(outcome, []).append(file_id)
            with db:
                db.execute(
                    "UPDATE metadata SET value=? WHERE key=?",
                    (canonical_json(schedule), key),
                )


def attach(ledger: AnnotationLedger, args, config: dict, client) -> None:
    """Recover a remote ID only after checking this shard's unique identity."""
    if not args.batch_number or not args.batch_id:
        raise InputError("Recovery requires --batch-number and --batch-id")
    row = ledger.connection.execute(
        "SELECT file_id, display_name, state FROM batch_jobs WHERE number=?",
        (args.batch_number,),
    ).fetchone()
    if row is None or row[2] != "submitting":
        raise InputError("Only an uncertain submitting batch can be attached")
    status = client.status(args.batch_id)
    if status.get("display_name") != row[1]:
        raise InputError("Remote batch display name does not match this submission")
    if config["provider"] == "openai" and status.get("input_file_id") != row[0]:
        raise InputError("Remote batch input file does not match")
    if config["provider"] == "google" and str(status.get("model", "")).removeprefix(
        "models/"
    ) != config["model"].removeprefix("models/"):
        raise InputError("Remote batch model does not match")
    with ledger.connection:
        ledger.connection.execute(
            "UPDATE batch_jobs SET remote_id=?, state='submitted' WHERE number=?",
            (args.batch_id, args.batch_number),
        )


def reset_uncertain(ledger: AnnotationLedger, number: int | None) -> None:
    """Release an uncertain shard only on an explicit, operator-checked reset.

    The old submission record remains marked abandoned. This action cannot
    determine whether the provider accepted a job; it never runs automatically.
    A subsequent submission can duplicate charges if the operator was wrong.
    """
    if number is None:
        raise InputError(
            "reset-uncertain requires --batch-number after confirming no remote job exists"
        )
    db = ledger.connection
    with db:
        row = db.execute(
            "SELECT state FROM batch_jobs WHERE number=?", (number,)
        ).fetchone()
        if row != ("submitting",):
            raise InputError("Only an uncertain submitting batch can be reset")
        db.execute(
            "UPDATE batch_requests SET job_number=NULL WHERE job_number=?", (number,)
        )
        db.execute("UPDATE batch_jobs SET state='abandoned' WHERE number=?", (number,))


def run(args, *, stop_event=None, expected_input_sha256: str | None = None) -> int:
    """Entry point selected by the shared --provider switch."""
    check_cancelled(stop_event)
    cleanup = getattr(args, "cleanup_remote_files", False)
    if cleanup and args.provider != "google":
        raise InputError("--cleanup-remote-files is available only for Google")
    if args.batch_size > MAX_BATCH_ROWS:
        raise InputError(
            f"--batch-size must not exceed {MAX_BATCH_ROWS} (bounded result-download memory)"
        )
    if args.dry_run and args.export_only:
        raise InputError("--dry-run and --export-only cannot be used together")
    if args.workers != DEFAULT_WORKERS or args.max_attempts != DEFAULT_MAX_ATTEMPTS:
        raise InputError(
            "--workers and --max-attempts apply to Typesafe; native batch concurrency uses --max-inflight-batches"
        )
    if args.endpoint != DEFAULT_ENDPOINT:
        raise InputError(
            "--endpoint is a Typesafe-only option; batch SDKs use official endpoints"
        )
    if args.dry_run:
        config = configuration(args, hash_input=False)
        for index, record in iter_jsonl(args.input):
            source_id, _ = source_identity(record, index)
            if (
                is_header_only_record(record)
                and config["header_only_policy"] == "exclude"
            ):
                continue
            request_for(record, source_id, config)
            preview = request_for(
                {"content": "<redacted legal text>"}, source_id, config
            )
            print(
                json.dumps(
                    {"provider": args.provider, "config": config, "request": preview},
                    indent=2,
                )
            )
            return 0
        raise InputError("No eligible rows found")
    distinct_paths(args.input, args.ledger, args.output)
    if args.export_only:
        if args.output is None:
            raise InputError("--export-only requires --output")
        print(
            json.dumps(
                {
                    "export": export_annotations_jsonl(
                        input_path=args.input,
                        ledger_path=args.ledger,
                        output_path=args.output,
                        include_excluded_header_only=args.include_excluded_header_only,
                        allow_partial=args.allow_partial_export,
                        overwrite=args.overwrite_output,
                        annotation_field=f"classification_{args.provider}",
                    )
                }
            )
        )
        return 0
    # Collection uses immutable prepared source metadata, not the corpus. It
    # remains possible while a large external input drive is unmounted.
    before = (
        file_stamp(args.input)
        if args.batch_action not in {"status", "attach", "collect", "reset-uncertain"}
        else None
    )
    config = configuration(args)
    if (
        expected_input_sha256 is not None
        and config["input_sha256"] != expected_input_sha256
    ):
        raise InputError("Input changed since preflight")
    check_cancelled(stop_event)
    action = args.batch_action or "submit"
    existed = args.ledger.exists()
    if not existed and action not in {"prepare", "submit"}:
        raise InputError("Prepare or submit a run before checking or collecting it")
    if not existed and args.max_records is None and not args.allow_full_run:
        raise InputError("New batches require --max-records N or --allow-full-run")
    if args.output and args.output.exists() and not args.overwrite_output:
        raise InputError("Output exists; use --overwrite-output deliberately")
    with closing(acquire_annotation_lock(args.ledger)):
        with closing(
            AnnotationLedger(args.ledger, run_metadata=config, resume=args.resume)
        ) as ledger:
            if ledger.connection.execute(
                "SELECT value FROM metadata WHERE key='batch_initialized'"
            ).fetchone() != ("1",):
                initialize(ledger, args.input, config, stop_event=stop_event)
            if before is not None:
                if file_stamp(args.input) != before:
                    raise InputError(
                        "Input changed during preparation; no requests submitted"
                    )
                with ledger.connection:
                    ledger.connection.execute(
                        "INSERT OR REPLACE INTO metadata VALUES ('input_stamp', ?)",
                        (json.dumps(before),),
                    )
            if action == "reset-uncertain":
                reset_uncertain(ledger, args.batch_number)
            elif action != "prepare":
                check_cancelled(stop_event)
                load_dotenv(args.env_file)
                key = api_key(args.provider, args.api_key_env)
                if action == "attach" and (not args.batch_number or not args.batch_id):
                    raise InputError("Recovery requires --batch-number and --batch-id")
                operation_row = ledger.connection.execute(
                    "SELECT value FROM metadata WHERE key='batch_operation_retry'"
                ).fetchone()
                operation_retry = json.loads(operation_row[0]) if operation_row else None
                if operation_retry and time.time() < operation_retry["retry_at"]:
                    uncertain = ledger.connection.execute(
                        "SELECT number FROM batch_jobs WHERE state='submitting' LIMIT 1"
                    ).fetchone()
                    if uncertain is not None:
                        raise InputError(
                            f"Batch {uncertain[0]} has an uncertain submission; "
                            "attach its remote ID before continuing"
                        )
                    print(
                        json.dumps(
                            {
                                "provider": args.provider,
                                "phase": "operation_wait",
                                "action": action,
                                "retry_at_utc": _retry_at_utc(operation_retry["retry_at"]),
                                "error": operation_retry["error"],
                            }
                        ),
                        flush=True,
                    )
                    return RETRYABLE_OPERATION_EXIT_CODE
                client_class = (
                    openai_batch.OpenAIBatchClient
                    if args.provider == "openai"
                    else google_batch.GoogleBatchClient
                )
                try:
                    client = client_class(key, timeout=args.timeout)
                except Exception:
                    raise InputError(
                        f"Cannot initialize {args.provider}; install its SDK extra and check local key configuration"
                    ) from None
                with closing(client):
                    try:
                        if action in {"retry", "submit"}:
                            if action == "retry":
                                with ledger.connection:
                                    reset = ledger.connection.execute(
                                        "UPDATE batch_requests SET error=NULL, job_number=NULL WHERE error IS NOT NULL"
                                    )
                                    if reset.rowcount:
                                        passes = _row_retry_passes(ledger.connection) + 1
                                        ledger.connection.execute(
                                            "INSERT OR REPLACE INTO metadata VALUES "
                                            "('row_retry_passes', ?)",
                                            (str(passes),),
                                        )
                            submit_ready(
                                ledger,
                                args.input,
                                config,
                                client,
                                batch_size=args.batch_size,
                                max_inflight=args.max_inflight_batches,
                                stop_event=stop_event,
                            )
                        elif action == "collect":
                            retry_provider_read(
                                lambda: collect_results(
                                    ledger,
                                    config,
                                    client,
                                    stop_event=stop_event,
                                    cleanup=cleanup,
                                ),
                                provider=args.provider,
                                stop_event=stop_event,
                            )
                        elif action == "attach":
                            retry_provider_read(
                                lambda: attach(ledger, args, config, client),
                                provider=args.provider,
                                stop_event=stop_event,
                            )
                        else:
                            for number, remote_id in ledger.connection.execute(
                                "SELECT number, remote_id FROM batch_jobs WHERE state='submitted'"
                            ):
                                status = retry_provider_read(
                                    lambda: client.status(remote_id),
                                    provider=args.provider,
                                    stop_event=stop_event,
                                )
                                print(
                                    json.dumps(
                                        {
                                            "batch_number": number,
                                            "id": remote_id,
                                            "status": status["status"],
                                        }
                                    )
                                )
                    except (InputError, ContractError, CancelledError):
                        raise
                    except Exception as exc:
                        if isinstance(exc, JevError) and not isinstance(
                            exc, RetryDelayExceeded
                        ):
                            raise
                        provider_error = (
                            exc.__context__
                            if isinstance(exc, RetryDelayExceeded)
                            and exc.__context__ is not None
                            else exc
                        )
                        details = provider_error_details(provider_error)
                        uncertain = ledger.connection.execute(
                            "SELECT 1 FROM batch_jobs WHERE state='submitting' LIMIT 1"
                        ).fetchone()
                        if (
                            action in {"collect", "status", "attach", "submit", "retry"}
                            and uncertain is None
                            and _transient_operation_error(provider_error)
                        ):
                            attempts = operation_retry["attempts"] + 1 if operation_retry else 1
                            retry_at = time.time() + _quota_delay(provider_error, attempts)
                            with ledger.connection:
                                ledger.connection.execute(
                                    "INSERT OR REPLACE INTO metadata VALUES ('batch_operation_retry', ?)",
                                    (
                                        canonical_json(
                                            {
                                                "retry_at": retry_at,
                                                "attempts": attempts,
                                                "action": action,
                                                "error": details,
                                            }
                                        ),
                                    ),
                                )
                            print(
                                json.dumps(
                                    {
                                        "provider": args.provider,
                                        "phase": "retryable_operation_failure",
                                        "action": action,
                                        "retry_at_utc": _retry_at_utc(retry_at),
                                        "error": details,
                                    }
                                ),
                                flush=True,
                            )
                            return RETRYABLE_OPERATION_EXIT_CODE
                        raise InputError(
                            f"{args.provider} batch operation failed: {canonical_json(details)}; "
                            "check provider status before retrying"
                        ) from None
                    else:
                        with ledger.connection:
                            ledger.connection.execute(
                                "DELETE FROM metadata WHERE key='batch_operation_retry'"
                            )
            progress = summary(ledger.connection)
            print(json.dumps(progress, sort_keys=True))
    check_cancelled(stop_event)
    complete = progress["annotated"] == progress["requests"] and not progress["errors"]
    if args.output and (complete or args.allow_partial_export):
        print(
            json.dumps(
                {
                    "export": export_annotations_jsonl(
                        input_path=args.input,
                        ledger_path=args.ledger,
                        output_path=args.output,
                        include_excluded_header_only=args.include_excluded_header_only,
                        allow_partial=args.allow_partial_export,
                        overwrite=args.overwrite_output,
                        annotation_field=f"classification_{args.provider}",
                    )
                }
            )
        )
    elif args.output:
        print(
            json.dumps(
                {
                    "export_deferred": str(args.output),
                    "reason": "Annotations are incomplete; collect remaining results before export",
                    "annotated": progress["annotated"],
                    "requests": progress["requests"],
                    "failed": progress["errors"],
                }
            ),
            flush=True,
        )
    return ROW_ERRORS_EXIT_CODE if progress["errors"] else 0
