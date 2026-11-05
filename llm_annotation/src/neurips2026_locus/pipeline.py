"""Shared CLI, input/ledger/export utilities, and the resumable Jev runner."""

from __future__ import annotations

import argparse
from concurrent.futures import (
    CancelledError,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
from threading import Event, current_thread, main_thread
import time
from typing import Any, Iterator, Mapping

from .contract import (
    PROMPT_VERSION,
    prompt_version_of,
    normalize_run_metadata,
    run_metadata_matches,
    PROVIDER_ANNOTATION_FIELDS,
    ContractError,
    compose_input_text,
    contract_sha256,
    request_builder_sha256,
    decode_json,
    is_header_only_record,
    parse_annotation,
)
from .jev import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    EnvironmentError,
    JevClient,
    JevError,
    JevHTTPError,
    JevResponse,
    JevResponseError,
    RetryDelayExceeded,
    get_api_key,
    load_dotenv,
)
from .files import (
    atomic_text_output,
    canonical_document_locator,
    file_stamp,
    opaque_document_id,
    project_root,
)

from .municipal_law_annotation.municipal_law_jev_adapter import (
    build_jev_request,
    build_jev_confidence_request,
    normalize_jev_annotation,
)

LEDGER_SCHEMA_VERSION = 1
DEFAULT_WORKERS = 4
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BATCH_SIZE = 5000
DEFAULT_MAX_INFLIGHT = 1
ROW_ERRORS_EXIT_CODE = 3
# Stop scheduling if rejections suggest a shared configuration problem.
# Already running workers still finish and checkpoint their outcomes.
MAX_CONSECUTIVE_REJECTIONS = 8


class InputError(ValueError):
    """The input or local run state cannot safely be used."""


@contextmanager
def cancellation_scope(stop_event=None):
    """Turn CLI Ctrl-C into cooperative cancellation; restore the caller's handler.

    An outer entry point can supply its event without installing another handler.
    Library calls from worker threads leave process signal handling to the caller.
    """
    stop = stop_event if stop_event is not None else Event()
    install_handler = stop_event is None and current_thread() is main_thread()
    previous = signal.getsignal(signal.SIGINT) if install_handler else None
    if install_handler:
        signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        yield stop
    finally:
        if install_handler:
            signal.signal(signal.SIGINT, previous)


def check_cancelled(stop_event):
    """Stop before starting another operation, without recording a row failure."""
    if stop_event is not None and stop_event.is_set():
        raise CancelledError()


def utc_now() -> str:
    """Record UTC timestamps independently of the computer's local timezone."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize hashes and ledger objects without non-JSON NaN or infinity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    """Hash raw input bytes so a resumed ledger cannot mix dataset versions."""

    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield every nonblank JSON object with its zero-based physical-row index."""
    with closing(indexed_jsonl(path)) as source:
        for index, _, record in source:
            yield index, record


def indexed_jsonl(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Yield byte offsets as well, so later batch shards can seek directly."""

    if not path.is_file():
        raise InputError(f"Input JSONL does not exist: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        source_index = -1
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            source_index += 1
            if not line.strip():
                continue
            try:
                record = decode_json(line.decode("utf-8"))
            except ValueError as exc:
                raise InputError(
                    f"Invalid JSON on source line {source_index + 1} of {path}"
                ) from exc
            if not isinstance(record, dict):
                raise InputError(
                    f"Expected an object on source line {source_index + 1} of {path}"
                )
            yield source_index, offset, record


def source_locator(record: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    """Retain the private hash/ledger locator without changing resume identity.

    Exported records use ``public_source_locator`` instead.
    """

    locator: dict[str, Any] = {"input_row_index": source_index}
    for key in (
        "document_id",
        "audit_id",
        "source_file",
        "source_line",
        "source_index",
        "custom_id",
    ):
        if key in record and record[key] is not None:
            locator[key] = record[key]
    return locator


def source_identity(record: Mapping[str, Any], source_index: int) -> tuple[str, str]:
    """Produce a stable identity and immutable content digest for one input row."""

    if "content" not in record:
        raise InputError("Each input record must contain a `content` field")
    for field in ("header", "content"):
        if record.get(field) is not None and not isinstance(record[field], str):
            raise InputError(
                f"Source row {source_index}: `{field}` must be text or null"
            )
    for field in PROVIDER_ANNOTATION_FIELDS.values():
        if field in record:
            raise InputError(f"Input already contains {field}; use the original input")
    basis = {
        "locator": source_locator(record, source_index),
        "header": record.get("header"),
        "content": record.get("content"),
    }
    digest = sha256(canonical_json(basis).encode("utf-8")).hexdigest()
    return f"locus-v1:{digest}", digest


_SOURCE_PATH_FIELDS = {
    "source_file",
    "source_path",
    "source_pdf",
    "file_path",
    "pdf_path",
    "corpus_path",
    "input_path",
    "original_path",
}
_PUBLIC_ID_FIELDS = {"document_id", "custom_id", "source_record_id", "audit_id"}


def _public_document_collection(record: Mapping[str, Any]) -> str:
    """Normalize a public collection alias without changing the private input."""
    for field in ("collection", "origin"):
        value = record.get(field)
        if value == "original":
            return "original"
        if isinstance(value, str) and value in {"additional", "ocr_addition"}:
            return "ocr_addition"
    locator = _source_document_path(record) or record.get("document_id")
    if isinstance(locator, str) and re.fullmatch(
        r"(?:new|fixes)/[^/]+/[a-zA-Z]{2}/(?:charters/)?[^/]+\.pdf",
        canonical_document_locator(locator),
    ):
        return "ocr_addition"
    return "original"


def _is_locator_metadata_key(key: str) -> bool:
    singular = key[:-1] if key.endswith("s") else key
    return singular in {"file", "path", "filename", "file_name"} or singular.endswith(
        ("_file", "_path", "_filename", "_file_name")
    )


def _is_web_url(value: Any) -> bool:
    return isinstance(value, str) and bool(re.match(r"https?://", value, re.I))


def _is_absolute_local_path(value: Any) -> bool:
    """Recognize explicit POSIX, Windows, home-relative, and file-URI locators."""
    return isinstance(value, str) and bool(
        re.match(r"(?:[/\\]|[A-Za-z]:[/\\]|~[/\\]|file:)", value, re.I)
    )


def _is_source_path(value: Any) -> bool:
    if not isinstance(value, str) or _is_web_url(value):
        return False
    return bool(
        "/" in value
        or "\\" in value
        or re.search(
            r"\.(?:jsonl?|pdf|txt|csv|xlsx?|xml|md|docx?|zip)(?:$|[#:])", value, re.I
        )
    )


def _source_document_path(record: Mapping[str, Any]) -> str | None:
    """Use the sanitizer's existing precedence for one collected locator."""
    for field in (
        "source_file",
        "source_path",
        "corpus_path",
        "pdf_path",
        "source_pdf",
    ):
        value = record.get(field)
        if isinstance(value, str) and value and not _is_web_url(value):
            return value
    for field, value in record.items():
        if field in _SOURCE_PATH_FIELDS:
            if isinstance(value, str) and value and not _is_web_url(value):
                return value
        elif _is_locator_metadata_key(field) and _is_source_path(value):
            return value
    return None


def public_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a public export without collected-file paths or path-bearing IDs.

    This is applied only at export time. Original inputs, source identity hashes,
    ledger rows, request bodies, legal text, and classification fields are not
    modified. It is locator minimization, not redaction of arbitrary free text.
    Locator lists keep their order and structure, with paths replaced by IDs.
    """
    collection = _public_document_collection(record)

    def document_id(value: str) -> str:
        if re.fullmatch(r"doc_[0-9a-f]{24}", value):
            return value
        return opaque_document_id(canonical_document_locator(value), collection)

    def clean(value: Any, *, locator_values: bool = False) -> Any:
        if isinstance(value, list):
            return [clean(item, locator_values=locator_values) for item in value]
        if not isinstance(value, Mapping):
            if _is_absolute_local_path(value) or (
                locator_values and _is_source_path(value)
            ):
                return document_id(value)
            return value
        if locator_values or any(_is_absolute_local_path(key) for key in value):
            keyed_values: dict[str, Any] = {}
            for key, item in value.items():
                public_key = (
                    document_id(key)
                    if _is_absolute_local_path(key)
                    or (locator_values and _is_source_path(key))
                    else key
                )
                if public_key in keyed_values:
                    raise InputError(
                        "Public locator metadata has colliding document-ID keys"
                    )
                keyed_values[public_key] = item
            value = keyed_values
        result: dict[str, Any] = {}
        source_document = _source_document_path(value)
        for key, item in value.items():
            # A heading hierarchy is legal text, including path-like headings.
            if key in {
                "header",
                "heading_path",
                "content",
                "classification",
            } or key.startswith("classification_"):
                result[key] = item
            elif key in {"source_document_slug", "source_document_slugs"}:
                continue
            elif re.search(
                r"(?:^|_)(?:dirs?|director(?:y|ies)|folders?|roots?)$", key, re.I
            ):
                # A run directory is execution metadata, not a source document.
                # Do not derive document_id from review_run_dir or similar keys.
                if _is_web_url(item):
                    result[key] = item
            elif key in _SOURCE_PATH_FIELDS:
                if isinstance(item, str) and item and not _is_web_url(item):
                    if source_document is None:
                        source_document = item
                else:
                    result[key] = clean(item, locator_values=True)
            elif key in _PUBLIC_ID_FIELDS and _is_source_path(item):
                result[key] = document_id(item)
            elif key in _PUBLIC_ID_FIELDS or (
                key.endswith("s") and key[:-1] in _PUBLIC_ID_FIELDS
            ):
                result[key] = clean(item, locator_values=True)
            elif _is_locator_metadata_key(key):
                if _is_source_path(item):
                    if source_document is None:
                        source_document = item
                else:
                    # Lists and keyed collections retain their structure and
                    # order, with each collected locator replaced by its ID.
                    result[key] = clean(item, locator_values=True)
            else:
                result[key] = clean(item, locator_values=locator_values)
        if source_document is not None:
            # Retain an existing non-path document identifier. Path-bearing
            # identifiers derive from the canonical source when one is known.
            original_id = value.get("document_id")
            if not original_id or _is_source_path(original_id):
                result["document_id"] = document_id(source_document)
            else:
                result.setdefault("document_id", original_id)
        return result

    return clean(record)


def public_source_locator(
    record: Mapping[str, Any], source_index: int
) -> dict[str, Any]:
    """Publish a minimized locator while leaving the private hash basis intact."""
    locator = source_locator(record, source_index)
    collection = _public_document_collection(record)
    if collection == "ocr_addition" or any(
        isinstance(record.get(field), str)
        and record[field] in {"original", "additional", "ocr_addition"}
        for field in ("collection", "origin")
    ):
        locator["collection"] = collection
    return public_source_record(locator)


def _public_annotation(
    annotation: Mapping[str, Any], record: Mapping[str, Any], source_index: int
) -> dict[str, Any]:
    result = dict(annotation)
    version = prompt_version_of(result)
    if version is not None:
        result.pop("contract_version", None)
        result["prompt_version"] = version
    details = []
    for name in ("provenance", "run_details"):
        if name not in annotation:
            continue
        value = annotation[name]
        if not isinstance(value, Mapping):
            raise InputError("Annotation run details must be an object")
        try:
            details.append(normalize_run_metadata(value))
        except ContractError as exc:
            raise InputError("Annotation run details do not match its protocol") from exc
    if len(details) == 2 and canonical_json(details[0]) != canonical_json(details[1]):
        raise InputError("Annotation has conflicting provenance and run_details")
    if details:
        result.pop("provenance", None)
        result["run_details"] = details[-1]
    if isinstance(annotation.get("source"), Mapping):
        result["source"] = dict(annotation["source"])
        if "locator" in annotation["source"]:
            result["source"]["locator"] = public_source_locator(record, source_index)
    return result


def limit_text(text: str, maximum: int | None) -> tuple[str, bool]:
    """Apply an explicit opt-in character cap and record that it changed input."""

    if maximum is None or len(text) <= maximum:
        return text, False
    return text[:maximum], True


def model_id(value: str) -> str:
    """Reject blank or malformed model overrides instead of silently defaulting."""
    if not value or any(
        ord(character) < 33 or ord(character) > 126 for character in value
    ):
        raise InputError("Model ID must be nonempty printable ASCII without whitespace")
    return value


def annotate_record(
    *,
    record: Mapping[str, Any],
    source_index: int,
    source_id: str,
    source_sha256: str,
    client: JevClient,
    model: str,
    max_text_chars: int | None,
    ledger_path: Path | None = None,
    stop_event=None,
) -> dict[str, Any]:
    """Classify, then obtain rubric confidence; checkpoint each validated response."""
    text = compose_input_text(record)
    if not text:
        raise InputError(
            f"Input row {source_index} has neither usable header nor content"
        )
    limited_text, text_truncated = limit_text(text, max_text_chars)
    connection: sqlite3.Connection | None = None

    def send(request: dict, validate):
        """Reuse one row-local connection while committing each response stage."""
        request_hash = sha256(canonical_json(request).encode("utf-8")).hexdigest()
        cached = None
        if connection is not None:
            cached = connection.execute(
                "SELECT response_json, response_sha256 FROM responses WHERE source_id=? AND request_sha256=?",
                (source_id, request_hash),
            ).fetchone()
        check_cancelled(stop_event)
        response = (
            JevResponse(decode_json(cached[0]), request_hash, cached[1])
            if cached
            else client.decide(request)
        )
        if response.request_sha256 != request_hash:
            raise InputError(
                "Jev transport request hash differs from the constructed request"
            )
        actual_model = response.body.get("model")
        requested_model = request["model"]
        if not isinstance(actual_model, str) or not actual_model.strip():
            raise InputError("Jev response omitted its model identifier")
        if actual_model in {"jev-latest", "jev-preview"}:
            raise InputError(
                "Jev response did not resolve its alias to a model version"
            )
        if (
            requested_model not in {"jev-latest", "jev-preview"}
            and actual_model != requested_model
        ):
            raise InputError(
                f"Requested model {requested_model!r}, but Jev returned {actual_model!r}"
            )
        try:
            validated = validate(response.body)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        if connection is not None and cached is None:
            connection.execute(
                "INSERT INTO responses VALUES (?, ?, ?, ?)",
                (
                    source_id,
                    request_hash,
                    canonical_json(response.body),
                    response.response_sha256,
                ),
            )
            connection.commit()
        metadata = {
            "request_sha256": request_hash,
            "response_sha256": response.response_sha256,
            "response_model": actual_model,
            "usage": response.body.get("usage"),
            "answers": response.body.get("answers"),
        }
        return response.body, validated, metadata

    try:
        # The worker owns this connection for both stages. FULL is explicit:
        # synchronous mode is connection-local and must not depend on a build
        # default. Each validated stage is committed before the next API call.
        if ledger_path is not None:
            connection = sqlite3.connect(ledger_path, timeout=60)
            connection.execute("PRAGMA synchronous=FULL")
        first, confidence_request, classification_metadata = send(
            build_jev_request(limited_text, model),
            # Pin the confidence stage to the model reported by the first stage.
            lambda body: build_jev_confidence_request(
                limited_text, body, body["model"]
            ),
        )
        _, final, confidence_metadata = send(
            confidence_request,
            lambda body: normalize_jev_annotation(first, body),
        )
        return {
            "prompt_version": PROMPT_VERSION,
            "source": {
                "id": source_id,
                "sha256": source_sha256,
                "locator": source_locator(record, source_index),
            },
            "final": final,
            "stage_metadata": {
                "classification": classification_metadata,
                "confidence": confidence_metadata,
            },
            "run_details": {
                "provider": "typesafe",
                "request_submission": "individual",
                "calls_per_record": 2,
                "annotated_at_utc": utc_now(),
                "endpoint": client.endpoint,
                "requested_model": model,
                "contract_sha256": contract_sha256(),
                "request_builder_sha256": request_builder_sha256(),
                "text_truncated": text_truncated,
                "input_text_characters": len(text),
                "sent_text_characters": len(limited_text),
            },
        }
    finally:
        if connection is not None:
            connection.close()


def ledger_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    """Validate a ledger before resuming it or using it as an export source."""

    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != str(LEDGER_SCHEMA_VERSION):
            raise InputError("Unsupported annotation ledger schema version")
        decoded = json.loads(metadata["run_metadata"])
        if (
            not isinstance(decoded, dict)
            or decoded.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION
        ):
            raise InputError("Invalid annotation ledger metadata")
        connection.execute(
            "SELECT source_id, source_index, source_sha256, status, annotation_json FROM annotations LIMIT 0"
        )
    except sqlite3.OperationalError as exc:
        raise InputError(f"Cannot read annotation ledger metadata: {exc}") from exc
    except (KeyError, json.JSONDecodeError, sqlite3.DatabaseError) as exc:
        raise InputError(
            "Invalid annotation ledger; use a ledger created by this program"
        ) from exc
    return decoded


def check_request_builders(connection: sqlite3.Connection) -> None:
    """Reject changed request/identity code, not comments or export-only changes.

    Older ledgers have no code fingerprint; keep them readable/resumable and
    leave their implementation provenance unknown. Do not silently backfill
    a new fingerprint onto work produced by an unrecorded implementation.
    """
    stored = connection.execute(
        "SELECT value FROM metadata WHERE key='request_builder_sha256'"
    ).fetchone()
    # Releases 0.8/0.9 used earlier fingerprints. This verified mapping accepts their
    # original requests without overwriting historical implementation provenance.
    legacy = {
        "fe48aee9b30b0b70d0bf7974b45fc549c2a05625fbb4e99eec2bf85d261252b6": "c5f726724fb131825ac61316f39aa43ba1e23a169a4aac6703f2bdb0eee4aec4",
        "7ad95a0934acce30a75a547b01e0f43a6ab14fc2bb82ef34a1b0a9fb8b7305d1": "c5f726724fb131825ac61316f39aa43ba1e23a169a4aac6703f2bdb0eee4aec4",
        "9379e067aa64204104dc2b817f2f80c4a0a774fa518b767b45773bfc3844522a": "c5f726724fb131825ac61316f39aa43ba1e23a169a4aac6703f2bdb0eee4aec4",
        "03690befa115d320881bc92a4c4474f42dd5768de1b35916ae5e8b167769c829": "c5f726724fb131825ac61316f39aa43ba1e23a169a4aac6703f2bdb0eee4aec4",
    }
    current = request_builder_sha256()
    if stored is not None and stored[0] != current and legacy.get(stored[0]) != current:
        raise InputError(
            "Request-building code changed since this ledger was created. "
            "Collect/export existing work without submitting, or use a new ledger/name."
        )


def distinct_paths(
    input_path: Path, ledger_path: Path, output_path: Path | None = None
) -> None:
    """Prevent an output or ledger from replacing its own source, including aliases."""

    lock_path = Path(str(ledger_path.resolve()) + ".lock.sqlite3")
    paths = [input_path, ledger_path, lock_path]
    if output_path is not None:
        paths.append(output_path)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path.resolve() == other.resolve() or (
                path.exists() and other.exists() and path.samefile(other)
            ):
                raise InputError(
                    "Input, ledger, lock, and output must refer to different files"
                )
    sidecars = {
        Path(str(database.resolve()) + suffix)
        for database in (ledger_path, lock_path)
        for suffix in ("-wal", "-shm", "-journal")
    }
    if input_path.resolve() in sidecars or (
        output_path is not None and output_path.resolve() in sidecars
    ):
        raise InputError("Input and output must not use a SQLite ledger sidecar path")


def acquire_annotation_lock(ledger_path: Path) -> sqlite3.Connection:
    """Allow one annotating process per ledger, with automatic release after a crash.

    A separate SQLite transaction supplies portable process locking without
    blocking the worker threads' checkpoint writes to the annotation ledger.
    The lock file stays in place; only the live transaction holds the lock.
    """

    ledger_path = ledger_path.resolve()
    if ledger_path.exists() and ledger_path.stat().st_nlink > 1:
        raise InputError(
            "Hard-linked ledgers cannot be locked safely; use a single ledger path"
        )
    lock_path = Path(str(ledger_path) + ".lock.sqlite3")
    if lock_path.is_symlink() or (lock_path.exists() and lock_path.stat().st_nlink > 1):
        raise InputError("The annotation lock file must not be a symbolic or hard link")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(lock_path, timeout=0)
    try:
        connection.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as exc:
        connection.close()
        if "locked" in str(exc).lower():
            raise InputError(
                f"Ledger is already in use by another annotation run: {ledger_path}"
            ) from exc
        raise
    except BaseException:
        connection.close()
        raise
    return connection


class AnnotationLedger:
    """SQLite transaction log keyed by immutable source-row identity.

    A committed ledger row is the source of truth. JSONL export is a separate,
    deterministic atomic step, so an interruption cannot duplicate labels or
    silently mix a changed input/configuration into a run.
    """

    def __init__(
        self, path: Path, *, run_metadata: Mapping[str, Any], resume: bool
    ) -> None:
        """Create a new ledger or require an exact semantic configuration match."""
        self.path = path
        self.run_metadata = dict(run_metadata)
        existed = path.exists()
        if existed and not resume:
            raise InputError(
                f"Ledger already exists: {path}. Pass --resume to continue it."
            )
        if resume and not existed:
            raise InputError(f"Cannot resume because the ledger does not exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        try:
            if existed:
                existing = ledger_metadata(self.connection)
                if not run_metadata_matches(existing, run_metadata):
                    raise InputError(
                        "Ledger metadata does not match this input or annotation configuration; "
                        "start a new ledger rather than mixing runs."
                    )
            self._initialize_schema(run_metadata, existed=existed)
        except BaseException:
            self.close()
            raise

    def _initialize_schema(
        self, run_metadata: Mapping[str, Any], *, existed: bool
    ) -> None:
        """Create/check durable tables; the constructor closes any failed setup."""
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                source_id TEXT PRIMARY KEY,
                source_index INTEGER NOT NULL,
                source_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                annotation_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            )
            """)
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS annotation_status ON annotations(status)"
        )
        # Optional in schema-1 ledgers created before request checkpointing.
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                source_id TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                response_json TEXT NOT NULL,
                response_sha256 TEXT NOT NULL,
                PRIMARY KEY (source_id, request_sha256)
            )
            """)
        if not existed:
            self.connection.execute(
                "INSERT INTO metadata(key,value) VALUES ('request_builder_sha256',?)",
                (request_builder_sha256(),),
            )
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(LEDGER_SCHEMA_VERSION),),
            )
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('run_metadata', ?)",
                (canonical_json(dict(run_metadata)),),
            )
        self.connection.commit()

    def close(self) -> None:
        """Release the database; a SQLite transaction context alone does not close it."""
        self.connection.close()

    @classmethod
    def open_readonly(cls, path: Path) -> "AnnotationLedger":
        """Open existing state without creating or upgrading a database."""
        ledger = cls.__new__(cls)
        ledger.path = path
        ledger.connection = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True
        )
        try:
            ledger.run_metadata = ledger_metadata(ledger.connection)
        except BaseException:
            ledger.close()
            raise
        return ledger

    def existing_status(self, source_id: str, source_sha256: str) -> str | None:
        """Distinguish completed, excluded and failed rows without loading JSON."""
        row = self.connection.execute(
            "SELECT source_sha256, status FROM annotations WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != source_sha256:
            raise InputError(
                "A source-id collision was detected in the annotation ledger"
            )
        if row[1] not in {"annotated", "header_only_excluded", "failed"}:
            raise InputError("Ledger contains an unrecognized annotation status")
        return row[1]

    def insert(
        self,
        *,
        source_id: str,
        source_index: int,
        source_sha256: str,
        status: str,
        annotation: Mapping[str, Any],
        replace_failed: bool = False,
    ) -> None:
        """Commit one final annotation or explicit historical exclusion."""
        if replace_failed:
            # Leave the old failure intact until the retry has an outcome.
            # DELETE and INSERT commit together; successful rows are protected.
            self.connection.execute(
                "DELETE FROM annotations WHERE source_id=? AND status='failed'",
                (source_id,),
            )
        self.connection.execute(
            """
            INSERT INTO annotations(source_id, source_index, source_sha256, status, annotation_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                source_index,
                source_sha256,
                status,
                canonical_json(dict(annotation)),
                utc_now(),
            ),
        )
        self.connection.commit()

    def annotation(
        self, source_id: str, source_sha256: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Read a checkpoint and revalidate current annotations before publication."""
        row = self.connection.execute(
            "SELECT source_sha256, status, annotation_json FROM annotations WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != source_sha256:
            raise InputError("A source-id collision was detected while exporting")
        if row[1] not in {"annotated", "header_only_excluded", "failed"}:
            raise InputError("Ledger contains an unrecognized annotation status")
        try:
            decoded = decode_json(row[2])
        except ValueError as exc:
            raise InputError("Ledger contains invalid annotation JSON") from exc
        if not isinstance(decoded, dict):
            raise InputError("Ledger has an invalid annotation JSON object")
        try:
            version = prompt_version_of(decoded)
            expected_version = prompt_version_of(self.run_metadata)
        except ContractError as exc:
            raise InputError("Ledger annotation has an invalid prompt version") from exc
        current_run = expected_version is not None
        if current_run and version != expected_version:
            raise InputError(
                "Ledger annotation prompt does not match its run metadata"
            )
        if current_run or version is not None:
            if row[1] == "annotated":
                try:
                    parse_annotation(decoded.get("final"))
                except ContractError as exc:
                    raise InputError(
                        "Ledger contains an invalid current annotation"
                    ) from exc
            elif row[1] == "failed":
                if decoded.get("status") != "failed" or not isinstance(
                    decoded.get("error"), dict
                ):
                    raise InputError("Ledger failure does not match its indexed row")
            elif decoded.get("status") != "header_only_excluded":
                raise InputError(
                    "Ledger exclusion status does not match its indexed row"
                )
            source = decoded.get("source")
            if not isinstance(source, dict) or (
                source.get("id"),
                source.get("sha256"),
            ) != (source_id, source_sha256):
                raise InputError(
                    "Ledger annotation source identity does not match its indexed row"
                )
        return row[1], decoded


def run_metadata(
    *,
    input_sha256: str,
    model: str,
    endpoint: str,
    header_only_policy: str,
    max_text_chars: int | None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Bind resumptions to the text, policy, model, and any fixed row selection."""
    metadata = {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "prompt_version": PROMPT_VERSION,
        "contract_sha256": contract_sha256(),
        "request_submission": "individual",
        "calls_per_record": 2,
        "provider": "typesafe",
        "model": model,
        "endpoint": endpoint,
        "header_only_policy": header_only_policy,
        "max_text_chars": max_text_chars,
    }
    # Omit an unbounded selection for compatibility with completed schema-1
    # ledgers. A bounded selection is immutable across all provider resumes.
    if max_records is not None:
        metadata["max_records"] = max_records
    return metadata


def annotate(
    *,
    input_path: Path,
    ledger_path: Path,
    resume: bool,
    model: str,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
    max_attempts: int,
    header_only_policy: str,
    max_text_chars: int | None,
    max_records: int | None,
    workers: int = 1,
    stop_event=None,
    expected_input_sha256: str | None = None,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Add missing rows to a durable ledger and return deterministic run counts."""

    stop_event = stop_event if stop_event is not None else Event()
    if header_only_policy not in {"exclude", "include"}:
        raise InputError("header_only_policy must be exclude or include")
    if max_records is not None and (type(max_records) is not int or max_records <= 0):
        raise InputError("max_records must be a positive integer")
    if max_text_chars is not None and max_text_chars <= 0:
        raise InputError("max_text_chars must be positive")
    if workers < 1:
        raise InputError("workers must be at least one")
    distinct_paths(input_path, ledger_path)
    input_stamp = file_stamp(input_path)
    input_hash = sha256_file(input_path)
    if file_stamp(input_path) != input_stamp:
        raise InputError("Input changed while hashing; no requests submitted")
    if expected_input_sha256 is not None and input_hash != expected_input_sha256:
        raise InputError("Input changed since preflight")
    run_lock = acquire_annotation_lock(ledger_path)
    try:
        ledger = AnnotationLedger(
            ledger_path,
            run_metadata=run_metadata(
                input_sha256=input_hash,
                model=model,
                endpoint=endpoint,
                header_only_policy=header_only_policy,
                max_text_chars=max_text_chars,
                max_records=max_records,
            ),
            resume=resume,
        )
    except BaseException:
        run_lock.close()
        raise
    try:
        check_request_builders(ledger.connection)
        client = JevClient(
            api_key,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            stop_event=stop_event,
            retry_rate_limits=True,
        )
    except BaseException:
        ledger.close()
        run_lock.close()
        raise
    counts = {
        "scanned": 0,
        "annotated": 0,
        "excluded_header_only": 0,
        "already_complete": 0,
        "failed": 0,
    }
    selected = 0
    first_failure: BaseException | None = None
    consecutive_rejections = 0
    pending: dict[Future[dict[str, Any]], tuple[int, str, str]] = {}
    last_progress = time.monotonic()

    def persist_completed(completed: set[Future[dict[str, Any]]]) -> None:
        """Commit every successful completed request before surfacing a failure."""

        nonlocal first_failure, last_progress, consecutive_rejections
        thousand_rows = False
        for future in completed:
            source_index, source_id, source_digest = pending.pop(future)
            if future.cancelled():
                continue
            try:
                annotation = future.result()
            except CancelledError:
                # A queued row or a not-yet-started confidence stage is resumable.
                # Validated stages are already committed by the worker.
                stop_event.set()
                continue
            except Exception as exc:
                row_rejection = isinstance(exc, JevHTTPError) and exc.row_rejection
                recorded = (
                    isinstance(exc, (ContractError, JevResponseError)) or row_rejection
                )
                if recorded:
                    ledger.insert(
                        source_id=source_id,
                        source_index=source_index,
                        source_sha256=source_digest,
                        status="failed",
                        annotation={
                            "prompt_version": PROMPT_VERSION,
                            "status": "failed",
                            "source": {"id": source_id, "sha256": source_digest},
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "http_status": getattr(exc, "status_code", None),
                                "code": getattr(exc, "error_code", None),
                            },
                        },
                        replace_failed=retry_failed,
                    )
                    counts["failed"] += 1
                    print(
                        json.dumps(
                            {
                                "provider": "typesafe",
                                "source_index": source_index,
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "http_status": getattr(exc, "status_code", None),
                            }
                        ),
                        flush=True,
                    )
                consecutive_rejections += 1
                if first_failure is None and (
                    not recorded
                    or consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS
                ):
                    first_failure = InputError(
                        f"Jev stopped at source row {source_index}: {exc}. "
                        + (
                            "Recorded failures require --retry-failed. "
                            if recorded
                            else "Inspect provider configuration before resuming. "
                        )
                        + "Successful stages are retained."
                    )
                continue
            consecutive_rejections = 0
            ledger.insert(
                source_id=source_id,
                source_index=source_index,
                source_sha256=source_digest,
                status="annotated",
                annotation=annotation,
                replace_failed=retry_failed,
            )
            counts["annotated"] += 1
            thousand_rows |= counts["annotated"] % 1000 == 0
        now = time.monotonic()
        if thousand_rows or now - last_progress >= 15:
            print(
                json.dumps(
                    {"provider": "typesafe", **counts, "inflight_rows": len(pending)}
                ),
                flush=True,
            )
            last_progress = now

    try:
        if retry_failed:
            with ledger.connection:
                if ledger.connection.execute(
                    "SELECT 1 FROM annotations WHERE status='failed' LIMIT 1"
                ).fetchone():
                    previous = ledger.connection.execute(
                        "SELECT value FROM metadata WHERE key='row_retry_passes'"
                    ).fetchone()
                    passes = int(previous[0]) if previous else 0
                    if passes < 0:
                        raise InputError("Ledger contains an invalid row retry counter")
                    ledger.connection.execute(
                        "INSERT OR REPLACE INTO metadata VALUES ('row_retry_passes', ?)",
                        (str(passes + 1),),
                    )
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="locus-jev"
        ) as executor:
            source = iter_jsonl(input_path)
            try:
                while max_records is None or selected < max_records:
                    if first_failure is not None or (
                        stop_event is not None and stop_event.is_set()
                    ):
                        break
                    if file_stamp(input_path) != input_stamp:
                        raise InputError(
                            "Input changed during annotation; restore the original input before resuming"
                        )
                    try:
                        source_index, record = next(source)
                    except StopIteration:
                        break
                    source_id, source_digest = source_identity(record, source_index)
                    excluded = (
                        is_header_only_record(record)
                        and header_only_policy == "exclude"
                    )
                    if not excluded:
                        selected += 1
                    status = ledger.existing_status(source_id, source_digest)
                    if status == "failed" and not retry_failed:
                        counts["scanned"] += 1
                        counts["failed"] += 1
                        continue
                    if status is not None and status != "failed":
                        counts["scanned"] += 1
                        counts["already_complete"] += 1
                        continue
                    counts["scanned"] += 1
                    if excluded:
                        ledger.insert(
                            source_id=source_id,
                            source_index=source_index,
                            source_sha256=source_digest,
                            status="header_only_excluded",
                            annotation={
                                "prompt_version": PROMPT_VERSION,
                                "source": {
                                    "id": source_id,
                                    "sha256": source_digest,
                                    "locator": source_locator(record, source_index),
                                },
                                "status": "header_only_excluded",
                                "reason": "The historical LOCUS v1.0 pipeline excluded rows with a header and no content.",
                            },
                        )
                        counts["excluded_header_only"] += 1
                        continue
                    future = executor.submit(
                        annotate_record,
                        record=record,
                        source_index=source_index,
                        source_id=source_id,
                        source_sha256=source_digest,
                        client=client,
                        model=model,
                        max_text_chars=max_text_chars,
                        ledger_path=ledger_path,
                        stop_event=stop_event,
                    )
                    pending[future] = (source_index, source_id, source_digest)
                    while len(pending) >= workers * 2:
                        completed, _ = wait(
                            pending, timeout=15, return_when=FIRST_COMPLETED
                        )
                        persist_completed(completed)
                        if stop_event is not None and stop_event.is_set():
                            break
            except BaseException as exc:
                # KeyboardInterrupt must cancel the queue before executor shutdown
                # waits for running requests to finish and checkpoint.
                if isinstance(exc, (KeyboardInterrupt, CancelledError)):
                    stop_event.set()
                if first_failure is None:
                    first_failure = exc
            finally:
                source.close()
                # A malformed later input row must not discard completed paid work.
                # Cancel only queued calls; running calls finish and checkpoint.
                if first_failure is not None or (
                    stop_event is not None and stop_event.is_set()
                ):
                    for future in pending:
                        future.cancel()
                while pending:
                    completed, _ = wait(
                        pending, timeout=15, return_when=FIRST_COMPLETED
                    )
                    persist_completed(completed)
        if first_failure is not None:
            raise first_failure
        # The reader may reach EOF while paid calls are still in flight.
        if file_stamp(input_path) != input_stamp:
            raise InputError(
                "Input changed during annotation; completed stages are checkpointed"
            )
    finally:
        try:
            client.close()
        finally:
            try:
                ledger.close()
            finally:
                run_lock.close()
    return counts


def public_annotation_identifiers(
    record: Mapping[str, Any], source_index: int, source_id: str
) -> dict[str, Any]:
    """Identify one section without repeating its legal text or source metadata."""
    context = {
        field: value
        for field, value in record.items()
        if field in _PUBLIC_ID_FIELDS
        or (field in {"collection", "origin"} and isinstance(value, str))
        or (
            isinstance(value, str)
            and (field in _SOURCE_PATH_FIELDS or _is_locator_metadata_key(field))
        )
    }
    public = public_source_record(context)
    identifiers = {
        field: public[field]
        for field in ("document_id", "audit_id", "custom_id", "source_record_id")
        if isinstance(public.get(field), str) and public[field].strip()
    }
    if "document_id" in identifiers:
        if type(record.get("index")) is int:
            identifiers["index"] = record["index"]
        else:
            identifiers["input_row_index"] = source_index
    elif "audit_id" not in identifiers:
        identifiers.update(source_id=source_id, input_row_index=source_index)
    return identifiers


def export_annotations_jsonl(
    *,
    input_path: Path,
    ledger_path: Path,
    output_path: Path,
    include_excluded_header_only: bool,
    allow_partial: bool,
    overwrite: bool,
    annotation_field: str = "classification_jev",
) -> dict[str, int]:
    """Export section identifiers and annotations for exactly the ledger selection.

    A bounded first-N run is complete when those selected rows are complete;
    rows beyond its immutable selection are not missing annotations. A failed
    selected row still requires an explicit partial export.
    """

    distinct_paths(input_path, ledger_path, output_path)
    if output_path.exists() and not overwrite:
        raise InputError(f"Refusing to overwrite existing output: {output_path}")
    if not ledger_path.is_file():
        raise InputError(
            f"Cannot export because the ledger does not exist: {ledger_path}"
        )
    ledger = AnnotationLedger.open_readonly(ledger_path)
    counts = {"exported": 0, "excluded_header_only": 0, "missing": 0, "failed": 0}
    try:
        metadata = ledger.run_metadata
        provider = metadata.get("provider", "jev")
        expected_field = PROVIDER_ANNOTATION_FIELDS.get(
            provider, f"classification_{provider}"
        )
        if annotation_field != expected_field:
            raise InputError("Export provider does not match the annotation ledger")
        before = file_stamp(input_path)
        if metadata.get("input_sha256") != sha256_file(input_path):
            raise InputError(
                "Ledger input hash does not match the source file; refusing to export mixed data"
            )
        maximum = metadata.get("max_records")
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise InputError("Ledger contains an invalid bounded selection")
        selected = 0
        with atomic_text_output(output_path, overwrite=overwrite) as handle:
            with closing(iter_jsonl(input_path)) as source:
                while maximum is None or selected < maximum:
                    try:
                        source_index, record = next(source)
                    except StopIteration:
                        break
                    source_id, source_digest = source_identity(record, source_index)
                    excluded = (
                        is_header_only_record(record)
                        and metadata.get("header_only_policy") == "exclude"
                    )
                    selected += not excluded
                    ledger_row = ledger.annotation(source_id, source_digest)
                    if ledger_row is None:
                        counts["missing"] += 1
                        if allow_partial:
                            continue
                        raise InputError(
                            "Ledger is incomplete for this input. Resume annotation or pass --allow-partial-export explicitly."
                        )
                    status, annotation = ledger_row
                    if status == "failed":
                        counts["failed"] += 1
                        if allow_partial:
                            continue
                        raise InputError(
                            "Ledger contains failed rows. Inspect the ledger and use "
                            "--retry-failed, or request --allow-partial-export explicitly."
                        )
                    if (
                        status == "header_only_excluded"
                        and not include_excluded_header_only
                    ):
                        counts["excluded_header_only"] += 1
                        continue
                    exported = public_annotation_identifiers(
                        record, source_index, source_id
                    )
                    exported[annotation_field] = _public_annotation(
                        annotation, record, source_index
                    )
                    handle.write(canonical_json(exported) + "\n")
                    counts["exported"] += 1
            if file_stamp(input_path) != before:
                raise InputError(
                    "Input changed during export; output was not published"
                )
    finally:
        ledger.close()
    return counts


def positive_int(value: str) -> int:
    """Argparse validator for nonzero row counts, worker counts, and token caps."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    """Reject nonpositive or nonfinite timeout and polling values."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def default_env_file() -> Path:
    """Use the checkout's .env from any working directory; installed copies use ./.env."""

    return project_root() / ".env"


def resolve_env_file(path: Path | None) -> Path:
    """An explicit --env-file must exist; the default .env is optional."""

    if path is None:
        return default_env_file()
    if not path.is_file():
        raise EnvironmentError(f"Environment file not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Expose one provider switch and explicit run/recovery controls."""
    from .batch import DEFAULT_GOOGLE_MODEL, DEFAULT_OPENAI_MODEL

    parser = argparse.ArgumentParser(
        description="Annotate LOCUS with Typesafe (Jev), OpenAI Batch, or Google Batch."
    )
    parser.add_argument(
        "--provider", choices=tuple(PROVIDER_ANNOTATION_FIELDS), default="typesafe"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Original LOCUS JSONL or JSONL.GZ input.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="SQLite run ledger; defaults to the checkout's runs/locus_<provider>.sqlite3 (working directory for installed wheels).",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional identifiers-and-annotations JSONL export path."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue an existing compatible ledger."
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry recorded failed rows; successful stages remain cached.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export an existing ledger without a key or API calls; requires --output.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Permit replacing --output atomically.",
    )
    parser.add_argument(
        "--allow-partial-export",
        action="store_true",
        help="Export only completed rows. The default rejects an incomplete ledger.",
    )
    parser.add_argument(
        "--include-excluded-header-only",
        action="store_true",
        help="Include header-only rows with an explicit excluded status during export.",
    )
    parser.add_argument(
        "--header-only-policy",
        choices=("exclude", "include"),
        default="include",
        help="Include header-only Structural candidates by default; exclude only for historical coverage.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=positive_int,
        help="Explicit opt-in input truncation limit.",
    )
    parser.add_argument(
        "--max-records",
        type=positive_int,
        help="Select a fixed first N eligible rows; the selection stays fixed on resume.",
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Explicitly authorize an unbounded API run.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_WORKERS,
        help="Concurrent row workers; reduce this if the Typesafe account rate-limits requests.",
    )
    parser.add_argument(
        "--model",
        type=model_id,
        help=f"Model override. Defaults: typesafe {DEFAULT_MODEL}; openai {DEFAULT_OPENAI_MODEL}; google {DEFAULT_GOOGLE_MODEL}.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        help="Batch token cap, including reasoning (default 4096).",
    )
    parser.add_argument(
        "--batch-action",
        choices=(
            "prepare",
            "submit",
            "status",
            "collect",
            "attach",
            "retry",
            "reset-uncertain",
        ),
        help="Google/OpenAI only; default submit. reset-uncertain is an explicit recovery action after checking that no remote job exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="Requests per uploaded shard; maximum 50000 (default 5000).",
    )
    parser.add_argument(
        "--max-inflight-batches",
        type=positive_int,
        default=DEFAULT_MAX_INFLIGHT,
        help="Maximum active remote jobs per provider (default 1).",
    )
    parser.add_argument(
        "--cleanup-remote-files",
        action="store_true",
        help="Google only: delete this run's input and result files after durable collection.",
    )
    parser.add_argument(
        "--batch-number",
        type=positive_int,
        help="Local batch number for recovery attachment.",
    )
    parser.add_argument(
        "--batch-id", help="Remote batch ID to attach after an ambiguous submission."
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help="HTTPS Typesafe endpoint."
    )
    parser.add_argument(
        "--timeout", type=positive_float, default=45.0, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Typesafe retry attempts for transient HTTP failures.",
    )
    parser.add_argument(
        "--api-key-env", help="Override the provider's API-key environment variable."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Local dotenv file; never committed. Default: the repository's .env, if present. A given path must exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the first request shape without an API call or writes.",
    )
    return parser


def dry_run(
    input_path: Path,
    *,
    model: str,
    header_only_policy: str,
    max_text_chars: int | None = None,
) -> int:
    """Preview the first eligible request with its legal text redacted."""
    for source_index, record in iter_jsonl(input_path):
        source_id, source_digest = source_identity(record, source_index)
        if is_header_only_record(record) and header_only_policy == "exclude":
            continue
        text = compose_input_text(record)
        if not text:
            raise InputError(
                f"Input row {source_index} has neither usable header nor content"
            )
        limited_text, truncated = limit_text(text, max_text_chars)
        payload = build_jev_request(limited_text, model)
        payload["state"]["excerpt"] = "<redacted; see source input>"
        print(
            json.dumps(
                {
                    "provider": "typesafe",
                    "source_id": source_id,
                    "source_sha256": source_digest,
                    "input_text_characters": len(text),
                    "sent_text_characters": len(limited_text),
                    "text_truncated": truncated,
                    "request": payload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print("No eligible rows were found.", file=sys.stderr)
    return 1


def main(
    argv: list[str] | None = None,
    *,
    stop_event=None,
    expected_input_sha256: str | None = None,
) -> int:
    """Run one provider; Ctrl-C drains active requests and returns exit code 130."""
    with cancellation_scope(stop_event) as stop:
        try:
            code = _main(
                argv, stop_event=stop, expected_input_sha256=expected_input_sha256
            )
            check_cancelled(stop)
            return code
        except (KeyboardInterrupt, CancelledError):
            stop.set()
            print(
                "Interrupted; completed work is checkpointed. Resume the same run to continue.",
                file=sys.stderr,
            )
            return 130


def _main(argv, *, stop_event, expected_input_sha256):
    """Dispatch one operation while the public entry point owns cancellation."""
    check_cancelled(stop_event)
    args = build_parser().parse_args(argv)
    args.ledger = (
        args.ledger or project_root() / "runs" / f"locus_{args.provider}.sqlite3"
    )
    try:
        args.env_file = resolve_env_file(args.env_file)
        if args.retry_failed and (args.export_only or args.dry_run):
            raise InputError("--retry-failed requires an annotation operation")
        if args.provider != "typesafe":
            from .batch import run

            if args.retry_failed:
                if args.batch_action not in (None, "submit", "retry"):
                    raise InputError("--retry-failed conflicts with --batch-action")
                args.batch_action = "retry"
            return run(
                args, stop_event=stop_event, expected_input_sha256=expected_input_sha256
            )
        if (
            any(
                (
                    args.batch_action,
                    args.max_output_tokens,
                    args.batch_number,
                    args.batch_id,
                    args.cleanup_remote_files,
                )
            )
            or args.batch_size != DEFAULT_BATCH_SIZE
            or args.max_inflight_batches != DEFAULT_MAX_INFLIGHT
        ):
            raise InputError("Batch options apply only to Google/OpenAI")
        if not args.export_only and (not args.dry_run or not args.model):
            load_dotenv(args.env_file)
        args.model = model_id(
            args.model or os.environ.get("TYPESAFE_MODEL_ID") or DEFAULT_MODEL
        )
        if args.dry_run:
            if args.export_only:
                raise InputError("--dry-run and --export-only cannot be used together")
            return dry_run(
                args.input,
                model=args.model,
                header_only_policy=args.header_only_policy,
                max_text_chars=args.max_text_chars,
            )
        distinct_paths(args.input, args.ledger, args.output)
        if args.output and args.output.exists() and not args.overwrite_output:
            raise InputError(f"Refusing to overwrite existing output: {args.output}")
        if args.export_only:
            if args.output is None:
                raise InputError("--export-only requires --output")
            counts = export_annotations_jsonl(
                input_path=args.input,
                ledger_path=args.ledger,
                output_path=args.output,
                include_excluded_header_only=args.include_excluded_header_only,
                allow_partial=args.allow_partial_export,
                overwrite=args.overwrite_output,
            )
            print(
                json.dumps(
                    {"export": counts, "output": str(args.output)}, sort_keys=True
                )
            )
            return 0
        if args.max_records is None and not args.allow_full_run:
            raise InputError(
                "Pass --max-records N for a bounded run or --allow-full-run for a full API run."
            )
        api_key, key_name = get_api_key(args.api_key_env)
        check_cancelled(stop_event)
        if key_name != (args.api_key_env or "TYPESAFE_API_KEY"):
            print(f"Using compatible API-key variable {key_name}.", file=sys.stderr)
        counts = annotate(
            input_path=args.input,
            ledger_path=args.ledger,
            resume=args.resume,
            model=args.model,
            endpoint=args.endpoint,
            api_key=api_key,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
            header_only_policy=args.header_only_policy,
            max_text_chars=args.max_text_chars,
            max_records=args.max_records,
            workers=args.workers,
            stop_event=stop_event,
            expected_input_sha256=expected_input_sha256,
            retry_failed=args.retry_failed,
        )
        print(
            json.dumps(
                {"annotation": counts, "ledger": str(args.ledger)}, sort_keys=True
            )
        )
        check_cancelled(stop_event)
        if args.output and (not counts["failed"] or args.allow_partial_export):
            export_counts = export_annotations_jsonl(
                input_path=args.input,
                ledger_path=args.ledger,
                output_path=args.output,
                include_excluded_header_only=args.include_excluded_header_only,
                allow_partial=args.allow_partial_export,
                overwrite=args.overwrite_output,
            )
            print(
                json.dumps(
                    {"export": export_counts, "output": str(args.output)},
                    sort_keys=True,
                )
            )
        return ROW_ERRORS_EXIT_CODE if counts["failed"] else 0
    except (
        EnvironmentError,
        InputError,
        ContractError,
        JevError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
