"""Read-only evidence checks for frozen LOCUS annotation ledgers.

This module never starts or resumes annotation jobs. A caller must supply actual
run dates separately; collection timestamps are not treated as run dates.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping


class ProvenanceError(ValueError):
    """An export disagrees with its ledger, or its evidence changed."""


_RUN_FIELDS = {
    "provider", "model", "reasoning_effort", "prompt_version",
    "contract_sha256", "request_builder_sha256", "input_sha256",
    "ledger_schema_version", "request_submission", "calls_per_record",
    "max_output_tokens", "max_text_chars", "max_records", "header_only_policy",
}
_META_FIELDS = {
    "schema_version", "request_builder_sha256", "batch_initialized",
    "batch_total", "row_retry_passes", "created_at", "provider", "schema",
}
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "provider_total_tokens")


def _object(value: str | Mapping[str, Any], description: str) -> dict[str, Any]:
    try:
        result = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"Invalid JSON object in {description}") from exc
    if not isinstance(result, dict):
        raise ProvenanceError(f"Expected a JSON object in {description}")
    return result


def _stamp(path: Path) -> dict[str, int] | None:
    try:
        value = path.stat()
    except FileNotFoundError:
        return None
    return {"device": value.st_dev, "inode": value.st_ino, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _annotation(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "final" in raw:
        return dict(raw)
    fields = [key for key in raw if key.startswith("classification_")]
    if len(fields) != 1 or not isinstance(raw[fields[0]], Mapping):
        raise ProvenanceError("Expected one classification object or an annotation with final labels")
    return dict(raw[fields[0]])


def _probabilities(annotation: Mapping[str, Any]) -> tuple[Any, Any]:
    stages = annotation.get("stage_metadata", {})
    if not isinstance(stages, Mapping):
        raise ProvenanceError("Invalid stage_metadata")
    classification = stages.get("classification", {})
    if not isinstance(classification, Mapping):
        raise ProvenanceError("Invalid classification stage metadata")
    answers = classification.get("answers", {})
    if not isinstance(answers, Mapping):
        raise ProvenanceError("Invalid classification answers")
    result = []
    for field in ("primary_function", "subcategory"):
        answer = answers.get(field, {})
        if not isinstance(answer, Mapping):
            raise ProvenanceError(f"Invalid {field} answer")
        result.append(answer.get("probabilities"))
    return tuple(result)


def _same(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ProvenanceError(f"Export differs from ledger: {description}")


def _same_source(exported: Mapping[str, Any], stored: Mapping[str, Any]) -> None:
    """Accept the exporter's optional public collection tag, preserving identity.

    public_source_locator adds this tag after the private source hash is made.
    It is not part of that hash's locator. All original locator fields, hashes,
    and any other source metadata must still match exactly.
    """
    actual = dict(exported)
    locator = actual.get("locator")
    original = stored.get("locator")
    if isinstance(locator, Mapping) and isinstance(original, Mapping):
        if "collection" in locator and "collection" not in original:
            if locator["collection"] not in {"original", "ocr_addition"}:
                raise ProvenanceError("Invalid public source collection")
            locator = dict(locator)
            locator.pop("collection")
            actual["locator"] = locator
    _same(actual, stored, "source locator")


def _tokens(metadata: Mapping[str, Any]) -> tuple[int | None, ...]:
    usage = metadata.get("usage", {})
    if usage is None:
        usage = {}
    if not isinstance(usage, Mapping):
        raise ProvenanceError("Recorded usage is not an object")
    candidates = (
        (metadata.get("input_tokens"), usage.get("input_tokens"), usage.get("promptTokenCount")),
        (metadata.get("output_tokens"), usage.get("output_tokens"), usage.get("candidatesTokenCount")),
        (metadata.get("reasoning_tokens"), usage.get("thoughtsTokenCount"),
         usage.get("output_tokens_details", {}).get("reasoning_tokens")
         if isinstance(usage.get("output_tokens_details", {}), Mapping) else None),
        (usage.get("total_tokens"), usage.get("totalTokenCount")),
    )
    result = []
    for field, values in zip(_TOKEN_FIELDS, candidates):
        present = [value for value in values if value is not None]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in present):
            raise ProvenanceError(f"Invalid recorded {field}")
        if len(set(present)) > 1:
            raise ProvenanceError(f"Conflicting recorded {field}")
        result.append(present[0] if present else None)
    return tuple(result)


class LedgerEvidence(AbstractContextManager):
    """Validate a frozen existing ledger without modifying its data.

    Opening uses SQLite ``mode=ro`` and ``query_only``; it deliberately does not
    use ``immutable``, which could hide committed WAL data. DB and WAL stamps
    and ``data_version`` are checked throughout and when the context exits.
    The caller should keep this context open until its release is staged.
    """

    def __init__(self, path: Path | str, kind: str = "harness") -> None:
        if kind not in {"harness", "individual"}:
            raise ProvenanceError(f"Unknown ledger kind: {kind}")
        self.path = Path(path).resolve(strict=True)
        self.kind = kind
        self._connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
            self._connection = connection
            connection.execute("PRAGMA query_only=ON")
            # A read-only SQLite connection may recreate absent WAL sidecars
            # after the last writer closed cleanly. Initialize that machinery
            # before watching for external changes; it is not a ledger edit.
            connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            connection.execute("BEGIN")
            opening_stamps = self._stamps()
            opening_version = connection.execute("PRAGMA data_version").fetchone()[0]
            # BEGIN is deferred: this read establishes the evidence snapshot.
            self._tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self._data_version = connection.execute("PRAGMA data_version").fetchone()[0]
            self._initial_stamps = self._stamps()
            if self._initial_stamps != opening_stamps or self._data_version != opening_version:
                raise ProvenanceError(f"Ledger changed during release validation: {self.path}")
            required = {"metadata", "annotations"} if kind == "harness" else {"meta", "rows"}
            if not required.issubset(self._tables):
                raise ProvenanceError(f"Not a {kind} LOCUS ledger: {self.path}")
            self.assert_unchanged()
        except BaseException:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            raise

    def _stamps(self) -> dict[str, Any]:
        return {"database": _stamp(self.path), "wal": _stamp(Path(str(self.path) + "-wal"))}

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ProvenanceError("Ledger evidence is closed")
        return self._connection

    def assert_unchanged(self) -> None:
        version = self.connection.execute("PRAGMA data_version").fetchone()[0]
        if self._stamps() != self._initial_stamps or version != self._data_version:
            raise ProvenanceError(f"Ledger changed during release validation: {self.path}")

    def check(self, source_id: str, raw_annotation: Mapping[str, Any]) -> None:
        """Compare one exported annotation with its exact retained ledger row."""
        self.assert_unchanged()
        if not isinstance(source_id, str) or not source_id:
            raise ProvenanceError("A nonempty source_id is required")
        exported = _annotation(raw_annotation)
        exported_source = exported.get("source")
        if not isinstance(exported_source, Mapping):
            raise ProvenanceError("Export omitted source identity")
        _same(exported_source.get("id"), source_id, "source.id")
        if self.kind == "harness":
            row = self.connection.execute(
                "SELECT source_sha256,status,annotation_json FROM annotations WHERE source_id=?", (source_id,)
            ).fetchone()
            if row is None:
                raise ProvenanceError(f"Source is absent from harness ledger: {source_id}")
            source_hash, status, annotation_json = row
            _same(source_id, "locus-v1:" + source_hash, "source ID hash")
            if status not in {"annotated", "failed"}:
                raise ProvenanceError(f"Nonterminal harness annotation status: {status}")
            stored = _object(annotation_json, "ledger annotation")
            stored_source = stored.get("source", {})
            if not isinstance(stored_source, Mapping):
                raise ProvenanceError("Ledger omitted source identity")
            _same(stored_source.get("id"), source_id, "ledger source.id")
            _same(stored_source.get("sha256"), source_hash, "ledger source.sha256")
            _same(exported_source.get("sha256"), source_hash, "source.sha256")
            _same_source(exported_source, stored_source)
            if "final" not in raw_annotation:
                locator = stored_source.get("locator", {})
                for field in ("document_id", "index", "audit_id", "input_row_index"):
                    if field in raw_annotation and field in locator:
                        _same(raw_annotation[field], locator[field], f"export {field}")
            _same(exported.get("final"), stored.get("final"), "final labels")
            _same(exported.get("prompt_version"), stored.get("prompt_version"), "prompt_version")
            if status == "failed":
                _same(exported.get("status"), stored.get("status"), "retained failure status")
                _same(exported.get("error"), stored.get("error"), "retained failure reason")
            _same(_probabilities(exported), _probabilities(stored), "Jev probability maps")
            for group in ("run_details", "final_metadata", "stage_metadata"):
                _same(exported.get(group), stored.get(group), group)
        else:
            row = self.connection.execute(
                "SELECT status,classification_json,metadata_json FROM rows WHERE source_id=?", (source_id,)
            ).fetchone()
            if row is None:
                raise ProvenanceError(f"Source is absent from individual ledger: {source_id}")
            if row[0] != "ok":
                raise ProvenanceError("Individual ledger has no successful export for this source")
            _same(exported.get("final"), _object(row[1], "individual classification"), "individual final labels")
            _same(exported.get("final_metadata"), _object(row[2], "individual metadata"), "individual final_metadata")
            metadata = self._recorded_metadata()
            config = metadata.get("run_metadata", {})
            details = exported.get("run_details", {})
            if not isinstance(details, Mapping):
                raise ProvenanceError("Invalid exported run_details")
            for field in ("provider", "reasoning_effort", "contract_sha256"):
                if field in config:
                    _same(details.get(field), config[field], f"individual {field}")
            if "model" in config:
                _same(details.get("requested_model"), config["model"], "individual requested_model")
        self.assert_unchanged()

    def _recorded_metadata(self) -> dict[str, Any]:
        table = "metadata" if self.kind == "harness" else "meta"
        result = {}
        for key, raw in self.connection.execute(f"SELECT key,value FROM {table}"):
            if key not in _META_FIELDS and key != "run_metadata":
                continue
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                value = raw
            if key == "run_metadata":
                if not isinstance(value, Mapping):
                    raise ProvenanceError("Ledger run_metadata is not an object")
                value = {name: item for name, item in value.items() if name in _RUN_FIELDS}
            result[key] = value
        return result

    def snapshot_metadata(self) -> dict[str, Any]:
        """Return an allowlisted metadata snapshot, without inferred run dates."""
        self.assert_unchanged()
        result = {"kind": self.kind, "metadata": self._recorded_metadata(),
                  "snapshot": self._initial_stamps, "actual_run_dates": None,
                  "run_dates_note": "Supply actual run start/end separately; annotation collection times are not run dates."}
        self.assert_unchanged()
        return result

    def _usage_candidates(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        if self.kind == "individual":
            for source, raw in self.connection.execute("SELECT source_id,metadata_json FROM rows WHERE metadata_json IS NOT NULL"):
                yield "individual_metadata", _object(raw, "individual response metadata")
            return
        if "responses" in self._tables:
            for source, request_hash, raw, response_hash in self.connection.execute(
                "SELECT source_id,request_sha256,response_json,response_sha256 FROM responses"
            ):
                response = _object(raw, "checkpointed response")
                yield "responses", {"request_sha256": request_hash, "response_sha256": response_hash,
                               "usage": response.get("usage")}
        if "batch_requests" in self._tables:
            for source, raw in self.connection.execute("SELECT source_id,result_json FROM batch_requests WHERE result_json IS NOT NULL"):
                result = _object(raw, "batch result")
                metadata = result.get("metadata")
                if metadata is not None:
                    yield "batch_results", _object(metadata, "batch response metadata")
        for source, raw in self.connection.execute("SELECT source_id,annotation_json FROM annotations"):
            annotation = _object(raw, "ledger annotation")
            if annotation.get("final_metadata") is not None:
                yield "annotation_final_metadata", _object(annotation["final_metadata"], "final response metadata")
            stages = annotation.get("stage_metadata", {})
            if not isinstance(stages, Mapping):
                raise ProvenanceError("Invalid recorded stage_metadata")
            for stage in stages.values():
                yield "annotation_stage_metadata", _object(stage, "stage response metadata")

    def usage_summary(self) -> dict[str, Any]:
        """Stream recorded usage by evidence source, without double-count claims.

        Checkpoints, batch results, and annotation metadata may describe the same
        response. Their separate totals must not be added. Global totals remain
        unknown if more than one populated source is present. These ledgers do
        not retain every HTTP attempt or overwritten result.
        """
        self.assert_unchanged()
        candidates = 0
        sources: dict[str, dict[str, Any]] = {}
        for origin, metadata in self._usage_candidates():
            if origin not in sources:
                sources[origin] = {"records": 0}
                for field in _TOKEN_FIELDS:
                    sources[origin][field] = None
                    sources[origin][field + "_records"] = 0
            summary = sources[origin]
            summary["records"] += 1
            for field, value in zip(_TOKEN_FIELDS, _tokens(metadata)):
                if value is not None:
                    summary[field] = (summary[field] or 0) + value
                    summary[field + "_records"] += 1
            candidates += 1
            if candidates % 10000 == 0:
                self.assert_unchanged()
        for summary in sources.values():
            for field in _TOKEN_FIELDS:
                summary[field + "_missing_records"] = summary["records"] - summary[field + "_records"]
        only_source = next(iter(sources.values())) if len(sources) == 1 else None
        result: dict[str, Any] = {
            "candidate_records": candidates,
            "retained_response_records": only_source["records"] if only_source else (0 if not sources else None),
            "sources": sources,
            "all_attempts_available": False,
            "overlapping_sources": len(sources) > 1,
            "coverage": "Recorded snapshots by ledger evidence source. Sources can overlap and must not be added. Global totals are null when multiple sources exist. Unrecorded HTTP attempts and overwritten responses are unavailable.",
            "reasoning_tokens_note": "Reasoning tokens are reported separately when recorded; OpenAI output_tokens already includes them. Google candidates and thoughts are separate recorded fields.",
        }
        for field in _TOKEN_FIELDS:
            result[field] = only_source[field] if only_source else None
            result[field + "_records"] = only_source[field + "_records"] if only_source else (0 if not sources else None)
            result[field + "_missing_records"] = only_source[field + "_missing_records"] if only_source else (0 if not sources else None)
        result["unavailable"] = not any(
            summary["input_tokens_records"] or summary["output_tokens_records"] for summary in sources.values()
        )
        if self.kind == "individual":
            result["recorded_attempt_counter_total"] = self.connection.execute("SELECT sum(attempts) FROM rows").fetchone()[0]
        self.assert_unchanged()
        return result

    def close(self, *, validate: bool = True) -> None:
        if self._connection is None:
            return
        try:
            if validate:
                self.assert_unchanged()
        finally:
            self._connection.close()
            self._connection = None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close(validate=exc_type is None)
        return False
