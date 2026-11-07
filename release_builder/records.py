"""Strict, offline row transformations for the LOCUS v1.1 release.

These functions do not read run ledgers, submit requests, or fill missing labels.
Human consensus fields remain independent because their eligibility differs.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any


class ReleaseError(ValueError):
    """An input cannot be represented faithfully by the release schema."""


FUNCTIONS = ("Context", "Rules", "Process", "Enforcement", "Structural")
TOPICS = ("Buildings", "Business", "Zoning", "Nuisance", "Other")
CONFIDENCES = ("Low", "Medium", "High")
PROVIDERS = {
    "google": ("classification_google", "gemini"),
    "openai": ("classification_openai", "openai"),
    "typesafe": ("classification_jev", "jev"),
}
_TYPES = {"city": "city", "county": "county", "both": "city-county", "other": "other"}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_GEOID = re.compile(r"(?:[0-9]{5}|[0-9]{7}|[0-9]{10})\Z")
_LABEL_FIELDS = {"primary_function", "subcategory", "confidence"}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ReleaseError(f"Invalid JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ReleaseError("JSON number is not finite")
    return parsed


def read_json(raw: str | bytes) -> Any:
    """Decode JSON without duplicate keys, NaN, Infinity, or numeric overflow."""
    try:
        return json.loads(raw, object_pairs_hook=_object,
                          parse_constant=_bad_constant, parse_float=_finite_float)
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, ReleaseError):
            raise
        raise ReleaseError(f"Invalid JSON: {exc}") from exc


def _dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**63:
        raise ReleaseError(f"{name} must be an int64 >= {minimum}")
    return value


def _state(value: Any) -> str:
    value = _text(value, "state").upper()
    if re.fullmatch(r"[A-Z]{2}", value) is None:
        raise ReleaseError("state must be a two-letter abbreviation")
    return value


def _csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames
            if names is None or len(set(names)) != len(names):
                raise ReleaseError(f"Missing or duplicate CSV column names in {path}")
            if not required.issubset(names):
                raise ReleaseError(f"Missing CSV columns in {path}: {sorted(required - set(names))}")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ReleaseError(f"Malformed CSV row ending at line {reader.line_num} in {path}")
                rows.append(row)
            if not rows:
                raise ReleaseError(f"Empty CSV: {path}")
            return rows
    except (csv.Error, UnicodeError) as exc:
        raise ReleaseError(f"Cannot read CSV {path}: {exc}") from exc


def read_counties(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read county release rows and document metadata without inferring geography."""
    source = _csv_rows(path, {"county_fips", "county_name", "state", "selected_document_id",
                              "selected_jurisdiction_name", "selected_city_county", "selection_reason"})
    counties = []
    documents: dict[str, dict[str, Any]] = {}
    seen = set()
    for row in source:
        fips = row["county_fips"]
        if re.fullmatch(r"[0-9]{5}", fips) is None or fips in seen:
            raise ReleaseError(f"Invalid or duplicate county_fips: {fips!r}")
        seen.add(fips)
        state = _state(row["state"])
        name = _text(row["county_name"], "county_name")
        reason = _text(row["selection_reason"], "selection_reason")
        document = row["selected_document_id"] or None
        jurisdiction = row["selected_jurisdiction_name"] or None
        raw_type = row["selected_city_county"]
        if document is None:
            if jurisdiction is not None or raw_type:
                raise ReleaseError(f"Uncovered county {fips} contains selected jurisdiction metadata")
            kind = None
        else:
            _text(document, "selected_document_id")
            _text(jurisdiction, "selected_jurisdiction_name")
            if raw_type not in _TYPES:
                raise ReleaseError(f"Invalid selected_city_county for {fips}: {raw_type!r}")
            kind = _TYPES[raw_type]
            if kind == "other" and not (state == "DC" and jurisdiction == "Washington"):
                raise ReleaseError("jurisdiction_type other is reserved for Washington, DC")
            metadata = documents.setdefault(document, {
                "jurisdiction_name": jurisdiction, "jurisdiction_type": kind, "state": state,
                "represented_county_fips": [], "represented_county_name": [],
            })
            if any(metadata[key] != value for key, value in (
                ("jurisdiction_name", jurisdiction), ("jurisdiction_type", kind), ("state", state)
            )):
                raise ReleaseError(f"Inconsistent county metadata for document {document}")
            metadata["represented_county_fips"].append(fips)
            metadata["represented_county_name"].append(name)
        counties.append({"county_fips": fips, "county_name": name, "state": state,
                         "document_id": document, "jurisdiction_name": jurisdiction,
                         "jurisdiction_type": kind, "selection_reason": reason, "section_count": None})
    for metadata in documents.values():
        ordered = sorted(zip(metadata["represented_county_fips"], metadata["represented_county_name"]))
        metadata["represented_county_fips"] = [item[0] for item in ordered]
        metadata["represented_county_name"] = [item[1] for item in ordered]
    return counties, documents


def section_key(raw: dict[str, Any]) -> tuple[str, int]:
    _dict(raw, "section")
    return _text(raw.get("document_id"), "document_id"), _integer(raw.get("index"), "index")


def audit_key(raw: dict[str, Any]) -> str:
    _dict(raw, "audit row")
    return _text(raw.get("audit_id"), "audit_id")


def section_record(raw: dict[str, Any], documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Transform source columns, preserving legal text and validating its geography."""
    document, index = section_key(raw)
    if document not in documents:
        raise ReleaseError(f"Section document is absent from county harmonization: {document}")
    metadata = documents[document]
    state = _state(raw.get("state"))
    if state != metadata["state"]:
        raise ReleaseError(f"Section and county states differ for {document}")
    content = raw.get("content")
    if not isinstance(content, str):
        raise ReleaseError("content must be a string")
    header = raw.get("header")
    if header is not None:
        if not isinstance(header, str):
            raise ReleaseError("header must be a string or null")
        header = re.sub(r"^#+(?=\s|$)", "", header.strip()).strip() or None
    geoid = raw.get("jurisdiction_geoid")
    if not isinstance(geoid, str):
        raise ReleaseError("jurisdiction_geoid must be a source GEOID string")
    if _GEOID.fullmatch(geoid) is None:
        parts = [part.strip() for part in geoid.split(";")]
        if len(parts) < 2 or len(set(parts)) != len(parts) or any(_GEOID.fullmatch(p) is None for p in parts):
            raise ReleaseError(f"Invalid jurisdiction_geoid for {document}: {geoid!r}")
        geoid = None
    return {"document_id": document, "section_index": index, "state": state,
            "jurisdiction_name": metadata["jurisdiction_name"],
            "jurisdiction_type": metadata["jurisdiction_type"], "jurisdiction_geoid": geoid,
            "represented_county_fips": list(metadata["represented_county_fips"]),
            "represented_county_name": list(metadata["represented_county_name"]),
            "header": header, "content": content}


def _human_label(value: str, vocabulary: tuple[str, ...], name: str) -> str | None:
    if value == "":
        return None
    normalized = {label.casefold(): label for label in vocabulary}.get(value.casefold())
    if normalized is None:
        raise ReleaseError(f"Invalid human {name}: {value!r}")
    return normalized


def read_human(path: Path) -> list[dict[str, Any]]:
    """Read independent human consensus and eligibility, preserving original text."""
    source = _csv_rows(path, {"audit_id", "text", "function", "topic", "confidence", "n_annotations",
                              "function_eligible", "topic_eligible", "confidence_eligible"})
    rows = []
    seen = set()
    for raw in source:
        identity = audit_key(raw)
        if identity in seen:
            raise ReleaseError(f"Duplicate audit_id: {identity}")
        seen.add(identity)
        text = _text(raw["text"], "human text")
        count = raw["n_annotations"]
        if re.fullmatch(r"[1-9][0-9]*", count) is None or int(count) >= 2**63:
            raise ReleaseError(f"Invalid n_annotations for {identity}: {count!r}")
        row = {"audit_id": identity, "text": text,
               "human_function": _human_label(raw["function"], FUNCTIONS, "function"),
               "human_topic": _human_label(raw["topic"], TOPICS, "topic"),
               "human_confidence": _human_label(raw["confidence"], CONFIDENCES, "confidence")}
        for field in ("function", "topic", "confidence"):
            value = raw[field + "_eligible"]
            if value not in ("0", "1"):
                raise ReleaseError(f"Invalid {field}_eligible for {identity}: {value!r}")
            row["human_" + field + "_eligible"] = value == "1"
        row["human_annotations"] = int(count)
        rows.append(row)
    return rows


def _probabilities(annotation: dict[str, Any], question: str, labels: tuple[str, ...]) -> dict[str, float]:
    try:
        answer = annotation["stage_metadata"]["classification"]["answers"][question]
        probabilities = answer["probabilities"]
    except (KeyError, TypeError):
        raise ReleaseError(f"Missing Jev {question} probabilities") from None
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ReleaseError(f"Jev {question} probabilities must contain exactly {labels}")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in probabilities.values()):
        raise ReleaseError(f"Jev {question} probabilities must be finite numbers in [0, 1]")
    return {label: float(probabilities[label]) for label in labels}


def _annotation_metadata(raw: dict[str, Any], annotation: dict[str, Any], provider: str) -> dict[str, Any]:
    source = _dict(annotation.get("source"), "annotation source")
    identity = _text(source.get("id"), "source.id")
    if not identity.startswith("locus-v1:") or _HEX.fullmatch(identity[9:]) is None:
        raise ReleaseError("source.id must be locus-v1:<sha256>")
    digest = source.get("sha256")
    if digest is not None and (not isinstance(digest, str) or _HEX.fullmatch(digest) is None
                               or identity != "locus-v1:" + digest):
        raise ReleaseError("source.sha256 does not match source.id")
    locator = source.get("locator")
    if locator is not None:
        locator = _dict(locator, "source.locator")
        _integer(locator.get("input_row_index"), "source.locator.input_row_index")
        for name in ("document_id", "audit_id"):
            if name in raw and locator.get(name) != raw[name]:
                raise ReleaseError(f"source.locator.{name} differs from annotation row")
    details = _dict(annotation.get("run_details"), "run_details")
    if details.get("provider") != provider:
        raise ReleaseError("run_details.provider differs from selected provider")
    _text(details.get("requested_model"), "requested_model")
    submission = details.get("request_submission")
    if submission not in ("batch", "individual"):
        raise ReleaseError("request_submission must be batch or individual")
    if provider == "typesafe" and submission != "individual":
        raise ReleaseError("Jev annotations require individual request provenance")
    expected_calls = 2 if provider == "typesafe" else 1
    if type(details.get("calls_per_record")) is not int or details["calls_per_record"] != expected_calls:
        raise ReleaseError(f"calls_per_record must be {expected_calls} for {provider}")
    for name in ("contract_sha256", "request_builder_sha256"):
        value = details.get(name)
        if not isinstance(value, str) or _HEX.fullmatch(value) is None:
            raise ReleaseError(f"{name} must be a sha256 digest")
    input_chars = _integer(details.get("input_text_characters"), "input_text_characters", minimum=1)
    sent_chars = _integer(details.get("sent_text_characters"), "sent_text_characters", minimum=1)
    if sent_chars > input_chars or type(details.get("text_truncated")) is not bool \
            or details["text_truncated"] != (sent_chars < input_chars):
        raise ReleaseError("Inconsistent text truncation provenance")
    # Provider-blocked exports can lack a collection timestamp. Preserve that
    # absence so manifest generation can report it rather than invent a date.
    stamp = details.get("annotated_at_utc")
    if stamp is not None:
        _text(stamp, "annotated_at_utc")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
        except ValueError:
            raise ReleaseError("annotated_at_utc must be a timezone-aware ISO timestamp") from None
    prompt = _text(annotation.get("prompt_version"), "prompt_version")
    return {"source_id": identity, "source_sha256": digest, "run_details": details,
            "prompt_version": prompt, "annotation": annotation}


def extract_annotation(raw: dict[str, Any], provider: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract one provider's labels and original provenance from a real export row."""
    _dict(raw, "annotation row")
    if provider not in PROVIDERS:
        raise ReleaseError(f"Unknown provider: {provider!r}")
    field, prefix = PROVIDERS[provider]
    present = {name for name, _ in PROVIDERS.values() if name in raw}
    if present != {field}:
        raise ReleaseError(f"Annotation row must contain exactly {field}")
    annotation = _dict(raw[field], field)
    final = _dict(annotation.get("final"), "final")
    if set(final) != _LABEL_FIELDS:
        raise ReleaseError("final must contain exactly primary_function, subcategory, confidence")
    status = annotation.get("status", "ANNOTATED")
    if status == "BLOCKED":
        if any(value is not None for value in final.values()):
            raise ReleaseError("BLOCKED annotations require three null labels")
        error = _dict(annotation.get("error"), "BLOCKED error")
        _text(error.get("provider_code"), "BLOCKED error.provider_code")
    elif status == "ANNOTATED":
        if annotation.get("error") is not None:
            raise ReleaseError("ANNOTATED annotation contains an error")
        function, topic, confidence = (final[name] for name in ("primary_function", "subcategory", "confidence"))
        if function not in FUNCTIONS or confidence not in CONFIDENCES:
            raise ReleaseError("Invalid function or confidence label")
        if function in ("Rules", "Enforcement"):
            if topic not in TOPICS:
                raise ReleaseError("Rules and Enforcement require a valid topic")
        elif topic is not None:
            raise ReleaseError("Context, Process, and Structural require a null topic")
    else:
        raise ReleaseError(f"Unsupported annotation status: {status!r}")
    labels = {prefix + "_function": final["primary_function"],
              prefix + "_topic": final["subcategory"], prefix + "_confidence": final["confidence"]}
    if provider == "typesafe":
        labels["jev_function_probs"] = (_probabilities(annotation, "primary_function", FUNCTIONS)
                                         if final["primary_function"] is not None else None)
        labels["jev_topic_probs"] = (_probabilities(annotation, "subcategory", TOPICS)
                                      if final["subcategory"] is not None else None)
    metadata = _annotation_metadata(raw, annotation, provider)
    metadata["status"] = status
    return labels, metadata


def consensus(labels: dict[str, Any]) -> dict[str, Any]:
    """Count answering models; a null topic from an answer is a real NA vote."""
    answered = [prefix for prefix in ("gemini", "openai", "jev")
                if labels.get(prefix + "_function") is not None]
    result = {}
    for field in ("function", "topic"):
        counts = Counter(labels.get(prefix + "_" + field) for prefix in answered)
        label, votes = counts.most_common(1)[0] if counts else (None, 0)
        result[field + "_majority"] = label if votes >= 2 else None
        result[field + "_votes"] = votes
    return result
