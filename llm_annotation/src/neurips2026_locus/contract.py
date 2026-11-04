"""Shared municipal annotation policy; legacy reference labels remain readable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .municipal_law_annotation.municipal_law_prompt_utils import load_prompt_parts
from .municipal_law_annotation.validate_municipal_law_annotation import (
    validate_annotation,
    _unique_object,
    _invalid_constant,
)

CONTRACT_VERSION = "municipal-law-2026-10-30-v1"
PROVIDER_ANNOTATION_FIELDS = {
    "typesafe": "classification_jev",
    "openai": "classification_openai",
    "google": "classification_google",
}
PRIMARY_FUNCTIONS = (
    "Context",
    "Rules",
    "Process",
    "Enforcement",
    "Structural",
)
SUBSTANTIVE_FUNCTIONS = frozenset({"Rules", "Enforcement"})
SUB_CATEGORIES = (
    "Zoning",
    "Nuisance",
    "Buildings",
    "Business",
    "Other",
)
TOPIC_SCHEMA_VERSION = "locus-topics-5-v1"
# Input compatibility only: never accept these old labels as new model outputs.
LEGACY_TOPIC_MAP = {
    "Land use": "Zoning",
    "Housing": "Zoning",
    "Noise/Nuisance": "Nuisance",
    "Public space": "Nuisance",
    "Building/Safety": "Buildings",
    "Business licensing": "Business",
    "Other": "Other",
}


_POLICY_DIR = Path(__file__).parent / "municipal_law_annotation"
CLASSIFICATION_SCHEMA = json.loads(
    (_POLICY_DIR / "municipal_law_annotation_schema.json").read_text(encoding="utf-8")
)


class ContractError(ValueError):
    """An annotation or input violates its declared contract."""


@dataclass(frozen=True)
class Classification:
    """Legacy human/reference layout, not the current model output schema."""

    is_substantive: int
    primary_function: str
    sub_category: str | None
    logic: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize a validated historical reference, not a current model response."""
        validate_classification(self)
        return asdict(self)


def validate_classification(classification: Classification) -> None:
    """Validate schema values plus cross-field LOCUS invariants."""

    if classification.primary_function not in PRIMARY_FUNCTIONS:
        raise ContractError("Unknown primary_function")
    expected_substantive = int(classification.primary_function in SUBSTANTIVE_FUNCTIONS)
    if (
        type(classification.is_substantive) is not int
        or classification.is_substantive != expected_substantive
    ):
        raise ContractError("is_substantive does not match primary_function")
    if expected_substantive and classification.sub_category not in SUB_CATEGORIES:
        raise ContractError("Substantive classifications require a valid sub_category")
    if not expected_substantive and classification.sub_category is not None:
        raise ContractError("Non-substantive classifications require null sub_category")
    if not isinstance(classification.logic, str) or not classification.logic.strip():
        raise ContractError("logic must be a non-empty sentence")


def is_header_only_record(record: Mapping[str, Any]) -> bool:
    """Match the historical production exclusion rule exactly."""

    header = text_value(record.get("header"))
    content = text_value(record.get("content"))
    return bool(header and not content)


def text_value(value: Any) -> str:
    """Trim only outer whitespace and reject implicit conversion of non-text."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContractError("Legal text fields must be strings or null")
    return value.strip()


def compose_input_text(record: Mapping[str, Any]) -> str:
    """Match the original header-plus-content model input construction."""

    pieces = [text_value(record.get(field)) for field in ("header", "content")]
    return "\n\n".join(piece for piece in pieces if piece)


@lru_cache(maxsize=1)
def contract_sha256() -> str:
    """Fingerprint the supplied policy, schema, and reference helper files."""
    digest = sha256(CONTRACT_VERSION.encode())
    for name in (
        "municipal_law_annotation_prompt_compact.txt",
        "municipal_law_annotation_schema.json",
        "municipal_law_prompt_utils.py",
        "municipal_law_jev_adapter.py",
        "validate_municipal_law_annotation.py",
    ):
        digest.update(name.encode())
        digest.update((_POLICY_DIR / name).read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def request_builder_sha256() -> str:
    """Fingerprint active request construction separately from the policy bundle.

    This conservative code digest can change without changing inference inputs.
    Per-request digests remain the evidence for the exact payload of each row.
    """
    digest = sha256(contract_sha256().encode())
    for name in (
        "contract.py",
        "pipeline.py",
        "generative.py",
        "batch.py",
        "openai_batch.py",
        "google_batch.py",
        "jev.py",
    ):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def parse_annotation(value: Any) -> dict[str, Any]:
    """Require the complete three-field schema and its conditional topic rule."""
    try:
        return validate_annotation(value)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def _finite_float(value: str) -> float:
    """Reject numeric overflow as well as literal NaN/Infinity tokens."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON numbers must be finite")
    return result


def decode_json(text: str) -> Any:
    """Reject duplicate keys and nonfinite numbers before validating the schema."""
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )


@lru_cache(maxsize=1)
def prompt_examples() -> tuple[str, ...]:
    """Whitespace-normalized examples for held-out leakage checks."""
    instructions, _ = load_prompt_parts()
    return tuple(
        " ".join(text.split())
        for text in re.findall(r"<excerpt>\n(.*?)\n</excerpt>", instructions, re.S)
    )


def example_overlap(text: str) -> bool:
    """Detect literal full-example overlap after whitespace normalization only."""
    normalized = " ".join(text.split())
    return any(example in normalized for example in prompt_examples())
