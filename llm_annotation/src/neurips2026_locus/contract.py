"""Shared municipal annotation policy and inference request fingerprints."""

from __future__ import annotations

import ast
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


_POLICY_DIR = Path(__file__).parent / "municipal_law_annotation"
CLASSIFICATION_SCHEMA = json.loads(
    (_POLICY_DIR / "municipal_law_annotation_schema.json").read_text(encoding="utf-8")
)


class ContractError(ValueError):
    """An annotation or input violates its declared contract."""


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


# Only functions that construct requests or checkpoint identities belong here.
# Export, CLI, retry, and documentation changes must not invalidate a run.
# Frozen prompt helpers are already covered byte-for-byte by contract_sha256().
# Add new request dependencies/settings here when changing construction; this
# registry is explicit, not automatic dependency discovery.
_REQUEST_CODE = {
    "contract.py": (
        "text_value",
        "compose_input_text",
        "_POLICY_DIR",
        "CLASSIFICATION_SCHEMA",
    ),
    "pipeline.py": (
        "canonical_json",
        "source_locator",
        "source_identity",
        "limit_text",
        "annotate_record",
    ),
    "generative.py": ("build_messages",),
    "batch.py": ("request_id", "request_for", "request_digest", "adapter"),
    "openai_batch.py": ("build_request", "REASONING_EFFORT"),
    "google_batch.py": ("build_request", "THINKING_LEVEL"),
    "jev.py": ("canonical_json_bytes",),
}


def _request_syntax(value):
    """Serialize syntax without comments, docstrings, annotations, or locations.

    Empty type_params is new in Python 3.12; omit it so Python 3.10 and 3.13
    produce the same digest. Request string literals remain significant.
    """
    if isinstance(value, ast.AST):
        fields = {}
        for key, item in ast.iter_fields(value):
            if key in {"annotation", "returns", "type_comment"} or (
                key == "type_params" and not item
            ):
                continue
            if (
                key == "body"
                and isinstance(item, list)
                and item
                and isinstance(item[0], ast.Expr)
                and isinstance(item[0].value, ast.Constant)
                and isinstance(item[0].value.value, str)
            ):
                item = item[1:]
            fields[key] = _request_syntax(item)
        return {"node": type(value).__name__, "fields": fields}
    if isinstance(value, list):
        return [_request_syntax(item) for item in value]
    return value


def _request_code_sha256(directory: Path) -> str:
    """Hash the declared request/identity functions and their imported bindings."""
    digest = sha256(b"locus-request-code-v2\0" + contract_sha256().encode())
    for filename, names in _REQUEST_CODE.items():
        tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
        selected = []
        found = set()
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            ):
                selected.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            ):
                selected.append(node)
                found.update(
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name) and target.id in names
                )
        if found != set(names):
            raise ContractError(
                f"Request fingerprint declarations do not match {filename}"
            )
        referenced = set()
        pending = [_request_syntax(selected)]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if item.get("node") == "Name":
                    referenced.add(item["fields"]["id"])
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        imports = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                aliases = [
                    alias
                    for alias in node.names
                    if (alias.asname or alias.name.split(".")[0]) in referenced
                ]
                if aliases:
                    imports.append(
                        ast.ImportFrom(
                            module=node.module, names=aliases, level=node.level
                        )
                        if isinstance(node, ast.ImportFrom)
                        else ast.Import(names=aliases)
                    )
        digest.update(filename.encode() + b"\0")
        digest.update(
            json.dumps(
                _request_syntax(imports + selected),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


@lru_cache(maxsize=1)
def request_builder_sha256() -> str:
    """Bind request construction and identities, not whole source-file bytes.

    Native batches also verify every reconstructed request against its exact
    prepared digest. Model, policy, input, cap, and selection remain ledger-bound.
    """
    return _request_code_sha256(Path(__file__).parent)


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
    """Reject data that cannot be written back as strict UTF-8 JSON."""
    value = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    _validate_unicode(value)
    return value


def _validate_unicode(value: Any) -> None:
    """Reject unpaired surrogates in metadata as well as model-input text.

    JSON escapes can otherwise create invalid Unicode after UTF-8 decoding,
    causing export to fail only after model requests have already been paid.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(
                "JSON strings must contain valid Unicode scalar values"
            ) from None
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode(key)
            _validate_unicode(item)
    elif isinstance(value, list):
        for item in value:
            _validate_unicode(item)


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
