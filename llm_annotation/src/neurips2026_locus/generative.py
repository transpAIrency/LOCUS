"""Shared prompt/schema rendering for the two generative batch adapters."""

from __future__ import annotations

from typing import Any, Mapping
from .contract import (
    CLASSIFICATION_SCHEMA,
    PROMPT_VERSION,
    ContractError,
    compose_input_text,
    contract_sha256,
    parse_annotation,
)
from .municipal_law_annotation.municipal_law_prompt_utils import load_prompt_parts

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
POLICY_VERSION = PROMPT_VERSION


def build_messages(
    record: Mapping[str, Any], *, max_text_chars: int | None = None
) -> tuple[str, list[str], dict]:
    """Keep instructions/examples separate from a literally inserted excerpt."""
    text = compose_input_text(record)
    if not text:
        raise ContractError("A classification request requires nonempty legal text")
    if max_text_chars is not None:
        if type(max_text_chars) is not int or max_text_chars < 1:
            raise ContractError("max_text_chars must be a positive integer")
        text = text[:max_text_chars]
    instructions, template = load_prompt_parts()
    return (
        instructions,
        [template.replace("{{EXCERPT}}", text, 1)],
        CLASSIFICATION_SCHEMA,
    )


contract_hash = contract_sha256
parse_classification = parse_annotation
