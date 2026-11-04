"""Render the shared annotation prompt and build requests without sending them.

The returned dictionaries are keyword arguments for OpenAI Responses
(`client.responses.create`) and Google Interactions
(`client.interactions.create`), respectively. No SDK or credentials are needed
to use this module. Validate parsed model results with validate_annotation
from the adjacent validate_municipal_law_annotation module.

API configuration references (checked October 2026):
https://developers.openai.com/api/docs/guides/structured-outputs
https://ai.google.dev/gemini-api/docs/text-generation
https://ai.google.dev/gemini-api/docs/structured-output
"""

import json
from functools import lru_cache
from pathlib import Path


_DIRECTORY = Path(__file__).resolve().parent
_DEFAULT_PROMPT = _DIRECTORY / "oct2026_codebook_short.txt"
_DEFAULT_SCHEMA = _DIRECTORY / "municipal_law_annotation_schema.json"
_INPUT_HEADING = "# Excerpt to classify\n"
_PLACEHOLDER = "{{EXCERPT}}"
_INPUT_BODY = "<excerpt>\n{{EXCERPT}}\n</excerpt>"


@lru_cache(maxsize=8)
def load_prompt_parts(prompt_path=None):
    """Return (instructions, input_template) from the shared UTF-8 prompt.

    Instructions and examples stay in the first part. The second part includes
    the final input heading and excerpt wrapper. Joining both parts reproduces
    the loaded prompt exactly. Malformed or ambiguous templates raise ValueError.
    """
    path = _DEFAULT_PROMPT if prompt_path is None else Path(prompt_path)
    prompt = path.read_text(encoding="utf-8")
    instructions, heading, body = prompt.rpartition(_INPUT_HEADING)
    if not heading or prompt.count(_INPUT_HEADING) != 1 or not instructions.strip():
        raise ValueError("prompt must have instructions and exactly one final excerpt heading")
    if _PLACEHOLDER in instructions or body.strip() != _INPUT_BODY:
        raise ValueError(
            "final input template must contain one {{EXCERPT}} inside <excerpt> tags"
        )
    return instructions, heading + body


def _render_parts(excerpt, prompt_path):
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError("excerpt must be a nonempty string")
    instructions, input_template = load_prompt_parts(prompt_path)
    # Validate only the template, then substitute once. Excerpt characters,
    # including braces, placeholder text, and XML tags, remain unchanged.
    return instructions, input_template.replace(_PLACEHOLDER, excerpt, 1)


def render_prompt(excerpt, prompt_path=None):
    """Return the full prompt with the excerpt inserted literally once."""
    instructions, user_input = _render_parts(excerpt, prompt_path)
    return instructions + user_input


def _load_schema(schema_path):
    path = _DEFAULT_SCHEMA if schema_path is None else Path(schema_path)
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("annotation schema must be a JSON object schema")
    return schema


def build_openai_request(excerpt, prompt_path=None, schema_path=None):
    """Build an OpenAI Responses request for GPT-6 Luna with low reasoning.

    The developer message contains the shared instructions and examples; the
    user message contains only the final excerpt section. Structured Outputs
    enforces the flat JSON schema. Application validation still enforces the
    relationship between primary_function and subcategory.
    """
    instructions, user_input = _render_parts(excerpt, prompt_path)
    return {
        "model": "gpt-6-luna",
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_input},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "municipal_law_annotation",
                "strict": True,
                "schema": _load_schema(schema_path),
            }
        },
    }


def build_gemini_request(excerpt, prompt_path=None, schema_path=None):
    """Build a Google Interactions request for Gemini 3.8 Flash, low thinking.

    This uses the Interactions API's parameter names, not Generate Content's.
    Instructions and examples are shared with the OpenAI request verbatim.
    """
    instructions, user_input = _render_parts(excerpt, prompt_path)
    return {
        "model": "gemini-3.8-flash",
        "system_instruction": instructions,
        "input": user_input,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _load_schema(schema_path),
        },
        "generation_config": {"thinking_level": "low"},
    }
