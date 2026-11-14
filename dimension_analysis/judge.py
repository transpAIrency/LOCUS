import asyncio
import os

from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from jinja2 import Environment, FileSystemLoader


template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
template_env = Environment(loader=FileSystemLoader(template_dir))

class Judgment(BaseModel):
    law_a_id: str
    law_b_id: str
    winner_id: str | None
    reasoning: str
    axis: str = "moralization"
    order: Literal["forward", "reverse"]


class PairwiseResponse(BaseModel):
    reasoning: str
    winner: Literal["A", "B", "Tie"]


MAX_TEXT_CHARS = 4000
TRUNCATION_NOTE = f" [TEXT TRUNCATED TO {MAX_TEXT_CHARS} CHARACTERS]"

AXES = ["enforcement_discretion", "opacity", "paternalism", "problem_salience"]


def prepare_text(text: str) -> str:
    """truncate law text and append a note if it exceeds the character limit."""
    if len(text) > MAX_TEXT_CHARS:
        return text[:MAX_TEXT_CHARS] + TRUNCATION_NOTE
    return text


def build_prompt(law_a: dict, law_b: dict, axis: str) -> str:
    """render the pairwise comparison prompt for the given axis."""
    template = template_env.get_template("system.tpl")

    text_a = prepare_text(law_a["content"])
    text_b = prepare_text(law_b["content"])

    return template.render(
        law_a=law_a,
        law_b=law_b,
        axis=axis,
        text_a=text_a,
        text_b=text_b,
    )


async def _judge_one_direction(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    law_a: dict,
    law_b: dict,
    axis: str,
    *,
    order: Literal["forward", "reverse"],
    model: str,
) -> Judgment:
    prompt_first, prompt_second = (law_a, law_b) if order == "forward" else (law_b, law_a)

    async with sem:
        prompt = build_prompt(prompt_first, prompt_second, axis)

        response = await client.responses.parse(
            model=model,
            input=prompt,
            text_format=PairwiseResponse,
        )

        output = response.output_parsed

        if output.winner == "A":
            winner_id = prompt_first["id"]
        elif output.winner == "B":
            winner_id = prompt_second["id"]
        else:
            winner_id = None

        return Judgment(
            law_a_id=law_a["id"],
            law_b_id=law_b["id"],
            winner_id=winner_id,
            reasoning=output.reasoning,
            axis=axis,
            order=order,
        )


async def judge_pair(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    law_a: dict,
    law_b: dict,
    axis: str,
    *,
    model: str = "gpt-5.4-mini",
) -> list[Judgment | BaseException]:
    return await asyncio.gather(
        _judge_one_direction(
            client, sem, law_a, law_b, axis,
            order="forward", model=model,
        ),
        _judge_one_direction(
            client, sem, law_a, law_b, axis,
            order="reverse", model=model,
        ),
        return_exceptions=True,
    )

