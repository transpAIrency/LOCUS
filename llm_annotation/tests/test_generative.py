"""Batch envelopes preserve the supplied single-request builders exactly."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neurips2026_locus import generative, google_batch, openai_batch
from neurips2026_locus.contract import ContractError
from neurips2026_locus.municipal_law_annotation import (
    municipal_law_prompt_utils as supplied,
)


class GenerativeTests(unittest.TestCase):
    def test_openai_batch_body_matches_supplied_responses_builder(self):
        row = {
            "header": " Header ",
            "content": "A {legal} {{EXCERPT}} rule.",
            "human_reference": "SECRET",
        }
        system, messages, schema = generative.build_messages(row)
        body = openai_batch.build_request(
            "id",
            model="gpt-6-luna",
            system_prompt=system,
            user_prompt=messages,
            schema=schema,
        )["body"]
        self.assertEqual(body.pop("max_output_tokens"), 4096)
        expected = supplied.build_openai_request(
            "Header\n\nA {legal} {{EXCERPT}} rule."
        )
        self.assertEqual(body, expected)
        self.assertNotIn("SECRET", str(body))

    def test_google_conversion_keeps_policy_schema_and_low_thinking(self):
        excerpt = "A test rule."
        system, messages, schema = generative.build_messages({"content": excerpt})
        converted = google_batch.build_request(
            "id",
            model="gemini-3.8-flash",
            system_prompt=system,
            user_prompt=messages,
            schema=schema,
        )["request"]
        original = supplied.build_gemini_request(excerpt)
        self.assertEqual(
            converted["systemInstruction"]["parts"],
            [{"text": original["system_instruction"]}],
        )
        self.assertEqual(
            converted["contents"],
            [{"role": "user", "parts": [{"text": original["input"]}]}],
        )
        self.assertEqual(
            converted["generationConfig"]["responseJsonSchema"],
            original["response_format"]["schema"],
        )
        self.assertEqual(
            converted["generationConfig"]["thinkingConfig"]["thinkingLevel"].lower(),
            original["generation_config"]["thinking_level"],
        )
        self.assertNotIn("response_format", converted)

    def test_literal_substitution_preserves_excerpt_bytes(self):
        text = '{ "a": 1 } {{EXCERPT}} </excerpt>\nNon-template content'
        system, messages, _ = generative.build_messages({"content": text})
        self.assertEqual(system + messages[0], supplied.render_prompt(text))
        self.assertTrue(messages[0].endswith(text + "\n</excerpt>\n"))

    def test_truncation_requires_explicit_positive_character_cap(self):
        row = {"content": "A legal rule."}
        self.assertIn(row["content"], generative.build_messages(row)[1][0])
        self.assertEqual(
            generative.build_messages(row, max_text_chars=7)[1][0],
            "# Excerpt to classify\n<excerpt>\nA legal\n</excerpt>\n",
        )
        for maximum in (0, -1, True, "7"):
            with self.assertRaises(ContractError):
                generative.build_messages(row, max_text_chars=maximum)
        for bad_row in ({}, {"content": {}}, {"header": 2}):
            with self.assertRaises(ContractError):
                generative.build_messages(bad_row)

    def test_policy_and_schema_are_identical_between_batch_providers(self):
        system, messages, schema = generative.build_messages({"content": "Section 1"})
        args = dict(
            model="test", system_prompt=system, user_prompt=messages, schema=schema
        )
        openai = openai_batch.build_request("id", **args)["body"]
        google = google_batch.build_request("id", **args)["request"]
        self.assertEqual(
            openai["input"][0]["content"],
            google["systemInstruction"]["parts"][0]["text"],
        )
        self.assertEqual(
            openai["input"][1]["content"], google["contents"][0]["parts"][0]["text"]
        )
        self.assertEqual(
            openai["text"]["format"]["schema"],
            google["generationConfig"]["responseJsonSchema"],
        )


if __name__ == "__main__":
    unittest.main()
