"""Behavioral tests for release row validation and consensus semantics."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
import tempfile
import unittest

try:
    from . import records
except ImportError:
    import records


def annotation(provider="google", function="Rules", topic="Business"):
    field, _ = records.PROVIDERS[provider]
    row = {"document_id": "doc_example", "index": 0, field: {
        "final": {"primary_function": function, "subcategory": topic, "confidence": "High"},
        "prompt_version": "fixture_prompt",
        "source": {"id": "locus-v1:" + "a" * 64, "sha256": "a" * 64,
                   "locator": {"document_id": "doc_example", "input_row_index": 2}},
        "run_details": {
            "provider": provider, "requested_model": "fixture_model",
            "request_submission": "individual" if provider == "typesafe" else "batch",
            "calls_per_record": 2 if provider == "typesafe" else 1,
            "contract_sha256": "b" * 64, "request_builder_sha256": "c" * 64,
            "input_text_characters": 20, "sent_text_characters": 16, "text_truncated": True,
            "annotated_at_utc": "2026-11-05T12:00:00Z",
        },
    }}
    if provider == "typesafe":
        row[field]["stage_metadata"] = {"classification": {"answers": {
            "primary_function": {"probabilities": dict.fromkeys(records.FUNCTIONS, 0.2)},
            "subcategory": {"probabilities": dict.fromkeys(records.TOPICS, 0.2)},
        }}}
    return row


class JsonAndIdentityTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for raw in ('{"a": 1, "a": 2}', '{"nested":{"x":0,"x":1}}',
                    '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(records.ReleaseError):
                records.read_json(raw)

    def test_section_key_rejects_boolean_and_overflow(self):
        for value in (True, -1, 2**63, "1", None):
            with self.subTest(value=value), self.assertRaises(records.ReleaseError):
                records.section_key({"document_id": "doc", "index": value})
        self.assertEqual(records.section_key({"document_id": "doc", "index": 0}), ("doc", 0))


class AnnotationTests(unittest.TestCase):
    def test_individual_export_with_source_id_only_is_accepted(self):
        row = annotation("openai")
        ann = row["classification_openai"]
        ann["source"] = {"id": "locus-v1:" + "a" * 64}
        ann["run_details"]["request_submission"] = "individual"
        labels, metadata = records.extract_annotation(row, "openai")
        self.assertEqual(labels["openai_topic"], "Business")
        self.assertIsNone(metadata["source_sha256"])
        self.assertIs(metadata["annotation"], ann)

    def test_source_digest_or_locator_mismatch_is_rejected(self):
        for mutate in (lambda a: a["source"].update(sha256="d" * 64),
                       lambda a: a["source"]["locator"].update(document_id="different")):
            row = annotation()
            mutate(row["classification_google"])
            with self.assertRaises(records.ReleaseError):
                records.extract_annotation(row, "google")

    def test_topic_applicability_is_enforced(self):
        for function, topic in (("Rules", None), ("Enforcement", ""), ("Process", "Other")):
            with self.subTest(function=function, topic=topic), self.assertRaises(records.ReleaseError):
                records.extract_annotation(annotation(function=function, topic=topic), "google")
        labels, _ = records.extract_annotation(annotation(function="Structural", topic=None), "google")
        self.assertIsNone(labels["gemini_topic"])

    def test_blocked_requires_nulls_and_explicit_reason(self):
        row = annotation()
        ann = row["classification_google"]
        ann["status"] = "BLOCKED"
        ann["final"] = dict.fromkeys(("primary_function", "subcategory", "confidence"))
        with self.assertRaises(records.ReleaseError):
            records.extract_annotation(row, "google")
        ann["error"] = {"provider_code": "CONTENT_BLOCK"}
        del ann["run_details"]["annotated_at_utc"]
        labels, metadata = records.extract_annotation(row, "google")
        self.assertEqual(set(labels.values()), {None})
        self.assertEqual(metadata["status"], "BLOCKED")
        ann["final"]["confidence"] = "High"
        with self.assertRaises(records.ReleaseError):
            records.extract_annotation(row, "google")

    def test_wrong_provider_and_mixed_annotation_fields_are_rejected(self):
        row = annotation()
        row["classification_openai"] = copy.deepcopy(row["classification_google"])
        with self.assertRaises(records.ReleaseError):
            records.extract_annotation(row, "google")
        row = annotation()
        row["classification_google"]["run_details"]["provider"] = "openai"
        with self.assertRaises(records.ReleaseError):
            records.extract_annotation(row, "google")

    def test_probability_values_are_preserved_without_normalization(self):
        row = annotation("typesafe")
        probabilities = row["classification_jev"]["stage_metadata"]["classification"]["answers"]["primary_function"]["probabilities"]
        probabilities.update(Context=0.19)
        labels, _ = records.extract_annotation(row, "typesafe")
        self.assertEqual(labels["jev_function_probs"]["Context"], 0.19)
        self.assertAlmostEqual(sum(labels["jev_function_probs"].values()), 0.99)

    def test_malformed_probability_maps_are_rejected(self):
        for update in ({"Context": float("nan")}, {"Context": float("inf")},
                       {"Context": True}, {"Context": -0.01}, {"Context": 1.01},
                       {"Extra": 0}):
            row = annotation("typesafe")
            row["classification_jev"]["stage_metadata"]["classification"]["answers"]["primary_function"]["probabilities"].update(update)
            with self.subTest(update=update), self.assertRaises(records.ReleaseError):
                records.extract_annotation(row, "typesafe")
        row = annotation("typesafe")
        del row["classification_jev"]["stage_metadata"]["classification"]["answers"]["subcategory"]["probabilities"]["Other"]
        with self.assertRaises(records.ReleaseError):
            records.extract_annotation(row, "typesafe")

    def test_inapplicable_jev_topic_probabilities_are_discarded(self):
        row = annotation("typesafe", function="Process", topic=None)
        row["classification_jev"]["stage_metadata"]["classification"]["answers"]["subcategory"] = {"probabilities": {"ignored": "bad"}}
        labels, _ = records.extract_annotation(row, "typesafe")
        self.assertIsNone(labels["jev_topic_probs"])
        self.assertIsNotNone(labels["jev_function_probs"])

    def test_truncation_provenance_must_be_consistent(self):
        row = annotation()
        row["classification_google"]["run_details"]["text_truncated"] = False
        with self.assertRaises(records.ReleaseError):
            records.extract_annotation(row, "google")


class ConsensusTests(unittest.TestCase):
    def test_missing_model_is_not_an_inapplicable_topic_vote(self):
        values = {"gemini_function": "Process", "gemini_topic": None,
                  "openai_function": None, "openai_topic": None,
                  "jev_function": "Rules", "jev_topic": "Other"}
        result = records.consensus(values)
        self.assertEqual(result, {"function_majority": None, "function_votes": 1,
                                  "topic_majority": None, "topic_votes": 1})

    def test_two_inapplicable_topics_are_a_topic_majority(self):
        values = {"gemini_function": "Process", "gemini_topic": None,
                  "openai_function": "Context", "openai_topic": None,
                  "jev_function": "Rules", "jev_topic": "Other"}
        self.assertEqual(records.consensus(values), {
            "function_majority": None, "function_votes": 1,
            "topic_majority": None, "topic_votes": 2})

    def test_votes_count_duplicate_labels_and_ties_have_no_majority(self):
        values = {"gemini_function": "Rules", "gemini_topic": "Other",
                  "openai_function": "Rules", "openai_topic": "Other",
                  "jev_function": "Enforcement", "jev_topic": "Business"}
        self.assertEqual(records.consensus(values), {
            "function_majority": "Rules", "function_votes": 2,
            "topic_majority": "Other", "topic_votes": 2})
        self.assertEqual(records.consensus({}), {
            "function_majority": None, "function_votes": 0,
            "topic_majority": None, "topic_votes": 0})


class CsvAndSectionTests(unittest.TestCase):
    def write_csv(self, directory, name, rows):
        path = Path(directory) / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def county_rows(self):
        return [
            {"county_fips": "02003", "county_name": "Second County", "state": "AK",
             "selected_document_id": "doc", "selected_jurisdiction_name": "Example",
             "selected_city_county": "both", "selection_reason": "selected"},
            {"county_fips": "02001", "county_name": "First County", "state": "AK",
             "selected_document_id": "doc", "selected_jurisdiction_name": "Example",
             "selected_city_county": "both", "selection_reason": "selected"},
            {"county_fips": "02005", "county_name": "Uncovered County", "state": "AK",
             "selected_document_id": "", "selected_jurisdiction_name": "",
             "selected_city_county": "", "selection_reason": "uncovered"},
        ]

    def test_county_lists_are_paired_sorted_and_uncovered_values_are_null(self):
        with tempfile.TemporaryDirectory() as directory:
            counties, documents = records.read_counties(self.write_csv(directory, "county.csv", self.county_rows()))
        self.assertEqual(documents["doc"]["represented_county_fips"], ["02001", "02003"])
        self.assertEqual(documents["doc"]["represented_county_name"], ["First County", "Second County"])
        self.assertEqual(documents["doc"]["jurisdiction_type"], "city-county")
        self.assertIsNone(counties[-1]["document_id"])
        self.assertIsNone(counties[-1]["section_count"])
        self.assertNotIn("selected_page_count", counties[0])

    def test_duplicate_counties_and_conflicting_document_metadata_are_rejected(self):
        for duplicate in (False, True):
            rows = self.county_rows()
            if duplicate:
                rows[1]["county_fips"] = rows[0]["county_fips"]
            else:
                rows[1]["selected_jurisdiction_name"] = "Different"
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(records.ReleaseError):
                records.read_counties(self.write_csv(directory, "county.csv", rows))

    def test_section_preserves_text_and_nulls_only_explicit_multiple_geoids(self):
        with tempfile.TemporaryDirectory() as directory:
            _, documents = records.read_counties(self.write_csv(directory, "county.csv", self.county_rows()))
        raw = {"document_id": "doc", "index": 0, "state": "ak", "header": "  ### Example  ",
               "content": "  legal text\n\n", "jurisdiction_geoid": "0200123"}
        row = records.section_record(raw, documents)
        self.assertEqual(row["header"], "Example")
        self.assertEqual(row["content"], raw["content"])
        self.assertEqual(row["jurisdiction_geoid"], "0200123")
        raw["jurisdiction_geoid"] = "0200123; 0200456"
        self.assertIsNone(records.section_record(raw, documents)["jurisdiction_geoid"])
        raw["jurisdiction_geoid"] = "unknown"
        with self.assertRaises(records.ReleaseError):
            records.section_record(raw, documents)

    def test_header_preserves_literal_hashes_and_removes_only_heading_marks(self):
        with tempfile.TemporaryDirectory() as directory:
            _, documents = records.read_counties(self.write_csv(directory, "county.csv", self.county_rows()))
        raw = {"document_id": "doc", "index": 0, "state": "ak",
               "content": "Body.", "jurisdiction_geoid": "0200123"}
        cases = (
            ("  #4 REBAR CONT.  ", "#4 REBAR CONT."),
            ("#14 Agricultural buildings", "#14 Agricultural buildings"),
            ("#None for residential units", "#None for residential units"),
            ("##4", "##4"),
            ("  ### Example  ", "Example"),
            ("###\tExample", "Example"),
            ("## #4 REBAR CONT.", "#4 REBAR CONT."),
            ("###", None),
            ("  ", None),
        )
        for header, expected in cases:
            with self.subTest(header=header):
                self.assertEqual(records.section_record({**raw, "header": header}, documents)["header"], expected)

    def test_human_consensus_fields_and_eligibility_stay_independent(self):
        source = [
            {"audit_id": "audit_1", "text": "text\nwith newline", "function": "rules", "topic": "",
             "confidence": "HIGH", "n_annotations": "3", "function_eligible": "1",
             "topic_eligible": "0", "confidence_eligible": "1"},
            {"audit_id": "audit_2", "text": "other text", "function": "", "topic": "business",
             "confidence": "", "n_annotations": "3", "function_eligible": "0",
             "topic_eligible": "1", "confidence_eligible": "0"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            rows = records.read_human(self.write_csv(directory, "human.csv", source))
        self.assertEqual(rows[0]["human_function"], "Rules")
        self.assertIsNone(rows[0]["human_topic"])
        self.assertFalse(rows[0]["human_topic_eligible"])
        self.assertEqual(rows[1]["human_topic"], "Business")
        self.assertIsNone(rows[1]["human_function"])
        self.assertTrue(rows[1]["human_topic_eligible"])
        self.assertEqual(rows[0]["text"], source[0]["text"])


if __name__ == "__main__":
    unittest.main()
