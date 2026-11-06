"""Regression checks for audit scoring and independent human eligibility."""

from copy import deepcopy
import unittest

from release_builder.audit import summarize_human_audit
from release_builder.records import ReleaseError


def row():
    result = {
        "audit_id": "one", "human_annotations": 3,
        "human_function": "Context", "human_topic": None, "human_confidence": "High",
        "human_function_eligible": True, "human_topic_eligible": True,
        "human_confidence_eligible": True,
    }
    for prefix in ("gemini", "openai", "jev"):
        result.update({prefix + "_function": "Context", prefix + "_topic": None,
                       prefix + "_confidence": "High"})
    return result


class AuditSummaryTests(unittest.TestCase):
    def test_eligible_null_topic_matches_a_real_answer(self):
        summary = summarize_human_audit([row()])
        topic = summary["model_agreement"]["google"]["topic"]
        self.assertEqual(topic, {"eligible": 1, "scored": 1, "matches": 1, "accuracy": 1.0})
        self.assertEqual(summary["annotation_count_distribution"], {"3": 1})

    def test_missing_answer_does_not_match_eligible_null_topic(self):
        missing = row()
        missing.update(gemini_function=None, gemini_topic=None, gemini_confidence=None)
        summary = summarize_human_audit([missing])
        for metric in summary["model_agreement"]["google"].values():
            self.assertEqual(metric, {"eligible": 1, "scored": 0, "matches": 0, "accuracy": None})

    def test_fields_have_independent_eligibility(self):
        first = row()
        first.update(human_function=None, human_function_eligible=False,
                     human_topic="Business", human_topic_eligible=True,
                     gemini_function="Rules", gemini_topic="Business")
        summary = summarize_human_audit([first])
        self.assertEqual(summary["fields"]["function"]["eligible"], 0)
        self.assertEqual(summary["model_agreement"]["google"]["topic"]["matches"], 1)
        self.assertEqual(summary["model_agreement"]["openai"]["topic"]["matches"], 0)

    def test_duplicate_audit_rows_are_rejected(self):
        with self.assertRaises(ReleaseError):
            summarize_human_audit([row(), deepcopy(row())])

    def test_eligible_function_or_confidence_requires_human_label(self):
        for field in ("function", "confidence"):
            invalid = row()
            invalid["human_" + field] = None
            with self.assertRaises(ReleaseError):
                summarize_human_audit([invalid])


if __name__ == "__main__":
    unittest.main()
