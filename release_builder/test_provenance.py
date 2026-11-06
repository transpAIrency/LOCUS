"""Fixture-only checks for frozen-ledger provenance validation."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from release_builder.provenance import LedgerEvidence, ProvenanceError


SOURCE_HASH = "a" * 64
SOURCE_ID = "locus-v1:" + SOURCE_HASH


def annotation():
    return {
        "source": {"id": SOURCE_ID, "sha256": SOURCE_HASH,
                   "locator": {"document_id": "doc", "index": 0}},
        "final": {"primary_function": "Rules", "subcategory": "Zoning", "confidence": "High"},
        "prompt_version": "fixture",
        "run_details": {"provider": "typesafe", "contract_sha256": "contract"},
        "stage_metadata": {"classification": {
            "request_sha256": "request", "response_sha256": "response",
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "answers": {
                "primary_function": {"probabilities": {"Context": 0.0, "Rules": 1.0}},
                "subcategory": {"probabilities": {"Zoning": 1.0, "Other": 0.0}},
            },
        }},
    }


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="locus-provenance-test-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "ledger.sqlite3"

    def harness(self, value=None, status="annotated", *, wal=False):
        value = annotation() if value is None else value
        connection = sqlite3.connect(self.path)
        if wal:
            connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE annotations (source_id TEXT PRIMARY KEY,source_sha256 TEXT,status TEXT,annotation_json TEXT);
            CREATE TABLE responses (source_id TEXT,request_sha256 TEXT,response_json TEXT,response_sha256 TEXT);
            CREATE TABLE batch_requests (source_id TEXT,result_json TEXT);
        """)
        metadata = {"provider": "typesafe", "model": "jev-fixture",
                    "max_output_tokens": 4096, "api_key": "must-not-leak"}
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("run_metadata", json.dumps(metadata)), ("api_key", "must-not-leak"),
            ("request_builder_sha256", "builder"), ("batch_total", "1"),
        ])
        connection.execute("INSERT INTO annotations VALUES (?,?,?,?)",
                           (SOURCE_ID, SOURCE_HASH, status, json.dumps(value)))
        connection.commit()
        return connection, value

    def test_harness_and_export_wrapper_match(self):
        writer, value = self.harness()
        writer.close()
        initial_bytes = self.path.read_bytes()
        with LedgerEvidence(self.path) as evidence:
            evidence.check(SOURCE_ID, value)
            evidence.check(SOURCE_ID, {"document_id": "doc", "index": 0, "classification_jev": value})
            self.assertEqual(evidence.connection.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                evidence.connection.execute("DELETE FROM annotations")
        self.assertEqual(initial_bytes, self.path.read_bytes())

    def test_rejects_source_final_and_probability_tampering(self):
        writer, value = self.harness()
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            for field in ("source", "final", "probability", "metadata"):
                changed = deepcopy(value)
                if field == "source":
                    changed["source"]["sha256"] = "b" * 64
                elif field == "final":
                    changed["final"]["confidence"] = "Low"
                elif field == "probability":
                    changed["stage_metadata"]["classification"]["answers"]["subcategory"]["probabilities"]["Zoning"] = 0.9
                else:
                    changed["run_details"]["contract_sha256"] = "changed"
                with self.subTest(field=field), self.assertRaises(ProvenanceError):
                    evidence.check(SOURCE_ID, changed)
            with self.assertRaises(ProvenanceError):
                evidence.check("missing", value)

    def test_exporter_public_collection_tag_preserves_source_identity(self):
        writer, value = self.harness()
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            for collection in ("original", "ocr_addition"):
                exported = deepcopy(value)
                exported["source"]["locator"]["collection"] = collection
                evidence.check(SOURCE_ID, exported)

    def test_public_collection_does_not_allow_other_source_changes(self):
        writer, value = self.harness()
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            for change in ({"collection": "unknown"},
                           {"collection": "ocr_addition", "document_id": "changed"},
                           {"collection": "ocr_addition", "unexpected": "value"}):
                exported = deepcopy(value)
                exported["source"]["locator"].update(change)
                with self.subTest(change=change), self.assertRaises(ProvenanceError):
                    evidence.check(SOURCE_ID, exported)

    def test_terminal_failure_is_checked_against_ledger(self):
        value = annotation()
        value["final"] = {"primary_function": None, "subcategory": None, "confidence": None}
        value.pop("stage_metadata")
        value["status"] = "failed"
        writer, value = self.harness(value, status="failed")
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            evidence.check(SOURCE_ID, value)

    def test_synthesized_block_reason_cannot_replace_retained_generic_failure(self):
        stored = annotation()
        stored.pop("final")
        stored.pop("stage_metadata")
        stored["status"] = "failed"
        stored["error"] = {"type": "ProviderBatchError", "message": "Provider returned an incomplete or invalid classification response"}
        writer, _ = self.harness(stored, status="failed")
        writer.close()
        changed = deepcopy(stored)
        changed["final"] = {"primary_function": None, "subcategory": None, "confidence": None}
        changed["status"] = "BLOCKED"
        changed["error"] = {"type": "ProviderContentBlock", "provider_code": "PROHIBITED_CONTENT", "message": "Provider blocked the prompt under its content policy"}
        with LedgerEvidence(self.path) as evidence:
            with self.assertRaises(ProvenanceError):
                evidence.check(SOURCE_ID, changed)

    def test_failed_annotation_reason_must_match_even_when_null_labels_match(self):
        stored = annotation()
        stored["final"] = {"primary_function": None, "subcategory": None, "confidence": None}
        stored.pop("stage_metadata")
        stored["status"] = "failed"
        stored["error"] = {"type": "ProviderBatchError", "message": "generic failure"}
        writer, _ = self.harness(stored, status="failed")
        writer.close()
        changed = deepcopy(stored)
        changed["error"]["provider_code"] = "PROHIBITED_CONTENT"
        with LedgerEvidence(self.path) as evidence:
            with self.assertRaisesRegex(ProvenanceError, "failure reason"):
                evidence.check(SOURCE_ID, changed)

    def test_metadata_allowlist_and_no_invented_run_dates(self):
        writer, _ = self.harness()
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            snapshot = evidence.snapshot_metadata()
            self.assertNotIn("must-not-leak", json.dumps(snapshot))
            self.assertEqual(snapshot["metadata"]["run_metadata"]["max_output_tokens"], 4096)
            self.assertIsNone(snapshot["actual_run_dates"])

    def test_overlapping_usage_sources_are_reported_separately(self):
        value = annotation()
        metadata = {"request_sha256": "request", "response_sha256": "response",
                    "input_tokens": 12, "output_tokens": 3,
                    "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}
        value["final_metadata"] = metadata
        writer, _ = self.harness(value)
        writer.execute("INSERT INTO responses VALUES (?,?,?,?)",
                       (SOURCE_ID, "request", json.dumps({"usage": metadata["usage"]}), "response"))
        writer.execute("INSERT INTO batch_requests VALUES (?,?)",
                       (SOURCE_ID, json.dumps({"metadata": metadata})))
        writer.commit()
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            result = evidence.usage_summary()
        self.assertEqual(result["candidate_records"], 4)
        self.assertIsNone(result["retained_response_records"])
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["output_tokens"])
        self.assertIsNone(result["provider_total_tokens"])
        self.assertEqual(result["sources"]["responses"]["input_tokens"], 12)
        self.assertEqual(result["sources"]["annotation_final_metadata"]["input_tokens"], 12)
        self.assertTrue(result["overlapping_sources"])
        self.assertIsNone(result["reasoning_tokens"])
        self.assertFalse(result["all_attempts_available"])

    def test_missing_usage_remains_unavailable(self):
        value = annotation()
        value.pop("stage_metadata")
        writer, _ = self.harness(value)
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            result = evidence.usage_summary()
        self.assertTrue(result["unavailable"])
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["output_tokens"])

    def test_conflicting_usage_fails(self):
        value = annotation()
        value["final_metadata"] = {"request_sha256": "request", "response_sha256": "response",
                                   "input_tokens": 12, "usage": {"input_tokens": 99, "output_tokens": 3}}
        writer, _ = self.harness(value)
        writer.close()
        with LedgerEvidence(self.path) as evidence:
            with self.assertRaises(ProvenanceError):
                evidence.usage_summary()

    def test_committed_wal_is_visible_and_mutation_is_detected(self):
        writer, value = self.harness(wal=True)
        with LedgerEvidence(self.path) as evidence:
            evidence.check(SOURCE_ID, value)
        evidence = LedgerEvidence(self.path)
        writer.execute("INSERT INTO metadata VALUES ('new','value')")
        writer.commit()
        with self.assertRaisesRegex(ProvenanceError, "changed"):
            evidence.assert_unchanged()
        with self.assertRaises(ProvenanceError):
            evidence.close()
        writer.close()

    def test_cleanly_closed_wal_ledger_can_be_opened_read_only(self):
        writer, value = self.harness(wal=True)
        writer.close()
        # The last writer checkpoints and removes its sidecars. Homebrew's
        # SQLite recreates them on the first read, even through mode=ro.
        # Builds that keep sidecars after a clean close do not exhibit this
        # absent-WAL case; exercise it on runtimes that produce it naturally.
        if any(Path(str(self.path) + suffix).exists() for suffix in ("-wal", "-shm")):
            self.skipTest("SQLite retained WAL sidecars after the last writer closed")
        self.assertFalse(Path(str(self.path) + "-wal").exists())
        self.assertFalse(Path(str(self.path) + "-shm").exists())
        initial_bytes = self.path.read_bytes()
        with LedgerEvidence(self.path) as evidence:
            evidence.check(SOURCE_ID, value)
            evidence.assert_unchanged()
        self.assertEqual(initial_bytes, self.path.read_bytes())

    def test_commit_during_snapshot_setup_is_rejected(self):
        writer, _ = self.harness(wal=True)

        class ConcurrentWriteEvidence(LedgerEvidence):
            def _stamps(self):
                stamps = super()._stamps()
                if not getattr(self, "injected_commit", False):
                    self.injected_commit = True
                    writer.execute("INSERT INTO metadata VALUES ('new','value')")
                    writer.commit()
                return stamps

        try:
            with self.assertRaisesRegex(ProvenanceError, "changed"):
                ConcurrentWriteEvidence(self.path)
        finally:
            writer.close()

    def test_individual_checks_and_attempt_coverage(self):
        connection = sqlite3.connect(self.path)
        connection.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE rows (source_id TEXT PRIMARY KEY,status TEXT,attempts INTEGER,classification_json TEXT,metadata_json TEXT);
        """)
        final = {"primary_function": "Rules", "subcategory": "Zoning", "confidence": "High"}
        metadata = {"request_sha256": "req", "response_sha256": "resp",
                    "usage": {"promptTokenCount": 10, "candidatesTokenCount": 2, "thoughtsTokenCount": 3, "totalTokenCount": 15}}
        config = {"provider": "google", "model": "fixture", "contract_sha256": "contract"}
        connection.execute("INSERT INTO meta VALUES ('run_metadata',?)", (json.dumps(config),))
        connection.execute("INSERT INTO rows VALUES (?,?,?,?,?)",
                           (SOURCE_ID, "ok", 3, json.dumps(final), json.dumps(metadata)))
        connection.commit()
        connection.close()
        exported = {"source": {"id": SOURCE_ID}, "final": final, "final_metadata": metadata,
                    "run_details": {"provider": "google", "requested_model": "fixture", "contract_sha256": "contract"}}
        with LedgerEvidence(self.path, kind="individual") as evidence:
            evidence.check(SOURCE_ID, exported)
            result = evidence.usage_summary()
            altered = deepcopy(exported)
            altered["final"]["subcategory"] = "Other"
            with self.assertRaises(ProvenanceError):
                evidence.check(SOURCE_ID, altered)
        self.assertEqual(result["input_tokens"], 10)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["reasoning_tokens"], 3)
        self.assertEqual(result["recorded_attempt_counter_total"], 3)
        self.assertFalse(result["all_attempts_available"])

    def test_unknown_ledger_kind_and_missing_file(self):
        with self.assertRaises(ProvenanceError):
            LedgerEvidence(self.path, kind="other")
        with self.assertRaises(FileNotFoundError):
            LedgerEvidence(self.path)


if __name__ == "__main__":
    unittest.main()
