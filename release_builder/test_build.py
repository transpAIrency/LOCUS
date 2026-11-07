"""Small offline release fixtures and end-to-end regression tests."""

from __future__ import annotations

from copy import deepcopy
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "llm_annotation" / "src"))

from neurips2026_locus.contract import PROMPT_VERSION, compose_input_text
from neurips2026_locus.pipeline import source_identity


PROVIDERS = ("google", "openai", "typesafe")
FIELDS = {"google": "classification_google", "openai": "classification_openai", "typesafe": "classification_jev"}
MODELS = {"google": "gemini-3.8-flash", "openai": "gpt-6-luna", "typesafe": "jev-1.14.0"}
FUNCTIONS = ("Context", "Rules", "Process", "Enforcement", "Structural")
TOPICS = ("Buildings", "Business", "Zoning", "Nuisance", "Other")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotation(record, position, provider, function="Rules", topic="Other"):
    sid, digest = source_identity(record, position)
    chars = len(compose_input_text(record))
    result = {
        "prompt_version": PROMPT_VERSION,
        "source": {"id": sid, "sha256": digest},
        "final": {"primary_function": function, "subcategory": topic, "confidence": "High"},
        "run_details": {
            "provider": provider,
            "requested_model": MODELS[provider],
            "request_submission": "individual" if provider == "typesafe" else "batch",
            "calls_per_record": 2 if provider == "typesafe" else 1,
            "reasoning_effort": None if provider == "typesafe" else "low",
            "contract_sha256": "a" * 64,
            "request_builder_sha256": "b" * 64,
            "input_text_characters": chars,
            "sent_text_characters": chars,
            "text_truncated": False,
            "annotated_at_utc": "2026-11-06T00:01:00Z",
        },
        "final_metadata": {"input_tokens": 100, "output_tokens": 10},
    }
    if provider == "typesafe":
        result["stage_metadata"] = {
            "classification": {
                "answers": {
                    "primary_function": {"choice": function, "probabilities": {name: float(name == function) for name in FUNCTIONS}},
                    "subcategory": {"choice": topic or "Other", "probabilities": {name: float(name == (topic or "Other")) for name in TOPICS}},
                },
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
            "confidence": {"usage": {"input_tokens": 50, "output_tokens": 5}},
        }
    if "audit_id" in record:
        return {"audit_id": record["audit_id"], FIELDS[provider]: result}
    return {"document_id": record["document_id"], "index": record["index"], FIELDS[provider]: result}


def make_ledger(path, input_path, provider, rows):
    with closing(sqlite3.connect(path)) as db, db:
        # Production ledgers use WAL and are closed before release validation.
        # Exercise the same clean-close/open lifecycle in every integration run.
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE annotations (
                source_id TEXT PRIMARY KEY, source_index INTEGER NOT NULL,
                source_sha256 TEXT NOT NULL, status TEXT NOT NULL,
                annotation_json TEXT NOT NULL, created_at_utc TEXT NOT NULL);
            CREATE TABLE responses (
                source_id TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                response_json TEXT NOT NULL, response_sha256 TEXT NOT NULL,
                PRIMARY KEY (source_id, request_sha256));
        """)
        metadata = {
            "provider": provider, "model": MODELS[provider],
            "requested_model": MODELS[provider], "prompt_version": PROMPT_VERSION,
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "contract_sha256": "a" * 64, "request_builder_sha256": "b" * 64,
            "max_text_chars": 16000, "reasoning_effort": None if provider == "typesafe" else "low",
        }
        db.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", "1"), ("request_builder_sha256", "b" * 64),
            ("run_metadata", json.dumps(metadata)),
        ])
        for position, row in enumerate(rows):
            cls = row[FIELDS[provider]]
            db.execute("INSERT INTO annotations VALUES (?,?,?,?,?,?)", (
                cls["source"]["id"], position, cls["source"]["sha256"],
                "annotated", json.dumps(cls), "2026-11-06T00:01:00Z",
            ))


class ReleaseFixture:
    def __init__(self, root):
        self.root = root
        self.sections = [
            {"document_id": "doc-ca", "index": 7, "state": "ca", "jurisdiction_geoid": "0600123", "header": " ##  Heading  ", "content": "  Unchanged body.\n"},
            {"document_id": "doc-ak", "index": 10, "state": "ak", "jurisdiction_geoid": "02001", "header": "# Later", "content": "Later body."},
            {"document_id": "doc-ak", "index": 0, "state": "ak", "jurisdiction_geoid": "02001", "header": "# First", "content": "First body."},
        ]
        self.source = root / "sections.jsonl"
        write_jsonl(self.source, self.sections)
        self.counties = root / "counties.csv"
        common = {"selected_page_count": "999", "selection_reason": "fixture"}
        write_csv(self.counties, [
            {"county_fips": "06003", "county_name": "Second County", "state": "CA", "selected_document_id": "doc-ca", "selected_jurisdiction_name": "Example", "selected_city_county": "city", **common},
            {"county_fips": "02001", "county_name": "First County", "state": "AK", "selected_document_id": "doc-ak", "selected_jurisdiction_name": "North", "selected_city_county": "county", **common},
            {"county_fips": "06001", "county_name": "Third County", "state": "CA", "selected_document_id": "doc-ca", "selected_jurisdiction_name": "Example", "selected_city_county": "city", **common},
            {"county_fips": "06005", "county_name": "Uncovered County", "state": "CA", "selected_document_id": "", "selected_jurisdiction_name": "", "selected_city_county": "", **common},
        ])
        self.human = root / "human.csv"
        write_csv(self.human, [
            {"audit_id": "AUDIT-1", "text": "Human example one.", "function": "Rules", "topic": "", "confidence": "High", "function_eligible": "1", "topic_eligible": "0", "confidence_eligible": "1", "n_annotations": "3"},
            {"audit_id": "AUDIT-2", "text": "Human example two.", "function": "", "topic": "Business", "confidence": "", "function_eligible": "0", "topic_eligible": "1", "confidence_eligible": "0", "n_annotations": "3"},
        ])
        self.audit_records = [
            {"audit_id": "AUDIT-1", "header": "", "content": "Human example one."},
            {"audit_id": "AUDIT-2", "header": "", "content": "Human example two."},
        ]
        self.audit_source = root / "audit_input.jsonl"
        write_jsonl(self.audit_source, self.audit_records)
        self.exports = {}
        self.ledgers = {}
        self.audit_exports = {}
        self.audit_ledgers = {}
        self.rows = {}
        for provider in PROVIDERS:
            # The AK document has two answering models with no applicable topic.
            rows = [annotation(row, i, provider, "Context" if i == 1 else "Rules", None if i == 1 else "Other") for i, row in enumerate(self.sections)]
            self.rows[provider] = rows
            self.exports[provider] = root / f"{provider}.jsonl"
            self.ledgers[provider] = root / f"{provider}.sqlite3"
            write_jsonl(self.exports[provider], list(reversed(rows)))
            make_ledger(self.ledgers[provider], self.source, provider, rows)
            audits = [annotation(row, i, provider) for i, row in enumerate(self.audit_records)]
            self.audit_exports[provider] = root / f"audit_{provider}.jsonl"
            self.audit_ledgers[provider] = root / f"audit_{provider}.sqlite3"
            write_jsonl(self.audit_exports[provider], list(reversed(audits)))
            make_ledger(self.audit_ledgers[provider], self.audit_source, provider, audits)

    def config(self):
        return {
            "publication": {
                "license": "cc-by-4.0",
                "repo_id": "test-owner/test-locus",
                "citation": "Test fixture citation.",
            },
            "inputs": {"sections": str(self.source), "counties": str(self.counties), "human": str(self.human)},
            "expected": {
                "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                "sections": 3, "documents": 2, "counties": 4,
                "covered_counties": 3, "human_audit": 2,
            },
            "models": {
                provider: {
                    "exports": [{"path": str(self.exports[provider]), "ledger": str(self.ledgers[provider]), "kind": "harness"}],
                    "audit": {"path": str(self.audit_exports[provider]), "ledger": str(self.audit_ledgers[provider]), "input": str(self.audit_source)},
                    "completed": True,
                    "run_start_utc": None, "run_end_utc": None,
                    "run_dates_note": "Run boundaries were not recorded for this synthetic test.",
                }
                for provider in PROVIDERS
            },
            "usage_scope": "recorded", "max_text_chars": 16000,
        }


@unittest.skipIf(pa is None, "pyarrow is required for release integration tests")
class ReleaseBuildTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.fixture = ReleaseFixture(self.root)
        self.output = self.root / "release"

    def run_build(self, config=None, output="default", **kwargs):
        from release_builder.build import build_release
        if output == "default":
            output = self.output
        return build_release(
            config or self.fixture.config(), output=output, base_dir=self.root,
            row_group_size=2, shard_target_bytes=1000,
            sample_size=kwargs.pop("sample_size", 20), seed=13,
            **kwargs,
        )

    def assert_rejected_without_output(self, config=None):
        from release_builder.records import ReleaseError
        with self.assertRaises((ReleaseError, ValueError)):
            self.run_build(config=config)
        self.assertFalse(self.output.exists())

    def test_full_tiny_release_has_exact_schema_sorting_and_faithful_values(self):
        manifest = self.run_build()
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        sections = pq.read_table(self.output / "sections")
        expected_columns = [
            "document_id", "section_index", "state", "jurisdiction_name", "jurisdiction_type",
            "jurisdiction_geoid", "represented_county_fips", "represented_county_name", "header", "content",
            "function_majority", "function_votes", "topic_majority", "topic_votes",
            "gemini_function", "gemini_topic", "gemini_confidence",
            "openai_function", "openai_topic", "openai_confidence",
            "jev_function", "jev_topic", "jev_confidence", "jev_function_probs", "jev_topic_probs",
        ]
        self.assertEqual(sections.column_names, expected_columns)
        self.assertEqual(sections.schema.field("section_index").type, pa.int64())
        self.assertEqual(sections.schema.field("function_votes").type, pa.int8())
        self.assertEqual(sections.schema.field("topic_votes").type, pa.int8())
        for struct_name, names in (("jev_function_probs", FUNCTIONS), ("jev_topic_probs", TOPICS)):
            dtype = sections.schema.field(struct_name).type
            self.assertEqual([field.name for field in dtype], list(names))
            self.assertTrue(all(field.type == pa.float32() for field in dtype))
        records = sections.to_pylist()
        self.assertEqual([(row["state"], row["document_id"], row["section_index"]) for row in records], [
            ("AK", "doc-ak", 0), ("AK", "doc-ak", 10), ("CA", "doc-ca", 7),
        ])
        ca = records[-1]
        self.assertEqual(ca["jurisdiction_geoid"], "0600123")
        self.assertEqual(ca["represented_county_fips"], ["06001", "06003"])
        self.assertEqual(ca["represented_county_name"], ["Third County", "Second County"])
        self.assertEqual(ca["header"], "Heading")
        self.assertEqual(ca["content"], "  Unchanged body.\n")
        self.assertEqual(ca["function_majority"], "Rules")
        self.assertEqual(ca["topic_majority"], "Other")
        self.assertEqual(ca["function_votes"], 3)
        self.assertEqual(ca["topic_votes"], 3)
        context = records[1]
        self.assertIsNone(context["topic_majority"])
        self.assertEqual(context["topic_votes"], 3)
        self.assertIsNone(context["jev_topic_probs"])
        self.assertIsNotNone(context["jev_function_probs"])
        labels = pq.read_table(self.output / "labels")
        self.assertEqual(labels.column_names, expected_columns[:2] + expected_columns[10:])
        self.assertEqual(len(labels.column_names), 17)
        self.assertEqual(labels.to_pylist(), [{name: row[name] for name in labels.column_names} for row in records])
        counties = pq.read_table(self.output / "counties")
        self.assertNotIn("selected_page_count", counties.column_names)
        by_fips = {row["county_fips"]: row for row in counties.to_pylist()}
        self.assertEqual(by_fips["02001"]["section_count"], 2)
        self.assertEqual(by_fips["06001"]["section_count"], 1)
        self.assertIsNone(by_fips["06005"]["section_count"])
        self.assertIsNone(by_fips["06005"]["jurisdiction_name"])
        audits = {row["audit_id"]: row for row in pq.read_table(self.output / "human_audit").to_pylist()}
        self.assertEqual(audits["AUDIT-1"]["human_function"], "Rules")
        self.assertIsNone(audits["AUDIT-1"]["human_topic"])
        self.assertFalse(audits["AUDIT-1"]["human_topic_eligible"])
        self.assertIsNone(audits["AUDIT-2"]["human_function"])
        self.assertEqual(audits["AUDIT-2"]["human_topic"], "Business")
        self.assertTrue(audits["AUDIT-2"]["human_topic_eligible"])
        summary = manifest["human_audit_summary"]
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["annotation_count_distribution"], {"3": 2})
        for field in ("function", "topic", "confidence"):
            self.assertEqual(summary["fields"][field]["eligible"], 1)
        for provider in PROVIDERS:
            scores = summary["model_agreement"][provider]
            self.assertEqual(scores["function"], {"eligible": 1, "scored": 1, "matches": 1, "accuracy": 1.0})
            self.assertEqual(scores["topic"], {"eligible": 1, "scored": 1, "matches": 0, "accuracy": 0.0})
            self.assertEqual(scores["confidence"], {"eligible": 1, "scored": 1, "matches": 1, "accuracy": 1.0})
        card = (self.output / "README.md").read_text(encoding="utf-8")
        self.assertIn("test-owner/test-locus", card)
        self.assertIn("Test fixture citation.", card)
        self.assertIn("cc-by-4.0", card)
        self.assertIn("load_dataset", card)
        self.assertIn("| function | 1/2 (50.0%) |", card)
        self.assertIn("| Gemini | function | 1/1 (100.0%) | 1/1 (100.0%) |", card)
        self.assertIn("| OpenAI | topic | 0/1 (0.0%) | 1/1 (100.0%) |", card)
        self.assertIn("| Jev | confidence | 1/1 (100.0%) | 1/1 (100.0%) |", card)
        for column in expected_columns:
            self.assertIn(f"`{column}`", card)
        for path in self.output.rglob("*.parquet"):
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                self.assertLessEqual(parquet.metadata.row_group(group).num_rows, 2)
                self.assertEqual(parquet.metadata.row_group(group).column(0).compression, "ZSTD")

    def test_city_county_jurisdiction_passes_full_build_and_readback(self):
        with self.fixture.counties.open(encoding="utf-8", newline="") as handle:
            counties = list(csv.DictReader(handle))
        for row in counties:
            if row["selected_document_id"] == "doc-ak":
                row["selected_city_county"] = "both"
        write_csv(self.fixture.counties, counties)

        manifest = self.run_build()

        county = next(row for row in pq.read_table(self.output / "counties").to_pylist()
                      if row["document_id"] == "doc-ak")
        self.assertEqual(county["jurisdiction_type"], "city-county")
        sections = [row for row in pq.read_table(self.output / "sections").to_pylist()
                    if row["document_id"] == "doc-ak"]
        self.assertEqual(len(sections), 2)
        self.assertTrue(all(row["jurisdiction_type"] == "city-county" for row in sections))
        self.assertEqual(manifest["validation"]["parquet_readback"]["counties"], 4)

    def test_validation_only_does_not_publish_release_or_mutate_inputs(self):
        originals = {path: path.read_bytes() for path in [self.fixture.source, self.fixture.counties, self.fixture.human]}
        result = self.run_build(output=None)
        self.assertIsInstance(result, dict)
        self.assertFalse(self.output.exists())
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_missing_publication_license_stops_before_loading_corpus(self):
        config = self.fixture.config()
        config["publication"]["license"] = None
        with patch("release_builder.build.Inputs.lines", side_effect=AssertionError("Corpus was read")):
            self.assert_rejected_without_output(config)

    def test_validation_only_does_not_require_publication_metadata(self):
        config = self.fixture.config()
        config.pop("publication")
        manifest = self.run_build(config=config, output=None)
        self.assertEqual(manifest["counts"]["human_audit"], 2)
        self.assertEqual(manifest["human_audit_summary"]["rows"], 2)
        self.assertFalse(self.output.exists())

    def test_duplicate_annotations_fail_atomically(self):
        rows = self.fixture.rows["google"]
        write_jsonl(self.fixture.exports["google"], rows + [rows[0]])
        self.assert_rejected_without_output()

    def test_duplicate_section_keys_fail_atomically(self):
        write_jsonl(self.fixture.source, self.fixture.sections + [self.fixture.sections[0]])
        self.assert_rejected_without_output()

    def test_extra_annotation_key_is_rejected(self):
        rows = deepcopy(self.fixture.rows["google"])
        extra = deepcopy(rows[0])
        extra["document_id"] = "not-in-source"
        write_jsonl(self.fixture.exports["google"], rows + [extra])
        self.assert_rejected_without_output()

    def test_source_content_identity_change_is_rejected_even_with_new_expected_hash(self):
        rows = deepcopy(self.fixture.sections)
        rows[0]["content"] = "Changed legal text."
        write_jsonl(self.fixture.source, rows)
        self.assert_rejected_without_output()

    def test_export_labels_changed_from_ledger_are_rejected(self):
        rows = deepcopy(self.fixture.rows["google"])
        rows[0][FIELDS["google"]]["final"]["subcategory"] = "Business"
        write_jsonl(self.fixture.exports["google"], rows)
        self.assert_rejected_without_output()

    def test_unfinished_run_cannot_be_called_final(self):
        config = self.fixture.config()
        config["models"]["openai"]["completed"] = False
        self.assert_rejected_without_output(config)

    def test_missing_export_is_rejected(self):
        self.fixture.exports["openai"].unlink()
        self.assert_rejected_without_output()

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("Preserve existing release.")
        from release_builder.records import ReleaseError
        with self.assertRaises((ReleaseError, ValueError)):
            self.run_build()
        self.assertEqual(marker.read_text(), "Preserve existing release.")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_missing_model_is_missing_and_does_not_cast_not_applicable_vote(self):
        # Removing Google from the Context row leaves exactly two NA topic votes.
        write_jsonl(self.fixture.exports["google"], [row for i, row in enumerate(self.fixture.rows["google"]) if i != 1])
        manifest = self.run_build()
        rows = pq.read_table(self.output / "sections").to_pylist()
        row = next(row for row in rows if row["document_id"] == "doc-ak" and row["section_index"] == 10)
        self.assertIsNone(row["gemini_function"])
        self.assertIsNone(row["gemini_topic"])
        self.assertIsNone(row["gemini_confidence"])
        self.assertEqual(row["function_votes"], 2)
        self.assertEqual(row["topic_votes"], 2)
        self.assertIsNone(row["topic_majority"])
        self.assertIsInstance(manifest, dict)

    def test_disjoint_batch_and_individual_recovery_exports_can_be_combined(self):
        provider = "google"
        recovered = deepcopy(self.fixture.rows[provider][1])
        cls = recovered[FIELDS[provider]]
        cls["source"] = {"id": cls["source"]["id"]}
        cls["run_details"]["request_submission"] = "individual"
        individual_export = self.root / "individual_google.jsonl"
        individual_ledger = self.root / "individual_google.sqlite3"
        write_jsonl(individual_export, [recovered])
        write_jsonl(self.fixture.exports[provider], [row for i, row in enumerate(self.fixture.rows[provider]) if i != 1])
        with closing(sqlite3.connect(individual_ledger)) as db, db:
            db.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE rows (
                    source_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    classification_json TEXT, metadata_json TEXT, attempts INTEGER);
            """)
            db.execute("INSERT INTO meta VALUES (?,?)", ("run_metadata", json.dumps({
                "provider": provider, "model": MODELS[provider], "reasoning_effort": "low",
                "contract_sha256": "a" * 64,
                "request_builder_sha256": "b" * 64,
                "input_sha256": hashlib.sha256(self.fixture.source.read_bytes()).hexdigest(),
            })))
            db.execute("INSERT INTO rows VALUES (?,?,?,?,?)", (
                cls["source"]["id"], "ok", json.dumps(cls["final"]), json.dumps(cls["final_metadata"]), 1,
            ))
        config = self.fixture.config()
        config["models"][provider]["exports"].append({"path": str(individual_export), "ledger": str(individual_ledger), "kind": "individual"})
        self.run_build(config)
        rows = pq.read_table(self.output / "sections").to_pylist()
        context = next(row for row in rows if row["document_id"] == "doc-ak" and row["section_index"] == 10)
        self.assertEqual(context["gemini_function"], "Context")
        self.assertEqual(context["function_votes"], 3)

    def _make_combined_openai_export(self):
        provider = "openai"
        recovered = deepcopy(self.fixture.rows[provider][1])
        cls = recovered[FIELDS[provider]]
        # The actual recovery exporter preserves source.id alone, rather than
        # fabricating the batch ledger's source.sha256 or source locator.
        cls["source"] = {"id": cls["source"]["id"]}
        cls["run_details"]["request_submission"] = "individual"
        ledger = self.root / "combined_openai_recovery.sqlite3"
        with closing(sqlite3.connect(ledger)) as db, db:
            db.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE rows (
                    source_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    classification_json TEXT, metadata_json TEXT, attempts INTEGER);
            """)
            db.execute("INSERT INTO meta VALUES (?,?)", ("run_metadata", json.dumps({
                "provider": provider, "model": MODELS[provider],
                "reasoning_effort": "low", "contract_sha256": "a" * 64,
                "request_builder_sha256": "b" * 64,
                "input_sha256": hashlib.sha256(self.fixture.source.read_bytes()).hexdigest(),
            })))
            db.execute("INSERT INTO rows VALUES (?,?,?,?,?)", (
                cls["source"]["id"], "ok", json.dumps(cls["final"]),
                json.dumps(cls["final_metadata"]), 1,
            ))
        rows = [deepcopy(row) for i, row in enumerate(self.fixture.rows[provider]) if i != 1]
        rows.append(recovered)
        write_jsonl(self.fixture.exports[provider], rows)
        config = self.fixture.config()
        config["models"][provider]["exports"][0]["individual_ledger"] = str(ledger)
        return config, recovered

    def test_one_combined_export_routes_individual_rows_to_recovery_ledger(self):
        config, recovered = self._make_combined_openai_export()
        manifest = self.run_build(config)
        self.assertEqual(len(config["models"]["openai"]["exports"]), 1)
        model = manifest["models"]["openai"]
        self.assertEqual(model["rows_annotated"], 3)
        self.assertEqual(model["rows_missing"], 0)
        self.assertEqual(model["rows_by_request_submission"], {"batch": 2, "individual": 1})
        context = next(row for row in pq.read_table(self.output / "sections").to_pylist()
                       if row["document_id"] == "doc-ak" and row["section_index"] == 10)
        self.assertEqual(context["openai_function"], "Context")
        self.assertEqual(context["function_votes"], 3)
        self.assertTrue(any(item["kind"] == "individual" for item in manifest["ledgers"]))
        # Release building must not rewrite the original export provenance.
        current = [json.loads(line) for line in self.fixture.exports["openai"].read_text().splitlines()]
        self.assertEqual(current[-1], recovered)
        self.assertEqual(set(current[-1][FIELDS["openai"]]["source"]), {"id"})

    def test_combined_recovery_is_ledger_checked_outside_random_sample(self):
        from release_builder.records import ReleaseError
        config, recovered = self._make_combined_openai_export()
        # With seed13 and sample_size1, the input reservoir retains source row0;
        # this recovery is source row1 and must still receive an exact check.
        valid_manifest = self.run_build(config, output=None, sample_size=1)
        self.assertNotIn(
            {"document_id": recovered["document_id"], "section_index": recovered["index"]},
            valid_manifest["validation"]["sampled_sections"],
        )
        recovered[FIELDS["openai"]]["final"]["confidence"] = "Low"
        rows = [json.loads(line) for line in self.fixture.exports["openai"].read_text().splitlines()]
        rows[-1] = recovered
        write_jsonl(self.fixture.exports["openai"], rows)
        with self.assertRaisesRegex(ReleaseError, "individual final labels"):
            self.run_build(config, sample_size=1)
        self.assertFalse(self.output.exists())

    def test_combined_export_requires_recovery_ledger(self):
        from release_builder.records import ReleaseError
        config, _ = self._make_combined_openai_export()
        config["models"]["openai"]["exports"][0].pop("individual_ledger")
        with self.assertRaisesRegex(ReleaseError, "requires individual_ledger"):
            self.run_build(config, sample_size=1)
        self.assertFalse(self.output.exists())

    def test_duplicate_keys_in_combined_export_are_rejected(self):
        config, _ = self._make_combined_openai_export()
        rows = [json.loads(line) for line in self.fixture.exports["openai"].read_text().splitlines()]
        write_jsonl(self.fixture.exports["openai"], rows + [rows[-1]])
        self.assert_rejected_without_output(config)

    def test_reconciled_harness_row_preserves_individual_transport_metadata(self):
        provider = "openai"
        rows = deepcopy(self.fixture.rows[provider])
        reconciled = rows[1][FIELDS[provider]]
        # Reconciliation retains the harness annotation contract while recording
        # the actual recovery transport and request identity in final_metadata.
        reconciled["final_metadata"].update({
            "request_submission": "individual",
            "request_id": "req-fixture-individual",
            "response_id": "resp-fixture-individual",
            "request_sha256": "c" * 64,
        })
        original_metadata = deepcopy(reconciled["final_metadata"])
        self.assertEqual(reconciled["run_details"]["request_submission"], "batch")
        self.assertNotIn("batch_id", original_metadata)
        with closing(sqlite3.connect(self.fixture.ledgers[provider])) as db, db:
            db.execute("UPDATE annotations SET annotation_json=? WHERE source_id=?", (
                json.dumps(reconciled), reconciled["source"]["id"],
            ))
        write_jsonl(self.fixture.exports[provider], rows)
        original_export = self.fixture.exports[provider].read_bytes()
        config = self.fixture.config()
        self.assertNotIn("individual_ledger", config["models"][provider]["exports"][0])
        manifest = self.run_build(config)
        self.assertEqual(manifest["models"][provider]["rows_annotated"], 3)
        self.assertEqual(manifest["models"][provider]["rows_by_request_submission"], {"batch": 3})
        self.assertTrue(all(item["kind"] == "harness" for item in manifest["ledgers"]))
        self.assertEqual(self.fixture.exports[provider].read_bytes(), original_export)
        current = [json.loads(line) for line in self.fixture.exports[provider].read_text().splitlines()]
        self.assertEqual(current[1][FIELDS[provider]]["final_metadata"], original_metadata)
        result = next(row for row in pq.read_table(self.output / "sections").to_pylist()
                      if row["document_id"] == "doc-ak" and row["section_index"] == 10)
        self.assertEqual(result["openai_function"], "Context")

    def _one_missing_openai_row_with_permanent_note(self):
        provider = "openai"
        source = self.fixture.rows[provider][1]
        failed = deepcopy(source[FIELDS[provider]])
        failed.pop("final")
        failed["error"] = {
            "provider_code": "invalid_annotation",
            "message": "Provider classification does not satisfy the annotation schema",
            "permanent": True,
        }
        with closing(sqlite3.connect(self.fixture.ledgers[provider])) as db, db:
            db.execute("UPDATE annotations SET status='failed',annotation_json=? WHERE source_id=?", (
                json.dumps(failed), failed["source"]["id"],
            ))
        write_jsonl(self.fixture.exports[provider], [
            row for i, row in enumerate(self.fixture.rows[provider]) if i != 1
        ])
        note = {
            "document_id": source["document_id"], "section_index": source["index"],
            "reason": failed["error"]["message"],
        }
        config = self.fixture.config()
        config["models"][provider]["permanent_failures"] = [note]
        return config, note

    def test_permanent_failure_note_is_preserved_for_missing_annotation(self):
        config, note = self._one_missing_openai_row_with_permanent_note()
        manifest = self.run_build(config)
        self.assertEqual(manifest["models"]["openai"]["permanent_failures"], [note])
        self.assertEqual(manifest["models"]["openai"]["rows_missing"], 1)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        result = next(row for row in pq.read_table(self.output / "sections").to_pylist()
                      if row["document_id"] == note["document_id"] and row["section_index"] == note["section_index"])
        self.assertIsNone(result["openai_function"])
        self.assertIsNone(result["openai_topic"])
        self.assertIsNone(result["openai_confidence"])
        self.assertEqual(result["function_votes"], 2)

    def test_permanent_failure_note_cannot_override_a_valid_annotation(self):
        from release_builder.records import ReleaseError
        source = self.fixture.rows["openai"][1]
        config = self.fixture.config()
        config["models"]["openai"]["permanent_failures"] = [{
            "document_id": source["document_id"], "section_index": source["index"],
            "reason": "Must not label a successful answer permanently failed.",
        }]
        with self.assertRaisesRegex(ReleaseError, "contradicts a valid annotation"):
            self.run_build(config)
        self.assertFalse(self.output.exists())

    def test_permanent_failure_notes_require_unique_known_keys_and_reasons(self):
        config, note = self._one_missing_openai_row_with_permanent_note()
        invalid_notes = [
            [note, deepcopy(note)],
            [{**note, "document_id": "unknown-source"}],
            [{**note, "reason": "  "}],
            [{**note, "section_index": True}],
        ]
        for notes in invalid_notes:
            with self.subTest(notes=notes):
                invalid_config = deepcopy(config)
                invalid_config["models"]["openai"]["permanent_failures"] = notes
                self.assert_rejected_without_output(invalid_config)

    def test_all_providers_missing_on_one_row_yields_zero_votes_and_null_probability_structs(self):
        for provider in PROVIDERS:
            write_jsonl(self.fixture.exports[provider], [row for i, row in enumerate(self.fixture.rows[provider]) if i != 1])
        self.run_build()
        row = next(row for row in pq.read_table(self.output / "sections").to_pylist()
                   if row["document_id"] == "doc-ak" and row["section_index"] == 10)
        self.assertEqual(row["function_votes"], 0)
        self.assertEqual(row["topic_votes"], 0)
        self.assertIsNone(row["function_majority"])
        self.assertIsNone(row["topic_majority"])
        for prefix in ("gemini", "openai", "jev"):
            for field in ("function", "topic", "confidence"):
                self.assertIsNone(row[f"{prefix}_{field}"])
        self.assertIsNone(row["jev_function_probs"])
        self.assertIsNone(row["jev_topic_probs"])

    def test_missing_audit_response_preserves_human_row_with_null_model_labels(self):
        provider = "openai"
        rows = [annotation(row, i, provider) for i, row in enumerate(self.fixture.audit_records)]
        write_jsonl(self.fixture.audit_exports[provider], rows[:1])
        manifest = self.run_build()
        rows = {row["audit_id"]: row for row in pq.read_table(self.output / "human_audit").to_pylist()}
        self.assertEqual(len(rows), 2)
        missing = rows["AUDIT-2"]
        self.assertEqual(missing["human_topic"], "Business")
        self.assertTrue(missing["human_topic_eligible"])
        for field in ("openai_function", "openai_topic", "openai_confidence"):
            self.assertIsNone(missing[field])
        self.assertEqual(missing["gemini_function"], "Rules")
        self.assertEqual(missing["jev_function"], "Rules")
        self.assertEqual(manifest["models"][provider]["human_audit_rows_missing"], 1)
        score = manifest["human_audit_summary"]["model_agreement"][provider]["topic"]
        self.assertEqual(score, {"eligible": 1, "scored": 0, "matches": 0, "accuracy": None})

    def _make_first_audit_topic_not_applicable(self):
        with self.fixture.human.open(encoding="utf-8", newline="") as handle:
            humans = list(csv.DictReader(handle))
        humans[0].update(function="Context", topic="", topic_eligible="1")
        write_csv(self.fixture.human, humans)
        for provider in PROVIDERS:
            rows = [
                annotation(record, position, provider,
                           "Context" if position == 0 else "Rules",
                           None if position == 0 else "Other")
                for position, record in enumerate(self.fixture.audit_records)
            ]
            write_jsonl(self.fixture.audit_exports[provider], rows)
            self.fixture.audit_ledgers[provider].unlink()
            make_ledger(self.fixture.audit_ledgers[provider], self.fixture.audit_source, provider, rows)
        return humans

    def test_eligible_null_human_topic_matches_answered_not_applicable_topic(self):
        self._make_first_audit_topic_not_applicable()
        manifest = self.run_build()
        summary = manifest["human_audit_summary"]
        self.assertEqual(summary["fields"]["topic"]["eligible"], 2)
        for provider in PROVIDERS:
            score = summary["model_agreement"][provider]["topic"]
            self.assertEqual(score, {"eligible": 2, "scored": 2, "matches": 1, "accuracy": 0.5})

    def test_missing_response_is_not_an_agreement_with_eligible_null_human_topic(self):
        self._make_first_audit_topic_not_applicable()
        with self.fixture.audit_exports["openai"].open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        write_jsonl(self.fixture.audit_exports["openai"], rows[1:])
        manifest = self.run_build()
        scores = manifest["human_audit_summary"]["model_agreement"]
        self.assertEqual(scores["openai"]["topic"], {"eligible": 2, "scored": 1, "matches": 0, "accuracy": 0.0})
        self.assertEqual(scores["openai"]["function"], {"eligible": 1, "scored": 0, "matches": 0, "accuracy": None})
        self.assertEqual(scores["google"]["topic"], {"eligible": 2, "scored": 2, "matches": 1, "accuracy": 0.5})

    def test_ledger_context_exit_failure_does_not_publish_release(self):
        from release_builder.provenance import LedgerEvidence, ProvenanceError
        original_exit = LedgerEvidence.__exit__

        def reject_changed_evidence(ledger, exc_type, exc_value, traceback):
            result = original_exit(ledger, exc_type, exc_value, traceback)
            if exc_type is None:
                raise ProvenanceError("Synthetic ledger mutation detected during final context exit")
            return result

        with patch.object(LedgerEvidence, "__exit__", reject_changed_evidence):
            self.assert_rejected_without_output()


if __name__ == "__main__":
    unittest.main()
