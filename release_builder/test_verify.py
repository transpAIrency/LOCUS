"""Readback checks must reject corrupted staged values before publication."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from release_builder.records import ReleaseError
from release_builder.verify import verify_outputs
from release_builder.test_build import ReleaseFixture


@unittest.skipIf(pa is None, "pyarrow is required for Parquet verification tests")
class ReadbackTests(unittest.TestCase):
    def setUp(self):
        from release_builder.build import build_release, schemas
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        root = Path(self.scratch.name)
        fixture = ReleaseFixture(root)
        self.output = root / "release"
        manifest = build_release(fixture.config(), self.output, root,
                                 row_group_size=2, shard_target_bytes=1000)
        self.schemas = schemas()
        self.counts = manifest["counts"]
        self.samples = self.read("sections")

    def read(self, subset):
        return pq.read_table(self.output / subset).to_pylist()

    def write(self, subset, rows):
        paths = sorted((self.output / subset).glob("*.parquet"))
        pq.write_table(pa.Table.from_pylist(rows, schema=self.schemas[subset]),
                       paths[0], compression="zstd", row_group_size=2)
        for extra in paths[1:]:
            extra.unlink()

    def verify(self):
        return verify_outputs(self.output, self.schemas, self.counts, self.samples)

    def mutate_both(self, mutate):
        for subset in ("sections", "labels"):
            rows = self.read(subset)
            mutate(rows[0])
            self.write(subset, rows)

    def test_valid_release_matches_across_shards_and_independent_human_fields(self):
        result = self.verify()
        self.assertEqual(result["sections"], 3)
        self.assertEqual(result["source_samples_verified"], 3)
        self.assertTrue(result["consensus_recomputed"])
        self.assertTrue(result["sections_labels_equal"])
        self.assertEqual(result["model_null_counts"], {"gemini": 0, "openai": 0, "jev": 0})

    def test_matching_wrong_consensus_in_both_subsets_is_rejected(self):
        self.mutate_both(lambda row: row.update(function_votes=2))
        with self.assertRaisesRegex(ReleaseError, "votes disagree"):
            self.verify()

    def test_label_subset_value_difference_is_rejected(self):
        rows = self.read("labels")
        rows[0]["gemini_topic"] = "Business"
        self.write("labels", rows)
        with self.assertRaisesRegex(ReleaseError, "sections and labels differ"):
            self.verify()

    def test_missing_function_with_other_labels_is_rejected(self):
        self.mutate_both(lambda row: row.update(gemini_function=None))
        with self.assertRaisesRegex(ReleaseError, "Missing emitted function"):
            self.verify()

    def test_required_jev_probability_struct_cannot_be_null(self):
        self.mutate_both(lambda row: row.update(jev_function_probs=None))
        with self.assertRaisesRegex(ReleaseError, "probability keys"):
            self.verify()

    def test_changed_source_text_is_rejected_by_full_sample_readback(self):
        rows = self.read("sections")
        rows[0]["content"] = "Changed after build."
        self.write("sections", rows)
        with self.assertRaisesRegex(ReleaseError, "source sample differs"):
            self.verify()

    def test_out_of_order_rows_are_rejected(self):
        for subset in ("sections", "labels"):
            self.write(subset, list(reversed(self.read(subset))))
        with self.assertRaisesRegex(ReleaseError, "unsorted or duplicated"):
            self.verify()

    def test_county_section_counts_are_recomputed(self):
        rows = self.read("counties")
        rows[0]["section_count"] = 999
        self.write("counties", rows)
        with self.assertRaisesRegex(ReleaseError, "section_count differs"):
            self.verify()

    def test_sample_probabilities_use_declared_float32_rounding(self):
        sample = deepcopy(self.samples[0])
        sample["jev_function_probs"]["Rules"] = 0.33
        sample["jev_function_probs"]["Context"] = 0.67
        for subset in ("sections", "labels"):
            rows = self.read(subset)
            rows[0]["jev_function_probs"] = sample["jev_function_probs"]
            self.write(subset, rows)
        self.samples[0] = sample
        self.assertEqual(self.verify()["source_samples_verified"], 3)


if __name__ == "__main__":
    unittest.main()
