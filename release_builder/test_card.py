"""Publication metadata and measured dataset-card content regression tests."""

import unittest

from release_builder.card import render_dataset_card, validate_publication
from release_builder.records import ReleaseError


def manifest():
    eligible = {"function": 226, "topic": 216, "confidence": 244}
    return {
        "counts": {"sections": 3073770, "labels": 3073770, "human_audit": 250,
                   "counties": 3144, "documents": 2360, "covered_counties": 2410,
                   "uncovered_counties": 734},
        "models": {
            "google": {"model_id": "gemini-3.8-flash", "rows_missing": 0},
            "openai": {"model_id": "gpt-6-luna", "reasoning_effort": "low", "rows_missing": 0},
            "typesafe": {"model_id": "jev-1.14.0", "rows_missing": 0},
        },
        "human_audit_summary": {
            "rows": 250, "annotation_count_distribution": {"3": 250},
            "fields": {name: {"eligible": count} for name, count in eligible.items()},
            "model_agreement": {
                provider: {name: {"eligible": count, "scored": count - (1 if provider == "google" else 0),
                                  "matches": count - 10} for name, count in eligible.items()}
                for provider in ("google", "openai", "typesafe")
            },
        },
    }


def publication():
    return {"license": "cc-by-4.0", "repo_id": "example/locus-release",
            "citation": "@dataset{fixture,\n  title = {Fixture citation only}\n}"}


class DatasetCardTests(unittest.TestCase):
    def test_complete_card_uses_explicit_publication_and_measured_release_counts(self):
        card = render_dataset_card(manifest(), publication())
        self.assertIn('license: "cc-by-4.0"', card)
        self.assertIn('pretty_name: "LOCUS v1.1"', card)
        self.assertIn('- "1M<n<10M"', card)
        self.assertIn("- text-classification", card)
        self.assertIn("- municipal-law", card)
        self.assertIn("  default: true", card)
        self.assertEqual(card.count("  default: true"), 1)
        for subset in ("sections", "labels", "human_audit", "counties"):
            self.assertIn(f"path: {subset}/*.parquet", card)
            self.assertIn(f'load_dataset(repo_id, "{subset}", split="train"', card)
        self.assertIn('repo_id = "example/locus-release"', card)
        self.assertIn("3,073,770", card)
        self.assertIn("2,360 distinct documents", card)
        self.assertIn("2,410 represented", card)
        self.assertIn("76.7%", card)
        self.assertIn("**734 without a selected document**", card)
        self.assertIn(publication()["citation"], card)

    def test_intro_links_models_and_missing_counts_follow_manifest(self):
        data = manifest()
        data["models"]["google"]["rows_missing"] = 60
        data["models"]["openai"]["rows_missing"] = 1
        card = render_dataset_card(data, publication())
        self.assertIn("Local Ordinance Corpus for the United States", card)
        self.assertIn("municipal and county legal text harmonized to county geography", card)
        self.assertIn("[Paper](https://arxiv.org/abs/2606.19334)", card)
        self.assertIn("[Code](https://github.com/transpAIrency/LOCUS)", card)
        self.assertIn("Gemini (`gemini-3.8-flash`)", card)
        self.assertIn("OpenAI (`gpt-6-luna`, low reasoning)", card)
        self.assertIn("Jev (`jev-1.14.0`)", card)
        self.assertIn("Gemini **60**; OpenAI **1**; Jev **0**", card)
        self.assertIn("| Jev | function |", card)
        self.assertNotIn("TypeSafe", card)
        data["models"]["google"]["rows_missing"] = 0
        self.assertIn("Gemini **0**; OpenAI **1**; Jev **0**", render_dataset_card(data, publication()))

    def test_human_annotation_counts_distinguish_uniform_and_mixed_audits(self):
        data = manifest()
        self.assertIn("Each excerpt has three human annotations.", render_dataset_card(data, publication()))
        data["human_audit_summary"]["annotation_count_distribution"] = {"3": 249, "2": 1}
        card = render_dataset_card(data, publication())
        self.assertIn("Human annotations per excerpt: 2 for 1 excerpt; 3 for 249 excerpts.", card)
        self.assertNotIn("Each excerpt has three human annotations.", card)

    def test_unverified_model_facts_are_rejected(self):
        for mutate in (
            lambda data: data.pop("models"),
            lambda data: data["models"]["google"].update(model_id=""),
            lambda data: data["models"]["openai"].pop("reasoning_effort"),
            lambda data: data["models"]["typesafe"].update(rows_missing=True),
            lambda data: data["models"]["google"].update(rows_missing=3073771),
        ):
            data = manifest()
            mutate(data)
            with self.assertRaises(ReleaseError):
                render_dataset_card(data, publication())

    def test_sections_glossary_has_exact_25_columns_in_release_order(self):
        from release_builder.build import SECTION_COLUMNS
        section = render_dataset_card(manifest(), publication()).split("### sections\n", 1)[1].split("### labels\n", 1)[0]
        actual = [line.split("`", 2)[1] for line in section.splitlines() if line.startswith("| `")]
        self.assertEqual(actual, SECTION_COLUMNS)

    def test_human_agreement_preserves_scored_coverage_and_does_not_claim_pairwise(self):
        card = render_dataset_card(manifest(), publication())
        self.assertIn("| function | 226/250 (90.4%) |", card)
        self.assertIn("| topic | 216/250 (86.4%) |", card)
        self.assertIn("| confidence | 244/250 (97.6%) |", card)
        self.assertIn("| Gemini | function | 216/225 (96.0%) | 225/226 (99.6%) |", card)
        self.assertIn("not pairwise or chance-corrected", card)
        self.assertIn("Missing model answers reduce scored coverage", card)
        self.assertIn("eligible blank human topic is a valid null", card)

    def test_missing_publication_fields_are_rejected_without_placeholders(self):
        for field in ("license", "repo_id", "citation"):
            pub = publication()
            pub[field] = None
            with self.subTest(field=field), self.assertRaisesRegex(ReleaseError, field):
                render_dataset_card(manifest(), pub)
        with self.assertRaises(ReleaseError):
            validate_publication(None)

    def test_custom_license_requires_name_and_link_and_keeps_explicit_scope(self):
        pub = publication()
        pub["license"] = "other"
        with self.assertRaisesRegex(ReleaseError, "license_name.*license_link"):
            validate_publication(pub)
        pub.update(license_name="fixture-license", license_link="LICENSE.txt",
                   license_description="Fixture scope text supplied by the owner.")
        card = render_dataset_card(manifest(), pub)
        self.assertIn('license_name: "fixture-license"', card)
        self.assertIn('license_link: "LICENSE.txt"', card)
        self.assertIn(pub["license_description"], card)

    def test_repo_id_does_not_accept_a_url_or_code_injection(self):
        for value in ("https://huggingface.co/datasets/example/locus", "owner/../../name", 'owner/name"\nprint(1)', "owner/name/extra"):
            pub = publication()
            pub["repo_id"] = value
            with self.subTest(value=value), self.assertRaises(ReleaseError):
                validate_publication(pub)

    def test_size_category_is_measured_even_for_small_test_releases(self):
        data = manifest()
        data["counts"].update(sections=3, labels=3)
        card = render_dataset_card(data, publication())
        self.assertIn('- "n<1K"', card)
        self.assertNotIn('- "1M<n<10M"', card)

    def test_invalid_summary_or_coverage_is_rejected(self):
        for mutate in (
            lambda data: data["counts"].update(covered_counties=2411),
            lambda data: data["human_audit_summary"]["fields"]["topic"].update(eligible=251),
            lambda data: data["human_audit_summary"]["model_agreement"]["google"]["topic"].update(matches=999),
            lambda data: data["human_audit_summary"].update(annotation_count_distribution={"3": 249}),
            lambda data: data.pop("human_audit_summary"),
        ):
            data = manifest()
            mutate(data)
            with self.assertRaises(ReleaseError):
                render_dataset_card(data, publication())

    def test_zero_scored_denominator_is_explicit_and_citation_fence_is_safe(self):
        data = manifest()
        data["human_audit_summary"]["model_agreement"]["google"]["topic"].update(scored=0, matches=0)
        pub = publication()
        pub["citation"] = "Exact citation including ``` delimiters."
        card = render_dataset_card(data, pub)
        self.assertIn("| Gemini | topic | 0/0 (not available) | 0/216 (0.0%) |", card)
        self.assertIn("````text\n" + pub["citation"] + "\n````", card)


if __name__ == "__main__":
    unittest.main()
