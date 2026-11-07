"""Render a publication-ready card from verified counts and explicit metadata.

Licensing, repository identity, and citation are release decisions supplied by
the caller. This module never infers them from filenames or model output.
"""

from __future__ import annotations

import json
import re

from .records import ReleaseError


_PROVIDERS = {"google": "Gemini", "openai": "OpenAI", "typesafe": "Jev"}
_FIELDS = ("function", "topic", "confidence")
_SUBSETS = ("sections", "labels", "human_audit", "counties")


def _require(condition, message):
    if not condition:
        raise ReleaseError(message)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value):
    return type(value) is int and value >= 0


def validate_publication(publication):
    """Validate explicit release decisions without touching the network."""
    _require(isinstance(publication, dict), "Publication metadata must be an object")
    missing = [name for name in ("license", "repo_id", "citation") if not _nonempty(publication.get(name))]
    _require(not missing, "Publication metadata needs explicit " + ", ".join(missing))
    result = dict(publication)
    license_id = result["license"]
    _require(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", license_id) is not None,
             "Publication license must be a Hugging Face license identifier")
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", result["repo_id"]) is not None,
             "Publication repo_id must have the form owner/dataset")
    for component in result["repo_id"].split("/"):
        _require(".." not in component and not component.endswith((".", "-")),
                 "Publication repo_id contains an invalid repository component")
    if license_id == "other":
        required = [name for name in ("license_name", "license_link") if not _nonempty(result.get(name))]
        _require(not required, "Custom license metadata needs explicit " + ", ".join(required))
    for name in ("pretty_name", "license_name", "license_link", "license_description"):
        if name in result and result[name] is not None:
            _require(_nonempty(result[name]), f"Publication {name} must be nonempty text")
    result.setdefault("pretty_name", "LOCUS v1.1")
    if result["pretty_name"] is None:
        result["pretty_name"] = "LOCUS v1.1"
    return result


def _yaml(value):
    # A JSON string is also a YAML quoted scalar, including its escape rules.
    return json.dumps(value, ensure_ascii=False)


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _size_category(rows):
    boundaries = ((1000, "n<1K"), (10000, "1K<n<10K"), (100000, "10K<n<100K"),
                  (1000000, "100K<n<1M"), (10000000, "1M<n<10M"),
                  (100000000, "10M<n<100M"), (1000000000, "100M<n<1B"))
    return next((name for maximum, name in boundaries if rows < maximum), "n>1B")


def _counts(manifest):
    counts = manifest.get("counts")
    _require(isinstance(counts, dict), "Dataset card needs verified counts")
    for name in (*_SUBSETS, "documents", "covered_counties", "uncovered_counties"):
        _require(_integer(counts.get(name)), f"Dataset card needs a nonnegative {name} count")
    _require(counts["counties"] > 0, "Dataset card needs a nonempty county universe")
    _require(counts["covered_counties"] + counts["uncovered_counties"] == counts["counties"],
             "Dataset card county coverage counts do not add up")
    _require(counts["labels"] == counts["sections"], "Dataset card sections/labels counts differ")
    return counts


def _summary(manifest, counts):
    summary = manifest.get("human_audit_summary")
    _require(isinstance(summary, dict), "Dataset card needs measured human_audit_summary")
    _require(summary.get("rows") == counts["human_audit"], "Dataset card human-audit summary count differs")
    fields = summary.get("fields")
    agreement = summary.get("model_agreement")
    distribution = summary.get("annotation_count_distribution")
    _require(isinstance(fields, dict) and isinstance(agreement, dict),
             "Dataset card human-audit summary needs fields and model_agreement")
    _require(isinstance(distribution, dict) and bool(distribution),
             "Dataset card human-audit summary needs annotation_count_distribution")
    annotations = {}
    for key, value in distribution.items():
        _require(isinstance(key, str) and re.fullmatch(r"[1-9][0-9]*", key) is not None and _integer(value),
                 "Invalid human-audit annotation-count distribution")
        annotations[int(key)] = value
    _require(sum(annotations.values()) == summary["rows"], "Human-audit annotation-count distribution differs from row count")
    for field in _FIELDS:
        item = fields.get(field)
        _require(isinstance(item, dict) and _integer(item.get("eligible"))
                 and item["eligible"] <= summary["rows"], f"Invalid human {field} eligibility summary")
    for provider in _PROVIDERS:
        item = agreement.get(provider)
        _require(isinstance(item, dict), f"Missing human-audit agreement for {provider}")
        for field in _FIELDS:
            score = item.get(field)
            _require(isinstance(score, dict) and all(_integer(score.get(name)) for name in ("eligible", "scored", "matches")),
                     f"Invalid human-audit agreement counts for {provider} {field}")
            _require(score["eligible"] == fields[field]["eligible"]
                     and score["matches"] <= score["scored"] <= score["eligible"],
                     f"Inconsistent human-audit agreement counts for {provider} {field}")
    return summary, annotations


def _rate(numerator, denominator):
    if denominator == 0:
        return f"{numerator:,}/0 (not available)"
    return f"{numerator:,}/{denominator:,} ({100 * numerator / denominator:.1f}%)"


def _models(manifest, counts):
    models = manifest.get("models")
    _require(isinstance(models, dict), "Dataset card needs measured model metadata")
    for provider in _PROVIDERS:
        model = models.get(provider)
        _require(isinstance(model, dict) and _nonempty(model.get("model_id")),
                 f"Dataset card needs a model_id for {provider}")
        _require(_integer(model.get("rows_missing"))
                 and model["rows_missing"] <= counts["sections"],
                 f"Dataset card needs a valid rows_missing count for {provider}")
    _require(_nonempty(models["openai"].get("reasoning_effort")),
             "Dataset card needs OpenAI reasoning_effort")
    return models


def _human_annotation_counts(annotations):
    if annotations == {3: sum(annotations.values())}:
        return "Each excerpt has three human annotations."
    return "Human annotations per excerpt: " + "; ".join(
        f"{number} for {total:,} {'excerpt' if total == 1 else 'excerpts'}"
        for number, total in sorted(annotations.items())) + "."


def _glossary():
    rows = [
        ("document_id", "string", "no", "Stable source document identifier; join key with section_index."),
        ("section_index", "int64", "no", "Original index within its document; gaps and an initial zero are allowed."),
        ("state", "string", "no", "Uppercase two-letter jurisdiction state abbreviation."),
        ("jurisdiction_name", "string", "no", "Selected jurisdiction name from county harmonization."),
        ("jurisdiction_type", "string", "no",
         "city, county, city-county, or other; city-county combines municipal/county "
         "governments or functions, and other is Washington, DC."),
        ("jurisdiction_geoid", "string", "yes", "Single 5-digit county, 7-digit place, or 10-digit county-subdivision GEOID; null when several apply. Leading zeros are retained."),
        ("represented_county_fips", "list<string>", "no", "Nonempty list of represented county FIPS codes sorted by FIPS."),
        ("represented_county_name", "list<string>", "no", "County names aligned with represented_county_fips."),
        ("header", "string", "yes", "Source header with leading Markdown # marks and surrounding whitespace removed."),
        ("content", "string", "no", "Unchanged source section content."),
        ("function_majority", "string", "yes", "Function receiving at least two matching model votes; otherwise null."),
        ("function_votes", "int8", "no", "Largest number of matching function votes, from 0 to 3."),
        ("topic_majority", "string", "yes", "Topic receiving at least two matching votes; null also represents a not-applicable majority."),
        ("topic_votes", "int8", "no", "Largest number of matching topic votes, including not applicable, from 0 to 3."),
    ]
    for prefix, model in (("gemini", "Gemini"), ("openai", "OpenAI"), ("jev", "Jev")):
        rows += [
            (prefix + "_function", "string", "yes", model + " primary function; null for an unavailable answer."),
            (prefix + "_topic", "string", "yes", model + " topic (the codebook subcategory); null when inapplicable or unavailable."),
            (prefix + "_confidence", "string", "yes", model + " rubric confidence: Low, Medium, or High; null when unavailable."),
        ]
    rows += [
        ("jev_function_probs", "struct<float32>", "yes", "Jev scores keyed by Context, Rules, Process, Enforcement, Structural; null iff jev_function is null."),
        ("jev_topic_probs", "struct<float32>", "yes", "Jev scores keyed by Buildings, Business, Zoning, Nuisance, Other; null iff jev_topic is null."),
    ]
    return ["| Column | Type | Nullable | Meaning |", "|---|---|---|---|"] + [
        f"| `{name}` | `{kind}` | {nullable} | {_cell(description)} |" for name, kind, nullable, description in rows
    ]


def render_dataset_card(manifest, publication):
    """Render verified release facts without filling unapproved metadata."""
    publication = validate_publication(publication)
    counts = _counts(manifest)
    summary, annotations = _summary(manifest, counts)
    models = _models(manifest, counts)
    lines = ["---", f"pretty_name: {_yaml(publication['pretty_name'])}", "language:", "- en",
             f"license: {_yaml(publication['license'])}"]
    if publication["license"] == "other":
        lines += [f"license_name: {_yaml(publication['license_name'])}",
                  f"license_link: {_yaml(publication['license_link'])}"]
    lines += ["size_categories:", "- " + _yaml(_size_category(counts["sections"])),
              "task_categories:", "- text-classification", "tags:", "- law", "- municipal-law",
              "- model-annotations", "configs:"]
    for subset in _SUBSETS:
        lines += [f"- config_name: {subset}"]
        if subset == "sections":
            lines += ["  default: true"]
        lines += ["  data_files:", "  - split: train", f"    path: {subset}/*.parquet"]
    lines += ["---", "", "# " + publication["pretty_name"], "",
              "LOCUS (Local Ordinance Corpus for the United States) contains municipal and county legal text "
              "harmonized to county geography. This release includes three models' annotations and a human audit "
              "for primary function, regulated topic where applicable, and rubric confidence.",
              "", "[Paper](https://arxiv.org/abs/2606.19334) · [Code](https://github.com/transpAIrency/LOCUS)",
              "", f"Annotations use Gemini (`{models['google']['model_id']}`), "
              f"OpenAI (`{models['openai']['model_id']}`, {models['openai']['reasoning_effort']} reasoning), "
              f"and Jev (`{models['typesafe']['model_id']}`).",
              "", "## Subsets and coverage", "",
              "| Subset | Rows | Contents |", "|---|---:|---|"]
    descriptions = {"sections": "Default: identifiers, jurisdiction, text, consensus, and all model labels (25 columns).",
                    "labels": "The section identifiers and label columns, without jurisdiction or legal text (17 columns).",
                    "human_audit": "Human consensus, independent eligibility flags, and the three models' labels.",
                    "counties": "The county harmonization universe, including counties without a selected document."}
    lines += [f"| `{name}` | {counts[name]:,} | {descriptions[name]} |" for name in _SUBSETS]
    lines += ["", f"The sections contain **{counts['documents']:,} distinct documents**. "
              f"The county table contains **{counts['counties']:,} counties**: "
              f"**{counts['covered_counties']:,} represented** ({100 * counts['covered_counties'] / counts['counties']:.1f}%) "
              f"and **{counts['uncovered_counties']:,} without a selected document**. "
              "These are counts of the released county selection, not a claim that every ordinance in a represented county is included.",
              "", "The `sections` and `labels` rows share the same order, sorted by state, document_id, and section_index. "
              "The composite key `(document_id, section_index)` is unique. Parquet files use Zstandard compression.",
              "", "## Loading", "", "```python", "from datasets import load_dataset", "",
              f"repo_id = {_yaml(publication['repo_id'])}",
              'sections = load_dataset(repo_id, "sections", split="train", streaming=True)',
              'labels = load_dataset(repo_id, "labels", split="train", streaming=True)',
              'human_audit = load_dataset(repo_id, "human_audit", split="train")',
              'counties = load_dataset(repo_id, "counties", split="train")',
              "```", "", "Streaming avoids downloading and materializing the entire sections table before iteration.",
              "", "## Column glossary", "", "### sections", ""]
    lines += _glossary()
    lines += ["", "### labels", "",
              "`document_id`, `section_index`, the four consensus columns, the nine per-model label columns, "
              "and the two Jev probability columns have the same definitions and types as `sections`.",
              "", "### human_audit", "", "| Column | Type | Meaning |", "|---|---|---|",
              "| `audit_id` | string | Stable audit excerpt identifier. |",
              "| `text` | string | Original human-audit excerpt text. |",
              "| `human_function`, `human_topic`, `human_confidence` | nullable string | Human majority consensus, recased to the release vocabularies. |",
              "| `human_function_eligible`, `human_topic_eligible`, `human_confidence_eligible` | bool | Independent flags identifying fields eligible for evaluation. |",
              "| `human_annotations` | int64 | Number of human annotations for the excerpt. |",
              "| `gemini_*`, `openai_*`, `jev_*`, `jev_*_probs` | as in sections | Model labels and scores joined by audit_id. |",
              "", "### counties", "", "| Column | Type | Meaning |", "|---|---|---|",
              "| `county_fips` | string | Five-digit county FIPS code, preserving leading zeros. |",
              "| `county_name`, `state` | string | County name and uppercase state abbreviation from harmonization. |",
              "| `document_id` | nullable string | Selected representative document; null for an uncovered county. |",
              "| `jurisdiction_name`, `jurisdiction_type` | nullable string | The selected jurisdiction, with the same definitions as sections. |",
              "| `selection_reason` | string | Recorded harmonization selection reason. |",
              "| `section_count` | nullable int64 | Released sections in the selected document; null when none is selected. |",
              "", "Pre-cleaning page counts are excluded because they do not describe the released text.",
              "", "## Labels, consensus, and missing values", "",
              "`topic` is the annotation codebook's `subcategory`. Function values are `Context`, `Rules`, `Process`, "
              "`Enforcement`, and `Structural`. Topic values are `Buildings`, `Business`, `Zoning`, `Nuisance`, and `Other`. "
              "Confidence values are `Low`, `Medium`, and `High`.",
              "", "Unavailable or unsuccessful model answers use real null values for all of that model's labels. "
              "A null topic with a non-null function means topic does not apply (`Context`, `Process`, or `Structural`). "
              "Topic is non-null exactly for answering models whose function is `Rules` or `Enforcement`. "
              "No ERROR strings or inferred labels are inserted. Per-model missing counts are in `manifest.json`.",
              "", "Section rows without a model answer: " + "; ".join(
                  f"{name} **{models[provider]['rows_missing']:,}**" for provider, name in _PROVIDERS.items()) + ".",
              "", "Consensus counts only models with valid answers. An answering model's inapplicable topic counts as "
              "a vote for not applicable; a missing model contributes no vote. A majority requires at least two matching votes. "
              "`topic_majority` null with `topic_votes` at least 2 means a majority for not applicable; at most 1 means no majority. "
              "All three agree when both vote counts equal 3. Confidence is reported separately for each model.",
              "", "## Human-audit agreement", "",
              f"The released audit contains **{summary['rows']:,} excerpts**. " + _human_annotation_counts(annotations),
              "", "The following table reports excerpts with human majority consensus for each field. These are "
              "consensus eligibility rates, not pairwise or chance-corrected inter-annotator agreement statistics.",
              "", "| Human field | Eligible consensus / audit excerpts |", "|---|---:|"]
    lines += [f"| {field} | {_rate(summary['fields'][field]['eligible'], summary['rows'])} |" for field in _FIELDS]
    lines += ["", "Human function, topic, and confidence have separate eligibility flags and must be evaluated independently. "
              "An eligible blank human topic is a valid null; an ineligible blank is unscored. Joint evaluations require all relevant eligibility flags. "
              "Shared `audit_id` values should remain together when splitting related human-label data.",
              "", "The next table reports exact model agreement with eligible human consensus. A row is scored only when "
              "that field is human-eligible and the model returned a valid function. Null topics from answering models are "
              "valid not-applicable answers and are compared as such. Missing model answers reduce scored coverage, "
              "and do not enter the agreement denominator.",
              "", "| Model | Field | Exact matches / scored | Scored / eligible |", "|---|---|---:|---:|"]
    for provider, name in _PROVIDERS.items():
        for field in _FIELDS:
            score = summary["model_agreement"][provider][field]
            lines += [f"| {name} | {field} | {_rate(score['matches'], score['scored'])} | {_rate(score['scored'], score['eligible'])} |"]
    lines += ["", "## Collection and limitations", "",
              "The providers use the shared codebook and identical header-plus-content input, capped at 16,000 characters. "
              "Truncation counts are measured during the build and recorded in `manifest.json`.",
              "", "Only Jev supplies probability scores. They are preserved as float32 without normalization; provider "
              "rounding means sums may differ from 1. They are provider scores, not verified calibrated probabilities. "
              "Evaluate calibration on the human audit before interpreting them as calibrated probabilities.",
              "", "Confidence follows a rubric with Low precedence. Compare confidence within each source; the rubric "
              "can produce more Low responses than the human audit. Confidence agreement measures rubric-label matching, not calibration.",
              "", "The shared Tennessee Mayor-Aldermanic model charter has several place GEOIDs and therefore a null "
              "`jurisdiction_geoid`. Represented counties come exclusively from county harmonization. Washington, DC has "
              "`jurisdiction_type` `other`. Uncovered counties retain null document and jurisdiction fields.",
              "", "## Provenance", "",
              "`manifest.json` records input hashes, exact model and prompt metadata, measured counts, missing answers, "
              "source-code hashes, and deterministic checks against run ledgers. Unavailable actual run dates remain null "
              "with an explanation; annotation collection timestamps are not substituted for run dates.",
              "", "## License", "", f"Declared dataset license: `{publication['license']}`."]
    if publication.get("license_description"):
        lines += ["", publication["license_description"]]
    if publication["license"] == "other":
        lines += ["", f"Custom license name: `{publication['license_name']}`. License text: {publication['license_link']}"]
    citation = publication["citation"]
    longest = max((len(match.group()) for match in re.finditer(r"`+", citation)), default=0)
    fence = "`" * max(3, longest + 1)
    syntax = "bibtex" if citation.lstrip().startswith("@") else "text"
    lines += ["", "## Citation", "", fence + syntax, citation, fence, ""]
    return "\n".join(lines)
