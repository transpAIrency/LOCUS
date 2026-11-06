"""Read back the staged Parquet release before it can be published.

The full scan projects identifiers and labels, avoiding a second full scan of
legal text. Full source columns are read only for row groups containing the
deterministically selected sample rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import re

from .records import CONFIDENCES, FUNCTIONS, TOPICS, ReleaseError


PREFIXES = ("gemini", "openai", "jev")
_SENTINEL = object()


def _require(condition, message):
    if not condition:
        raise ReleaseError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _probabilities(value, names, description):
    _require(isinstance(value, dict) and set(value) == set(names),
             f"Invalid emitted probability keys: {description}")
    _require(all(type(number) in (int, float) and math.isfinite(number) and 0 <= number <= 1
                 for number in value.values()), f"Invalid emitted probability values: {description}")


def _model_labels(row, description):
    missing = Counter()
    answered = []
    for prefix in PREFIXES:
        function, topic, confidence = (row[prefix + "_" + name]
                                        for name in ("function", "topic", "confidence"))
        if function is None:
            _require(topic is None and confidence is None,
                     f"Missing emitted function has other labels: {description}, {prefix}")
            missing[prefix] += 1
        else:
            _require(function in FUNCTIONS and confidence in CONFIDENCES,
                     f"Invalid emitted function or confidence: {description}, {prefix}")
            if function in ("Rules", "Enforcement"):
                _require(topic in TOPICS, f"Emitted topic is required: {description}, {prefix}")
            else:
                _require(topic is None, f"Emitted topic must be null: {description}, {prefix}")
            answered.append((function, topic))
        if prefix == "jev":
            for value, vocabulary, name in ((function, FUNCTIONS, "function"), (topic, TOPICS, "topic")):
                probabilities = row["jev_" + name + "_probs"]
                if value is None:
                    _require(probabilities is None,
                             f"Emitted Jev probabilities require a matching label: {description}, {name}")
                else:
                    _probabilities(probabilities, vocabulary, f"{description}, {name}")
    return missing, answered


def _independent_consensus(row, answered, description):
    for offset, name in ((0, "function"), (1, "topic")):
        votes = Counter(answer[offset] for answer in answered)
        count = max(votes.values(), default=0)
        winners = [label for label, total in votes.items() if total == count]
        majority = winners[0] if count >= 2 else None
        _require(type(row[name + "_votes"]) is int and row[name + "_votes"] == count,
                 f"Emitted {name} votes disagree with model labels: {description}")
        _require(row[name + "_majority"] == majority,
                 f"Emitted {name} majority disagrees with model labels: {description}")


def _parquet_files(root, subset):
    paths = sorted((root / subset).glob("*.parquet"))
    _require(bool(paths), f"No emitted Parquet files for {subset}")
    return paths


def _rows(root, subset, columns=None):
    import pyarrow.parquet as pq
    for path in _parquet_files(root, subset):
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()


def _count(expected_counts, name, actual):
    if name in expected_counts:
        _require(actual == expected_counts[name],
                 f"Emitted {name} count differs: expected {expected_counts[name]}, found {actual}")


def verify_outputs(root, schemas, expected_counts, expected_samples):
    """Validate staged values, ordering, cross-subset equality, and source samples.

    ``expected_samples`` contains full section rows captured from the validated
    source join. Arrow casts them to the declared schema before comparison so
    the expected float32 probability rounding is compared exactly.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    label_columns = list(schemas["labels"].names)
    expected_sample_rows = pa.Table.from_pylist(expected_samples, schema=schemas["sections"]).to_pylist()
    expected_by_key = {}
    for row in expected_sample_rows:
        key = (row["document_id"], row["section_index"])
        _require(key not in expected_by_key, f"Duplicate expected sample key: {key}")
        expected_by_key[key] = row
    sample_locations = defaultdict(dict)
    found_samples = set()
    labels = iter(_rows(root, "labels", label_columns))
    prior_sort = None
    documents = {}
    document_counts = Counter()
    model_nulls = Counter({prefix: 0 for prefix in PREFIXES})
    sections_count = 0

    for path in _parquet_files(root, "sections"):
        file = pq.ParquetFile(path)
        for group_index in range(file.num_row_groups):
            position = 0
            for batch in file.iter_batches(batch_size=4096, row_groups=[group_index],
                                           columns=["state", *label_columns]):
                for row in batch.to_pylist():
                    key = (row["document_id"], row["section_index"])
                    state = row["state"]
                    _require(_text(key[0]) and type(key[1]) is int and 0 <= key[1] < 2**63,
                             f"Invalid emitted section identity at row {sections_count}")
                    _require(isinstance(state, str) and re.fullmatch(r"[A-Z]{2}", state) is not None,
                             f"Invalid emitted state for {key}")
                    if key[0] in documents:
                        _require(documents[key[0]] == state, f"Emitted document spans states: {key[0]}")
                    else:
                        documents[key[0]] = state
                    sort = (state, *key)
                    _require(prior_sort is None or prior_sort < sort,
                             f"Emitted sections are unsorted or duplicated: {key}")
                    prior_sort = sort
                    other = next(labels, _SENTINEL)
                    projection = {name: row[name] for name in label_columns}
                    _require(other is not _SENTINEL and other == projection,
                             f"Emitted sections and labels differ at {key}")
                    missing, answered = _model_labels(row, str(key))
                    model_nulls.update(missing)
                    _independent_consensus(row, answered, str(key))
                    if key in expected_by_key:
                        _require(key not in found_samples, f"Duplicate emitted sample key: {key}")
                        sample_locations[(path, group_index)][position] = key
                        found_samples.add(key)
                    sections_count += 1
                    document_counts[key[0]] += 1
                    position += 1
    _require(next(labels, _SENTINEL) is _SENTINEL, "Emitted labels has extra rows")
    _count(expected_counts, "sections", sections_count)
    _count(expected_counts, "labels", sections_count)
    _count(expected_counts, "documents", len(documents))
    _require(found_samples == set(expected_by_key), "Expected sample rows are missing from emitted sections")

    # At most one full row group is resident, including its original legal text.
    for (path, group_index), positions in sample_locations.items():
        table = pq.ParquetFile(path).read_row_group(group_index)
        for position, key in positions.items():
            actual = table.slice(position, 1).to_pylist()[0]
            _require(actual == expected_by_key[key], f"Emitted source sample differs at {key}")
        del table

    human_count = 0
    human_ids = set()
    human_nulls = Counter({prefix: 0 for prefix in PREFIXES})
    for row in _rows(root, "human_audit"):
        identity = row["audit_id"]
        _require(_text(identity) and identity not in human_ids, "Invalid or duplicate emitted audit_id")
        human_ids.add(identity)
        _require(_text(row["text"]), f"Invalid emitted human text: {identity}")
        _require(type(row["human_annotations"]) is int and row["human_annotations"] > 0,
                 f"Invalid emitted human annotation count: {identity}")
        for name, vocabulary in (("function", FUNCTIONS), ("topic", TOPICS), ("confidence", CONFIDENCES)):
            value = row["human_" + name]
            _require(value is None or value in vocabulary, f"Invalid emitted human {name}: {identity}")
            _require(type(row["human_" + name + "_eligible"]) is bool,
                     f"Invalid emitted human {name} eligibility: {identity}")
        missing, _ = _model_labels(row, identity)
        human_nulls.update(missing)
        human_count += 1
    _count(expected_counts, "human_audit", human_count)

    county_count = covered = 0
    county_ids = set()
    selected_documents = set()
    for row in _rows(root, "counties"):
        fips, state = row["county_fips"], row["state"]
        _require(isinstance(fips, str) and re.fullmatch(r"[0-9]{5}", fips) is not None
                 and fips not in county_ids, "Invalid or duplicate emitted county_fips")
        county_ids.add(fips)
        _require(isinstance(state, str) and re.fullmatch(r"[A-Z]{2}", state) is not None,
                 f"Invalid emitted county state: {fips}")
        _require(_text(row["county_name"]) and _text(row["selection_reason"]),
                 f"Invalid emitted county name or selection reason: {fips}")
        document = row["document_id"]
        if document is None:
            _require(all(row[name] is None for name in ("jurisdiction_name", "jurisdiction_type", "section_count")),
                     f"Uncovered emitted county has selected metadata: {fips}")
        else:
            _require(document in documents and documents[document] == state,
                     f"Emitted county document or state differs from sections: {fips}")
            _require(_text(row["jurisdiction_name"]) and row["jurisdiction_type"] in ("city", "county", "consolidated", "other"),
                     f"Invalid emitted county jurisdiction: {fips}")
            _require(row["section_count"] == document_counts[document],
                     f"Emitted county section_count differs from sections: {fips}")
            selected_documents.add(document)
            covered += 1
        county_count += 1
    _require(selected_documents == set(documents), "Emitted counties and section document sets differ")
    _count(expected_counts, "counties", county_count)
    _count(expected_counts, "covered_counties", covered)
    _count(expected_counts, "uncovered_counties", county_count - covered)

    return {"sections": sections_count, "labels": sections_count,
            "human_audit": human_count, "counties": county_count,
            "documents": len(documents), "source_samples_verified": len(found_samples),
            "source_sample_row_groups_read": len(sample_locations),
            "model_null_counts": dict(model_nulls), "human_model_null_counts": dict(human_nulls),
            "sections_labels_equal": True, "consensus_recomputed": True, "strict_sort_verified": True}
