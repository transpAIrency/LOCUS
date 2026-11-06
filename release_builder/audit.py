"""Measured human-consensus coverage and model agreement for the dataset card."""

from __future__ import annotations

from collections import Counter

from .records import PROVIDERS, ReleaseError


def summarize_human_audit(rows):
    """Score fields independently on eligible human consensus and valid answers.

    An eligible human null topic is a real not-applicable label. A model whose
    function is null has no valid answer and is excluded from every field's
    scored denominator. These two cases must never be counted as agreement.
    """
    fields = ("function", "topic", "confidence")
    eligible = Counter({field: 0 for field in fields})
    distribution = Counter()
    models = {
        provider: {field: {"eligible": 0, "scored": 0, "matches": 0, "accuracy": None}
                   for field in fields}
        for provider in PROVIDERS
    }
    seen = set()
    for row in rows:
        identity = row["audit_id"]
        if identity in seen:
            raise ReleaseError(f"Duplicate audit ID in summary: {identity}")
        seen.add(identity)
        distribution[row["human_annotations"]] += 1
        for field in fields:
            if not row["human_" + field + "_eligible"]:
                continue
            target = row["human_" + field]
            if target is None and field != "topic":
                raise ReleaseError(f"Eligible human {field} has no label: {identity}")
            eligible[field] += 1
            for provider, (_, prefix) in PROVIDERS.items():
                metric = models[provider][field]
                metric["eligible"] += 1
                if row.get(prefix + "_function") is None:
                    continue
                metric["scored"] += 1
                metric["matches"] += row[prefix + "_" + field] == target
    for metrics in models.values():
        for metric in metrics.values():
            if metric["scored"]:
                metric["accuracy"] = metric["matches"] / metric["scored"]
    return {
        "rows": len(seen),
        "annotation_count_distribution": {str(n): count for n, count in sorted(distribution.items())},
        "fields": {field: {"eligible": eligible[field]} for field in fields},
        "model_agreement": models,
    }
