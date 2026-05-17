"""
mutation.py — Deterministic mutation policy engine.

Maps metric signals → mutation tags + few-shot example selection.
No LLM calls — pure rule-based logic.
"""

from typing import List, Dict, Optional

from src.config import (
    FEW_SHOT_FN_COUNT,
    FEW_SHOT_FP_COUNT,
    FEW_SHOT_TN_COUNT,
)

# All valid mutation tags (strict enum)
VALID_TAGS = frozenset({
    "few_shot_fn",        # high FN → add missed phishing examples
    "few_shot_balanced",  # high FP → add balanced FP/TN examples
    "add_constraints",    # high FP or low consistency → explicit rules
    "enrich_role",        # low accuracy → improve role description
    "prompt_enrichment",  # high FN or low accuracy → add context / signals
})


def decide_mutation(
    metrics:     Dict,
    batch:       List[Dict],
    predictions: List[Dict],
    fn_threshold: float,
    fp_threshold: float,
    acc_threshold: float,
    consistency_threshold: float,
) -> Dict:
    """
    Deterministically decide which mutation tags to fire and which
    example emails to pass to Agent 2.

    Args:
        metrics:     output of compute_metrics() (includes consistency).
        batch:       original batch rows with ground-truth labels.
        predictions: Agent 1 first-pass predictions [{id, label, reason}].
        fn_threshold: dynamic False Negative rate threshold
        fp_threshold: dynamic False Positive rate threshold
        acc_threshold: dynamic accuracy threshold
        consistency_threshold: dynamic consistency threshold

    Returns:
        {
            "tags":     List[str],  # subset of VALID_TAGS
            "examples": List[Dict], # annotated email rows for few-shot
        }
    """
    tags:     set  = set()
    examples: List[Dict] = []

    fn_rate    = metrics.get("fn_rate",    0.0)
    fp_rate    = metrics.get("fp_rate",    0.0)
    accuracy   = metrics.get("accuracy",   1.0)
    consistency= metrics.get("consistency")   # may be None (not sampled)

    # Build lookup for quick access
    pred_map     = {p["id"]: p["label"] for p in predictions}
    true_map     = {row["id"]: int(row["label"]) for row in batch}
    batch_by_id  = {row["id"]: row for row in batch}

    # ── Rule 1: High False Negative Rate ──────────────────────────────────────
    if fn_rate > fn_threshold:
        tags.add("few_shot_fn")
        tags.add("prompt_enrichment")

        # FN examples: true=1, predicted=0
        fn_rows = [
            _annotate(batch_by_id[eid], true_map[eid], pred_map.get(eid, 0))
            for eid in pred_map
            if true_map.get(eid) == 1 and pred_map.get(eid) == 0
        ][:FEW_SHOT_FN_COUNT]
        examples.extend(fn_rows)

    # ── Rule 2: High False Positive Rate ─────────────────────────────────────
    if fp_rate > fp_threshold:
        tags.add("few_shot_balanced")
        tags.add("add_constraints")

        # FP examples: true=0, predicted=1
        fp_rows = [
            _annotate(batch_by_id[eid], true_map[eid], pred_map.get(eid, 1))
            for eid in pred_map
            if true_map.get(eid) == 0 and pred_map.get(eid) == 1
        ][:FEW_SHOT_FP_COUNT]

        # TN examples: true=0, predicted=0 (correct, for contrast)
        tn_rows = [
            _annotate(batch_by_id[eid], true_map[eid], pred_map.get(eid, 0))
            for eid in pred_map
            if true_map.get(eid) == 0 and pred_map.get(eid) == 0
        ][:FEW_SHOT_TN_COUNT]

        examples.extend(fp_rows)
        examples.extend(tn_rows)

    # ── Rule 3: Low Consistency (only when sampled) ───────────────────────────
    if consistency is not None and consistency < consistency_threshold:
        tags.add("add_constraints")

    # ── Rule 4: Low Accuracy ──────────────────────────────────────────────────
    if accuracy < acc_threshold:
        tags.add("enrich_role")
        tags.add("prompt_enrichment")

    return {
        "tags":     sorted(tags),   # sorted for deterministic ordering
        "examples": examples,
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _annotate(row: Dict, true_label: int, predicted_label: int) -> Dict:
    """Return a copy of the row with true/predicted labels attached."""
    return {
        **row,
        "true_label":      true_label,
        "predicted_label": predicted_label,
    }
