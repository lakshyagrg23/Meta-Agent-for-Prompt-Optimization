"""
metrics.py — Evaluation and sampled consistency measurement.

compute_metrics: accuracy, F1, precision, recall, TP/TN/FP/FN, rates.
compute_consistency: reruns Agent 1 N times and measures label stability.
"""

from typing import List, Dict, Optional, Callable

from src.config import CONSISTENCY_RUNS


# ── Core Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(predictions: List[Dict], batch: List[Dict]) -> Dict:
    """
    Compute classification metrics.

    Args:
        predictions: list of {id, label} from Agent 1 (first pass).
        batch:       list of {id, label} ground-truth rows.

    Returns:
        dict with keys:
            tp, tn, fp, fn,
            accuracy, precision, recall, f1,
            fn_rate, fp_rate,
            consistency (None — filled in by loop.py if sampled),
            consistency_sampled (False by default)
    """
    # Build lookup: id → predicted label
    pred_map = {p["id"]: p["label"] for p in predictions}

    tp = tn = fp = fn = 0

    for row in batch:
        true_label = int(row["label"])
        pred_label = pred_map.get(row["id"])

        if pred_label is None:
            # Missing prediction — skip to not skew metrics
            continue

        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        else:  # true=1, pred=0
            fn += 1

    n          = tp + tn + fp + fn
    accuracy   = (tp + tn) / n if n > 0 else 0.0
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fn_rate    = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fp_rate    = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fn_rate":   round(fn_rate,   4),
        "fp_rate":   round(fp_rate,   4),
        # consistency is populated by loop.py
        "consistency":         None,
        "consistency_sampled": False,
    }


# ── Consistency ────────────────────────────────────────────────────────────────

def compute_consistency(
    classify_fn: Callable[[str, List[Dict]], List[Dict]],
    system_prompt: str,
    batch: List[Dict],
    runs: int = CONSISTENCY_RUNS,
) -> float:
    """
    Measure label consistency by running classify_fn `runs` times.
    Uses temperature=0 (set in config) so variance reflects prompt ambiguity.

    Returns:
        Fraction of emails where all `runs` predictions agree [0.0, 1.0].
    """
    print(f"  [metrics] Measuring consistency over {runs} runs …")

    # Collect label lists per email id
    all_runs: List[Dict[int, int]] = []
    for run_idx in range(runs):
        preds      = classify_fn(system_prompt, batch)
        label_map  = {p["id"]: p["label"] for p in preds}
        all_runs.append(label_map)

    consistent_count = 0
    valid_count = 0
    for row in batch:
        labels = [run.get(row["id"]) for run in all_runs]
        if None in labels:
            continue
        valid_count += 1
        if all(l == labels[0] for l in labels):
            consistent_count += 1

    score = consistent_count / valid_count if valid_count else 0.0
    return round(score, 4)
