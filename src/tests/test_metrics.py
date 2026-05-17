"""
Tests for src/evaluation/metrics.py

Covers: normal case, perfect classifier, all-wrong classifier, all-phishing
batch, all-safe batch, empty inputs, length mismatch, consistency
calculation, attach_consistency, FPR/FNR correctness, and summary format.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    EvaluationMetrics,
    MetricsEngine,
    empty_metrics,
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
)
from src.core.constants import LABEL_PHISHING, LABEL_SAFE


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Normal case
# ---------------------------------------------------------------------------

def test_normal_case():
    """Mixed batch with known ground truth."""
    # 4 phishing, 4 safe; model gets 3/4 phishing right, 4/4 safe right
    labels      = ["PHISHING", "PHISHING", "PHISHING", "PHISHING", "SAFE", "SAFE", "SAFE", "SAFE"]
    predictions = ["PHISHING", "PHISHING", "PHISHING", "SAFE",     "SAFE", "SAFE", "SAFE", "SAFE"]
    #  TP=3  FN=1  FP=0  TN=4
    m = MetricsEngine.compute_metrics(predictions, labels)

    assert m.sample_count == 8
    assert m.phishing_count == 4
    assert m.safe_count == 4

    assert _approx(m.accuracy, 7/8)
    assert _approx(m.precision, 1.0)           # 3/(3+0)
    assert _approx(m.recall, 3/4)              # 3/(3+1)
    assert _approx(m.f1, 2*(1.0*(3/4))/(1.0+(3/4)))
    assert _approx(m.false_positive_rate, 0.0) # 0/(0+4)
    assert _approx(m.false_negative_rate, 1/4) # 1/(1+3)
    assert _approx(m.consistency, 1.0)          # single run default
    print("test_normal_case PASSED")


# ---------------------------------------------------------------------------
# Perfect classifier
# ---------------------------------------------------------------------------

def test_perfect_classifier():
    labels = ["PHISHING", "SAFE", "PHISHING", "SAFE"]
    m = MetricsEngine.compute_metrics(labels, labels)

    assert _approx(m.accuracy, 1.0)
    assert _approx(m.precision, 1.0)
    assert _approx(m.recall, 1.0)
    assert _approx(m.f1, 1.0)
    assert _approx(m.false_positive_rate, 0.0)
    assert _approx(m.false_negative_rate, 0.0)
    print("test_perfect_classifier PASSED")


# ---------------------------------------------------------------------------
# All wrong
# ---------------------------------------------------------------------------

def test_all_wrong():
    labels      = ["PHISHING", "PHISHING", "SAFE", "SAFE"]
    predictions = ["SAFE",     "SAFE",     "PHISHING", "PHISHING"]
    # TP=0  FN=2  FP=2  TN=0
    m = MetricsEngine.compute_metrics(predictions, labels)

    assert _approx(m.accuracy, 0.0)
    assert _approx(m.precision, 0.0)    # 0/(0+2) = 0
    assert _approx(m.recall, 0.0)       # 0/(0+2) = 0
    assert _approx(m.f1, 0.0)
    assert _approx(m.false_positive_rate, 1.0)  # 2/(2+0)
    assert _approx(m.false_negative_rate, 1.0)  # 2/(0+2)
    print("test_all_wrong PASSED")


# ---------------------------------------------------------------------------
# All-phishing batch (no safe examples → FPR denominator = 0)
# ---------------------------------------------------------------------------

def test_all_phishing_batch():
    labels      = ["PHISHING", "PHISHING", "PHISHING"]
    predictions = ["PHISHING", "PHISHING", "SAFE"]
    m = MetricsEngine.compute_metrics(predictions, labels)

    assert _approx(m.false_positive_rate, 0.0)  # FP+TN = 0 → safe 0.0
    assert _approx(m.recall, 2/3)
    print("test_all_phishing_batch PASSED")


# ---------------------------------------------------------------------------
# All-safe batch (no phishing → FNR denominator = 0, precision = 0)
# ---------------------------------------------------------------------------

def test_all_safe_batch():
    labels      = ["SAFE", "SAFE", "SAFE"]
    predictions = ["PHISHING", "SAFE", "SAFE"]
    # FP=1, TN=2, TP=0, FN=0
    m = MetricsEngine.compute_metrics(predictions, labels)

    assert _approx(m.false_negative_rate, 0.0)  # FN+TP = 0 → safe 0.0
    assert _approx(m.precision, 0.0)            # no true positives
    assert _approx(m.false_positive_rate, 1/3)
    print("test_all_safe_batch PASSED")


# ---------------------------------------------------------------------------
# Edge: empty inputs
# ---------------------------------------------------------------------------

def test_empty_inputs():
    m = MetricsEngine.compute_metrics([], [])
    assert m.is_empty()
    assert m.accuracy == 0.0
    assert m.f1 == 0.0
    print("test_empty_inputs PASSED")


# ---------------------------------------------------------------------------
# Edge: length mismatch
# ---------------------------------------------------------------------------

def test_length_mismatch():
    m = MetricsEngine.compute_metrics(["PHISHING"], ["PHISHING", "SAFE"])
    assert m.is_empty()
    print("test_length_mismatch PASSED")


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

def test_consistency_perfect():
    """All runs agree → 1.0."""
    runs = [
        ["PHISHING", "SAFE", "PHISHING"],
        ["PHISHING", "SAFE", "PHISHING"],
    ]
    c = MetricsEngine.compute_consistency(runs)
    assert _approx(c, 1.0)
    print("test_consistency_perfect PASSED")


def test_consistency_partial():
    """Two runs, disagree on one out of three → 2/3."""
    runs = [
        ["PHISHING", "SAFE",     "PHISHING"],
        ["PHISHING", "PHISHING", "PHISHING"],  # position 1 differs
    ]
    c = MetricsEngine.compute_consistency(runs)
    assert _approx(c, 2/3)
    print("test_consistency_partial PASSED")


def test_consistency_single_run():
    """Single run is trivially consistent."""
    c = MetricsEngine.compute_consistency([["PHISHING", "SAFE"]])
    assert _approx(c, 1.0)
    print("test_consistency_single_run PASSED")


def test_consistency_empty():
    """Empty list → 1.0 (no evidence of inconsistency)."""
    c = MetricsEngine.compute_consistency([])
    assert _approx(c, 1.0)
    print("test_consistency_empty PASSED")


# ---------------------------------------------------------------------------
# attach_consistency
# ---------------------------------------------------------------------------

def test_attach_consistency():
    labels = ["PHISHING", "SAFE", "PHISHING"]
    preds  = ["PHISHING", "SAFE", "SAFE"]
    m = MetricsEngine.compute_metrics(preds, labels)
    assert _approx(m.consistency, 1.0)  # default before attachment

    run1 = ["PHISHING", "SAFE", "SAFE"]
    run2 = ["PHISHING", "SAFE", "PHISHING"]  # position 2 differs
    m2 = MetricsEngine.attach_consistency(m, [run1, run2])

    assert _approx(m2.consistency, 2/3)
    assert _approx(m.consistency, 1.0)  # original unchanged (copy semantics)
    print("test_attach_consistency PASSED")


# ---------------------------------------------------------------------------
# FNR = 1 - recall (always)
# ---------------------------------------------------------------------------

def test_fnr_equals_one_minus_recall():
    labels      = ["PHISHING", "PHISHING", "PHISHING", "SAFE"]
    predictions = ["PHISHING", "SAFE",     "SAFE",     "SAFE"]
    m = MetricsEngine.compute_metrics(predictions, labels)
    assert _approx(m.false_negative_rate, 1.0 - m.recall), (
        f"FNR={m.false_negative_rate} should equal 1-recall={1-m.recall}"
    )
    print("test_fnr_equals_one_minus_recall PASSED")


# ---------------------------------------------------------------------------
# empty_metrics sentinel
# ---------------------------------------------------------------------------

def test_empty_metrics_sentinel():
    m = empty_metrics()
    assert m.is_empty()
    for field in ("accuracy", "precision", "recall", "f1",
                  "false_positive_rate", "false_negative_rate", "consistency"):
        assert getattr(m, field) == 0.0, f"{field} should be 0.0"
    print("test_empty_metrics_sentinel PASSED")


# ---------------------------------------------------------------------------
# summary() format
# ---------------------------------------------------------------------------

def test_summary_format():
    labels = ["PHISHING", "SAFE"]
    m = MetricsEngine.compute_metrics(labels, labels)
    s = m.summary()
    for key in ("acc=", "prec=", "rec=", "f1=", "fpr=", "fnr=", "cons=", "n="):
        assert key in s, f"Missing '{key}' in summary: {s}"
    print("test_summary_format PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_normal_case()
    test_perfect_classifier()
    test_all_wrong()
    test_all_phishing_batch()
    test_all_safe_batch()
    test_empty_inputs()
    test_length_mismatch()
    test_consistency_perfect()
    test_consistency_partial()
    test_consistency_single_run()
    test_consistency_empty()
    test_attach_consistency()
    test_fnr_equals_one_minus_recall()
    test_empty_metrics_sentinel()
    test_summary_format()
    print("\nAll metrics tests passed.")
