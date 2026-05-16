"""
Tests for src/evaluation/evaluator.py

Covers: full pipeline with and without repeated predictions, weight
resolution, score ordering, empty inputs, token count passthrough,
summary format, consistency merge, custom weights, and module-level
compute_optimization_score.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    compute_optimization_score,
    DEFAULT_WEIGHTS,
    _resolve_weights,
)
from src.evaluation.metrics import empty_metrics


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Basic pipeline — no repeated predictions
# ---------------------------------------------------------------------------

def test_no_repeated_predictions():
    """Pipeline runs cleanly without repeated predictions."""
    preds  = ["PHISHING", "SAFE", "PHISHING", "SAFE"]
    labels = ["PHISHING", "SAFE", "SAFE",     "SAFE"]

    result = Evaluator.evaluate(
        predictions=preds,
        labels=labels,
        prompt_token_count=128,
    )

    assert isinstance(result, EvaluationResult)
    assert result.consistency_result is None        # not computed
    assert result.metrics.consistency == 1.0        # default from MetricsEngine
    assert result.prompt_token_count == 128
    assert result.metrics.sample_count == 4
    assert result.optimization_score != 0.0
    print("test_no_repeated_predictions PASSED")


# ---------------------------------------------------------------------------
# Pipeline with repeated predictions
# ---------------------------------------------------------------------------

def test_with_repeated_predictions():
    """Consistency is computed and merged into metrics."""
    preds  = ["PHISHING", "SAFE", "PHISHING"]
    labels = ["PHISHING", "SAFE", "SAFE"]

    rp = [
        ["PHISHING", "PHISHING"],   # sample 0: consistent
        ["SAFE",     "SAFE"],       # sample 1: consistent
        ["PHISHING", "SAFE"],       # sample 2: split → 0.5
    ]

    result = Evaluator.evaluate(
        predictions=preds,
        labels=labels,
        repeated_predictions=rp,
        prompt_token_count=200,
    )

    assert result.consistency_result is not None
    # batch consistency = (1.0 + 1.0 + 0.5) / 3 = 0.833…
    expected_cons = (1.0 + 1.0 + 0.5) / 3
    assert _approx(result.metrics.consistency, expected_cons), \
        f"Got {result.metrics.consistency}, expected {expected_cons}"
    print("test_with_repeated_predictions PASSED")


# ---------------------------------------------------------------------------
# Consistency is merged into metrics (not left as default 1.0)
# ---------------------------------------------------------------------------

def test_consistency_overrides_default():
    rp = [
        ["PHISHING", "SAFE"],   # 0.5
        ["SAFE",     "SAFE"],   # 1.0
    ]
    result = Evaluator.evaluate(
        predictions=["PHISHING", "SAFE"],
        labels=["PHISHING", "SAFE"],
        repeated_predictions=rp,
    )
    assert result.metrics.consistency < 1.0, \
        "Consistency should be < 1.0 when repetitions disagree"
    print("test_consistency_overrides_default PASSED")


# ---------------------------------------------------------------------------
# Score ordering: better metrics → higher score
# ---------------------------------------------------------------------------

def test_score_ordering():
    """A perfect classifier should score higher than an imperfect one."""
    labels = ["PHISHING", "PHISHING", "SAFE", "SAFE"]

    perfect = Evaluator.evaluate(predictions=labels, labels=labels)
    imperfect = Evaluator.evaluate(
        predictions=["SAFE", "SAFE", "SAFE", "SAFE"],
        labels=labels,
    )

    assert perfect.optimization_score > imperfect.optimization_score, (
        f"Perfect={perfect.optimization_score:.4f} should > "
        f"Imperfect={imperfect.optimization_score:.4f}"
    )
    print("test_score_ordering PASSED")


# ---------------------------------------------------------------------------
# Empty inputs are handled safely
# ---------------------------------------------------------------------------

def test_empty_predictions():
    result = Evaluator.evaluate(predictions=[], labels=[])
    assert result.metrics.is_empty()
    assert result.optimization_score == 0.0
    print("test_empty_predictions PASSED")


# ---------------------------------------------------------------------------
# Token count passthrough
# ---------------------------------------------------------------------------

def test_token_count_passthrough():
    result = Evaluator.evaluate(
        predictions=["PHISHING"],
        labels=["PHISHING"],
        prompt_token_count=999,
    )
    assert result.prompt_token_count == 999
    print("test_token_count_passthrough PASSED")


# ---------------------------------------------------------------------------
# Weights used are stored on result
# ---------------------------------------------------------------------------

def test_default_weights_stored():
    result = Evaluator.evaluate(["PHISHING"], ["PHISHING"])
    assert result.weights_used == DEFAULT_WEIGHTS
    print("test_default_weights_stored PASSED")


def test_custom_weights_stored():
    custom = {"f1": 1.0, "recall": 0.0, "precision": 0.0,
              "false_negative_rate": 0.0, "false_positive_rate": 0.0,
              "consistency": 0.0}
    result = Evaluator.evaluate(["PHISHING"], ["PHISHING"], weights=custom)
    assert result.weights_used["f1"] == 1.0
    print("test_custom_weights_stored PASSED")


# ---------------------------------------------------------------------------
# Custom weights change the score
# ---------------------------------------------------------------------------

def test_custom_weights_affect_score():
    """Weighting only F1 should produce a different score than defaults."""
    preds  = ["PHISHING", "SAFE", "PHISHING"]
    labels = ["PHISHING", "SAFE", "SAFE"]

    default_result = Evaluator.evaluate(predictions=preds, labels=labels)
    f1_only_weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    f1_only_weights["f1"] = 1.0
    custom_result = Evaluator.evaluate(
        predictions=preds, labels=labels, weights=f1_only_weights
    )

    # F1-only score should equal the F1 value directly
    assert _approx(custom_result.optimization_score, custom_result.metrics.f1), \
        f"score={custom_result.optimization_score}, f1={custom_result.metrics.f1}"
    assert not _approx(default_result.optimization_score, custom_result.optimization_score)
    print("test_custom_weights_affect_score PASSED")


# ---------------------------------------------------------------------------
# Summary format
# ---------------------------------------------------------------------------

def test_summary_contains_expected_parts():
    result = Evaluator.evaluate(
        predictions=["PHISHING", "SAFE"],
        labels=["PHISHING", "SAFE"],
        prompt_token_count=55,
    )
    s = result.summary()
    for key in ("score=", "acc=", "f1=", "cons=", "n=", "tokens="):
        assert key in s, f"Missing '{key}' in summary: {s}"
    assert "tokens=55" in s
    print("test_summary_contains_expected_parts PASSED")


# ---------------------------------------------------------------------------
# Module-level compute_optimization_score
# ---------------------------------------------------------------------------

def test_module_level_score_perfect():
    """Perfect metrics should yield a score close to 1.0."""
    from src.evaluation.metrics import MetricsEngine
    labels = ["PHISHING", "SAFE", "PHISHING", "SAFE"]
    metrics = MetricsEngine.compute_metrics(labels, labels)
    score = compute_optimization_score(metrics)
    # With defaults: 0.30*1 + 0.25*1 + 0.15*1 - 0.20*0 - 0.05*0 + 0.05*1 = 0.75
    assert _approx(score, 0.75), f"Got {score}"
    print("test_module_level_score_perfect PASSED")


def test_module_level_score_worst():
    """All-wrong predictions → very low or negative score."""
    from src.evaluation.metrics import MetricsEngine
    labels = ["PHISHING", "PHISHING", "SAFE", "SAFE"]
    preds  = ["SAFE",     "SAFE",     "PHISHING", "PHISHING"]
    metrics = MetricsEngine.compute_metrics(preds, labels)
    score = compute_optimization_score(metrics)
    assert score < 0.5, f"Expected low score, got {score}"
    print("test_module_level_score_worst PASSED")


# ---------------------------------------------------------------------------
# _resolve_weights
# ---------------------------------------------------------------------------

def test_resolve_weights_none():
    resolved = _resolve_weights(None)
    assert resolved == DEFAULT_WEIGHTS
    assert resolved is not DEFAULT_WEIGHTS  # must be a copy
    print("test_resolve_weights_none PASSED")


def test_resolve_weights_partial_override():
    resolved = _resolve_weights({"f1": 0.99})
    assert resolved["f1"] == 0.99
    assert resolved["recall"] == DEFAULT_WEIGHTS["recall"]  # unchanged
    print("test_resolve_weights_partial_override PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_no_repeated_predictions()
    test_with_repeated_predictions()
    test_consistency_overrides_default()
    test_score_ordering()
    test_empty_predictions()
    test_token_count_passthrough()
    test_default_weights_stored()
    test_custom_weights_stored()
    test_custom_weights_affect_score()
    test_summary_contains_expected_parts()
    test_module_level_score_perfect()
    test_module_level_score_worst()
    test_resolve_weights_none()
    test_resolve_weights_partial_override()
    print("\nAll evaluator tests passed.")
