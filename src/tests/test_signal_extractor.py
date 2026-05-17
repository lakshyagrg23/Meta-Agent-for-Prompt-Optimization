"""
Tests for src/critic/signal_extractor.py

Covers:
    - high false negatives (fn_rate > threshold)
    - high false positives (fp_rate > threshold)
    - low accuracy
    - consistency failure (inconsistent)
    - plateau detection
      - exactly at window boundary
      - strictly inside plateau
      - improvement just above delta (no plateau)
      - improvement exactly at delta (no plateau — < not <=)
      - insufficient history
    - stable optimization (no signals)
    - multiple simultaneous signals
    - CriticSignals.any_active and active_names helpers
    - CriticSignals.summary format
    - custom threshold overrides
    - default thresholds are applied when thresholds=None
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.critic.signal_extractor import (
    CriticSignals,
    CriticThresholds,
    SignalExtractor,
)
from src.evaluation.metrics import EvaluationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(
    fnr: float = 0.0,
    fpr: float = 0.0,
    accuracy: float = 1.0,
    consistency: float = 1.0,
    f1: float = 1.0,
    recall: float = 1.0,
    precision: float = 1.0,
) -> EvaluationMetrics:
    """Build an EvaluationMetrics with overridable fields."""
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        consistency=consistency,
    )


def _defaults() -> CriticThresholds:
    return CriticThresholds(
        fn_rate=0.20,
        fp_rate=0.15,
        min_accuracy=0.75,
        min_consistency=0.80,
        plateau_window=3,
        plateau_delta=0.01,
    )


def _extract(
    m: EvaluationMetrics,
    history: list | None = None,
    thresholds: CriticThresholds | None = None,
) -> CriticSignals:
    return SignalExtractor.extract_signals(
        metrics=m,
        score_history=history if history is not None else [],
        thresholds=thresholds if thresholds is not None else _defaults(),
    )


# ---------------------------------------------------------------------------
# high_fn
# ---------------------------------------------------------------------------

def test_high_fn_fires_above_threshold():
    """FNR = 0.21 > threshold 0.20 → high_fn = True."""
    s = _extract(_metrics(fnr=0.21))
    assert s.high_fn is True
    print("test_high_fn_fires_above_threshold PASSED")


def test_high_fn_not_fired_at_threshold():
    """FNR exactly at threshold (0.20) → high_fn = False (> not >=)."""
    s = _extract(_metrics(fnr=0.20))
    assert s.high_fn is False
    print("test_high_fn_not_fired_at_threshold PASSED")


def test_high_fn_not_fired_below_threshold():
    """FNR = 0.10 < threshold 0.20 → high_fn = False."""
    s = _extract(_metrics(fnr=0.10))
    assert s.high_fn is False
    print("test_high_fn_not_fired_below_threshold PASSED")


# ---------------------------------------------------------------------------
# high_fp
# ---------------------------------------------------------------------------

def test_high_fp_fires_above_threshold():
    """FPR = 0.16 > threshold 0.15 → high_fp = True."""
    s = _extract(_metrics(fpr=0.16))
    assert s.high_fp is True
    print("test_high_fp_fires_above_threshold PASSED")


def test_high_fp_not_fired_at_threshold():
    """FPR exactly at threshold → high_fp = False."""
    s = _extract(_metrics(fpr=0.15))
    assert s.high_fp is False
    print("test_high_fp_not_fired_at_threshold PASSED")


def test_high_fp_not_fired_below_threshold():
    """FPR = 0.05 → high_fp = False."""
    s = _extract(_metrics(fpr=0.05))
    assert s.high_fp is False
    print("test_high_fp_not_fired_below_threshold PASSED")


# ---------------------------------------------------------------------------
# low_accuracy
# ---------------------------------------------------------------------------

def test_low_accuracy_fires_below_threshold():
    """accuracy = 0.74 < threshold 0.75 → low_accuracy = True."""
    s = _extract(_metrics(accuracy=0.74))
    assert s.low_accuracy is True
    print("test_low_accuracy_fires_below_threshold PASSED")


def test_low_accuracy_not_fired_at_threshold():
    """accuracy exactly at threshold → low_accuracy = False (< not <=)."""
    s = _extract(_metrics(accuracy=0.75))
    assert s.low_accuracy is False
    print("test_low_accuracy_not_fired_at_threshold PASSED")


def test_low_accuracy_not_fired_above_threshold():
    """accuracy = 0.90 → low_accuracy = False."""
    s = _extract(_metrics(accuracy=0.90))
    assert s.low_accuracy is False
    print("test_low_accuracy_not_fired_above_threshold PASSED")


# ---------------------------------------------------------------------------
# inconsistent
# ---------------------------------------------------------------------------

def test_inconsistent_fires_below_threshold():
    """consistency = 0.79 < threshold 0.80 → inconsistent = True."""
    s = _extract(_metrics(consistency=0.79))
    assert s.inconsistent is True
    print("test_inconsistent_fires_below_threshold PASSED")


def test_inconsistent_not_fired_at_threshold():
    """consistency exactly at threshold → inconsistent = False."""
    s = _extract(_metrics(consistency=0.80))
    assert s.inconsistent is False
    print("test_inconsistent_not_fired_at_threshold PASSED")


def test_inconsistent_not_fired_above_threshold():
    """consistency = 0.95 → inconsistent = False."""
    s = _extract(_metrics(consistency=0.95))
    assert s.inconsistent is False
    print("test_inconsistent_not_fired_above_threshold PASSED")


# ---------------------------------------------------------------------------
# plateau detection
# ---------------------------------------------------------------------------

def test_plateau_insufficient_history_no_signal():
    """
    Less than k+1 scores → cannot determine plateau → plateau = False.
    k=3 needs 4 scores; only 3 provided.
    """
    history = [0.60, 0.61, 0.62]   # len=3, need 4
    s = _extract(_metrics(), history=history)
    assert s.plateau is False
    print("test_plateau_insufficient_history_no_signal PASSED")


def test_plateau_exactly_at_window_boundary():
    """
    Exactly k+1 scores available and improvement < delta → plateau = True.
    k=3: compares history[-1] vs history[-4].
    history = [0.700, 0.701, 0.702, 0.703]
    Δ = |0.703 - 0.700| = 0.003 < delta 0.01 → plateau
    """
    history = [0.700, 0.701, 0.702, 0.703]
    s = _extract(_metrics(), history=history)
    assert s.plateau is True
    print("test_plateau_exactly_at_window_boundary PASSED")


def test_plateau_longer_history_stagnant():
    """
    More than k+1 scores; last k+1 entries are stagnant.
    history[-1]=0.800  history[-4]=0.799  Δ=0.001 < 0.01 → plateau
    """
    history = [0.50, 0.60, 0.70, 0.799, 0.799, 0.800, 0.800]
    s = _extract(_metrics(), history=history)
    # history[-1] = 0.800, history[-(3+1)] = history[-4] = 0.799
    delta = abs(0.800 - 0.799)
    assert delta < 0.01
    assert s.plateau is True
    print("test_plateau_longer_history_stagnant PASSED")


def test_plateau_not_fired_when_improving():
    """
    Improvement = 0.05 > delta 0.01 → plateau = False.
    history = [0.700, 0.720, 0.740, 0.750]
    Δ = |0.750 - 0.700| = 0.05 ≥ 0.01 → no plateau
    """
    history = [0.700, 0.720, 0.740, 0.750]
    s = _extract(_metrics(), history=history)
    assert s.plateau is False
    print("test_plateau_not_fired_when_improving PASSED")


def test_plateau_not_fired_when_improvement_equals_delta():
    """
    |Δ| = delta exactly → plateau = False (< not <=).
    Use values exact in binary float (multiples of 0.25) so the diff is exact.
    history = [0.50, 0.52, 0.54, 0.51]
    delta=0.01: |0.51 - 0.50| = 0.01 exactly → NOT a plateau (< not <=)
    """
    import math
    # 0.50 and 0.51 are NOT exact in binary; use 0.00 and 0.01 with a
    # purpose-built threshold to sidestep float representation entirely.
    t = CriticThresholds(plateau_window=1, plateau_delta=0.10)
    # window=1: compares history[-1] vs history[-2]
    history = [0.700, 0.800]   # |0.800 - 0.700| = 0.100 = delta → NOT plateau
    improvement = abs(history[-1] - history[-2])
    assert math.isclose(improvement, 0.10, rel_tol=1e-9), f"got {improvement}"
    s = _extract(_metrics(), history=history, thresholds=t)
    assert s.plateau is False
    print("test_plateau_not_fired_when_improvement_equals_delta PASSED")


def test_plateau_empty_history():
    """Empty history → plateau = False (no history to detect plateau)."""
    s = _extract(_metrics(), history=[])
    assert s.plateau is False
    print("test_plateau_empty_history PASSED")


def test_plateau_single_score():
    """Single score → plateau = False (need k+1 = 4 scores)."""
    s = _extract(_metrics(), history=[0.700])
    assert s.plateau is False
    print("test_plateau_single_score PASSED")


def test_plateau_custom_window():
    """
    Custom window k=1: only needs 2 scores.
    |history[-1] - history[-2]| = 0.001 < delta 0.01 → plateau
    """
    t = CriticThresholds(plateau_window=1, plateau_delta=0.01)
    history = [0.800, 0.801]
    s = _extract(_metrics(), history=history, thresholds=t)
    assert s.plateau is True
    print("test_plateau_custom_window PASSED")


def test_plateau_regression_in_history_not_plateau():
    """
    Score went down then recovered — net window delta is positive enough.
    history = [0.700, 0.650, 0.690, 0.760]
    Δ = |0.760 - 0.700| = 0.060 > 0.01 → no plateau
    """
    history = [0.700, 0.650, 0.690, 0.760]
    s = _extract(_metrics(), history=history)
    assert s.plateau is False
    print("test_plateau_regression_in_history_not_plateau PASSED")


# ---------------------------------------------------------------------------
# Stable optimization — no signals should fire
# ---------------------------------------------------------------------------

def test_stable_optimization_no_signals():
    """All metrics well above thresholds and history improving → no signals."""
    history = [0.700, 0.730, 0.760, 0.800]
    m = _metrics(fnr=0.05, fpr=0.05, accuracy=0.90, consistency=0.95)
    s = _extract(m, history=history)
    assert s.high_fn is False
    assert s.high_fp is False
    assert s.low_accuracy is False
    assert s.inconsistent is False
    assert s.plateau is False
    assert s.any_active is False
    assert s.active_names() == []
    print("test_stable_optimization_no_signals PASSED")


# ---------------------------------------------------------------------------
# Multiple simultaneous signals
# ---------------------------------------------------------------------------

def test_multiple_signals_simultaneous():
    """High FNR + plateau can fire together."""
    history = [0.600, 0.601, 0.601, 0.601]  # plateau
    m = _metrics(fnr=0.35, accuracy=0.80)   # high_fn also
    s = _extract(m, history=history)
    assert s.high_fn is True
    assert s.plateau is True
    assert s.any_active is True
    assert "high_fn" in s.active_names()
    assert "plateau" in s.active_names()
    print("test_multiple_signals_simultaneous PASSED")


def test_all_signals_fire_together():
    """Construct inputs that trigger all five signals simultaneously."""
    t = CriticThresholds(
        fn_rate=0.10,
        fp_rate=0.10,
        min_accuracy=0.90,
        min_consistency=0.90,
        plateau_window=2,
        plateau_delta=0.05,
    )
    history = [0.500, 0.501, 0.501]   # plateau (|0.501-0.500|=0.001 < 0.05, window=2)
    m = _metrics(
        fnr=0.50,       # > fn_rate 0.10 → high_fn
        fpr=0.50,       # > fp_rate 0.10 → high_fp
        accuracy=0.50,  # < min_accuracy 0.90 → low_accuracy
        consistency=0.50,  # < min_consistency 0.90 → inconsistent
    )
    s = _extract(m, history=history, thresholds=t)
    assert s.high_fn is True
    assert s.high_fp is True
    assert s.low_accuracy is True
    assert s.inconsistent is True
    assert s.plateau is True
    assert len(s.active_names()) == 5
    print("test_all_signals_fire_together PASSED")


# ---------------------------------------------------------------------------
# CriticSignals helpers
# ---------------------------------------------------------------------------

def test_any_active_false_when_none():
    s = CriticSignals(high_fn=False, high_fp=False, low_accuracy=False,
                      inconsistent=False, plateau=False)
    assert s.any_active is False
    print("test_any_active_false_when_none PASSED")


def test_any_active_true_on_single_signal():
    s = CriticSignals(high_fn=False, high_fp=False, low_accuracy=False,
                      inconsistent=False, plateau=True)
    assert s.any_active is True
    print("test_any_active_true_on_single_signal PASSED")


def test_active_names_returns_correct_subset():
    s = CriticSignals(high_fn=True, high_fp=False, low_accuracy=True,
                      inconsistent=False, plateau=False)
    names = s.active_names()
    assert names == ["high_fn", "low_accuracy"]
    print("test_active_names_returns_correct_subset PASSED")


def test_summary_contains_all_keys():
    s = CriticSignals(high_fn=True, high_fp=False, low_accuracy=False,
                      inconsistent=True, plateau=False)
    txt = s.summary()
    for key in ("high_fn=", "high_fp=", "low_acc=", "inconsistent=", "plateau="):
        assert key in txt, f"Missing '{key}' in summary: {txt}"
    print("test_summary_contains_all_keys PASSED")


# ---------------------------------------------------------------------------
# Custom thresholds and None thresholds
# ---------------------------------------------------------------------------

def test_default_thresholds_applied_when_none():
    """Passing thresholds=None must use CriticThresholds defaults."""
    m = _metrics(fnr=0.25)   # above default fn_rate=0.20
    s = SignalExtractor.extract_signals(
        metrics=m,
        score_history=[],
        thresholds=None,
    )
    assert s.high_fn is True
    print("test_default_thresholds_applied_when_none PASSED")


def test_custom_threshold_raises_signal_that_default_would_not():
    """
    Default fn_rate=0.20 would NOT fire for fnr=0.15.
    Custom fn_rate=0.10 DOES fire for fnr=0.15.
    """
    m = _metrics(fnr=0.15)
    s_default = _extract(m)
    assert s_default.high_fn is False

    strict = CriticThresholds(fn_rate=0.10)
    s_strict = _extract(m, thresholds=strict)
    assert s_strict.high_fn is True
    print("test_custom_threshold_raises_signal_that_default_would_not PASSED")


def test_lenient_threshold_suppresses_signal():
    """
    fnr=0.30 normally fires high_fn, but a lenient threshold (fn_rate=0.50)
    suppresses it.
    """
    m = _metrics(fnr=0.30)
    lenient = CriticThresholds(fn_rate=0.50)
    s = _extract(m, thresholds=lenient)
    assert s.high_fn is False
    print("test_lenient_threshold_suppresses_signal PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # high_fn
    test_high_fn_fires_above_threshold()
    test_high_fn_not_fired_at_threshold()
    test_high_fn_not_fired_below_threshold()
    # high_fp
    test_high_fp_fires_above_threshold()
    test_high_fp_not_fired_at_threshold()
    test_high_fp_not_fired_below_threshold()
    # low_accuracy
    test_low_accuracy_fires_below_threshold()
    test_low_accuracy_not_fired_at_threshold()
    test_low_accuracy_not_fired_above_threshold()
    # inconsistent
    test_inconsistent_fires_below_threshold()
    test_inconsistent_not_fired_at_threshold()
    test_inconsistent_not_fired_above_threshold()
    # plateau
    test_plateau_insufficient_history_no_signal()
    test_plateau_exactly_at_window_boundary()
    test_plateau_longer_history_stagnant()
    test_plateau_not_fired_when_improving()
    test_plateau_not_fired_when_improvement_equals_delta()
    test_plateau_empty_history()
    test_plateau_single_score()
    test_plateau_custom_window()
    test_plateau_regression_in_history_not_plateau()
    # stable optimization
    test_stable_optimization_no_signals()
    # multiple signals
    test_multiple_signals_simultaneous()
    test_all_signals_fire_together()
    # helpers
    test_any_active_false_when_none()
    test_any_active_true_on_single_signal()
    test_active_names_returns_correct_subset()
    test_summary_contains_all_keys()
    # thresholds
    test_default_thresholds_applied_when_none()
    test_custom_threshold_raises_signal_that_default_would_not()
    test_lenient_threshold_suppresses_signal()
    print("\nAll signal_extractor tests passed.")
