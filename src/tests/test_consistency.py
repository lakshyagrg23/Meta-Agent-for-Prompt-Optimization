"""
Tests for src/evaluation/consistency.py

Covers: normal batch, perfect consistency, zero consistency, single
repetition, single sample, empty inputs, tie-breaking rules,
varying repetition counts, consistency_to_signal, and result fields.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.consistency import (
    ConsistencyResult,
    compute_consistency,
    compute_sample_consistency,
    get_majority_label,
    consistency_to_signal,
)
from src.core.constants import LABEL_PHISHING, LABEL_SAFE


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# get_majority_label
# ---------------------------------------------------------------------------

def test_majority_clear_winner():
    assert get_majority_label(["PHISHING", "PHISHING", "SAFE"]) == "PHISHING"
    assert get_majority_label(["SAFE", "SAFE", "PHISHING"]) == "SAFE"
    print("test_majority_clear_winner PASSED")


def test_majority_tie_prefers_phishing():
    """Tie between SAFE and PHISHING → prefer PHISHING."""
    result = get_majority_label(["SAFE", "PHISHING"])
    assert result == "PHISHING", f"Expected PHISHING, got {result}"
    print("test_majority_tie_prefers_phishing PASSED")


def test_majority_tie_alphabetical_fallback():
    """Tie between two non-PHISHING labels → alphabetically first."""
    result = get_majority_label(["SAFE", "UNKNOWN", "SAFE", "UNKNOWN", "OTHER", "OTHER"])
    # SAFE, UNKNOWN, OTHER each appear twice → no PHISHING → sorted → "OTHER"
    assert result == "OTHER", f"Expected OTHER, got {result}"
    print("test_majority_tie_alphabetical_fallback PASSED")


def test_majority_single_element():
    assert get_majority_label(["PHISHING"]) == "PHISHING"
    assert get_majority_label(["SAFE"]) == "SAFE"
    print("test_majority_single_element PASSED")


def test_majority_empty_raises():
    raised = False
    try:
        get_majority_label([])
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for empty input"
    print("test_majority_empty_raises PASSED")


# ---------------------------------------------------------------------------
# compute_sample_consistency
# ---------------------------------------------------------------------------

def test_sample_consistency_unanimous():
    score = compute_sample_consistency(["PHISHING", "PHISHING", "PHISHING"])
    assert _approx(score, 1.0)
    print("test_sample_consistency_unanimous PASSED")


def test_sample_consistency_partial():
    # majority=PHISHING (2/3), score = 2/3
    score = compute_sample_consistency(["PHISHING", "PHISHING", "SAFE"])
    assert _approx(score, 2 / 3)
    print("test_sample_consistency_partial PASSED")


def test_sample_consistency_split():
    # tie → PHISHING; 1/2 match PHISHING
    score = compute_sample_consistency(["PHISHING", "SAFE"])
    assert _approx(score, 0.5)
    print("test_sample_consistency_split PASSED")


def test_sample_consistency_single():
    score = compute_sample_consistency(["SAFE"])
    assert _approx(score, 1.0)
    print("test_sample_consistency_single PASSED")


def test_sample_consistency_empty():
    assert compute_sample_consistency([]) == 0.0
    print("test_sample_consistency_empty PASSED")


# ---------------------------------------------------------------------------
# compute_consistency (batch)
# ---------------------------------------------------------------------------

def test_batch_from_docstring():
    """Example from module docstring."""
    rp = [
        ["PHISHING", "PHISHING", "SAFE"],   # majority=PHISHING, 2/3
        ["SAFE", "SAFE", "SAFE"],            # majority=SAFE,     3/3
    ]
    result = compute_consistency(rp)

    assert result.sample_count == 2
    assert result.majority_labels == ["PHISHING", "SAFE"]
    assert _approx(result.per_sample_scores[0], 2 / 3)
    assert _approx(result.per_sample_scores[1], 1.0)
    expected_batch = (2 / 3 + 1.0) / 2
    assert _approx(result.batch_consistency, expected_batch)
    print("test_batch_from_docstring PASSED")


def test_batch_perfect():
    rp = [
        ["PHISHING", "PHISHING"],
        ["SAFE", "SAFE"],
        ["PHISHING", "PHISHING"],
    ]
    result = compute_consistency(rp)
    assert _approx(result.batch_consistency, 1.0)
    assert all(_approx(s, 1.0) for s in result.per_sample_scores)
    print("test_batch_perfect PASSED")


def test_batch_worst_case():
    """Every sample is a perfect 50/50 split (tie → PHISHING, score = 0.5)."""
    rp = [
        ["PHISHING", "SAFE"],
        ["PHISHING", "SAFE"],
    ]
    result = compute_consistency(rp)
    assert _approx(result.batch_consistency, 0.5)
    print("test_batch_worst_case PASSED")


def test_batch_single_sample():
    rp = [["PHISHING", "PHISHING", "SAFE"]]
    result = compute_consistency(rp)
    assert result.sample_count == 1
    assert _approx(result.batch_consistency, 2 / 3)
    print("test_batch_single_sample PASSED")


def test_batch_single_repetition():
    """Each sample has only one prediction → unanimously consistent."""
    rp = [["PHISHING"], ["SAFE"], ["PHISHING"]]
    result = compute_consistency(rp)
    assert _approx(result.batch_consistency, 1.0)
    assert result.repetitions_per_sample == 1
    print("test_batch_single_repetition PASSED")


def test_batch_uniform_repetition_count():
    rp = [["PHISHING", "SAFE"], ["SAFE", "SAFE"]]
    result = compute_consistency(rp)
    assert result.repetitions_per_sample == 2  # all same length
    print("test_batch_uniform_repetition_count PASSED")


def test_batch_varying_repetition_count():
    rp = [["PHISHING", "SAFE"], ["SAFE", "SAFE", "SAFE"]]
    result = compute_consistency(rp)
    assert result.repetitions_per_sample == -1  # signals non-uniform
    print("test_batch_varying_repetition_count PASSED")


def test_batch_empty_input():
    result = compute_consistency([])
    assert result.sample_count == 0
    assert result.batch_consistency == 0.0
    assert result.per_sample_scores == []
    assert result.majority_labels == []
    print("test_batch_empty_input PASSED")


def test_majority_labels_order_preserved():
    """majority_labels must be in the same order as the input samples."""
    rp = [
        ["SAFE", "SAFE"],
        ["PHISHING", "PHISHING"],
        ["SAFE"],
    ]
    result = compute_consistency(rp)
    assert result.majority_labels == ["SAFE", "PHISHING", "SAFE"]
    print("test_majority_labels_order_preserved PASSED")


# ---------------------------------------------------------------------------
# consistency_to_signal
# ---------------------------------------------------------------------------

def test_signal_below_threshold():
    rp = [["PHISHING", "SAFE"], ["PHISHING", "SAFE"]]  # 0.5 consistency
    result = compute_consistency(rp)
    assert consistency_to_signal(result, threshold=0.80) is True
    print("test_signal_below_threshold PASSED")


def test_signal_above_threshold():
    rp = [["PHISHING", "PHISHING"], ["SAFE", "SAFE"]]  # 1.0 consistency
    result = compute_consistency(rp)
    assert consistency_to_signal(result, threshold=0.80) is False
    print("test_signal_above_threshold PASSED")


def test_signal_exactly_at_threshold():
    """Exactly at threshold is NOT inconsistent (< not <=)."""
    rp = [["PHISHING", "SAFE"], ["PHISHING", "PHISHING"]]
    # sample 0: 0.5, sample 1: 1.0 → batch = 0.75
    result = compute_consistency(rp)
    assert _approx(result.batch_consistency, 0.75)
    assert consistency_to_signal(result, threshold=0.75) is False
    assert consistency_to_signal(result, threshold=0.76) is True
    print("test_signal_exactly_at_threshold PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_majority_clear_winner()
    test_majority_tie_prefers_phishing()
    test_majority_tie_alphabetical_fallback()
    test_majority_single_element()
    test_majority_empty_raises()
    test_sample_consistency_unanimous()
    test_sample_consistency_partial()
    test_sample_consistency_split()
    test_sample_consistency_single()
    test_sample_consistency_empty()
    test_batch_from_docstring()
    test_batch_perfect()
    test_batch_worst_case()
    test_batch_single_sample()
    test_batch_single_repetition()
    test_batch_uniform_repetition_count()
    test_batch_varying_repetition_count()
    test_batch_empty_input()
    test_majority_labels_order_preserved()
    test_signal_below_threshold()
    test_signal_above_threshold()
    test_signal_exactly_at_threshold()
    print("\nAll consistency tests passed.")
