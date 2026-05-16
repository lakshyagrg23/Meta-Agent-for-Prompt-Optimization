"""
Tests for src/optimization/acceptance.py

Covers: J(S) formula correctness, PromptCost normalisation, score delta
comparison, should_accept rule (above/below/exactly-at epsilon), epsilon
validation, zero token cost, ceiling edge cases, and score_delta helper.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.optimization.acceptance import (
    AcceptanceStrategy,
    W_F1, W_RECALL, W_CONSISTENCY, W_COST,
    DEFAULT_TOKEN_BUDGET_CEILING,
    _normalise_token_count,
)
from src.evaluation.metrics import EvaluationMetrics


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def _make_metrics(
    f1: float = 0.0,
    recall: float = 0.0,
    consistency: float = 1.0,
    accuracy: float = 0.0,
    precision: float = 0.0,
    false_positive_rate: float = 0.0,
    false_negative_rate: float = 0.0,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        consistency=consistency,
    )


# ---------------------------------------------------------------------------
# _normalise_token_count
# ---------------------------------------------------------------------------

def test_normalise_within_budget():
    assert _approx(_normalise_token_count(512, 2048), 512 / 2048)
    print("test_normalise_within_budget PASSED")


def test_normalise_at_ceiling():
    assert _approx(_normalise_token_count(2048, 2048), 1.0)
    print("test_normalise_at_ceiling PASSED")


def test_normalise_above_ceiling_clamped():
    assert _approx(_normalise_token_count(4096, 2048), 1.0)
    print("test_normalise_above_ceiling_clamped PASSED")


def test_normalise_zero_tokens():
    assert _approx(_normalise_token_count(0, 2048), 0.0)
    print("test_normalise_zero_tokens PASSED")


def test_normalise_zero_ceiling():
    """Zero ceiling → no cost penalty."""
    assert _approx(_normalise_token_count(512, 0), 0.0)
    print("test_normalise_zero_ceiling PASSED")


def test_normalise_negative_ceiling():
    assert _approx(_normalise_token_count(512, -1), 0.0)
    print("test_normalise_negative_ceiling PASSED")


# ---------------------------------------------------------------------------
# compute_score — J(S) formula
# ---------------------------------------------------------------------------

def test_score_formula_perfect_no_cost():
    """Perfect metrics, zero tokens → J = 0.4 + 0.3 + 0.2 = 0.9."""
    m = _make_metrics(f1=1.0, recall=1.0, consistency=1.0)
    score = AcceptanceStrategy.compute_score(m, prompt_token_count=0)
    expected = W_F1 * 1.0 + W_RECALL * 1.0 + W_CONSISTENCY * 1.0 - W_COST * 0.0
    assert _approx(score, expected), f"Got {score}, expected {expected}"
    print("test_score_formula_perfect_no_cost PASSED")


def test_score_formula_zero_metrics():
    """All-zero metrics, max cost → J = -0.1."""
    m = _make_metrics(f1=0.0, recall=0.0, consistency=0.0)
    score = AcceptanceStrategy.compute_score(
        m,
        prompt_token_count=DEFAULT_TOKEN_BUDGET_CEILING,
        token_budget_ceiling=DEFAULT_TOKEN_BUDGET_CEILING,
    )
    expected = -W_COST * 1.0   # = -0.1
    assert _approx(score, expected), f"Got {score}, expected {expected}"
    print("test_score_formula_zero_metrics PASSED")


def test_score_formula_mixed():
    """Known values: f1=0.8, recall=0.7, consistency=0.9, cost=512/2048."""
    m = _make_metrics(f1=0.8, recall=0.7, consistency=0.9)
    token_count = 512
    ceiling = 2048
    prompt_cost = token_count / ceiling
    expected = (
        W_F1 * 0.8
        + W_RECALL * 0.7
        + W_CONSISTENCY * 0.9
        - W_COST * prompt_cost
    )
    score = AcceptanceStrategy.compute_score(m, token_count, ceiling)
    assert _approx(score, expected), f"Got {score}, expected {expected}"
    print("test_score_formula_mixed PASSED")


def test_score_cost_term_reduces_score():
    """Adding token cost should reduce J(S)."""
    m = _make_metrics(f1=0.8, recall=0.7, consistency=0.9)
    no_cost = AcceptanceStrategy.compute_score(m, prompt_token_count=0)
    with_cost = AcceptanceStrategy.compute_score(m, prompt_token_count=1024)
    assert with_cost < no_cost
    print("test_score_cost_term_reduces_score PASSED")


def test_score_default_token_count_is_zero_cost():
    """Default token_count=0 → PromptCost=0, cost term is absent."""
    m = _make_metrics(f1=0.6, recall=0.5, consistency=0.8)
    score = AcceptanceStrategy.compute_score(m)
    expected = W_F1 * 0.6 + W_RECALL * 0.5 + W_CONSISTENCY * 0.8
    assert _approx(score, expected)
    print("test_score_default_token_count_is_zero_cost PASSED")


# ---------------------------------------------------------------------------
# should_accept
# ---------------------------------------------------------------------------

def test_accept_clear_improvement():
    accepted = AcceptanceStrategy.should_accept(0.600, 0.650, epsilon=0.01)
    assert accepted is True
    print("test_accept_clear_improvement PASSED")


def test_reject_no_improvement():
    accepted = AcceptanceStrategy.should_accept(0.650, 0.630, epsilon=0.01)
    assert accepted is False
    print("test_reject_no_improvement PASSED")


def test_reject_insufficient_improvement():
    """Delta = 0.005 < epsilon 0.01 → reject."""
    accepted = AcceptanceStrategy.should_accept(0.700, 0.705, epsilon=0.01)
    assert accepted is False
    print("test_reject_insufficient_improvement PASSED")


def test_accept_exactly_at_epsilon():
    """Delta == epsilon (>=) → accept."""
    accepted = AcceptanceStrategy.should_accept(0.700, 0.710, epsilon=0.01)
    assert accepted is True
    print("test_accept_exactly_at_epsilon PASSED")


def test_accept_epsilon_zero_equal_scores():
    """epsilon=0.0 and identical scores → accept (>= 0)."""
    accepted = AcceptanceStrategy.should_accept(0.700, 0.700, epsilon=0.0)
    assert accepted is True
    print("test_accept_epsilon_zero_equal_scores PASSED")


def test_reject_regression():
    """Candidate is strictly worse → always reject."""
    accepted = AcceptanceStrategy.should_accept(0.800, 0.750, epsilon=0.0)
    assert accepted is False
    print("test_reject_regression PASSED")


def test_negative_epsilon_raises():
    raised = False
    try:
        AcceptanceStrategy.should_accept(0.5, 0.6, epsilon=-0.01)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for negative epsilon"
    print("test_negative_epsilon_raises PASSED")


# ---------------------------------------------------------------------------
# score_delta
# ---------------------------------------------------------------------------

def test_score_delta_positive():
    delta = AcceptanceStrategy.score_delta(0.600, 0.650)
    assert _approx(delta, 0.050)
    print("test_score_delta_positive PASSED")


def test_score_delta_negative():
    delta = AcceptanceStrategy.score_delta(0.700, 0.650)
    assert _approx(delta, -0.050)
    print("test_score_delta_negative PASSED")


def test_score_delta_zero():
    delta = AcceptanceStrategy.score_delta(0.700, 0.700)
    assert _approx(delta, 0.0)
    print("test_score_delta_zero PASSED")


# ---------------------------------------------------------------------------
# End-to-end: compute_score → should_accept
# ---------------------------------------------------------------------------

def test_end_to_end_acceptance():
    """Simulate one optimization step: current vs. improved candidate."""
    current_m = _make_metrics(f1=0.72, recall=0.68, consistency=0.85)
    candidate_m = _make_metrics(f1=0.78, recall=0.75, consistency=0.88)

    current_score = AcceptanceStrategy.compute_score(current_m, prompt_token_count=300)
    candidate_score = AcceptanceStrategy.compute_score(candidate_m, prompt_token_count=320)

    delta = AcceptanceStrategy.score_delta(current_score, candidate_score)
    accepted = AcceptanceStrategy.should_accept(current_score, candidate_score, epsilon=0.01)

    assert delta > 0, f"Expected positive delta, got {delta}"
    assert accepted is True
    print("test_end_to_end_acceptance PASSED")


def test_end_to_end_rejection_cost_penalty():
    """Higher token cost on candidate should reduce its score advantage."""
    good_m = _make_metrics(f1=0.80, recall=0.75, consistency=0.90)

    # Candidate has marginally better classification but much higher token cost.
    candidate_m = _make_metrics(f1=0.81, recall=0.76, consistency=0.91)

    current_score = AcceptanceStrategy.compute_score(
        good_m, prompt_token_count=200, token_budget_ceiling=2048
    )
    candidate_score = AcceptanceStrategy.compute_score(
        candidate_m, prompt_token_count=2048, token_budget_ceiling=2048  # max cost
    )

    # Cost penalty (-0.1) should overcome the tiny metric gains.
    assert candidate_score < current_score, (
        f"Expected cost penalty to dominate: current={current_score:.4f}, "
        f"candidate={candidate_score:.4f}"
    )
    accepted = AcceptanceStrategy.should_accept(current_score, candidate_score, epsilon=0.01)
    assert accepted is False
    print("test_end_to_end_rejection_cost_penalty PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_normalise_within_budget()
    test_normalise_at_ceiling()
    test_normalise_above_ceiling_clamped()
    test_normalise_zero_tokens()
    test_normalise_zero_ceiling()
    test_normalise_negative_ceiling()
    test_score_formula_perfect_no_cost()
    test_score_formula_zero_metrics()
    test_score_formula_mixed()
    test_score_cost_term_reduces_score()
    test_score_default_token_count_is_zero_cost()
    test_accept_clear_improvement()
    test_reject_no_improvement()
    test_reject_insufficient_improvement()
    test_accept_exactly_at_epsilon()
    test_accept_epsilon_zero_equal_scores()
    test_reject_regression()
    test_negative_epsilon_raises()
    test_score_delta_positive()
    test_score_delta_negative()
    test_score_delta_zero()
    test_end_to_end_acceptance()
    test_end_to_end_rejection_cost_penalty()
    print("\nAll acceptance tests passed.")
