"""
src/optimization/acceptance.py
--------------------------------
Candidate acceptance strategy for the adaptive prompt optimization loop.

This module defines the **specific research optimization objective J(S)**
that governs whether a refined candidate PromptState replaces the current
one.  It is deliberately separate from the general-purpose
:func:`~src.evaluation.evaluator.compute_optimization_score` in
``evaluator.py``, which is used for logging and monitoring.

Optimization objective
----------------------
::

    J(S) = 0.4 * F1
         + 0.3 * Recall
         + 0.2 * Consistency
         - 0.1 * PromptCost

Where::

    PromptCost = clamp(prompt_token_count / token_budget_ceiling, 0.0, 1.0)

All four components are in [0.0, 1.0], so J(S) ∈ [−0.1, 1.0].

Rationale for weights
---------------------
* **F1 (0.4)** — primary signal; balances precision and recall.
* **Recall (0.3)** — catching phishing is operationally more critical than
  minimising false alarms; given higher weight than precision.
* **Consistency (0.2)** — prompt stability is essential for research
  reproducibility; an inconsistent prompt that scores well on one batch
  may fail on another.
* **PromptCost (−0.1)** — mild regularisation; discourages prompt bloat
  without dominating the objective.

Acceptance rule
---------------
A candidate is accepted if and only if::

    J(candidate) - J(current) >= epsilon

where ``epsilon`` is a caller-supplied minimum improvement threshold
(e.g. 0.01).  Equality is accepted (>=) to allow the loop to escape
plateaus when a neutral-score refinement improves a non-score dimension.

Design constraints
------------------
* **Stateless** — all methods are static; no hidden state.
* **Deterministic** — same inputs always produce the same decision.
* **Transparent** — no hidden heuristics; every term in J(S) is explicit.
"""

from __future__ import annotations

import logging

from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Objective weights — single source of truth
# ---------------------------------------------------------------------------

#: Weight on F1 score.
W_F1: float = 0.4

#: Weight on recall (sensitivity to phishing).
W_RECALL: float = 0.3

#: Weight on consistency (prompt stability).
W_CONSISTENCY: float = 0.2

#: Penalty weight on normalised prompt token cost.
W_COST: float = 0.1

#: Default ceiling used to normalise raw token counts into [0, 1].
#: Prompts at or above this threshold receive a PromptCost of 1.0.
DEFAULT_TOKEN_BUDGET_CEILING: int = 2048


# ---------------------------------------------------------------------------
# AcceptanceStrategy
# ---------------------------------------------------------------------------

class AcceptanceStrategy:
    """
    Determines whether a refined :class:`~src.core.prompt_state.PromptState`
    candidate should replace the current state.

    All methods are static.  The class holds no instance state.

    Usage pattern in the optimization loop::

        current_score = AcceptanceStrategy.compute_score(current_metrics, current_tokens)
        candidate_score = AcceptanceStrategy.compute_score(candidate_metrics, candidate_tokens)

        if AcceptanceStrategy.should_accept(current_score, candidate_score, epsilon=0.01):
            current_state = candidate_state
            current_score = candidate_score
    """

    @staticmethod
    def compute_score(
        metrics: EvaluationMetrics,
        prompt_token_count: int = 0,
        token_budget_ceiling: int = DEFAULT_TOKEN_BUDGET_CEILING,
    ) -> float:
        """
        Compute the composite optimization objective J(S).

        ::

            J(S) = 0.4 * F1
                 + 0.3 * Recall
                 + 0.2 * Consistency
                 - 0.1 * PromptCost

            PromptCost = clamp(token_count / ceiling, 0.0, 1.0)

        Args:
            metrics:              :class:`~src.evaluation.metrics.EvaluationMetrics`
                                  for the prompt state being scored.
            prompt_token_count:   Raw token count of the rendered prompt.
                                  Pass 0 to exclude the cost term (PromptCost = 0).
            token_budget_ceiling: Denominator used to normalise token count.
                                  Counts at or above this value produce
                                  PromptCost = 1.0 (maximum penalty).

        Returns:
            Scalar J(S) in [−0.1, 1.0].  Higher is better.

        Example::

            score = AcceptanceStrategy.compute_score(metrics, prompt_token_count=312)
        """
        prompt_cost = _normalise_token_count(prompt_token_count, token_budget_ceiling)

        score = (
            W_F1          * metrics.f1
            + W_RECALL    * metrics.recall
            + W_CONSISTENCY * metrics.consistency
            - W_COST      * prompt_cost
        )

        logger.debug(
            "J(S): f1=%.3f recall=%.3f cons=%.3f cost=%.3f → score=%.4f",
            metrics.f1,
            metrics.recall,
            metrics.consistency,
            prompt_cost,
            score,
        )
        return score

    @staticmethod
    def should_accept(
        current_score: float,
        candidate_score: float,
        epsilon: float,
    ) -> bool:
        """
        Decide whether to accept the candidate PromptState.

        Acceptance rule::

            candidate_score - current_score >= epsilon

        The ``>=`` (not ``>``) means a candidate that matches the current
        score exactly is accepted when ``epsilon = 0.0``, which lets the
        optimization loop escape a plateau if a non-score dimension
        (e.g. structural improvement) warrants the swap.

        Args:
            current_score:   J(S) of the current (incumbent) PromptState.
            candidate_score: J(S) of the refined candidate PromptState.
            epsilon:         Minimum required score improvement.  Must be
                             non-negative.  Typical value: 0.01.

        Returns:
            ``True`` if the candidate should replace the current state,
            ``False`` otherwise.

        Raises:
            ValueError: If *epsilon* is negative (would allow degradation).

        Example::

            accepted = AcceptanceStrategy.should_accept(
                current_score=0.712,
                candidate_score=0.731,
                epsilon=0.01,
            )
            # True — improvement of 0.019 >= 0.01
        """
        if epsilon < 0:
            raise ValueError(
                f"epsilon must be non-negative; got {epsilon}. "
                "A negative epsilon would allow accepting worse candidates."
            )

        delta = candidate_score - current_score
        accepted = delta >= epsilon

        logger.debug(
            "Accept decision: current=%.4f candidate=%.4f Δ=%.4f epsilon=%.4f → %s",
            current_score,
            candidate_score,
            delta,
            epsilon,
            "ACCEPT" if accepted else "REJECT",
        )
        return accepted

    @staticmethod
    def score_delta(
        current_score: float,
        candidate_score: float,
    ) -> float:
        """
        Return the raw score improvement of the candidate over the current state.

        Convenience helper for logging and history tracking without
        repeating the subtraction at every call site.

        Args:
            current_score:   J(S) of the incumbent PromptState.
            candidate_score: J(S) of the candidate PromptState.

        Returns:
            ``candidate_score - current_score``.  Positive means improvement.
        """
        return candidate_score - current_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_token_count(
    token_count: int,
    ceiling: int,
) -> float:
    """
    Normalise a raw token count into [0.0, 1.0].

    ::

        PromptCost = clamp(token_count / ceiling, 0.0, 1.0)

    A ceiling of 0 or negative is treated as unconstrained and always
    returns 0.0 (no cost penalty applied).

    Args:
        token_count: Raw token count from the prompt renderer.
        ceiling:     Maximum expected token count (denominator).

    Returns:
        Normalised cost in [0.0, 1.0].
    """
    if ceiling <= 0 or token_count <= 0:
        return 0.0
    return min(token_count / ceiling, 1.0)