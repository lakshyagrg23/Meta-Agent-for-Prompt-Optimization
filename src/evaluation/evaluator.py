"""
src/evaluation/evaluator.py
----------------------------
Unified evaluation pipeline — central entry point for the optimization loop.

Responsibilities
----------------
1. Accept predictions, labels, optional repeated predictions, and prompt
   token count.
2. Compute classification metrics via :class:`~src.evaluation.metrics.MetricsEngine`.
3. Compute sample-major consistency via
   :func:`~src.evaluation.consistency.compute_consistency`.
4. Merge consistency into the :class:`~src.evaluation.metrics.EvaluationMetrics`
   object.
5. Compute a scalar composite optimization score from the merged metrics.
6. Return a single :class:`EvaluationResult` containing everything the
   optimization loop needs.

Composite score formula
-----------------------
The optimization score balances classification performance with consistency::

    score = w_f1 * f1
          + w_recall * recall
          + w_precision * precision
          - w_fnr * false_negative_rate
          - w_fpr * false_positive_rate
          + w_consistency * consistency

Default weights are conservative (recall and FNR weighted most heavily)
because missing phishing is operationally worse than a false alarm.
Weights are configurable at call time so the optimization loop can tune
the objective without touching this module.

Design constraints
------------------
* **Stateless** — all methods are static; no side effects.
* **Deterministic** — same inputs always produce the same score.
* **Modular** — metric computation, consistency computation, and scoring
  are separate steps; each can be called independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.evaluation.consistency import ConsistencyResult, compute_consistency
from src.evaluation.metrics import (
    EvaluationMetrics,
    MetricsEngine,
    empty_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default composite score weights
# ---------------------------------------------------------------------------

#: Default weights for :func:`compute_optimization_score`.
#: Keys map to :class:`EvaluationMetrics` float fields.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "f1":                   0.30,
    "recall":               0.25,   # catching phishing is most critical
    "precision":            0.15,
    "false_negative_rate": -0.20,   # penalty: missed phishing
    "false_positive_rate": -0.05,   # penalty: false alarm
    "consistency":          0.05,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Complete result of one evaluation pass through the pipeline.

    Attributes:
        metrics:              Full :class:`EvaluationMetrics` with consistency
                              already merged in.
        consistency_result:   Raw :class:`ConsistencyResult` from the
                              sample-major consistency module, or ``None``
                              if no repeated predictions were supplied.
        optimization_score:   Scalar composite score in approximately
                              [−0.25, 1.0]; used by the acceptance strategy
                              to compare current vs. candidate prompts.
        prompt_token_count:   Token count of the rendered prompt, as supplied
                              by the caller.  0 if not provided.
        weights_used:         The weight dict that produced *optimization_score*,
                              stored for reproducibility.
    """

    metrics: EvaluationMetrics
    consistency_result: Optional[ConsistencyResult]
    optimization_score: float
    prompt_token_count: int
    weights_used: Dict[str, float]

    def summary(self) -> str:
        """
        Return a one-line log summary combining metrics and score.

        Example::

            "score=0.731 | acc=0.872 prec=0.891 rec=0.854 f1=0.872 fpr=0.109 fnr=0.146 cons=0.960 n=250 | tokens=312"
        """
        return (
            f"score={self.optimization_score:.4f} | "
            f"{self.metrics.summary()} | "
            f"tokens={self.prompt_token_count}"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Stateless evaluation pipeline orchestrator.

    All methods are static.  The class exists purely for namespace clarity
    and to mirror the :class:`~src.evaluation.metrics.MetricsEngine` pattern
    already established in this codebase.

    Typical usage inside the optimization loop::

        result = Evaluator.evaluate(
            predictions=model_outputs,
            labels=batch_labels,
            repeated_predictions=repeated_outputs,   # optional
            prompt_token_count=state.get_total_token_count(),
        )

        if AcceptanceStrategy.should_accept(
            current_score=prev_result.optimization_score,
            candidate_score=result.optimization_score,
            epsilon=SCORE_DELTA_THRESHOLD,
        ):
            current_state = candidate_state
    """

    @staticmethod
    def evaluate(
        predictions: List[str],
        labels: List[str],
        repeated_predictions: Optional[List[List[str]]] = None,
        prompt_token_count: int = 0,
        weights: Optional[Dict[str, float]] = None,
    ) -> EvaluationResult:
        """
        Run the full evaluation pipeline for one optimization step.

        Pipeline steps:

        1. Compute classification metrics from *predictions* and *labels*.
        2. If *repeated_predictions* is supplied, compute sample-major
           consistency and merge it into the metrics object.
        3. Compute the composite optimization score.
        4. Return a fully populated :class:`EvaluationResult`.

        Args:
            predictions:          Model-predicted labels for the batch, one
                                  per example (``"PHISHING"`` or ``"SAFE"``).
            labels:               Ground-truth labels, same length as
                                  *predictions*.
            repeated_predictions: Optional per-sample repeated predictions in
                                  sample-major format:
                                  ``repeated_predictions[i][j]`` = sample
                                  ``i``'s ``j``-th prediction.  Used to
                                  compute sample-level consistency.  Pass
                                  ``None`` to skip (consistency defaults to
                                  1.0 from the single-run metrics engine).
            prompt_token_count:   Token count of the rendered prompt.
                                  Stored for logging; not used in scoring.
            weights:              Optional weight override dict.  Missing keys
                                  fall back to :data:`DEFAULT_WEIGHTS`.
                                  Pass ``None`` to use all defaults.

        Returns:
            :class:`EvaluationResult` with all fields populated.

        Example::

            result = Evaluator.evaluate(
                predictions=["PHISHING", "SAFE", "PHISHING"],
                labels=["PHISHING", "SAFE", "SAFE"],
                repeated_predictions=[
                    ["PHISHING", "PHISHING"],
                    ["SAFE", "SAFE"],
                    ["PHISHING", "SAFE"],
                ],
                prompt_token_count=312,
            )
            print(result.summary())
        """
        resolved_weights = _resolve_weights(weights)

        # Step 1: classification metrics (consistency field defaults to 1.0).
        metrics = MetricsEngine.compute_metrics(predictions, labels)

        # Step 2: consistency (optional).
        consistency_result: Optional[ConsistencyResult] = None

        if repeated_predictions is not None:
            consistency_result = compute_consistency(repeated_predictions)
            metrics = Evaluator._merge_consistency(metrics, consistency_result)
        else:
            logger.debug(
                "No repeated predictions supplied — consistency defaults to 1.0."
            )

        # Step 3: composite score.
        score = compute_optimization_score(metrics, resolved_weights)

        logger.debug(
            "Evaluation complete: %s (score=%.4f, tokens=%d)",
            metrics.summary(),
            score,
            prompt_token_count,
        )

        return EvaluationResult(
            metrics=metrics,
            consistency_result=consistency_result,
            optimization_score=score,
            prompt_token_count=prompt_token_count,
            weights_used=resolved_weights,
        )

    @staticmethod
    def _merge_consistency(
        metrics: EvaluationMetrics,
        consistency_result: ConsistencyResult,
    ) -> EvaluationMetrics:
        """
        Return a copy of *metrics* with the consistency field overwritten by
        the batch consistency from *consistency_result*.

        Uses :meth:`~src.evaluation.metrics.MetricsEngine.attach_consistency`
        indirectly via a direct field copy to avoid re-importing the engine.

        Args:
            metrics:            Base metrics from :class:`MetricsEngine`.
            consistency_result: Result from the sample-major consistency module.

        Returns:
            New :class:`EvaluationMetrics` with updated consistency; all
            other fields are unchanged.
        """
        import copy
        updated = copy.copy(metrics)
        updated.consistency = consistency_result.batch_consistency
        return updated


# ---------------------------------------------------------------------------
# Composite scoring (module-level so it can be called independently)
# ---------------------------------------------------------------------------

def compute_optimization_score(
    metrics: EvaluationMetrics,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a scalar optimization score from :class:`EvaluationMetrics`.

    The score is a weighted linear combination of metric fields::

        score = Σ  weight_k * metric_k

    Positive weights reward good metrics (f1, recall, precision, consistency).
    Negative weights penalise bad metrics (false_negative_rate, false_positive_rate).

    The result is in approximately [−0.25, 1.0] with :data:`DEFAULT_WEIGHTS`,
    but the caller may supply custom weights that change this range.

    Args:
        metrics: Populated :class:`EvaluationMetrics` instance.
        weights: Optional weight override.  Missing keys use :data:`DEFAULT_WEIGHTS`.
                 Pass ``None`` to use all defaults.

    Returns:
        Scalar float score.  Higher is better.

    Example::

        score = compute_optimization_score(metrics)
        # or with custom weights:
        score = compute_optimization_score(metrics, {"f1": 1.0, "recall": 0.0, ...})
    """
    resolved = _resolve_weights(weights)
    score = 0.0
    for metric_name, weight in resolved.items():
        value = getattr(metrics, metric_name, 0.0)
        score += weight * value
    return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Merge caller-supplied *weights* with :data:`DEFAULT_WEIGHTS`.

    Caller values take precedence; any key absent from *weights* falls back
    to the default.

    Args:
        weights: Partial or complete weight override, or ``None``.

    Returns:
        Complete weight dict with all keys from :data:`DEFAULT_WEIGHTS`.
    """
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(weights)
    return merged
