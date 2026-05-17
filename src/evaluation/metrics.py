"""
src/evaluation/metrics.py
--------------------------
Deterministic phishing classification metrics engine.

PHISHING is the positive class throughout.  All metrics use
``pos_label="PHISHING"`` so that precision, recall, and F1 measure the
system's ability to correctly identify phishing emails — the operationally
critical direction.

Metric definitions
------------------
Given the confusion matrix entries TP, FP, TN, FN (w.r.t. PHISHING):

* **accuracy**           = (TP + TN) / total
* **precision**          = TP / (TP + FP)          — of predicted positives, how many are real
* **recall**             = TP / (TP + FN)           — of real positives, how many were caught
* **f1**                 = 2 * precision * recall / (precision + recall)
* **false_positive_rate** = FP / (FP + TN)          — legitimate emails misclassified as phishing
* **false_negative_rate** = FN / (TP + FN)          — phishing emails missed (= 1 - recall)
* **consistency**        = mean pairwise agreement across repeated runs on the same inputs

All divide-by-zero cases return 0.0 with a logged warning rather than
raising exceptions, making the engine safe to call mid-optimisation loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.core.constants import LABEL_PHISHING, LABEL_SAFE, LABEL_LIST

logger = logging.getLogger(__name__)

# The positive class label used throughout.
POSITIVE_LABEL: str = LABEL_PHISHING
NEGATIVE_LABEL: str = LABEL_SAFE

# Labels list used in every sklearn call so output is always in [SAFE, PHISHING] order.
_LABELS: List[str] = LABEL_LIST


# ---------------------------------------------------------------------------
# EvaluationMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    """
    Holds all evaluation metrics for one classification run.

    All float fields are in the range [0.0, 1.0].  A value of 0.0 is used
    as the safe sentinel when a metric is undefined (e.g. precision when
    there are zero predicted positives).

    Attributes:
        accuracy:            Fraction of all examples correctly classified.
        precision:           Fraction of PHISHING predictions that are correct.
        recall:              Fraction of true PHISHING emails that were detected.
        f1:                  Harmonic mean of precision and recall.
        false_positive_rate: Fraction of SAFE emails incorrectly flagged as PHISHING.
        false_negative_rate: Fraction of PHISHING emails missed (1 − recall).
        consistency:         Mean pairwise label agreement across repeated inference
                             runs on the same inputs.  1.0 = perfectly consistent.
                             Set to 1.0 when only a single run is provided.
        sample_count:        Number of examples in the evaluated batch.
        phishing_count:      Number of true PHISHING examples in the batch.
        safe_count:          Number of true SAFE examples in the batch.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float

    false_positive_rate: float
    false_negative_rate: float

    consistency: float

    # Optional informational fields — not used in signal extraction but
    # useful for logging and debugging.
    sample_count: int = 0
    phishing_count: int = 0
    safe_count: int = 0

    def is_empty(self) -> bool:
        """Return True if metrics were computed over zero samples."""
        return self.sample_count == 0

    def summary(self) -> str:
        """
        Return a compact one-line summary suitable for logging.

        Example::

            "acc=0.872 prec=0.891 rec=0.854 f1=0.872 fpr=0.109 fnr=0.146 cons=0.960 n=250"
        """
        return (
            f"acc={self.accuracy:.3f} "
            f"prec={self.precision:.3f} "
            f"rec={self.recall:.3f} "
            f"f1={self.f1:.3f} "
            f"fpr={self.false_positive_rate:.3f} "
            f"fnr={self.false_negative_rate:.3f} "
            f"cons={self.consistency:.3f} "
            f"n={self.sample_count}"
        )


# ---------------------------------------------------------------------------
# Zero-metrics sentinel
# ---------------------------------------------------------------------------

def empty_metrics() -> EvaluationMetrics:
    """
    Return an all-zero :class:`EvaluationMetrics` instance.

    Used as a safe default when evaluation cannot proceed (e.g. empty batch).

    Returns:
        :class:`EvaluationMetrics` with all floats = 0.0 and sample_count = 0.
    """
    return EvaluationMetrics(
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
        consistency=0.0,
        sample_count=0,
        phishing_count=0,
        safe_count=0,
    )


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """
    Stateless, deterministic metrics computation engine.

    All methods are static.  No mutable state is kept between calls.
    """

    @staticmethod
    def compute_metrics(
        predictions: List[str],
        labels: List[str],
    ) -> EvaluationMetrics:
        """
        Compute all classification metrics for one prediction run.

        PHISHING is the positive class for precision, recall, F1, FPR, and FNR.
        Both ``predictions`` and ``labels`` must use exactly ``"PHISHING"``
        and ``"SAFE"`` as values (case-sensitive).

        Args:
            predictions: Model-predicted labels, one per example.
            labels:      Ground-truth labels, same length as *predictions*.

        Returns:
            Populated :class:`EvaluationMetrics` instance.
            Returns :func:`empty_metrics` if either list is empty or lengths differ.

        Example::

            metrics = MetricsEngine.compute_metrics(
                predictions=["PHISHING", "SAFE", "PHISHING"],
                labels=["PHISHING", "SAFE", "SAFE"],
            )
            print(metrics.summary())
        """
        if not predictions or not labels:
            logger.warning("compute_metrics called with empty lists — returning zero metrics.")
            return empty_metrics()

        if len(predictions) != len(labels):
            logger.warning(
                "predictions length (%d) != labels length (%d) — returning zero metrics.",
                len(predictions), len(labels),
            )
            return empty_metrics()

        n = len(labels)
        phishing_count = labels.count(POSITIVE_LABEL)
        safe_count = labels.count(NEGATIVE_LABEL)

        accuracy = accuracy_score(labels, predictions)
        precision = MetricsEngine._safe_precision(labels, predictions)
        recall = MetricsEngine._safe_recall(labels, predictions)
        f1 = MetricsEngine._safe_f1(labels, predictions)
        fpr, fnr = MetricsEngine._compute_rates(labels, predictions)

        return EvaluationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            false_positive_rate=float(fpr),
            false_negative_rate=float(fnr),
            consistency=1.0,  # single-run default; use compute_consistency() for multi-run
            sample_count=n,
            phishing_count=phishing_count,
            safe_count=safe_count,
        )

    @staticmethod
    def compute_consistency(runs: List[List[str]]) -> float:
        """
        Measure label consistency across multiple inference runs on the same inputs.

        Consistency is defined as the mean fraction of examples where all runs
        agree on the predicted label.  This captures how stable the LLM's
        classification behaviour is for a given prompt.

        Args:
            runs: List of prediction lists, each the same length.
                  Each inner list is one complete inference pass over the batch.

        Returns:
            Float in [0.0, 1.0].  Returns 1.0 for a single run (trivially
            consistent) or when *runs* is empty.

        Example::

            run1 = ["PHISHING", "SAFE", "PHISHING"]
            run2 = ["PHISHING", "SAFE", "SAFE"]      # disagree on example 2
            score = MetricsEngine.compute_consistency([run1, run2])
            # → 0.667  (2 out of 3 positions agree)
        """
        if not runs or len(runs) == 1:
            return 1.0

        n_examples = len(runs[0])
        if n_examples == 0:
            return 1.0

        # Validate all runs are the same length.
        if any(len(r) != n_examples for r in runs):
            logger.warning(
                "compute_consistency: not all runs have the same length (%s) — returning 0.0.",
                [len(r) for r in runs],
            )
            return 0.0

        agreed = sum(
            1
            for i in range(n_examples)
            if len({run[i] for run in runs}) == 1   # all runs agree at position i
        )
        return agreed / n_examples

    @staticmethod
    def attach_consistency(
        metrics: EvaluationMetrics,
        runs: List[List[str]],
    ) -> EvaluationMetrics:
        """
        Return a copy of *metrics* with the consistency field populated.

        This is the intended pattern when consistency is measured separately
        from the main evaluation pass:

        1. Call :meth:`compute_metrics` → base metrics (consistency=1.0)
        2. Run additional inference passes
        3. Call :meth:`attach_consistency` → final metrics

        Args:
            metrics: Existing :class:`EvaluationMetrics` to update.
            runs:    All inference runs, **including the original one**.

        Returns:
            New :class:`EvaluationMetrics` with ``consistency`` updated.
            All other fields are unchanged.
        """
        import copy
        updated = copy.copy(metrics)
        updated.consistency = MetricsEngine.compute_consistency(runs)
        return updated

    # ------------------------------------------------------------------
    # Private helpers — safe wrappers around sklearn functions
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_precision(labels: List[str], predictions: List[str]) -> float:
        """
        Compute precision with zero_division=0.0 guard.

        Returns 0.0 when there are no predicted positives rather than
        raising a ZeroDivisionError.
        """
        try:
            return float(
                precision_score(
                    labels,
                    predictions,
                    pos_label=POSITIVE_LABEL,
                    average="binary",
                    zero_division=0,
                    labels=_LABELS,
                )
            )
        except Exception as exc:
            logger.warning("precision_score failed: %s — returning 0.0.", exc)
            return 0.0

    @staticmethod
    def _safe_recall(labels: List[str], predictions: List[str]) -> float:
        """
        Compute recall with zero_division=0.0 guard.

        Returns 0.0 when there are no true positives in the batch.
        """
        try:
            return float(
                recall_score(
                    labels,
                    predictions,
                    pos_label=POSITIVE_LABEL,
                    average="binary",
                    zero_division=0,
                    labels=_LABELS,
                )
            )
        except Exception as exc:
            logger.warning("recall_score failed: %s — returning 0.0.", exc)
            return 0.0

    @staticmethod
    def _safe_f1(labels: List[str], predictions: List[str]) -> float:
        """
        Compute F1 with zero_division=0.0 guard.

        Returns 0.0 when both precision and recall are 0.
        """
        try:
            return float(
                f1_score(
                    labels,
                    predictions,
                    pos_label=POSITIVE_LABEL,
                    average="binary",
                    zero_division=0,
                    labels=_LABELS,
                )
            )
        except Exception as exc:
            logger.warning("f1_score failed: %s — returning 0.0.", exc)
            return 0.0

    @staticmethod
    def _compute_rates(
        labels: List[str],
        predictions: List[str],
    ) -> tuple[float, float]:
        """
        Compute false-positive rate (FPR) and false-negative rate (FNR).

        FPR = FP / (FP + TN)  — fraction of SAFE emails incorrectly flagged.
        FNR = FN / (FN + TP)  — fraction of PHISHING emails missed (= 1 - recall).

        Both values are taken directly from the confusion matrix so they are
        consistent with the sklearn binary-classification convention.

        Args:
            labels:      Ground-truth label list.
            predictions: Predicted label list.

        Returns:
            Tuple of (fpr, fnr), each in [0.0, 1.0].  Returns (0.0, 0.0)
            when the denominator would be zero.
        """
        try:
            cm = confusion_matrix(labels, predictions, labels=_LABELS)
            # cm layout with labels=[SAFE, PHISHING]:
            #            Pred SAFE   Pred PHISHING
            # True SAFE  [ TN          FP ]
            # True PHISH [ FN          TP ]
            tn, fp, fn, tp = cm.ravel()
        except Exception as exc:
            logger.warning("confusion_matrix failed: %s — returning (0.0, 0.0).", exc)
            return 0.0, 0.0

        fpr = _safe_divide(fp, fp + tn)
        fnr = _safe_divide(fn, fn + tp)
        return fpr, fnr


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Return numerator / denominator, or 0.0 if denominator is zero.

    Args:
        numerator:   Dividend.
        denominator: Divisor.

    Returns:
        Division result, or 0.0 on zero-denominator.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator