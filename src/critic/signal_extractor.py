"""
src/critic/signal_extractor.py
--------------------------------
Deterministic conversion of evaluation metrics into optimization signals.

No randomness, no external API calls, no LLM involvement.
All outputs are fully determined by the inputs.

Signal definitions
------------------

high_fn
    False-negative rate exceeds ``thresholds.fn_rate``.
    Phishing emails are being missed; few-shot refinement is likely needed.

high_fp
    False-positive rate exceeds ``thresholds.fp_rate``.
    Legitimate emails are being over-flagged; instruction refinement is needed.

low_accuracy
    Overall accuracy falls below ``thresholds.min_accuracy``.
    General performance is poor; role or instruction refinement is needed.

inconsistent
    Output consistency falls below ``thresholds.min_consistency``.
    The prompt produces unstable outputs; CoT refinement is typically triggered.

plateau
    The absolute improvement in the composite score across the last ``k``
    iterations is smaller than ``thresholds.plateau_delta``.
    The current refinement strategy has stagnated; a different operator is needed.

Plateau detection formula
--------------------------
::

    plateau = True  if  |score_history[-1] - score_history[-(k+1)]| < delta
                        AND  len(score_history) >= k + 1

    plateau = False if  len(score_history) < k + 1
                        (not enough history to detect a plateau)

Signal priority
---------------
Priority is defined by ``MutationPolicy``, not here.  ``SignalExtractor``
reports what is observed; it does not decide what to do about it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CriticThresholds — all tuneable threshold values in one place
# ---------------------------------------------------------------------------

@dataclass
class CriticThresholds:
    """
    Threshold values that govern signal extraction.

    All fields have research-grounded defaults that can be overridden per
    experiment via ``configs/experiment_config.yaml`` or direct instantiation.

    Attributes:
        fn_rate:         False-negative rate above which ``high_fn`` fires.
        fp_rate:         False-positive rate above which ``high_fp`` fires.
        min_accuracy:    Accuracy below which ``low_accuracy`` fires.
        min_consistency: Consistency below which ``inconsistent`` fires.
        plateau_window:  Number of consecutive iterations (k) used for
                         plateau detection.
        plateau_delta:   Minimum score improvement required to NOT be
                         considered a plateau.
    """

    fn_rate: float = 0.20
    fp_rate: float = 0.15
    min_accuracy: float = 0.75
    min_consistency: float = 0.80
    plateau_window: int = 3
    plateau_delta: float = 0.01


# ---------------------------------------------------------------------------
# CriticSignals — structured output of the extraction pass
# ---------------------------------------------------------------------------

@dataclass
class CriticSignals:
    """
    Boolean optimization signals derived from one evaluation pass.

    Each field maps to a distinct failure mode.  ``True`` means the failure
    mode is currently active; ``False`` means it is not.

    Attributes:
        high_fn:      False-negative rate exceeds the configured threshold.
        high_fp:      False-positive rate exceeds the configured threshold.
        low_accuracy: Overall accuracy is below the configured threshold.
        inconsistent: Output consistency is below the configured threshold.
        plateau:      Score improvement has stagnated across k iterations.
    """

    high_fn: bool
    high_fp: bool
    low_accuracy: bool
    inconsistent: bool
    plateau: bool

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def any_active(self) -> bool:
        """Return True if at least one signal is active."""
        return self.high_fn or self.high_fp or self.low_accuracy or self.inconsistent or self.plateau

    def summary(self) -> str:
        """
        Return a compact one-line summary for logging.

        Example::

            "high_fn=True high_fp=False low_acc=False inconsistent=False plateau=True"
        """
        return (
            f"high_fn={self.high_fn} "
            f"high_fp={self.high_fp} "
            f"low_acc={self.low_accuracy} "
            f"inconsistent={self.inconsistent} "
            f"plateau={self.plateau}"
        )

    def active_names(self) -> List[str]:
        """
        Return a list of the names of currently active signals.

        Useful for structured logging and mutation history tracking.

        Returns:
            List of signal name strings, e.g. ``["high_fn", "plateau"]``.
            Empty list if no signals are active.
        """
        names: List[str] = []
        if self.high_fn:
            names.append("high_fn")
        if self.high_fp:
            names.append("high_fp")
        if self.low_accuracy:
            names.append("low_accuracy")
        if self.inconsistent:
            names.append("inconsistent")
        if self.plateau:
            names.append("plateau")
        return names


# ---------------------------------------------------------------------------
# SignalExtractor
# ---------------------------------------------------------------------------

class SignalExtractor:
    """
    Deterministic conversion of evaluation metrics into optimization signals.

    All methods are static.  No mutable state is held between calls.
    No randomness, no external I/O, no LLM calls.
    """

    @staticmethod
    def extract_signals(
        metrics: EvaluationMetrics,
        score_history: List[float],
        thresholds: Optional[CriticThresholds] = None,
    ) -> CriticSignals:
        """
        Compute all five optimization signals from metrics and score history.

        Args:
            metrics:       Evaluation metrics for the current prompt state.
            score_history: Ordered list of composite J(S) scores, one per
                           completed optimization iteration.  The most recent
                           score is the last element.  May be empty.
            thresholds:    Threshold configuration.  Defaults to
                           :class:`CriticThresholds` with default values.

        Returns:
            :class:`CriticSignals` with all five boolean signals populated.

        Example::

            thresholds = CriticThresholds(fn_rate=0.20, plateau_window=3)
            signals = SignalExtractor.extract_signals(metrics, history, thresholds)
            if signals.high_fn:
                # trigger few-shot refinement
        """
        if thresholds is None:
            thresholds = CriticThresholds()

        signals = CriticSignals(
            high_fn=SignalExtractor._detect_high_fn(metrics, thresholds),
            high_fp=SignalExtractor._detect_high_fp(metrics, thresholds),
            low_accuracy=SignalExtractor._detect_low_accuracy(metrics, thresholds),
            inconsistent=SignalExtractor._detect_inconsistency(metrics, thresholds),
            plateau=SignalExtractor._detect_plateau(score_history, thresholds),
        )

        logger.debug("Signals extracted: %s", signals.summary())
        return signals

    # ------------------------------------------------------------------
    # Private signal detectors — one method per signal for isolability
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_high_fn(
        metrics: EvaluationMetrics,
        thresholds: CriticThresholds,
    ) -> bool:
        """
        Return True when the false-negative rate exceeds the threshold.

        FNR > tau_fn  →  high_fn = True

        A high false-negative rate means phishing emails are being missed —
        the most safety-critical failure mode in this domain.

        Args:
            metrics:    Current evaluation metrics.
            thresholds: Configured signal thresholds.

        Returns:
            True if FNR > ``thresholds.fn_rate``.
        """
        result = metrics.false_negative_rate > thresholds.fn_rate
        logger.debug(
            "_detect_high_fn: fnr=%.3f threshold=%.3f → %s",
            metrics.false_negative_rate, thresholds.fn_rate, result,
        )
        return result

    @staticmethod
    def _detect_high_fp(
        metrics: EvaluationMetrics,
        thresholds: CriticThresholds,
    ) -> bool:
        """
        Return True when the false-positive rate exceeds the threshold.

        FPR > tau_fp  →  high_fp = True

        A high false-positive rate means legitimate emails are being over-flagged,
        degrading user trust and increasing alert fatigue.

        Args:
            metrics:    Current evaluation metrics.
            thresholds: Configured signal thresholds.

        Returns:
            True if FPR > ``thresholds.fp_rate``.
        """
        result = metrics.false_positive_rate > thresholds.fp_rate
        logger.debug(
            "_detect_high_fp: fpr=%.3f threshold=%.3f → %s",
            metrics.false_positive_rate, thresholds.fp_rate, result,
        )
        return result

    @staticmethod
    def _detect_low_accuracy(
        metrics: EvaluationMetrics,
        thresholds: CriticThresholds,
    ) -> bool:
        """
        Return True when overall accuracy falls below the threshold.

        accuracy < tau_acc  →  low_accuracy = True

        Low accuracy reflects insufficient task understanding and typically
        triggers role or instruction enrichment refinement.

        Args:
            metrics:    Current evaluation metrics.
            thresholds: Configured signal thresholds.

        Returns:
            True if accuracy < ``thresholds.min_accuracy``.
        """
        result = metrics.accuracy < thresholds.min_accuracy
        logger.debug(
            "_detect_low_accuracy: acc=%.3f threshold=%.3f → %s",
            metrics.accuracy, thresholds.min_accuracy, result,
        )
        return result

    @staticmethod
    def _detect_inconsistency(
        metrics: EvaluationMetrics,
        thresholds: CriticThresholds,
    ) -> bool:
        """
        Return True when output consistency falls below the threshold.

        consistency < tau_cons  →  inconsistent = True

        Low consistency indicates the prompt produces unstable predictions
        across repeated inference runs, reducing reproducibility.

        Args:
            metrics:    Current evaluation metrics.
            thresholds: Configured signal thresholds.

        Returns:
            True if consistency < ``thresholds.min_consistency``.
        """
        result = metrics.consistency < thresholds.min_consistency
        logger.debug(
            "_detect_inconsistency: cons=%.3f threshold=%.3f → %s",
            metrics.consistency, thresholds.min_consistency, result,
        )
        return result

    @staticmethod
    def _detect_plateau(
        score_history: List[float],
        thresholds: CriticThresholds,
    ) -> bool:
        """
        Return True when score improvement has stagnated across k iterations.

        Formula::

            plateau = True  if  len(history) >= k + 1
                                AND  |history[-1] - history[-(k+1)]| < delta

        The window comparison ``history[-1]`` vs ``history[-(k+1)]`` measures
        the total score change over the last ``k`` iterations.  Using the
        endpoints rather than pairwise comparisons is robust to single-step
        oscillations (a score may dip then recover without meaningful progress).

        Returns False (no plateau) when fewer than ``k + 1`` scores are
        available — there is not yet enough history to make a determination.

        Args:
            score_history: Ordered score list; most recent score last.
            thresholds:    Configured signal thresholds.

        Returns:
            True if the optimization has plateaued, False otherwise.
        """
        k: int = thresholds.plateau_window
        delta: float = thresholds.plateau_delta

        if len(score_history) < k + 1:
            logger.debug(
                "_detect_plateau: insufficient history (have %d, need %d) → False",
                len(score_history), k + 1,
            )
            return False

        recent: float = score_history[-1]
        earlier: float = score_history[-(k + 1)]
        improvement: float = abs(recent - earlier)

        result = improvement < delta
        logger.debug(
            "_detect_plateau: score[-%d]=%.4f score[-1]=%.4f Δ=%.4f delta=%.4f → %s",
            k + 1, earlier, recent, improvement, delta, result,
        )
        return result