from dataclasses import dataclass
from src.evaluation.metrics import EvaluationMetrics


@dataclass
class CriticSignals:
    high_fn: bool
    high_fp: bool
    low_accuracy: bool
    inconsistent: bool
    plateau: bool


class SignalExtractor:
    """
    Converts evaluation metrics into deterministic refinement signals.
    """

    @staticmethod
    def extract_signals(
        metrics: EvaluationMetrics
    ) -> CriticSignals:
        """
        Generate deterministic optimization signals.
        """