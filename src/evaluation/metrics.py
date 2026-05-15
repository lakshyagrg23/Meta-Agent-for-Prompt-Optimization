from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    false_positive_rate: float
    false_negative_rate: float

    consistency: float


class MetricsEngine:
    """
    Deterministic evaluation metrics computation.
    """

    @staticmethod
    def compute_metrics(
        predictions: List[str],
        labels: List[str]
    ) -> EvaluationMetrics:
        """
        Compute phishing classification metrics.
        """