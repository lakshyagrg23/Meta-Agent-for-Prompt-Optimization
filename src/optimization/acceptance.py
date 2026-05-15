from src.evaluation.metrics import EvaluationMetrics


class AcceptanceStrategy:
    """
    Determines whether refined PromptState should be accepted.
    """

    @staticmethod
    def compute_score(
        metrics: EvaluationMetrics
    ) -> float:
        """
        Composite optimization objective.
        """

    @staticmethod
    def should_accept(
        current_score: float,
        candidate_score: float,
        epsilon: float
    ) -> bool:
        """
        Accept candidate only if improvement exceeds threshold.
        """