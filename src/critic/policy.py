from enum import Enum
from src.critic.signal_extractor import CriticSignals


class RefinementOperator(Enum):
    REFINE_ROLE = "refine_role"
    REFINE_ENRICHMENT = "refine_enrichment"
    REFINE_COT = "refine_cot"
    REFINE_FEWSHOT = "refine_fewshot"


class MutationPolicy:
    """
    Deterministically maps signals to refinement operators.
    """

    @staticmethod
    def select_operator(
        signals: CriticSignals
    ) -> RefinementOperator:
        """
        Select refinement operator using deterministic policy.
        """