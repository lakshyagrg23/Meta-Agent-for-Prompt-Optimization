from abc import ABC, abstractmethod
from src.core.prompt_state import PromptState


class MutationOperator(ABC):
    """
    Abstract base class for prompt refinement operators.
    """

    @abstractmethod
    def refine(
        self,
        prompt_state: PromptState,
        error_summary: dict
    ) -> PromptState:
        """
        Generate refined PromptState candidate.
        """