from src.mutations.base_operator import MutationOperator
from src.core.prompt_state import PromptState


class RefineFewShotOperator(MutationOperator):
    """
    Adaptive bounded few-shot refinement.

    Responsibilities:
    - generate new examples
    - replace least relevant examples if capacity full
    - maintain token budget
    - refine examples based on current error patterns
    """

    def refine(
        self,
        prompt_state: PromptState,
        error_summary: dict
    ) -> PromptState:
        """
        Generate refined few-shot component.
        """

    def compute_relevance_scores(
        self,
        prompt_state: PromptState,
        error_summary: dict
    ):
        """
        Score current examples against current failure patterns.
        """

    def replace_least_relevant_example(
        self,
        prompt_state: PromptState,
        new_example
    ):
        """
        Replace weakest example while respecting memory bounds.
        """