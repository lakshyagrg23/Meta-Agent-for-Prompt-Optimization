"""
src/mutations/base_operator.py
-------------------------------
Abstract base class for all prompt refinement operators.

Architectural contract
----------------------
- Operators NEVER mutate the original PromptState directly.
- The `mutate` wrapper handles cloning and central validation.
- Subclasses implement `_mutate(candidate, context)` to modify the clone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

from src.core.prompt_state import PromptState
from src.core.validator import PromptValidator, ValidationResult
from src.critic.signal_extractor import CriticSignals


@dataclass
class MutationContext:
    """Contextual information needed for mutation."""
    signals: CriticSignals
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationResult:
    """The result of applying a mutation operator."""
    candidate_state: PromptState
    validation_result: ValidationResult
    operator_name: str
    success: bool
    mutation_summary: str


class BaseMutationOperator(ABC):
    """
    Abstract base class for all refinement operators.
    
    Responsibilities:
    - Clone state safely
    - Preserve immutability of original state
    - Validate candidate state
    - Centralize validation flow
    """

    def mutate(
        self,
        state: PromptState,
        context: MutationContext,
    ) -> MutationResult:
        """
        Execute mutation safely on a clone of the state and validate it.
        
        Args:
            state: The original, read-only PromptState.
            context: Context containing signals and extra metadata.
            
        Returns:
            A MutationResult containing the new candidate and validation results.
        """
        # 1. Clone state safely to preserve immutability
        candidate = state.clone()

        # 2. Delegate to subclass to mutate the cloned candidate
        summary = self._mutate(candidate, context)
        
        # Double-check that we still have a distinct object
        assert candidate is not state, "Operator mutated original state or broke clone pattern."

        # 3. Validate candidate state
        validation_result = PromptValidator.validate_state(candidate)

        # 4. Centralize validation flow (success logic)
        success = validation_result.is_valid
        
        if not success:
            summary = f"Validation failed with {len(validation_result.errors)} errors."

        return MutationResult(
            candidate_state=candidate,
            validation_result=validation_result,
            operator_name=self.__class__.__name__,
            success=success,
            mutation_summary=summary
        )

    @abstractmethod
    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        """
        Subclasses implement this to perform the actual mutation logic.
        
        IMPORTANT: Modify `candidate` IN-PLACE. Do not modify the original state.
        
        Args:
            candidate: A cloned PromptState ready to be modified.
            context: Signals and context guiding the mutation.
            
        Returns:
            A string summarizing the changes made.
        """
        pass