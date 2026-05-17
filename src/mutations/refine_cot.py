"""
src/mutations/refine_cot.py
----------------------------
Refinement operator for the chain-of-thought (CoT) component of a prompt.

Ensures that the LLM focuses purely on concise reasoning behavior, explicitly
avoiding verbose multi-step chains, task duplication, role framing, or output
formatting instructions.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.prompt_state import PromptState
from src.mutations.base_operator import BaseMutationOperator, MutationContext
from src.llm.refinement_generation import (
    BaseRefinementGenerator,
    DeterministicPlaceholderGenerator,
    RefinementRequest,
    RefinementTarget
)

logger = logging.getLogger(__name__)


class RefineCoTOperator(BaseMutationOperator):
    """
    Operator dedicated to refining the PromptState.cot component.
    
    Dependency Injection:
        Accepts a `BaseRefinementGenerator`. Defaults to 
        `DeterministicPlaceholderGenerator` if none is provided.
    """

    def __init__(self, generator: Optional[BaseRefinementGenerator] = None):
        self.generator = generator or DeterministicPlaceholderGenerator()

    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        """
        In-place mutation logic for the cloned PromptState's cot component.
        
        Args:
            candidate: A cloned PromptState ready to be modified.
            context: Context containing signals guiding the mutation.
            
        Returns:
            A string summarizing the changes made.
        """
        active_signals = context.signals.active_names()
        
        # Add strict CoT boundaries to prevent cross-contamination and bloat
        framing_constraint = (
            "CoT refinement must remain concise and lightweight. Guide reasoning "
            "behavior only. Avoid verbose multi-step chains. Avoid task duplication. "
            "Avoid role framing. Avoid output formatting instructions."
        )
        
        summary_text = f"{framing_constraint} Active issues: {', '.join(active_signals)}" if active_signals else framing_constraint

        request = RefinementRequest(
            target_component=RefinementTarget.COT,
            current_content=candidate.cot.content,
            token_budget=candidate.cot.token_budget,
            failure_summary=summary_text,
            optimization_signals=active_signals,
            contextual_examples=[]
        )

        response = self.generator.generate_refinement(request)

        # 1. Mutate ONLY candidate.cot.content
        candidate.cot.content = response.refined_content
        
        # 2. Increment revision metadata
        candidate.cot.revision_count += 1
        
        if hasattr(candidate.metadata, "mutation_history"):
            candidate.metadata.mutation_history.append("RefineCoTOperator applied.")

        # 3. Return mutation summary
        return (
            f"Refined CoT (Tokens: {response.token_count}/{candidate.cot.token_budget}, "
            f"Truncated: {response.truncated})"
        )
