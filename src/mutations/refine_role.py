"""
src/mutations/refine_role.py
-----------------------------
Refinement operator for the role component of a prompt.

Ensures the persona and task framing remain concise and relevant to
cybersecurity analysis, without bleeding into instruction enrichment or few-shot
memory areas.
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


class RefineRoleOperator(BaseMutationOperator):
    """
    Operator dedicated to refining the PromptState.role component.
    
    Dependency Injection:
        Accepts a `BaseRefinementGenerator`. Defaults to 
        `DeterministicPlaceholderGenerator` if none is provided.
    """

    def __init__(self, generator: Optional[BaseRefinementGenerator] = None):
        self.generator = generator or DeterministicPlaceholderGenerator()

    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        """
        In-place mutation logic for the cloned PromptState's role component.
        
        Args:
            candidate: A cloned PromptState ready to be modified.
            context: Context containing signals guiding the mutation.
            
        Returns:
            A string summarizing the changes made.
        """
        active_signals = context.signals.active_names()
        
        # Add lightweight cybersecurity framing constraints directly to the summary
        framing_constraint = (
            "Role refinement must remain lightweight: concise cybersecurity "
            "analyst framing, no large instructions, no task duplication."
        )
        
        summary_text = f"{framing_constraint} Active issues: {', '.join(active_signals)}" if active_signals else framing_constraint

        request = RefinementRequest(
            target_component=RefinementTarget.ROLE,
            current_content=candidate.role.content,
            token_budget=candidate.role.token_budget,
            failure_summary=summary_text,
            optimization_signals=active_signals,
            contextual_examples=[]
        )

        response = self.generator.generate_refinement(request)

        # 1. Mutate ONLY candidate.role.content
        candidate.role.content = response.refined_content
        
        # 2. Increment revision metadata
        candidate.role.revision_count += 1
        
        # Metadata on PromptState can also be tracked, but requirement says 
        # "increment revision metadata" which usually means the component's revision counter.
        if hasattr(candidate.metadata, "mutation_history"):
            candidate.metadata.mutation_history.append("RefineRoleOperator applied.")

        # 3. Return mutation summary
        return (
            f"Refined role (Tokens: {response.token_count}/{candidate.role.token_budget}, "
            f"Truncated: {response.truncated})"
        )
