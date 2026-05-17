"""
src/mutations/refine_enrichment.py
-----------------------------------
Refinement operator for the instruction enrichment component of a prompt.

Ensures that the LLM focuses purely on phishing detection heuristics and
guidance, explicitly avoiding persona framing, chain-of-thought bloat,
or general task duplication.
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


class RefineEnrichmentOperator(BaseMutationOperator):
    """
    Operator dedicated to refining the PromptState.instruction_enrichment component.
    
    Dependency Injection:
        Accepts a `BaseRefinementGenerator`. Defaults to 
        `DeterministicPlaceholderGenerator` if none is provided.
    """

    def __init__(self, generator: Optional[BaseRefinementGenerator] = None):
        self.generator = generator or DeterministicPlaceholderGenerator()

    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        """
        In-place mutation logic for the cloned PromptState's instruction_enrichment component.
        
        Args:
            candidate: A cloned PromptState ready to be modified.
            context: Context containing signals guiding the mutation.
            
        Returns:
            A string summarizing the changes made.
        """
        active_signals = context.signals.active_names()
        
        # Add strict enrichment boundaries to prevent cross-contamination of concerns
        framing_constraint = (
            "Enrichment refinement must focus only on phishing detection guidance. "
            "Avoid role framing. Avoid task duplication. Avoid chain-of-thought "
            "reasoning instructions. Prefer concise semantic refinement over accumulation."
        )
        
        summary_text = f"{framing_constraint} Active issues: {', '.join(active_signals)}" if active_signals else framing_constraint

        request = RefinementRequest(
            target_component=RefinementTarget.ENRICHMENT,
            current_content=candidate.instruction_enrichment.content,
            token_budget=candidate.instruction_enrichment.token_budget,
            failure_summary=summary_text,
            optimization_signals=active_signals,
            contextual_examples=[]
        )

        response = self.generator.generate_refinement(request)

        # 1. Mutate ONLY candidate.instruction_enrichment.content
        candidate.instruction_enrichment.content = response.refined_content
        
        # 2. Increment revision metadata
        candidate.instruction_enrichment.revision_count += 1
        
        if hasattr(candidate.metadata, "mutation_history"):
            candidate.metadata.mutation_history.append("RefineEnrichmentOperator applied.")

        # 3. Return mutation summary
        return (
            f"Refined enrichment (Tokens: {response.token_count}/{candidate.instruction_enrichment.token_budget}, "
            f"Truncated: {response.truncated})"
        )
