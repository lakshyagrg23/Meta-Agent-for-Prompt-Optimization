"""
src/mutations/refine_fewshot.py
--------------------------------
Refinement operator for the few-shot component of a prompt.

Prioritizes resolving false negative failures by analyzing the failure report,
generating a concise reasoning explanation via the LLM, and replacing the
oldest examples if the memory capacity bounds are reached.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.prompt_state import PromptState, FewShotExample
from src.mutations.base_operator import BaseMutationOperator, MutationContext
from src.llm.refinement_generation import (
    BaseRefinementGenerator,
    DeterministicPlaceholderGenerator,
    RefinementRequest,
    RefinementTarget
)
from src.critic.error_analysis import ErrorAnalyzer

logger = logging.getLogger(__name__)


class RefineFewShotOperator(BaseMutationOperator):
    """
    Operator dedicated to refining the PromptState.few_shot component.
    
    Dependency Injection:
        Accepts a `BaseRefinementGenerator`. Defaults to 
        `DeterministicPlaceholderGenerator` if none is provided.
    """

    def __init__(self, generator: Optional[BaseRefinementGenerator] = None):
        self.generator = generator or DeterministicPlaceholderGenerator()

    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        """
        In-place mutation logic for the cloned PromptState's few_shot component.
        
        Args:
            candidate: A cloned PromptState ready to be modified.
            context: Context containing failure_report and signals guiding the mutation.
            
        Returns:
            A string summarizing the changes made.
        """
        report = context.extra.get("failure_report")
        if not report:
            return "No failure report found; few_shot refinement skipped."
            
        # 1. Retrieve top failure cases (ErrorAnalyzer strictly prioritizes FALSE_NEGATIVE)
        top_cases = ErrorAnalyzer.get_top_failure_cases(report, limit=1)
        if not top_cases:
            return "No valid failure cases found; few_shot refinement skipped."
            
        case = top_cases[0]
        
        # 2. Build RefinementRequest to generate a concise reasoning string
        active_signals = context.signals.active_names()
        
        # Constraint to avoid uncontrolled growth
        framing_constraint = (
            f"Generate a concise reason why this {case.category.name} should be "
            f"labeled as {case.true_label}. Keep examples concise and avoid uncontrolled prompt growth."
        )
        
        summary_text = f"{framing_constraint} Active issues: {', '.join(active_signals)}" if active_signals else framing_constraint

        # We allocate a small subset of the few_shot budget specifically for generating the reason string
        # To avoid overflow, we enforce a strict 30 token bound per reasoning step just for the request.
        reason_budget = min(30, candidate.few_shot.token_budget)

        request = RefinementRequest(
            target_component=RefinementTarget.FEWSHOT,
            current_content="", # We want a fresh reason
            token_budget=reason_budget,
            failure_summary=summary_text,
            optimization_signals=active_signals,
            contextual_examples=[{"email": str(case.email), "true_label": case.true_label}]
        )

        # 3. Call refinement generator
        response = self.generator.generate_refinement(request)

        # 4. Build concise structured FewShotExample object
        new_example = FewShotExample(
            email=case.email,
            label=case.true_label,
            reason=response.refined_content
        )
        
        fs = candidate.few_shot
        
        # 5. Enforce bounded memory capacity: replace lowest-relevance (oldest/index 0) when full
        action_taken = "Added"
        if len(fs.examples) >= fs.max_examples:
            if fs.max_examples > 0:
                fs.examples.pop(0)
                action_taken = "Replaced oldest and added"
            else:
                return "Capacity is 0; cannot add few-shot examples."
                
        # Mutate ONLY candidate.few_shot
        fs.examples.append(new_example)
        
        # 6. Increment revision metadata
        fs.revision_count += 1
        
        if hasattr(candidate.metadata, "mutation_history"):
            candidate.metadata.mutation_history.append("RefineFewShotOperator applied.")

        # 7. Return mutation summary
        return (
            f"{action_taken} {case.true_label} example "
            f"(Tokens: {response.token_count}/{reason_budget}, Truncated: {response.truncated})"
        )