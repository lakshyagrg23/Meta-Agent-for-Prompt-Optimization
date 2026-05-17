"""
src/optimization/candidate_generator.py
-----------------------------------------
Translates a selected refinement operator into a candidate PromptState.

Responsibility
--------------
This module has exactly one job: given the current PromptState and the
operator chosen by the refinement policy, produce a candidate PromptState
by invoking the operator and returning the result.

It does NOT:
* compute metrics
* extract signals
* decide which operator to use
* evaluate the candidate
* accept or reject the candidate

All of those concerns belong to other modules.

Clone-first guarantee
---------------------
``CandidateGenerator.generate`` delegates to ``operator.refine``, whose
base class (``MutationOperator``) asserts at runtime that the returned
object is a distinct instance from the input state.  This module therefore
inherits the clone-first safety guarantee without needing to enforce it
itself.  The assertion in ``MutationOperator.refine`` will raise immediately
if any operator attempts in-place mutation.
"""

from __future__ import annotations

from src.core.prompt_state import PromptState
from src.critic.signal_extractor import CriticSignals
from src.mutations.base_operator import BaseMutationOperator, MutationContext, MutationResult


class CandidateGenerator:
    """
    Thin adapter that invokes a refinement operator and returns the candidate result.
    """

    @staticmethod
    def generate(
        state: PromptState,
        operator: BaseMutationOperator,
        signals: CriticSignals,
    ) -> MutationResult:
        """
        Produce a candidate PromptState via *operator*.

        Args:
            state:    The current incumbent PromptState.
            operator: The refinement operator chosen by the policy layer.
            signals:  Active optimization signals forwarded to the operator.

        Returns:
            A MutationResult containing the candidate state and validation info.
        """
        context = MutationContext(signals=signals)
        return operator.mutate(state, context)
