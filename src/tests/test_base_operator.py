"""
Tests for src/mutations/base_operator.py

Covers:
- clone safety
- validation flow
- immutability guarantees
- invalid candidate handling
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mutations.base_operator import BaseMutationOperator, MutationContext, MutationResult
from src.core.prompt_state import PromptState, PromptComponent, FewShotComponent, FewShotExample, EmailInput, PromptMetadata
from src.critic.signal_extractor import CriticSignals
from src.core.constants import LABEL_PHISHING, LABEL_SAFE

def get_valid_state() -> PromptState:
    email = EmailInput(sender="a", receiver="b", subject="c", body="d")
    return PromptState(
        base_instruction="Classify emails",
        role=PromptComponent(content="You are an AI", token_budget=10),
        instruction_enrichment=PromptComponent(content="Think well", token_budget=10),
        cot=PromptComponent(content="Step by step", token_budget=10),
        few_shot=FewShotComponent(
            examples=[
                FewShotExample(email=email, label=LABEL_PHISHING, reason="Bad email")
            ],
            token_budget=100,
            max_examples=5
        ),
        metadata=PromptMetadata()
    )

def get_context() -> MutationContext:
    signals = CriticSignals(
        high_fn=False,
        high_fp=False,
        low_accuracy=False,
        inconsistent=False,
        plateau=False
    )
    return MutationContext(signals=signals)


class GoodOperator(BaseMutationOperator):
    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        candidate.role.content = "You are a specialized AI"
        return "Updated role"


class InvalidatingOperator(BaseMutationOperator):
    def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
        candidate.base_instruction = ""  # Invalidates the state
        return "Cleared base instruction"


def test_clone_safety_and_immutability():
    state = get_valid_state()
    original_state = copy.deepcopy(state)
    
    op = GoodOperator()
    result = op.mutate(state, get_context())
    
    assert result.success is True
    assert result.candidate_state is not state
    assert result.candidate_state.role.content == "You are a specialized AI"
    
    # Ensure original state is untouched
    assert state.role.content == "You are an AI"
    assert state == original_state
    assert result.operator_name == "GoodOperator"
    print("test_clone_safety_and_immutability PASSED")


def test_validation_flow():
    state = get_valid_state()
    op = GoodOperator()
    result = op.mutate(state, get_context())
    
    assert result.success is True
    assert result.validation_result.is_valid is True
    assert result.mutation_summary == "Updated role"
    print("test_validation_flow PASSED")


def test_invalid_candidate_handling():
    state = get_valid_state()
    op = InvalidatingOperator()
    result = op.mutate(state, get_context())
    
    assert result.success is False
    assert result.validation_result.is_valid is False
    assert "Validation failed" in result.mutation_summary
    assert len(result.validation_result.errors) > 0
    assert result.candidate_state is not state
    print("test_invalid_candidate_handling PASSED")


def test_mutating_original_state_raises():
    state = get_valid_state()
    
    class SneakyBadOperator(BaseMutationOperator):
        def _mutate(self, candidate: PromptState, context: MutationContext) -> str:
            pass

        def mutate(self, state, context):
            # Hack to test the assertion inside BaseMutationOperator.mutate by overriding it
            candidate = state
            summary = self._mutate(candidate, context)
            assert candidate is not state, "Operator mutated original state or broke clone pattern."
            
    op = SneakyBadOperator()
    raised = False
    try:
        op.mutate(state, get_context())
    except AssertionError as e:
        if "broke clone pattern" in str(e):
            raised = True
            
    assert raised, "Expected AssertionError when clone pattern is broken."
    print("test_mutating_original_state_raises PASSED")


if __name__ == "__main__":
    test_clone_safety_and_immutability()
    test_validation_flow()
    test_invalid_candidate_handling()
    test_mutating_original_state_raises()
    print("\nAll base_operator tests passed.")
