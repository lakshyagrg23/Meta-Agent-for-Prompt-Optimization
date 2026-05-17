"""
Tests for src/mutations/refine_role.py

Covers:
- successful role mutation
- original state immutability
- token budget preservation
- isolated component mutation
- deterministic placeholder compatibility
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mutations.refine_role import RefineRoleOperator
from src.mutations.base_operator import MutationContext
from src.core.prompt_state import PromptState, PromptComponent, FewShotComponent, FewShotExample, EmailInput, PromptMetadata
from src.critic.signal_extractor import CriticSignals
from src.core.constants import LABEL_PHISHING, LABEL_SAFE

def get_valid_state() -> PromptState:
    email = EmailInput(sender="a", receiver="b", subject="c", body="d")
    return PromptState(
        base_instruction="Classify emails",
        role=PromptComponent(content="You are an AI", token_budget=10, revision_count=0),
        instruction_enrichment=PromptComponent(content="Think well", token_budget=10, revision_count=0),
        cot=PromptComponent(content="Step by step", token_budget=10, revision_count=0),
        few_shot=FewShotComponent(
            examples=[
                FewShotExample(email=email, label=LABEL_PHISHING, reason="Bad email")
            ],
            token_budget=100,
            max_examples=5,
            revision_count=0
        ),
        metadata=PromptMetadata()
    )

def get_context() -> MutationContext:
    signals = CriticSignals(
        high_fn=True,  # Simulate one active signal
        high_fp=False,
        low_accuracy=False,
        inconsistent=False,
        plateau=False
    )
    return MutationContext(signals=signals)

def test_successful_role_mutation():
    state = get_valid_state()
    op = RefineRoleOperator()
    
    result = op.mutate(state, get_context())
    
    assert result.success is True
    # The deterministic placeholder outputs: "Refined role fixing 1 signals."
    assert "Refined role" in result.candidate_state.role.content
    assert result.candidate_state.role.revision_count == 1
    assert "RefineRoleOperator" in result.operator_name
    print("test_successful_role_mutation PASSED")

def test_isolated_component_mutation():
    state = get_valid_state()
    op = RefineRoleOperator()
    
    result = op.mutate(state, get_context())
    
    c = result.candidate_state
    assert c.role.content != state.role.content
    assert c.base_instruction == state.base_instruction
    assert c.instruction_enrichment.content == state.instruction_enrichment.content
    assert c.cot.content == state.cot.content
    assert len(c.few_shot.examples) == len(state.few_shot.examples)
    print("test_isolated_component_mutation PASSED")

def test_token_budget_preservation():
    state = get_valid_state()
    budget_before = state.role.token_budget
    
    op = RefineRoleOperator()
    result = op.mutate(state, get_context())
    
    assert result.candidate_state.role.token_budget == budget_before
    print("test_token_budget_preservation PASSED")

def test_original_state_immutability():
    state = get_valid_state()
    original_state = copy.deepcopy(state)
    
    op = RefineRoleOperator()
    result = op.mutate(state, get_context())
    
    # Check that original state has not been modified
    assert state.role.content == original_state.role.content
    assert state.role.revision_count == original_state.role.revision_count
    assert result.candidate_state is not state
    print("test_original_state_immutability PASSED")

def test_deterministic_placeholder_compatibility():
    state = get_valid_state()
    op = RefineRoleOperator()
    
    # Run twice on identical inputs
    result1 = op.mutate(state, get_context())
    result2 = op.mutate(state, get_context())
    
    assert result1.candidate_state.role.content == result2.candidate_state.role.content
    assert result1.success == result2.success
    assert result1.mutation_summary == result2.mutation_summary
    print("test_deterministic_placeholder_compatibility PASSED")


if __name__ == "__main__":
    test_successful_role_mutation()
    test_isolated_component_mutation()
    test_token_budget_preservation()
    test_original_state_immutability()
    test_deterministic_placeholder_compatibility()
    print("\nAll refine_role tests passed.")
