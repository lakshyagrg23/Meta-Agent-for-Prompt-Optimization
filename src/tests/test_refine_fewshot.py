"""
Tests for src/mutations/refine_fewshot.py

Covers:
- successful example insertion
- bounded capacity enforcement (replacement behavior)
- FN prioritization
- isolated component mutation
- original state immutability
- deterministic placeholder compatibility
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mutations.refine_fewshot import RefineFewShotOperator
from src.mutations.base_operator import MutationContext
from src.core.prompt_state import PromptState, PromptComponent, FewShotComponent, FewShotExample, EmailInput, PromptMetadata
from src.critic.signal_extractor import CriticSignals
from src.core.constants import LABEL_PHISHING, LABEL_SAFE
from src.critic.error_analysis import FailureAnalysisReport, FailureCase, FailureCategory

def get_valid_state(max_ex: int = 5) -> PromptState:
    email = EmailInput(sender="a", receiver="b", subject="c", body="d")
    return PromptState(
        base_instruction="Classify emails",
        role=PromptComponent(content="You are an AI", token_budget=10, revision_count=0),
        instruction_enrichment=PromptComponent(content="Focus on domains", token_budget=15, revision_count=0),
        cot=PromptComponent(content="Think step by step", token_budget=10, revision_count=0),
        few_shot=FewShotComponent(
            examples=[
                FewShotExample(email=email, label=LABEL_SAFE, reason="Good email")
            ],
            token_budget=100,
            max_examples=max_ex,
            revision_count=0
        ),
        metadata=PromptMetadata()
    )

def get_report() -> FailureAnalysisReport:
    cases = [
        FailureCase("fp email", LABEL_SAFE, LABEL_PHISHING, FailureCategory.FALSE_POSITIVE, []),
        FailureCase("fn email", LABEL_PHISHING, LABEL_SAFE, FailureCategory.FALSE_NEGATIVE, ["urgency"]),
    ]
    return FailureAnalysisReport(
        total_failures=2,
        false_positives=1,
        false_negatives=1,
        dominant_category=None,
        cases=cases,
        heuristics_summary={}
    )

def get_context(report=None) -> MutationContext:
    signals = CriticSignals(
        high_fn=True,
        high_fp=False,
        low_accuracy=False,
        inconsistent=False,
        plateau=False
    )
    if report is None:
        report = get_report()
    return MutationContext(signals=signals, extra={"failure_report": report})

def test_successful_example_insertion():
    state = get_valid_state()
    op = RefineFewShotOperator()
    
    result = op.mutate(state, get_context())
    
    assert result.success is True
    # Should append a new example
    assert len(result.candidate_state.few_shot.examples) == 2
    assert result.candidate_state.few_shot.revision_count == 1
    assert "RefineFewShotOperator" in result.operator_name
    print("test_successful_example_insertion PASSED")

def test_fn_prioritization():
    # Our report has one FP and one FN. The FN has an 'urgency' heuristic so it will be top.
    state = get_valid_state()
    op = RefineFewShotOperator()
    result = op.mutate(state, get_context())
    
    fs = result.candidate_state.few_shot
    new_ex = fs.examples[-1]
    
    # It must have picked the FN email
    assert new_ex.email == "fn email"
    assert new_ex.label == LABEL_PHISHING
    assert "Added" in result.mutation_summary
    print("test_fn_prioritization PASSED")

def test_bounded_capacity_enforcement():
    # Set max_examples to 1. The existing example should be dropped.
    state = get_valid_state(max_ex=1)
    op = RefineFewShotOperator()
    result = op.mutate(state, get_context())
    
    fs = result.candidate_state.few_shot
    assert len(fs.examples) == 1
    new_ex = fs.examples[0]
    
    assert new_ex.email == "fn email"  # The new one replaced the old one
    assert "Replaced oldest" in result.mutation_summary
    print("test_bounded_capacity_enforcement PASSED")

def test_isolated_component_mutation():
    state = get_valid_state()
    op = RefineFewShotOperator()
    
    result = op.mutate(state, get_context())
    
    c = result.candidate_state
    assert len(c.few_shot.examples) != len(state.few_shot.examples)
    assert c.cot.content == state.cot.content
    assert c.base_instruction == state.base_instruction
    assert c.role.content == state.role.content
    assert c.instruction_enrichment.content == state.instruction_enrichment.content
    print("test_isolated_component_mutation PASSED")

def test_original_state_immutability():
    state = get_valid_state()
    original_state = copy.deepcopy(state)
    
    op = RefineFewShotOperator()
    result = op.mutate(state, get_context())
    
    # Check that original state has not been modified
    assert len(state.few_shot.examples) == len(original_state.few_shot.examples)
    assert state.few_shot.revision_count == original_state.few_shot.revision_count
    assert result.candidate_state is not state
    print("test_original_state_immutability PASSED")

def test_deterministic_placeholder_compatibility():
    state = get_valid_state()
    op = RefineFewShotOperator()
    
    result1 = op.mutate(state, get_context())
    result2 = op.mutate(state, get_context())
    
    assert len(result1.candidate_state.few_shot.examples) == len(result2.candidate_state.few_shot.examples)
    assert result1.success == result2.success
    assert result1.mutation_summary == result2.mutation_summary
    assert result1.candidate_state.few_shot.examples[-1].reason == result2.candidate_state.few_shot.examples[-1].reason
    print("test_deterministic_placeholder_compatibility PASSED")

def test_missing_report_handling():
    state = get_valid_state()
    op = RefineFewShotOperator()
    
    # Create context without report
    ctx = MutationContext(signals=CriticSignals(False, False, False, False, False))
    result = op.mutate(state, ctx)
    
    assert "skipped" in result.mutation_summary
    assert len(result.candidate_state.few_shot.examples) == 1  # No change
    print("test_missing_report_handling PASSED")


if __name__ == "__main__":
    test_successful_example_insertion()
    test_fn_prioritization()
    test_bounded_capacity_enforcement()
    test_isolated_component_mutation()
    test_original_state_immutability()
    test_deterministic_placeholder_compatibility()
    test_missing_report_handling()
    print("\nAll refine_fewshot tests passed.")
