"""
Tests for src/core/validator.py

Covers:
- Valid states
- Empty base instruction
- Token overflow in components and few_shot
- Valid component types
- Valid few-shot example labels
- Few-shot capacity constraints
- Renderability (EmailInput malformed)
- Non-mutating guarantees
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.prompt_state import PromptState, PromptComponent, FewShotComponent, FewShotExample, EmailInput, PromptMetadata
from src.core.validator import PromptValidator, ValidationResult, ValidationError
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

def test_valid_state():
    state = get_valid_state()
    res = PromptValidator.validate_state(state)
    if not res.is_valid:
        print("Valid state failed with errors:")
        for e in res.errors:
            print(f"- {e.component}: {e.message}")
    assert res.is_valid
    assert len(res.errors) == 0
    print("test_valid_state PASSED")

def test_empty_base_instruction():
    state = get_valid_state()
    state.base_instruction = "   "
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    assert any(e.component == "base_instruction" for e in res.errors)
    print("test_empty_base_instruction PASSED")

def test_token_overflow_component():
    state = get_valid_state()
    state.role.content = "One Two Three Four Five Six"
    state.role.token_budget = 5
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "role")
    assert "Token budget exceeded" in err.message
    print("test_token_overflow_component PASSED")

def test_invalid_component_type():
    state = get_valid_state()
    state.cot = "This should be a PromptComponent, not a string"
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "cot")
    assert "must be a PromptComponent" in err.message
    print("test_invalid_component_type PASSED")

def test_fewshot_capacity_exceeded():
    state = get_valid_state()
    state.few_shot.max_examples = 1
    # Adding a second example
    state.few_shot.examples.append(
        FewShotExample(email="test", label=LABEL_SAFE, reason="good")
    )
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "few_shot")
    assert "Capacity exceeded" in err.message
    print("test_fewshot_capacity_exceeded PASSED")

def test_invalid_fewshot_label():
    state = get_valid_state()
    state.few_shot.examples[0].label = "NOT_A_VALID_LABEL"
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "few_shot")
    assert "invalid label" in err.message
    print("test_invalid_fewshot_label PASSED")

def test_malformed_email_input():
    state = get_valid_state()
    state.few_shot.examples[0].email = EmailInput(sender="a", receiver="b", subject=None, body="d")
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "few_shot")
    assert "must have string subject and body" in err.message
    print("test_malformed_email_input PASSED")

def test_empty_email_string():
    state = get_valid_state()
    state.few_shot.examples[0].email = "   "
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "few_shot")
    assert "empty email string" in err.message
    print("test_empty_email_string PASSED")

def test_fewshot_token_overflow():
    state = get_valid_state()
    state.few_shot.token_budget = 2
    res = PromptValidator.validate_state(state)
    assert not res.is_valid
    err = next(e for e in res.errors if e.component == "few_shot")
    assert "Token budget exceeded" in err.message
    print("test_fewshot_token_overflow PASSED")

def test_never_mutate_state():
    state = get_valid_state()
    state_copy = copy.deepcopy(state)
    
    # Introduce some errors
    state.role.token_budget = 0
    state.role.content = "lots of words"
    
    PromptValidator.validate_state(state)
    
    # Check that state hasn't been auto-fixed or mutated
    assert state.role.token_budget == 0
    assert state.role.content == "lots of words"
    print("test_never_mutate_state PASSED")


if __name__ == "__main__":
    test_valid_state()
    test_empty_base_instruction()
    test_token_overflow_component()
    test_invalid_component_type()
    test_fewshot_capacity_exceeded()
    test_invalid_fewshot_label()
    test_malformed_email_input()
    test_empty_email_string()
    test_fewshot_token_overflow()
    test_never_mutate_state()
    print("\nAll validator tests passed.")
