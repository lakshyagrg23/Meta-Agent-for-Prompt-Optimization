"""
Tests for src/llm/refinement_generation.py

Covers:
- valid responses
- budget overflow and truncation
- empty responses
- negative budget bounds
- deterministic placeholder generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.refinement_generation import (
    RefinementTarget,
    RefinementRequest,
    DeterministicPlaceholderGenerator
)

def _make_request(content: str, budget: int) -> RefinementRequest:
    return RefinementRequest(
        target_component=RefinementTarget.ROLE,
        current_content=content,
        token_budget=budget,
        failure_summary="Bad accuracy",
        optimization_signals=["low_accuracy"],
    )

def test_valid_responses_within_budget():
    req = _make_request("standard", budget=20)
    gen = DeterministicPlaceholderGenerator()
    resp = gen.generate_refinement(req)
    
    assert resp.within_budget is True
    assert resp.truncated is False
    assert resp.refined_content == "Refined role fixing 1 signals."
    assert resp.token_count == 5
    print("test_valid_responses_within_budget PASSED")

def test_budget_overflow_and_truncation():
    # OVERFLOW_TEST triggers the placeholder to generate many words
    req = _make_request("OVERFLOW_TEST", budget=3)
    gen = DeterministicPlaceholderGenerator()
    resp = gen.generate_refinement(req)
    
    assert resp.within_budget is False
    assert resp.truncated is True
    assert resp.token_count == 3
    # The output should be exactly 3 words of "overflowing"
    assert resp.refined_content == "overflowing overflowing overflowing"
    print("test_budget_overflow_and_truncation PASSED")

def test_truncation_to_zero_budget():
    req = _make_request("OVERFLOW_TEST", budget=0)
    gen = DeterministicPlaceholderGenerator()
    resp = gen.generate_refinement(req)
    
    assert resp.within_budget is False
    assert resp.truncated is True
    assert resp.token_count == 0
    assert resp.refined_content == ""
    print("test_truncation_to_zero_budget PASSED")

def test_empty_responses():
    req = _make_request("", budget=10)
    gen = DeterministicPlaceholderGenerator()
    resp = gen.generate_refinement(req)
    
    assert resp.within_budget is True
    assert resp.truncated is False
    assert resp.token_count == 0
    assert resp.refined_content == ""
    print("test_empty_responses PASSED")

def test_negative_budget_raises():
    req = _make_request("standard", budget=-5)
    gen = DeterministicPlaceholderGenerator()
    
    raised = False
    try:
        gen.generate_refinement(req)
    except ValueError as e:
        if "cannot be negative" in str(e):
            raised = True
            
    assert raised, "Expected ValueError on negative budget"
    print("test_negative_budget_raises PASSED")
    
def test_deterministic_behavior():
    req = _make_request("standard", budget=10)
    gen = DeterministicPlaceholderGenerator()
    resp1 = gen.generate_refinement(req)
    resp2 = gen.generate_refinement(req)
    
    assert resp1.refined_content == resp2.refined_content
    assert resp1.token_count == resp2.token_count
    assert resp1.within_budget == resp2.within_budget
    assert resp1.truncated == resp2.truncated
    print("test_deterministic_behavior PASSED")

if __name__ == "__main__":
    test_valid_responses_within_budget()
    test_budget_overflow_and_truncation()
    test_truncation_to_zero_budget()
    test_empty_responses()
    test_negative_budget_raises()
    test_deterministic_behavior()
    print("\nAll refinement_generation tests passed.")
