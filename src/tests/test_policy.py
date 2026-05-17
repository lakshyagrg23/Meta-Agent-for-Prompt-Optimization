"""
Tests for src/critic/policy.py

Covers:
    - Every signal → operator mapping (one signal active in isolation)
    - Priority conflicts (multiple signals, highest wins)
        - high_fn beats all others
        - high_fp beats low_accuracy, inconsistent, plateau
        - low_accuracy beats inconsistent and plateau
        - inconsistent beats plateau
    - NO_OP when no signals are active
    - Deterministic repeatability (same inputs → same output, always)
    - RefinementDecision fields are correctly populated
    - RefinementDecision.is_no_op() and summary()
    - Frozen dataclass (immutability)
    - MutationPolicy alias resolves to RefinementPolicy
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.critic.policy import (
    RefinementDecision,
    RefinementOperator,
    RefinementPolicy,
    MutationPolicy,
)
from src.critic.signal_extractor import CriticSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signals(
    high_fn: bool = False,
    high_fp: bool = False,
    low_accuracy: bool = False,
    inconsistent: bool = False,
    plateau: bool = False,
) -> CriticSignals:
    return CriticSignals(
        high_fn=high_fn,
        high_fp=high_fp,
        low_accuracy=low_accuracy,
        inconsistent=inconsistent,
        plateau=plateau,
    )


def _select(**kwargs) -> RefinementDecision:
    return RefinementPolicy.select_operator(_signals(**kwargs))


# ---------------------------------------------------------------------------
# Single-signal mappings (each signal in isolation)
# ---------------------------------------------------------------------------

def test_high_fn_maps_to_refine_fewshot():
    d = _select(high_fn=True)
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    assert d.target_component == "few_shot"
    assert d.priority == 5
    print("test_high_fn_maps_to_refine_fewshot PASSED")


def test_high_fp_maps_to_refine_enrichment():
    d = _select(high_fp=True)
    assert d.operator is RefinementOperator.REFINE_ENRICHMENT
    assert d.target_component == "instruction_enrichment"
    assert d.priority == 4
    print("test_high_fp_maps_to_refine_enrichment PASSED")


def test_low_accuracy_maps_to_refine_role():
    d = _select(low_accuracy=True)
    assert d.operator is RefinementOperator.REFINE_ROLE
    assert d.target_component == "role"
    assert d.priority == 3
    print("test_low_accuracy_maps_to_refine_role PASSED")


def test_inconsistent_maps_to_refine_cot():
    d = _select(inconsistent=True)
    assert d.operator is RefinementOperator.REFINE_COT
    assert d.target_component == "cot"
    assert d.priority == 2
    print("test_inconsistent_maps_to_refine_cot PASSED")


def test_plateau_maps_to_refine_cot():
    d = _select(plateau=True)
    assert d.operator is RefinementOperator.REFINE_COT
    assert d.target_component == "cot"
    assert d.priority == 1
    print("test_plateau_maps_to_refine_cot PASSED")


def test_no_signals_returns_no_op():
    d = _select()  # all False
    assert d.operator is RefinementOperator.NO_OP
    assert d.target_component == ""
    assert d.priority == 0
    assert d.is_no_op() is True
    print("test_no_signals_returns_no_op PASSED")


# ---------------------------------------------------------------------------
# Priority conflicts — high_fn beats everything
# ---------------------------------------------------------------------------

def test_priority_high_fn_beats_high_fp():
    d = _select(high_fn=True, high_fp=True)
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    assert d.priority == 5
    print("test_priority_high_fn_beats_high_fp PASSED")


def test_priority_high_fn_beats_low_accuracy():
    d = _select(high_fn=True, low_accuracy=True)
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    print("test_priority_high_fn_beats_low_accuracy PASSED")


def test_priority_high_fn_beats_inconsistent():
    d = _select(high_fn=True, inconsistent=True)
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    print("test_priority_high_fn_beats_inconsistent PASSED")


def test_priority_high_fn_beats_plateau():
    d = _select(high_fn=True, plateau=True)
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    print("test_priority_high_fn_beats_plateau PASSED")


def test_priority_high_fn_beats_all():
    """All signals active — high_fn (priority 5) must win."""
    d = _select(
        high_fn=True, high_fp=True,
        low_accuracy=True, inconsistent=True, plateau=True,
    )
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    assert d.priority == 5
    print("test_priority_high_fn_beats_all PASSED")


# ---------------------------------------------------------------------------
# Priority conflicts — high_fp beats lower-priority signals
# ---------------------------------------------------------------------------

def test_priority_high_fp_beats_low_accuracy():
    d = _select(high_fp=True, low_accuracy=True)
    assert d.operator is RefinementOperator.REFINE_ENRICHMENT
    assert d.priority == 4
    print("test_priority_high_fp_beats_low_accuracy PASSED")


def test_priority_high_fp_beats_inconsistent():
    d = _select(high_fp=True, inconsistent=True)
    assert d.operator is RefinementOperator.REFINE_ENRICHMENT
    print("test_priority_high_fp_beats_inconsistent PASSED")


def test_priority_high_fp_beats_plateau():
    d = _select(high_fp=True, plateau=True)
    assert d.operator is RefinementOperator.REFINE_ENRICHMENT
    print("test_priority_high_fp_beats_plateau PASSED")


def test_priority_high_fp_beaten_by_high_fn():
    """high_fp is NOT highest; high_fn wins when both active."""
    d = _select(high_fn=True, high_fp=True)
    assert d.operator is not RefinementOperator.REFINE_ENRICHMENT
    assert d.operator is RefinementOperator.REFINE_FEWSHOT
    print("test_priority_high_fp_beaten_by_high_fn PASSED")


# ---------------------------------------------------------------------------
# Priority conflicts — low_accuracy beats inconsistent and plateau
# ---------------------------------------------------------------------------

def test_priority_low_accuracy_beats_inconsistent():
    d = _select(low_accuracy=True, inconsistent=True)
    assert d.operator is RefinementOperator.REFINE_ROLE
    assert d.priority == 3
    print("test_priority_low_accuracy_beats_inconsistent PASSED")


def test_priority_low_accuracy_beats_plateau():
    d = _select(low_accuracy=True, plateau=True)
    assert d.operator is RefinementOperator.REFINE_ROLE
    print("test_priority_low_accuracy_beats_plateau PASSED")


def test_priority_low_accuracy_beats_both_low_priority():
    d = _select(low_accuracy=True, inconsistent=True, plateau=True)
    assert d.operator is RefinementOperator.REFINE_ROLE
    print("test_priority_low_accuracy_beats_both_low_priority PASSED")


# ---------------------------------------------------------------------------
# Priority conflicts — inconsistent beats plateau
# ---------------------------------------------------------------------------

def test_priority_inconsistent_beats_plateau():
    d = _select(inconsistent=True, plateau=True)
    assert d.operator is RefinementOperator.REFINE_COT
    assert d.priority == 2   # inconsistent has priority 2, plateau has 1
    print("test_priority_inconsistent_beats_plateau PASSED")


def test_plateau_only_returns_priority_1():
    """When plateau is the sole active signal, priority should be 1."""
    d = _select(plateau=True)
    assert d.priority == 1
    print("test_plateau_only_returns_priority_1 PASSED")


# ---------------------------------------------------------------------------
# Deterministic repeatability
# ---------------------------------------------------------------------------

def test_same_inputs_always_same_output():
    """Calling select_operator twice with the same signals must be identical."""
    s = _signals(high_fn=True, plateau=True)
    d1 = RefinementPolicy.select_operator(s)
    d2 = RefinementPolicy.select_operator(s)
    assert d1 == d2
    assert d1.operator is d2.operator
    assert d1.priority == d2.priority
    assert d1.rationale == d2.rationale
    print("test_same_inputs_always_same_output PASSED")


def test_no_op_same_instance_returned():
    """NO_OP is a module-level singleton — identical calls return equal objects."""
    d1 = _select()
    d2 = _select()
    assert d1 == d2
    assert d1.is_no_op() and d2.is_no_op()
    print("test_no_op_same_instance_returned PASSED")


def test_all_false_to_all_true_is_deterministic():
    """Exhaustive: flip every signal on one at a time, check stability."""
    combos = [
        dict(high_fn=True),
        dict(high_fp=True),
        dict(low_accuracy=True),
        dict(inconsistent=True),
        dict(plateau=True),
    ]
    for kwargs in combos:
        d1 = _select(**kwargs)
        d2 = _select(**kwargs)
        assert d1 == d2, f"Non-deterministic for {kwargs}"
    print("test_all_false_to_all_true_is_deterministic PASSED")


# ---------------------------------------------------------------------------
# RefinementDecision field correctness
# ---------------------------------------------------------------------------

def test_rationale_is_non_empty_for_every_operator():
    """Every active-signal path must produce a non-empty rationale."""
    cases = [
        dict(high_fn=True),
        dict(high_fp=True),
        dict(low_accuracy=True),
        dict(inconsistent=True),
        dict(plateau=True),
    ]
    for kwargs in cases:
        d = _select(**kwargs)
        assert d.rationale, f"Empty rationale for {kwargs}"
    print("test_rationale_is_non_empty_for_every_operator PASSED")


def test_no_op_rationale_is_non_empty():
    d = _select()
    assert d.rationale
    print("test_no_op_rationale_is_non_empty PASSED")


def test_target_component_empty_for_no_op():
    d = _select()
    assert d.target_component == ""
    print("test_target_component_empty_for_no_op PASSED")


def test_target_component_non_empty_for_all_operators():
    """Every operator except NO_OP must name a real target component."""
    cases = [
        dict(high_fn=True),
        dict(high_fp=True),
        dict(low_accuracy=True),
        dict(inconsistent=True),
        dict(plateau=True),
    ]
    for kwargs in cases:
        d = _select(**kwargs)
        assert d.target_component, f"Empty target_component for {kwargs}"
    print("test_target_component_non_empty_for_all_operators PASSED")


# ---------------------------------------------------------------------------
# RefinementDecision helpers
# ---------------------------------------------------------------------------

def test_is_no_op_true_for_no_op():
    d = _select()
    assert d.is_no_op() is True
    print("test_is_no_op_true_for_no_op PASSED")


def test_is_no_op_false_for_active_signal():
    d = _select(high_fn=True)
    assert d.is_no_op() is False
    print("test_is_no_op_false_for_active_signal PASSED")


def test_summary_contains_operator_name():
    d = _select(high_fn=True)
    assert "REFINE_FEWSHOT" in d.summary()
    print("test_summary_contains_operator_name PASSED")


def test_summary_contains_priority():
    d = _select(high_fp=True)
    assert "priority=4" in d.summary()
    print("test_summary_contains_priority PASSED")


def test_summary_contains_target_component():
    d = _select(low_accuracy=True)
    assert "role" in d.summary()
    print("test_summary_contains_target_component PASSED")


def test_decision_is_frozen():
    """RefinementDecision is frozen — mutation must raise AttributeError."""
    d = _select(high_fn=True)
    raised = False
    try:
        d.operator = RefinementOperator.NO_OP  # type: ignore[misc]
    except (AttributeError, TypeError):
        raised = True
    assert raised, "Expected frozen dataclass to reject attribute assignment"
    print("test_decision_is_frozen PASSED")


# ---------------------------------------------------------------------------
# MutationPolicy alias
# ---------------------------------------------------------------------------

def test_mutation_policy_alias_is_refinement_policy():
    """MutationPolicy must be the same object as RefinementPolicy."""
    assert MutationPolicy is RefinementPolicy
    print("test_mutation_policy_alias_is_refinement_policy PASSED")


def test_mutation_policy_alias_produces_same_decisions():
    s = _signals(high_fp=True)
    d1 = RefinementPolicy.select_operator(s)
    d2 = MutationPolicy.select_operator(s)
    assert d1 == d2
    print("test_mutation_policy_alias_produces_same_decisions PASSED")


# ---------------------------------------------------------------------------
# RefinementOperator enum
# ---------------------------------------------------------------------------

def test_all_five_operators_exist():
    names = {op.name for op in RefinementOperator}
    expected = {"REFINE_ROLE", "REFINE_ENRICHMENT", "REFINE_COT", "REFINE_FEWSHOT", "NO_OP"}
    assert names == expected, f"Missing operators: {expected - names}"
    print("test_all_five_operators_exist PASSED")


def test_operator_values_are_stable_strings():
    assert RefinementOperator.REFINE_ROLE.value == "refine_role"
    assert RefinementOperator.REFINE_ENRICHMENT.value == "refine_enrichment"
    assert RefinementOperator.REFINE_COT.value == "refine_cot"
    assert RefinementOperator.REFINE_FEWSHOT.value == "refine_fewshot"
    assert RefinementOperator.NO_OP.value == "no_op"
    print("test_operator_values_are_stable_strings PASSED")


# ---------------------------------------------------------------------------
# Priority ordering integrity check
# ---------------------------------------------------------------------------

def test_priority_ordering_is_strict_5_to_0():
    """
    Activate each signal in isolation and verify priorities form the
    strict sequence 5 > 4 > 3 > 2 > 1 > 0.
    """
    priorities = [
        _select(high_fn=True).priority,       # 5
        _select(high_fp=True).priority,       # 4
        _select(low_accuracy=True).priority,  # 3
        _select(inconsistent=True).priority,  # 2
        _select(plateau=True).priority,       # 1
        _select().priority,                   # 0  (NO_OP)
    ]
    for i in range(len(priorities) - 1):
        assert priorities[i] > priorities[i + 1], (
            f"Priority ordering violated at index {i}: "
            f"{priorities[i]} not > {priorities[i+1]}"
        )
    print("test_priority_ordering_is_strict_5_to_0 PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # single-signal mappings
    test_high_fn_maps_to_refine_fewshot()
    test_high_fp_maps_to_refine_enrichment()
    test_low_accuracy_maps_to_refine_role()
    test_inconsistent_maps_to_refine_cot()
    test_plateau_maps_to_refine_cot()
    test_no_signals_returns_no_op()
    # priority — high_fn wins
    test_priority_high_fn_beats_high_fp()
    test_priority_high_fn_beats_low_accuracy()
    test_priority_high_fn_beats_inconsistent()
    test_priority_high_fn_beats_plateau()
    test_priority_high_fn_beats_all()
    # priority — high_fp
    test_priority_high_fp_beats_low_accuracy()
    test_priority_high_fp_beats_inconsistent()
    test_priority_high_fp_beats_plateau()
    test_priority_high_fp_beaten_by_high_fn()
    # priority — low_accuracy
    test_priority_low_accuracy_beats_inconsistent()
    test_priority_low_accuracy_beats_plateau()
    test_priority_low_accuracy_beats_both_low_priority()
    # priority — inconsistent vs plateau
    test_priority_inconsistent_beats_plateau()
    test_plateau_only_returns_priority_1()
    # determinism
    test_same_inputs_always_same_output()
    test_no_op_same_instance_returned()
    test_all_false_to_all_true_is_deterministic()
    # field correctness
    test_rationale_is_non_empty_for_every_operator()
    test_no_op_rationale_is_non_empty()
    test_target_component_empty_for_no_op()
    test_target_component_non_empty_for_all_operators()
    # helpers
    test_is_no_op_true_for_no_op()
    test_is_no_op_false_for_active_signal()
    test_summary_contains_operator_name()
    test_summary_contains_priority()
    test_summary_contains_target_component()
    test_decision_is_frozen()
    # alias
    test_mutation_policy_alias_is_refinement_policy()
    test_mutation_policy_alias_produces_same_decisions()
    # enum
    test_all_five_operators_exist()
    test_operator_values_are_stable_strings()
    # ordering integrity
    test_priority_ordering_is_strict_5_to_0()
    print("\nAll policy tests passed.")
