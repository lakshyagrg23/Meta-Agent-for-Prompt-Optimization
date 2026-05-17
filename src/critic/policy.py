"""
src/critic/policy.py
----------------------
Deterministic refinement policy: maps CriticSignals to a RefinementDecision.

No randomness, no LLM calls, no mutable state.
All outputs are fully determined by the inputs.

Priority ordering (highest to lowest)
---------------------------------------
1. high_fn      → REFINE_FEWSHOT     (missed phishing — safety critical)
2. high_fp      → REFINE_ENRICHMENT  (over-flagging — degrades trust)
3. low_accuracy → REFINE_ROLE        (weak task framing)
4. inconsistent → REFINE_COT         (unstable outputs)
5. plateau      → REFINE_COT         (stagnation — introduce structured reasoning)
0. (none)       → NO_OP              (no action needed)

When multiple signals are active simultaneously, only the highest-priority
signal's operator is returned.  This prevents conflicting simultaneous
refinements and keeps the optimization trajectory deterministic.

Design notes
------------
* ``RefinementPolicy`` contains only static methods — it is stateless.
* ``RefinementDecision`` carries the full decision context (operator,
  target component name, human-readable rationale, numeric priority) so
  that the optimization loop can log a complete audit trail without
  reaching back into the policy.
* ``MutationPolicy`` is kept as a backwards-compatible alias for any code
  that was written against the original stub name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.critic.signal_extractor import CriticSignals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RefinementOperator enum
# ---------------------------------------------------------------------------

class RefinementOperator(Enum):
    """
    Identifies which prompt component a refinement pass should target.

    Values are stable string tags used in logging and mutation history.
    """

    REFINE_ROLE = "refine_role"
    """Refine the role specification component."""

    REFINE_ENRICHMENT = "refine_enrichment"
    """Refine the instruction enrichment component."""

    REFINE_COT = "refine_cot"
    """Refine (or introduce) the chain-of-thought component."""

    REFINE_FEWSHOT = "refine_fewshot"
    """Refine the adaptive few-shot memory component."""

    NO_OP = "no_op"
    """No refinement needed; current prompt state is retained."""


# ---------------------------------------------------------------------------
# RefinementDecision dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementDecision:
    """
    Complete, self-describing output of the refinement policy.

    Carrying the full decision context here means the optimization loop
    never needs to re-query the policy or interpret raw signal bits — it
    can log, record, and act on a single structured object.

    Attributes:
        operator:         Which refinement operator to invoke.
        target_component: Name of the PromptState attribute to refine,
                          matching the field names on :class:`PromptState`
                          (e.g. ``"few_shot"``, ``"instruction_enrichment"``).
                          ``""`` for NO_OP.
        rationale:        Human-readable explanation of why this operator
                          was chosen.  Used for audit logging and mutation
                          history tracking.
        priority:         Numeric priority of the driving signal (higher
                          number = higher priority).  0 for NO_OP.
    """

    operator: RefinementOperator
    target_component: str
    rationale: str
    priority: int

    def is_no_op(self) -> bool:
        """Return True if no refinement will be performed."""
        return self.operator is RefinementOperator.NO_OP

    def summary(self) -> str:
        """
        Return a compact one-line summary for logging.

        Example::

            "op=REFINE_FEWSHOT target=few_shot priority=5 | Missed phishing detected (high_fn)"
        """
        return (
            f"op={self.operator.name} "
            f"target={self.target_component!r} "
            f"priority={self.priority} | "
            f"{self.rationale}"
        )


# ---------------------------------------------------------------------------
# Priority table — single source of truth for the mapping
# ---------------------------------------------------------------------------
#
# Each entry: (signal_attribute, priority, operator, target_component, rationale)
# Ordered highest-priority first.  select_operator() iterates this list and
# returns on the first active signal — the ordering IS the policy.

_PRIORITY_TABLE: tuple = (
    (
        "high_fn",
        5,
        RefinementOperator.REFINE_FEWSHOT,
        "few_shot",
        "Missed phishing emails detected (high_fn): "
        "adaptive few-shot refinement to expose the model to failure patterns.",
    ),
    (
        "high_fp",
        4,
        RefinementOperator.REFINE_ENRICHMENT,
        "instruction_enrichment",
        "Excessive false positives detected (high_fp): "
        "refine instruction enrichment to sharpen phishing decision boundaries.",
    ),
    (
        "low_accuracy",
        3,
        RefinementOperator.REFINE_ROLE,
        "role",
        "Low overall accuracy (low_accuracy): "
        "refine role specification to improve domain-oriented task framing.",
    ),
    (
        "inconsistent",
        2,
        RefinementOperator.REFINE_COT,
        "cot",
        "Inconsistent outputs detected (inconsistent): "
        "introduce or refine chain-of-thought to stabilise predictions.",
    ),
    (
        "plateau",
        1,
        RefinementOperator.REFINE_COT,
        "cot",
        "Optimization plateau detected (plateau): "
        "refine chain-of-thought to escape stagnation with structured reasoning.",
    ),
)

_NO_OP_DECISION = RefinementDecision(
    operator=RefinementOperator.NO_OP,
    target_component="",
    rationale="No active optimization signals — current prompt state is retained.",
    priority=0,
)


# ---------------------------------------------------------------------------
# RefinementPolicy
# ---------------------------------------------------------------------------

class RefinementPolicy:
    """
    Deterministic policy that maps :class:`CriticSignals` to a
    :class:`RefinementDecision`.

    All methods are static.  No mutable state exists between calls.
    Calling ``select_operator`` with the same signals always returns the
    same decision — no randomness, no LLM involvement.

    Priority ordering (highest → lowest):
        high_fn > high_fp > low_accuracy > inconsistent > plateau > (none)
    """

    @staticmethod
    def select_operator(signals: CriticSignals) -> RefinementDecision:
        """
        Return the highest-priority :class:`RefinementDecision` for *signals*.

        Iterates the priority table from highest to lowest priority.  Returns
        the first decision whose corresponding signal is active.  Returns
        ``NO_OP`` if no signals are active.

        Args:
            signals: Active optimization signals from
                     :class:`~src.critic.signal_extractor.SignalExtractor`.

        Returns:
            :class:`RefinementDecision` with operator, target component,
            rationale, and priority fully populated.

        Example::

            signals = CriticSignals(
                high_fn=True, high_fp=True,
                low_accuracy=False, inconsistent=False, plateau=False,
            )
            decision = RefinementPolicy.select_operator(signals)
            # decision.operator  → RefinementOperator.REFINE_FEWSHOT  (high_fn wins)
            # decision.priority  → 5
        """
        for signal_attr, priority, operator, target, rationale in _PRIORITY_TABLE:
            if getattr(signals, signal_attr):
                decision = RefinementDecision(
                    operator=operator,
                    target_component=target,
                    rationale=rationale,
                    priority=priority,
                )
                logger.debug(
                    "Policy decision: %s (signal=%s)",
                    decision.summary(),
                    signal_attr,
                )
                return decision

        logger.debug("Policy decision: NO_OP — no active signals.")
        return _NO_OP_DECISION


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

#: Alias retained for code that was written against the original stub name.
#: Prefer ``RefinementPolicy`` in all new code.
MutationPolicy = RefinementPolicy