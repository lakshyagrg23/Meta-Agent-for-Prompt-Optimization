"""
src/configs/objective_weights.py
---------------------------------
Canonical source of truth for the composite optimization objective J(S).

All modules that compute or compare J(S) MUST import from here.
Do NOT hardcode these constants anywhere else.

Objective formula
-----------------
::

    J(S) = W_F1          * F1
         + W_RECALL      * Recall
         + W_CONSISTENCY * Consistency
         - W_COST        * PromptCost

    PromptCost = clamp(token_count / token_budget_ceiling, 0.0, 1.0)

    J(S) ∈ [−W_COST, W_F1 + W_RECALL + W_CONSISTENCY]
          = [−0.10,  0.90]  with the weights below

Final paper weights (as of May 2026)
-------------------------------------
+---------------+--------+------------------------------------------------------+
| Term          | Weight | Rationale                                            |
+===============+========+======================================================+
| F1            |  0.35  | Primary balanced classification signal               |
| Recall        |  0.40  | Missing phishing is operationally more costly than   |
|               |        | false alarms; recall dominates the objective         |
| Consistency   |  0.15  | Prompt stability required for research               |
|               |        | reproducibility                                      |
| PromptCost    | −0.10  | Mild regulariser; discourages prompt bloat without   |
|               |        | dominating the objective                             |
+---------------+--------+------------------------------------------------------+

Weights sum (positive terms): 0.35 + 0.40 + 0.15 = 0.90  ✓
"""

# ---------------------------------------------------------------------------
# J(S) weights — edit ONLY here, never at call sites
# ---------------------------------------------------------------------------

#: Weight on F1 score (harmonic mean of precision and recall).
W_F1: float = 0.35

#: Weight on phishing recall.  Given the highest weight because missing a
#: phishing email carries greater operational risk than a false alarm.
W_RECALL: float = 0.40

#: Weight on output consistency across repeated inference runs.
W_CONSISTENCY: float = 0.15

#: Penalty weight on normalised prompt token cost.
#: Subtracted from J(S) to discourage unnecessary prompt expansion.
W_COST: float = 0.10

# ---------------------------------------------------------------------------
# Token-budget ceiling
# ---------------------------------------------------------------------------

#: Default denominator used to normalise raw token counts into [0.0, 1.0].
#: Prompts at or above this threshold receive the maximum cost penalty (1.0).
DEFAULT_TOKEN_BUDGET_CEILING: int = 2048
