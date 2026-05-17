"""
src/optimization/optimization_loop.py
---------------------------------------
Thin orchestration shell for the adaptive prompt optimization process.

Architectural mandate
---------------------
This file MUST remain a pure orchestrator.  It coordinates calls between
the other modules but contains ZERO business logic of its own.

The following logic MUST NOT appear here:
* Metric computation  → ``src.evaluation.metrics`` / ``src.evaluation.evaluator``
* Signal extraction   → ``src.critic.signal_extractor``
* Refinement policy   → ``src.critic.policy``
* Candidate mutation  → ``src.mutations.*`` / ``src.optimization.candidate_generator``
* Prompt rendering    → ``src.core.renderer``
* Acceptance logic    → ``src.optimization.acceptance``
* Dataset loading     → ``src.dataset.loader`` / ``src.dataset.sampler``
* LLM inference       → ``src.llm.inference``

If you feel the urge to put any of the above directly in this class, stop
and put it in the correct module instead.

Orchestration sequence (per iteration)
---------------------------------------
::

    1.  sample_batch()              ← DatasetSampler
    2.  render_prompt()             ← PromptRenderer
    3.  run_inference()             ← LLMInference
    4.  evaluate()                  ← Evaluator
    5.  extract_signals()           ← SignalExtractor
    6.  select_operator()           ← MutationPolicy
    7.  generate_candidate()        ← CandidateGenerator   (clone-first)
    8.  render + infer candidate    ← PromptRenderer, LLMInference
    9.  evaluate_candidate()        ← Evaluator  (SAME batch as step 4)
    10. accept_or_rollback()        ← AcceptanceStrategy
    11. log_iteration()             ← Logger / history tracking
    12. check_convergence()         ← plateau detection, early stop

Mutation / rollback safety note
---------------------------------
Candidate generation (step 7) is safe because ``CandidateGenerator.generate``
delegates to ``MutationOperator.refine``, which asserts at runtime that the
returned object is a distinct instance from the incumbent state.  The loop
therefore never needs to defensively clone before calling the operator.
"""

from __future__ import annotations

from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.acceptance import AcceptanceStrategy
from src.critic.signal_extractor import SignalExtractor
from src.critic.policy import RefinementPolicy, RefinementDecision, RefinementOperator


class OptimizationLoop:
    """
    Thin orchestration shell for iterative prompt refinement.

    Each public method delegates entirely to a specialist module.  This class
    contains no metric logic, no mutation logic, no rendering logic, and no
    signal extraction logic.

    Parameters
    ----------
    (Inject dependencies via constructor once the full pipeline is wired up.
    Keeping this as a stub avoids premature coupling to specific LLM clients
    or dataset formats.)
    """

    def run(self) -> None:
        """
        Execute the iterative optimization process.

        Stub — implement by following the orchestration sequence documented
        in the module docstring.  Do not add business logic here; route every
        concern to the appropriate specialist module.
        """
        raise NotImplementedError(
            "OptimizationLoop.run() is not yet implemented.  "
            "Wire up the orchestration sequence from the module docstring."
        )