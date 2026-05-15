"""
src/llm/refinement_generation.py
---------------------------------
Constrained semantic refinement generation for adaptive prompt optimization.

Responsibility
--------------
Build meta-prompts for each component type, call the LLM, and return a
structured ``LLMRefinementResponse``.  Nothing else.

    error_summary + current content
            ↓
    _build_*_meta_prompt()
            ↓
    LLMClient.generate()          ← deterministic, low temperature
            ↓
    _strip_refined_content()
            ↓
    LLMRefinementResponse(refined_content, raw_response)

What this module does NOT do
-----------------------------
* No evaluation logic.
* No acceptance / rejection of candidates.
* No mutation of ``PromptState`` — callers (mutation operators) handle that.
* No few-shot example parsing into ``FewShotExample`` — that is the
  ``RefineFewShotOperator``'s responsibility.

Error summary keys (convention across the system)
--------------------------------------------------
``error_summary`` is a plain ``dict`` produced by the critic layer.
Expected keys (all optional — generation degrades gracefully if absent):

    "high_fp"          bool    — many false positives (legit → phishing)
    "high_fn"          bool    — many false negatives (phishing → legit)
    "low_accuracy"     bool    — overall accuracy below threshold
    "inconsistent"     bool    — volatile predictions across runs
    "plateau"          bool    — no improvement in recent iterations
    "fp_examples"      list    — sample misclassified legitimate emails
    "fn_examples"      list    — sample misclassified phishing emails
    "iteration"        int     — current optimization iteration
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient, LLMError
from src.llm.schemas import LLMRefinementResponse, LLMRequestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Refinement-specific exception
# ---------------------------------------------------------------------------

class RefinementGenerationError(Exception):
    """
    Raised when the LLM call for a refinement step fails.

    Attributes:
        component:  Which component was being refined (e.g. ``"role"``).
        cause:      The original ``LLMError``.
    """

    def __init__(self, message: str, component: str, cause: Exception) -> None:
        super().__init__(message)
        self.component = component
        self.cause = cause


# ---------------------------------------------------------------------------
# Generation config defaults
# ---------------------------------------------------------------------------

# Low temperature: near-deterministic, controlled generation.
# Slightly above 0 to avoid degenerate repetition on weaker models.
_REFINE_CONFIG = LLMRequestConfig(temperature=0.1, max_tokens=200)

# Few-shot examples can be slightly longer (email + label + reason).
_FEWSHOT_CONFIG = LLMRequestConfig(temperature=0.1, max_tokens=120)

# Max misclassified examples to include in error context (keep prompt lean).
_MAX_ERROR_EXAMPLES = 3


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RefinementGenerator:
    """
    Generates refined content for each prompt component via LLM calls.

    All four ``generate_*`` methods follow the same contract:

    Args:
        current_content:  The existing text of the component being refined.
        error_summary:    Critic-produced dict describing current failures.
        token_budget:     Upper token limit for the refined output.

    Returns:
        ``LLMRefinementResponse`` with ``refined_content`` (the new text,
        stripped and ready to use) and ``raw_response`` (preserved for
        logging/debugging).

    Raises:
        ``RefinementGenerationError``: If the LLM call fails.

    Example::

        gen = RefinementGenerator(client)
        response = gen.generate_role("You are an analyst.", error_summary, 50)
        new_role_text = response.refined_content
    """

    def __init__(
        self,
        client: LLMClient,
        refine_config: Optional[LLMRequestConfig] = None,
        fewshot_config: Optional[LLMRequestConfig] = None,
    ) -> None:
        self._client = client
        self._refine_config = refine_config or _REFINE_CONFIG
        self._fewshot_config = fewshot_config or _FEWSHOT_CONFIG

    # ------------------------------------------------------------------
    # Role refinement
    # ------------------------------------------------------------------

    def generate_role_refinement(
        self,
        current_content: str,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> LLMRefinementResponse:
        """
        Generate a refined role description for the phishing detection LLM.

        The role sets the LLM's persona and expertise frame.  Refinement
        makes it more specific to the observed failure mode.

        Args:
            current_content: Existing role text.
            error_summary:   Critic signals and misclassified examples.
            token_budget:    Max tokens the refined role should use.

        Returns:
            ``LLMRefinementResponse`` — ``refined_content`` is the new role text.
        """
        meta_prompt = self._build_role_meta_prompt(
            current_content, error_summary, token_budget
        )
        return self._call_and_wrap(meta_prompt, component="role")

    # ------------------------------------------------------------------
    # Chain-of-thought refinement
    # ------------------------------------------------------------------

    def generate_cot_refinement(
        self,
        current_content: str,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> LLMRefinementResponse:
        """
        Generate refined chain-of-thought reasoning instructions.

        CoT guides the model through intermediate reasoning steps before
        producing a label.  Refinement sharpens the focus on signals
        correlated with current failure patterns.

        Args:
            current_content: Existing CoT instruction text.
            error_summary:   Critic signals and misclassified examples.
            token_budget:    Max tokens the refined CoT should use.

        Returns:
            ``LLMRefinementResponse`` — ``refined_content`` is the new CoT text.
        """
        meta_prompt = self._build_cot_meta_prompt(
            current_content, error_summary, token_budget
        )
        return self._call_and_wrap(meta_prompt, component="cot")

    # ------------------------------------------------------------------
    # Instruction enrichment refinement
    # ------------------------------------------------------------------

    def generate_enrichment_refinement(
        self,
        current_content: str,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> LLMRefinementResponse:
        """
        Generate refined instruction enrichment (classification guidelines).

        Instruction enrichment adds actionable detection criteria on top of
        the fixed base instruction.  Refinement targets the specific signal
        types that are currently mis-weighted.

        Args:
            current_content: Existing enrichment / guidelines text.
            error_summary:   Critic signals and misclassified examples.
            token_budget:    Max tokens the refined enrichment should use.

        Returns:
            ``LLMRefinementResponse`` — ``refined_content`` is the new guidelines text.
        """
        meta_prompt = self._build_enrichment_meta_prompt(
            current_content, error_summary, token_budget
        )
        return self._call_and_wrap(meta_prompt, component="instruction_enrichment")

    # ------------------------------------------------------------------
    # Few-shot example generation
    # ------------------------------------------------------------------

    def generate_fewshot_refinement(
        self,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> LLMRefinementResponse:
        """
        Generate a new labeled few-shot example targeting current failures.

        The LLM is asked to produce a single ``Email / Label / Reason``
        block in the canonical format.  Parsing into a ``FewShotExample``
        dataclass is the caller's (``RefineFewShotOperator``'s) responsibility.

        Output format guaranteed in the meta-prompt::

            Email: <email text>
            Label: PHISHING | LEGITIMATE
            Reason: <brief justification>

        Args:
            error_summary: Critic signals and misclassified examples.
            token_budget:  Max tokens the generated example should use.

        Returns:
            ``LLMRefinementResponse`` — ``refined_content`` contains the raw
            ``Email / Label / Reason`` block for downstream parsing.
        """
        meta_prompt = self._build_fewshot_meta_prompt(error_summary, token_budget)
        return self._call_and_wrap(
            meta_prompt, component="few_shot", config=self._fewshot_config
        )

    # ------------------------------------------------------------------
    # Meta-prompt builders
    # ------------------------------------------------------------------

    def _build_role_meta_prompt(
        self,
        current: str,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> str:
        error_ctx = _format_error_context(error_summary)
        return textwrap.dedent(f"""
            You are a prompt engineering expert optimizing a phishing email detection system.

            CURRENT ROLE DESCRIPTION:
            {current.strip()}

            OBSERVED FAILURE PATTERNS:
            {error_ctx}

            TASK:
            Write an improved role description for the phishing detection AI.

            REQUIREMENTS:
            - Describe specific expertise relevant to email security.
            - Address the observed failure patterns directly.
            - Be concise — stay within {token_budget} tokens.
            - Return ONLY the new role description. No preamble, no explanation.

            NEW ROLE DESCRIPTION:
        """).strip()

    def _build_cot_meta_prompt(
        self,
        current: str,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> str:
        error_ctx = _format_error_context(error_summary)
        return textwrap.dedent(f"""
            You are a prompt engineering expert optimizing a phishing email detection system.

            CURRENT REASONING APPROACH:
            {current.strip()}

            OBSERVED FAILURE PATTERNS:
            {error_ctx}

            TASK:
            Write improved chain-of-thought reasoning instructions for the classifier.

            REQUIREMENTS:
            - Guide step-by-step analysis before producing a label.
            - Highlight signals most correlated with the current failure mode.
            - Be actionable and concise — stay within {token_budget} tokens.
            - Return ONLY the reasoning instructions. No preamble, no explanation.

            NEW REASONING APPROACH:
        """).strip()

    def _build_enrichment_meta_prompt(
        self,
        current: str,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> str:
        error_ctx = _format_error_context(error_summary)
        return textwrap.dedent(f"""
            You are a prompt engineering expert optimizing a phishing email detection system.

            CURRENT CLASSIFICATION GUIDELINES:
            {current.strip()}

            OBSERVED FAILURE PATTERNS:
            {error_ctx}

            TASK:
            Write improved classification guidelines for the phishing detector.

            REQUIREMENTS:
            - Include concrete, actionable detection criteria.
            - Address the specific signal types contributing to current failures.
            - Stay within {token_budget} tokens.
            - Return ONLY the new guidelines. No preamble, no explanation.

            NEW GUIDELINES:
        """).strip()

    def _build_fewshot_meta_prompt(
        self,
        error_summary: Dict[str, Any],
        token_budget: int,
    ) -> str:
        error_ctx = _format_error_context(error_summary)
        return textwrap.dedent(f"""
            You are a phishing detection expert creating training examples.

            OBSERVED FAILURE PATTERNS:
            {error_ctx}

            TASK:
            Generate one labeled email example that targets the observed failure patterns.
            The example should cover an edge case the classifier is currently missing.

            OUTPUT FORMAT (use exactly this format, nothing else):
            Email: <email text>
            Label: PHISHING
            Reason: <one-sentence justification>

            OR:
            Email: <email text>
            Label: LEGITIMATE
            Reason: <one-sentence justification>

            REQUIREMENTS:
            - Email text should be realistic but concise (under {token_budget} tokens total).
            - Label must be exactly PHISHING or LEGITIMATE.
            - Reason must be a single sentence.
            - Return ONLY the Email / Label / Reason block. Nothing else.

            EXAMPLE:
        """).strip()

    # ------------------------------------------------------------------
    # LLM call + response wrapping
    # ------------------------------------------------------------------

    def _call_and_wrap(
        self,
        meta_prompt: str,
        component: str,
        config: Optional[LLMRequestConfig] = None,
    ) -> LLMRefinementResponse:
        """
        Call the LLM with *meta_prompt* and return a wrapped response.

        Args:
            meta_prompt: Fully-built meta-prompt string.
            component:   Name of the component being refined (for errors/logs).
            config:      Override generation config; defaults to ``_refine_config``.

        Returns:
            ``LLMRefinementResponse`` with stripped ``refined_content``.

        Raises:
            RefinementGenerationError: If the LLM call fails.
        """
        effective_config = config or self._refine_config
        try:
            raw = self._client.generate(meta_prompt, effective_config)
        except LLMError as exc:
            raise RefinementGenerationError(
                message=f"Refinement LLM call failed for component '{component}': {exc}",
                component=component,
                cause=exc,
            ) from exc

        refined = _strip_refined_content(raw)

        logger.debug(
            "refinement | component=%s | raw_len=%d | refined_len=%d",
            component,
            len(raw),
            len(refined),
        )

        return LLMRefinementResponse(
            refined_content=refined,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RefinementGenerator(client={self._client!r})"


# ---------------------------------------------------------------------------
# Error context formatter
# ---------------------------------------------------------------------------

def _format_error_context(error_summary: Dict[str, Any]) -> str:
    """
    Build a human-readable error context block from *error_summary*.

    Gracefully handles missing keys — generates useful context from
    whatever signals are present.

    Args:
        error_summary: Critic-produced dict (see module docstring for keys).

    Returns:
        Formatted multi-line string describing current failure patterns.
    """
    lines: List[str] = []

    # Boolean signal flags
    signal_descriptions = {
        "high_fp":       "High false-positive rate: legitimate emails misclassified as phishing.",
        "high_fn":       "High false-negative rate: phishing emails misclassified as legitimate.",
        "low_accuracy":  "Overall accuracy is below the target threshold.",
        "inconsistent":  "Predictions are inconsistent across repeated runs.",
        "plateau":       "No improvement observed over recent iterations.",
    }
    for key, description in signal_descriptions.items():
        if error_summary.get(key):
            lines.append(f"- {description}")

    # Iteration context
    iteration = error_summary.get("iteration")
    if iteration is not None:
        lines.append(f"- Current optimization iteration: {iteration}.")

    # Misclassified example snippets (capped to keep the meta-prompt lean)
    fp_examples: List[str] = error_summary.get("fp_examples", [])
    fn_examples: List[str] = error_summary.get("fn_examples", [])

    if fp_examples:
        lines.append(
            "- False-positive examples (legitimate emails wrongly flagged):"
        )
        for ex in fp_examples[:_MAX_ERROR_EXAMPLES]:
            snippet = str(ex)[:80].replace("\n", " ")
            lines.append(f"    • {snippet}")

    if fn_examples:
        lines.append(
            "- False-negative examples (phishing emails missed):"
        )
        for ex in fn_examples[:_MAX_ERROR_EXAMPLES]:
            snippet = str(ex)[:80].replace("\n", " ")
            lines.append(f"    • {snippet}")

    if not lines:
        lines.append("- No specific failure patterns identified yet.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response content stripper
# ---------------------------------------------------------------------------

def _strip_refined_content(raw: str) -> str:
    """
    Strip meta-prompt echoes, preamble, and trailing whitespace from *raw*.

    Many models echo part of the prompt or add a brief preamble before the
    actual content.  This function removes common lead-in patterns so that
    ``refined_content`` contains only the usable text.

    Args:
        raw: Raw LLM response string.

    Returns:
        Cleaned content string.  Falls back to the original stripped value
        if no patterns are detected.
    """
    text = raw.strip()

    # Remove common preamble patterns the model might prepend.
    lead_ins = [
        "new role description:",
        "new reasoning approach:",
        "new guidelines:",
        "example:",
        "here is",
        "here's",
        "sure,",
        "certainly,",
        "of course,",
    ]
    lower = text.lower()
    for lead in lead_ins:
        if lower.startswith(lead):
            text = text[len(lead):].strip()
            break

    return text
