"""
src/utils/token_utils.py
------------------------
Lightweight token-counting utilities for PromptState components.

All token estimates are routed through ``estimate_token_count`` so that
upgrading from whitespace splitting to a real tokeniser (e.g. tiktoken)
requires changing **exactly one function**.

Typical upgrade path::

    # Replace the body of estimate_token_count with:
    import tiktoken
    _ENC = tiktoken.encoding_for_model("gpt-4o")

    def estimate_token_count(text: str) -> int:
        return len(_ENC.encode(text))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    # Import only for type hints; avoids a circular dependency at runtime.
    from src.core.prompt_state import FewShotComponent, FewShotExample, PromptComponent


# ---------------------------------------------------------------------------
# Primary estimator – single entry point for the tokenisation strategy
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in *text*.

    **Current strategy**: whitespace splitting (``str.split()``).
    This over-counts slightly for punctuation-heavy text and under-counts
    for CJK scripts, but is dependency-free and adequate for budget checks.

    Args:
        text: Any string whose token count is needed.

    Returns:
        Estimated token count as a non-negative integer.  Empty or
        whitespace-only strings return 0.

    Note:
        To switch to a real tokeniser, replace the body of *this function
        only* — all other helpers delegate here automatically.
    """
    return len(text.split())


# ---------------------------------------------------------------------------
# Few-shot helpers
# ---------------------------------------------------------------------------

def count_example_tokens(example: "FewShotExample") -> int:
    """Count tokens for a single :class:`FewShotExample`.

    Accounts for all three text-bearing fields:
    ``email``, ``label``, and ``reason``.

    Args:
        example: A :class:`~src.core.prompt_state.FewShotExample` instance.

    Returns:
        Combined token estimate for the example.
    """
    from src.core.prompt_state import EmailInput
    
    email_text = ""
    if isinstance(example.email, str):
        email_text = example.email
    elif isinstance(example.email, EmailInput):
        email_text = f"{example.email.subject} {example.email.body}"
    elif isinstance(example.email, dict):
        email_text = f"{example.email.get('subject', '')} {example.email.get('body', '')}"
        
    return (
        estimate_token_count(email_text)
        + estimate_token_count(example.label)
        + estimate_token_count(example.reason)
    )


def count_fewshot_tokens(few_shot: "FewShotComponent") -> int:
    """Count total tokens consumed by all examples in a :class:`FewShotComponent`.

    Args:
        few_shot: A :class:`~src.core.prompt_state.FewShotComponent` instance.

    Returns:
        Sum of token estimates across every contained example.
        Returns 0 if the examples list is empty.
    """
    return sum(count_example_tokens(ex) for ex in few_shot.examples)


# ---------------------------------------------------------------------------
# Component-level helper
# ---------------------------------------------------------------------------

def count_component_tokens(component: "PromptComponent") -> int:
    """Count tokens in a single :class:`PromptComponent`.

    Args:
        component: A :class:`~src.core.prompt_state.PromptComponent` instance.

    Returns:
        Token estimate for ``component.content``.
    """
    return estimate_token_count(component.content)


# ---------------------------------------------------------------------------
# Full PromptState total
# ---------------------------------------------------------------------------

def count_total_prompt_tokens(
    base_instruction: str,
    role: "PromptComponent",
    instruction_enrichment: "PromptComponent",
    cot: "PromptComponent",
    few_shot: "FewShotComponent",
) -> int:
    """Compute the aggregate token count across all PromptState components.

    Accepts individual components rather than the full :class:`PromptState`
    object to keep this module free of circular imports.

    Args:
        base_instruction: The fixed base instruction string.
        role: Role :class:`PromptComponent`.
        instruction_enrichment: Instruction enrichment :class:`PromptComponent`.
        cot: Chain-of-thought :class:`PromptComponent`.
        few_shot: :class:`FewShotComponent` holding all few-shot examples.

    Returns:
        Total estimated token count across every component.

    Example::

        total = count_total_prompt_tokens(
            state.base_instruction,
            state.role,
            state.instruction_enrichment,
            state.cot,
            state.few_shot,
        )
    """
    return (
        estimate_token_count(base_instruction)
        + count_component_tokens(role)
        + count_component_tokens(instruction_enrichment)
        + count_component_tokens(cot)
        + count_fewshot_tokens(few_shot)
    )
