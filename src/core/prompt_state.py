"""
src/core/prompt_state.py
------------------------
Core dataclasses for structured bounded prompt memory.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class EmailInput:
    """
    Structured representation of an email to be classified.

    Using explicit fields instead of raw text prevents ambiguity
    about where the subject ends and the body begins, and allows
    the renderer to produce a consistently formatted prompt section.

    Attributes:
        sender:   Email address (or display name) of the sender.
        receiver: Email address (or display name) of the recipient.
        subject:  Subject line of the email.
        body:     Full body text of the email.
    """

    sender: str
    receiver: str
    subject: str
    body: str


@dataclass
class FewShotExample:
    """
    A single labeled example used in few-shot prompting.

    The ``email`` field accepts either a plain string (legacy / simple
    usage) or a fully structured :class:`EmailInput` object.  The
    renderer handles both cases transparently.

    Attributes:
        email:            Email content — plain string or :class:`EmailInput`.
        label:            Ground-truth classification label (e.g. ``"PHISHING"``).
        reason:           Human-readable explanation of the label decision.
        relevance_score:  Operator-assigned score used during bounded
                          replacement; higher means more relevant to keep.
    """

    email: Union[str, "EmailInput"]
    label: str
    reason: str
    relevance_score: float = 0.0


@dataclass
class PromptComponent:
    content: str
    token_budget: int
    revision_count: int = 0


@dataclass
class FewShotComponent:
    examples: List[FewShotExample]
    token_budget: int
    max_examples: int
    revision_count: int = 0


@dataclass
class PromptMetadata:
    iteration: int = 0
    score_history: List[float] = field(default_factory=list)
    active_signals: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)


@dataclass
class PromptState:
    """
    Structured bounded prompt memory.

    Components evolve iteratively through constrained refinement.
    """

    base_instruction: str

    role: PromptComponent
    instruction_enrichment: PromptComponent
    cot: PromptComponent

    few_shot: FewShotComponent

    metadata: PromptMetadata

    def clone(self) -> "PromptState":
        """
        Return a fully independent deep copy of this PromptState.

        Uses copy.deepcopy so that every nested dataclass
        (PromptComponent, FewShotComponent, FewShotExample,
        PromptMetadata) and every mutable collection (lists)
        is duplicated — no shared references remain between
        the original and the clone.
        """
        return copy.deepcopy(self)

    def get_total_token_count(self) -> int:
        """
        Compute approximate total token usage across all components.

        Delegates to :func:`src.utils.token_utils.count_total_prompt_tokens`
        so that the tokenisation strategy is defined in exactly one place.
        Swapping to tiktoken only requires changing ``estimate_token_count``
        in token_utils.py — nothing here needs to change.
        """
        from src.utils.token_utils import count_total_prompt_tokens

        return count_total_prompt_tokens(
            self.base_instruction,
            self.role,
            self.instruction_enrichment,
            self.cot,
            self.few_shot,
        )

    def update_component(
        self,
        component_name: str,
        new_content: str,
    ) -> None:
        """
        Replace the content of a named PromptComponent in-place.

        Args:
            component_name: One of 'role', 'instruction_enrichment', 'cot'.
            new_content: Replacement content string.

        Raises:
            AttributeError: If component_name does not exist on this state.
            TypeError: If the named attribute is not a PromptComponent.
        """
        component = getattr(self, component_name)
        if not isinstance(component, PromptComponent):
            raise TypeError(
                f"'{component_name}' is not a PromptComponent; "
                f"got {type(component).__name__}."
            )
        component.content = new_content

    def increment_revision(
        self,
        component_name: str,
    ) -> None:
        """
        Increment the revision_count of a named component.

        Works for both PromptComponent ('role', 'instruction_enrichment',
        'cot') and FewShotComponent ('few_shot').

        Args:
            component_name: Name of the component attribute to increment.

        Raises:
            AttributeError: If component_name does not exist on this state.
            TypeError: If the named attribute lacks a revision_count field.
        """
        component = getattr(self, component_name)
        if not hasattr(component, "revision_count"):
            raise TypeError(
                f"'{component_name}' has no revision_count attribute."
            )
        component.revision_count += 1