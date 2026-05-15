"""
src/core/renderer.py
--------------------
Deterministic rendering of PromptState into a final LLM prompt string.

The rendering order is fixed and must never change without a corresponding
architecture update:

    1. ROLE
    2. BASE INSTRUCTION
    3. INSTRUCTION ENRICHMENT
    4. FEW-SHOT EXAMPLES
    5. CHAIN OF THOUGHT
    6. EMAIL

Every call with identical inputs produces byte-for-byte identical output.
No randomness, timestamps, or hidden state is introduced.
"""

from __future__ import annotations

from typing import List

from src.core.prompt_state import FewShotExample, PromptState

# Section header template — single source of truth for formatting.
_HEADER = "[{title}]"


class PromptRenderer:
    """
    Converts a :class:`~src.core.prompt_state.PromptState` into a
    deterministic, fully-formed prompt string ready for LLM inference.

    Design constraints
    ------------------
    * **Stateless** — all methods are static; no instance state exists.
    * **Deterministic** — same inputs always produce the same output.
    * **Order-fixed** — the six-section layout is enforced by the
      implementation and cannot drift silently.
    * **No extra whitespace** — sections are joined with exactly one blank
      line; trailing whitespace is stripped from every section.
    """

    @staticmethod
    def render_prompt(prompt_state: PromptState, email: str) -> str:
        """
        Render a complete LLM prompt from *prompt_state* and an *email*.

        Sections are assembled in the canonical order and joined with a
        single blank line separator.  Empty components produce empty
        section bodies but the section header is still emitted so that
        downstream parsers see a consistent skeleton.

        Args:
            prompt_state: The current :class:`PromptState` to render.
            email: Raw email text to be classified.

        Returns:
            A single multi-line string representing the full prompt.

        Example::

            prompt = PromptRenderer.render_prompt(state, raw_email_text)
            response = llm.generate(prompt)
        """
        sections: List[str] = [
            PromptRenderer._render_role(prompt_state),
            PromptRenderer._render_base_instruction(prompt_state),
            PromptRenderer._render_instruction_enrichment(prompt_state),
            PromptRenderer._render_few_shot(prompt_state),
            PromptRenderer._render_cot(prompt_state),
            PromptRenderer._render_email(email),
        ]
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Private section renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_role(prompt_state: PromptState) -> str:
        """
        Render the ROLE section.

        Args:
            prompt_state: Source state.

        Returns:
            Section string with header and role content.
        """
        return PromptRenderer._section("ROLE", prompt_state.role.content)

    @staticmethod
    def _render_base_instruction(prompt_state: PromptState) -> str:
        """
        Render the BASE INSTRUCTION section.

        The base instruction is the fixed anchor of the prompt and is
        never mutated by any refinement operator.

        Args:
            prompt_state: Source state.

        Returns:
            Section string with header and base instruction text.
        """
        return PromptRenderer._section("TASK", prompt_state.base_instruction)

    @staticmethod
    def _render_instruction_enrichment(prompt_state: PromptState) -> str:
        """
        Render the INSTRUCTION ENRICHMENT section.

        Args:
            prompt_state: Source state.

        Returns:
            Section string with header and enrichment content.
        """
        return PromptRenderer._section(
            "GUIDELINES", prompt_state.instruction_enrichment.content
        )

    @staticmethod
    def _render_few_shot(prompt_state: PromptState) -> str:
        """
        Render the FEW-SHOT EXAMPLES section.

        Examples are rendered in list order (index 0 first).  Each example
        is formatted as a numbered block with ``Email:``, ``Label:``, and
        ``Reason:`` fields.  The order is deterministic because
        :class:`~src.core.prompt_state.FewShotComponent` is an ordered list.

        Args:
            prompt_state: Source state.

        Returns:
            Section string containing all formatted examples, or an empty
            section body if no examples are present.
        """
        examples = prompt_state.few_shot.examples
        if not examples:
            return PromptRenderer._section("EXAMPLES", "")

        rendered_examples = [
            PromptRenderer._render_single_example(i + 1, ex)
            for i, ex in enumerate(examples)
        ]
        body = "\n\n".join(rendered_examples)
        return PromptRenderer._section("EXAMPLES", body)

    @staticmethod
    def _render_single_example(index: int, example: FewShotExample) -> str:
        """
        Format one :class:`FewShotExample` as a numbered block.

        Output format::

            Example {index}:
            Email: {email}
            Label: {label}
            Reason: {reason}

        Args:
            index: 1-based position of this example in the list.
            example: The :class:`FewShotExample` to render.

        Returns:
            Formatted string for this single example.
        """
        lines = [
            f"Example {index}:",
            f"Email: {example.email.strip()}",
            f"Label: {example.label.strip()}",
            f"Reason: {example.reason.strip()}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_cot(prompt_state: PromptState) -> str:
        """
        Render the CHAIN OF THOUGHT section.

        Args:
            prompt_state: Source state.

        Returns:
            Section string with header and CoT instructions.
        """
        return PromptRenderer._section(
            "REASONING APPROACH", prompt_state.cot.content
        )

    @staticmethod
    def _render_email(email: str) -> str:
        """
        Render the EMAIL section containing the text to classify.

        Args:
            email: Raw email body text.

        Returns:
            Section string with header and stripped email text.
        """
        return PromptRenderer._section("EMAIL TO CLASSIFY", email.strip())

    # ------------------------------------------------------------------
    # Formatting primitive
    # ------------------------------------------------------------------

    @staticmethod
    def _section(title: str, body: str) -> str:
        """
        Compose a single titled section string.

        Format::

            [TITLE]
            body text

        Trailing whitespace is stripped from both the title and body.
        The header and body are separated by exactly one newline.

        Args:
            title: Section title rendered inside brackets.
            body: Section content.  May be multi-line.

        Returns:
            Formatted section string.
        """
        header = _HEADER.format(title=title.strip())
        return f"{header}\n{body.strip()}"