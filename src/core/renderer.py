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

Emails are structured :class:`~src.core.prompt_state.EmailInput` objects
with explicit ``sender``, ``receiver``, ``subject``, and ``body`` fields.
Plain strings are also accepted for backward compatibility.

Every call with identical inputs produces byte-for-byte identical output.
No randomness, timestamps, or hidden state is introduced.
"""

from __future__ import annotations

from typing import Dict, List, Union

from src.core.prompt_state import EmailInput, FewShotExample, PromptState

# Convenience alias used in type hints throughout this module.
_EmailArg = Union[str, EmailInput, Dict[str, str]]

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
    def render_prompt(
        prompt_state: PromptState,
        email: _EmailArg,
    ) -> str:
        """
        Render a complete LLM prompt from *prompt_state* and an *email*.

        Sections are assembled in the canonical order and joined with a
        single blank line separator.  Empty components produce empty
        section bodies but the section header is still emitted so that
        downstream parsers see a consistent skeleton.

        Args:
            prompt_state: The current :class:`PromptState` to render.
            email: The email to classify.  Accepts:
                   * :class:`EmailInput` — structured object (preferred).
                   * :class:`dict` — keys ``sender``, ``receiver``,
                     ``subject``, ``body`` are coerced to
                     :class:`EmailInput` automatically.
                   * :class:`str` — plain body text (legacy convenience).

        Returns:
            A single multi-line string representing the full prompt.

        Example::

            email = EmailInput(
                sender="attacker@evil.com",
                receiver="victim@corp.com",
                subject="Urgent: verify your account",
                body="Click here immediately to avoid suspension.",
            )
            prompt = PromptRenderer.render_prompt(state, email)
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

        When ``example.email`` is a plain string the output is::

            Example {index}:
            Email: {email}
            Label: {label}
            Reason: {reason}

        When ``example.email`` is an :class:`EmailInput` the email portion
        expands to its four structured fields before the label and reason::

            Example {index}:
            Sender: {sender}
            Receiver: {receiver}
            Subject: {subject}
            Body: {body}
            Label: {label}
            Reason: {reason}

        Args:
            index:   1-based position of this example in the list.
            example: The :class:`FewShotExample` to render.

        Returns:
            Formatted string for this single example.
        """
        header = f"Example {index}:"

        if isinstance(example.email, EmailInput):
            email_lines = PromptRenderer._render_email_fields(example.email)
        else:
            email_lines = [f"Email: {example.email.strip()}"]

        lines = (
            [header]
            + email_lines
            + [
                f"Label: {example.label.strip()}",
                f"Reason: {example.reason.strip()}",
            ]
        )
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
    def _render_email(email: _EmailArg) -> str:
        """
        Render the EMAIL section containing the message to classify.

        Accepts a structured :class:`EmailInput`, a plain ``dict`` with
        the same four keys, or a plain string (renders as ``Body:`` only).
        All paths produce output under the ``[EMAIL]`` header.

        Args:
            email: Structured :class:`EmailInput`, compatible ``dict``,
                   or a plain string.

        Returns:
            Section string under the ``[EMAIL]`` header.

        Raises:
            KeyError: If a ``dict`` is passed but is missing one of the
                      required keys: ``sender``, ``receiver``, ``subject``,
                      ``body``.
        """
        coerced = PromptRenderer._coerce_email_input(email)
        if isinstance(coerced, EmailInput):
            body = "\n".join(PromptRenderer._render_email_fields(coerced))
        else:
            body = f"Body: {coerced.strip()}"
        return PromptRenderer._section("EMAIL", body)

    @staticmethod
    def _coerce_email_input(email: _EmailArg) -> Union[EmailInput, str]:
        """
        Normalise any supported email representation to either an
        :class:`EmailInput` or a plain string.

        * ``EmailInput`` is returned as-is.
        * A ``dict`` with keys ``sender``, ``receiver``, ``subject``,
          ``body`` is converted to :class:`EmailInput`.
        * Any other string is returned unchanged.

        Args:
            email: Raw email argument from the caller.

        Returns:
            :class:`EmailInput` or ``str``.

        Raises:
            KeyError: If a ``dict`` is missing a required field.
        """
        if isinstance(email, EmailInput):
            return email
        if isinstance(email, dict):
            try:
                return EmailInput(
                    sender=email["sender"],
                    receiver=email["receiver"],
                    subject=email["subject"],
                    body=email["body"],
                )
            except KeyError as exc:
                raise KeyError(
                    f"Email dict is missing required field: {exc}. "
                    f"Expected keys: sender, receiver, subject, body."
                ) from exc
        return email  # plain str

    @staticmethod
    def _render_email_fields(email: EmailInput) -> List[str]:
        """
        Produce an ordered list of ``Field: value`` lines for an
        :class:`EmailInput` object.

        The order is fixed: Sender → Receiver → Subject → Body.  This
        helper is shared by both the top-level email section and
        individual few-shot example rendering so the format is always
        identical.

        Args:
            email: A fully populated :class:`EmailInput` instance.

        Returns:
            List of strings, one per field, ready to be joined with
            newlines.
        """
        return [
            f"Sender: {email.sender.strip()}",
            f"Receiver: {email.receiver.strip()}",
            f"Subject: {email.subject.strip()}",
            f"Body: {email.body.strip()}",
        ]

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