"""
src/llm/inference.py
---------------------
Phishing classification inference pipeline.

Responsibility
--------------
Orchestrate the three-step classification workflow:

    Rendered prompt → LLMClient.generate() → raw text
                   → ClassificationParser.parse() → LLMClassificationResponse

This module owns *nothing* except the wiring between client and parser.
No evaluation, no scoring, no dataset access.

Public surface
--------------
``PhishingInferenceEngine``     — primary class; classify single or batch.
``classify_email``              — module-level shorthand for single calls.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.llm.client import LLMClient, LLMError
from src.llm.parser import ClassificationParser
from src.llm.schemas import LLMClassificationResponse, LLMRequestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference-specific exception
# ---------------------------------------------------------------------------

class InferenceError(Exception):
    """
    Raised when the inference pipeline cannot complete a classification.

    Wraps ``LLMError`` subclasses so callers can catch inference failures
    without importing the client exception hierarchy.

    Attributes:
        prompt_preview: First 120 characters of the offending prompt,
                        for logging and debugging.
        cause:          The original ``LLMError`` (also available via
                        ``__cause__``).
    """

    def __init__(self, message: str, prompt_preview: str, cause: Exception) -> None:
        super().__init__(message)
        self.prompt_preview = prompt_preview
        self.cause = cause


# ---------------------------------------------------------------------------
# Deterministic request config defaults
# ---------------------------------------------------------------------------

# Default configuration used for all classification calls unless overridden.
# temperature=0.0 → greedy decoding; max_tokens sized for label + reason.
_DEFAULT_CONFIG = LLMRequestConfig(temperature=0.0, max_tokens=300)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class PhishingInferenceEngine:
    """
    Stateless phishing classification inference pipeline.

    Wraps an ``LLMClient`` and exposes a clean ``classify_email`` / ``classify_batch``
    interface.  The engine carries no mutable state after construction —
    the same instance can safely be reused across iterations of the
    optimization loop.

    Args:
        client:         Configured ``LLMClient`` instance.
        request_config: Generation parameters applied to every call.
                        Defaults to ``temperature=0.0, max_tokens=300``
                        for deterministic, concise outputs.

    Example::

        client = LLMClient(LLMClientConfig(provider="openai", model="gpt-4o"))
        engine = PhishingInferenceEngine(client)
        result = engine.classify_email(rendered_prompt)
        print(result.label, result.reason)
    """

    def __init__(
        self,
        client: LLMClient,
        request_config: Optional[LLMRequestConfig] = None,
    ) -> None:
        self._client = client
        self._config = request_config if request_config is not None else _DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Single-email inference
    # ------------------------------------------------------------------

    def classify_email(self, rendered_prompt: str) -> LLMClassificationResponse:
        """
        Classify a single email from its fully-rendered prompt.

        Workflow:
            1. Send ``rendered_prompt`` to ``LLMClient.generate()``.
            2. Parse the raw text with ``ClassificationParser.parse()``.
            3. Return the structured ``LLMClassificationResponse``.

        The parser **never raises**, so a valid dataclass is always
        returned.  If the LLM call itself fails, ``InferenceError`` is
        raised so the caller can decide on retry / skip behaviour.

        Args:
            rendered_prompt: Output of ``PromptRenderer.render_prompt()``.

        Returns:
            ``LLMClassificationResponse`` with label, reason, and raw text.

        Raises:
            InferenceError: If the LLM client call fails for any reason
                            (timeout, auth error, provider error).
        """
        raw_text = self._call_llm(rendered_prompt)
        response = ClassificationParser.parse(raw_text)

        logger.debug(
            "classify_email | label=%s | prompt_len=%d | response_len=%d",
            response.label,
            len(rendered_prompt),
            len(raw_text),
        )

        return response

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        rendered_prompts: List[str],
    ) -> List[LLMClassificationResponse]:
        """
        Classify a list of emails, one rendered prompt per email.

        Calls ``classify_email()`` sequentially.  Failures on individual prompts
        raise ``InferenceError`` and halt the batch — callers should handle
        this or wrap each call individually if partial results are acceptable.

        Args:
            rendered_prompts: List of fully-rendered prompt strings.

        Returns:
            List of ``LLMClassificationResponse`` in the same order as input.

        Raises:
            InferenceError: On the first prompt that fails an LLM call.
        """
        results: List[LLMClassificationResponse] = []
        for idx, prompt in enumerate(rendered_prompts):
            try:
                result = self.classify_email(prompt)
            except InferenceError as exc:
                logger.error(
                    "classify_batch | failed at index %d/%d | %s",
                    idx,
                    len(rendered_prompts),
                    exc,
                )
                raise
            results.append(result)

        logger.info(
            "classify_batch | completed %d/%d",
            len(results),
            len(rendered_prompts),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Delegate to ``LLMClient.generate()`` and translate any ``LLMError``
        into ``InferenceError``.

        Args:
            prompt: Rendered prompt string.

        Returns:
            Raw response string from the provider.

        Raises:
            InferenceError: Wrapping the original ``LLMError``.
        """
        try:
            return self._client.generate(prompt, self._config)
        except LLMError as exc:
            preview = prompt[:120].replace("\n", " ")
            raise InferenceError(
                message=f"LLM call failed: {exc}",
                prompt_preview=preview,
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PhishingInferenceEngine("
            f"client={self._client!r}, "
            f"config={self._config!r})"
        )


# ---------------------------------------------------------------------------
# Module-level shorthand
# ---------------------------------------------------------------------------

def classify_email(
    rendered_prompt: str,
    client: LLMClient,
    request_config: Optional[LLMRequestConfig] = None,
) -> LLMClassificationResponse:
    """
    Classify a single email without instantiating the engine explicitly.

    Useful for one-off calls in tests or scripts.

    Args:
        rendered_prompt: Output of ``PromptRenderer.render_prompt()``.
        client:          Configured ``LLMClient`` instance.
        request_config:  Optional generation parameters.

    Returns:
        ``LLMClassificationResponse`` — never ``None``.

    Raises:
        InferenceError: If the LLM call fails.
    """
    engine = PhishingInferenceEngine(client, request_config)
    return engine.classify_email(rendered_prompt)
