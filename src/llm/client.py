"""
src/llm/client.py
-----------------
Provider-agnostic LLM client.

Responsibilities
----------------
* Expose a single call surface: ``LLMClient.generate(prompt, config) -> str``
* Route calls to the correct backend based on a ``provider`` tag.
* Enforce determinism settings (temperature, seed where supported).
* Raise typed, catchable exceptions — no silent swallowing of errors.
* Return raw text only.  All parsing lives upstream in ``inference.py``.

Supported providers (set ``provider`` in ``LLMClientConfig``)
--------------------------------------------------------------
* ``"openai"``  — OpenAI Chat Completions API  (gpt-4o, gpt-4-turbo, …)
* ``"gemini"``  — Google Gemini API            (gemini-1.5-pro, …)
* ``"local"``   — Local model via OpenAI-compatible REST endpoint

Adding a new provider
---------------------
1. Subclass ``_BaseBackend`` and implement ``call()``.
2. Register the subclass in ``LLMClient._REGISTRY``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Type

from src.llm.schemas import LLMRequestConfig


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM client errors."""


class LLMProviderError(LLMError):
    """Raised when the upstream provider returns an error response."""


class LLMTimeoutError(LLMError):
    """Raised when a provider call exceeds its deadline."""


class LLMConfigurationError(LLMError):
    """Raised for missing API keys or invalid configuration."""


class LLMUnsupportedProviderError(LLMError):
    """Raised when an unknown provider string is requested."""


# ---------------------------------------------------------------------------
# Client-level configuration
# ---------------------------------------------------------------------------

class LLMClientConfig:
    """
    Top-level configuration for ``LLMClient``.

    Attributes:
        provider:    One of ``"openai"``, ``"gemini"``, ``"local"``.
        model:       Provider-specific model name.
        api_key:     API key. Defaults to the relevant env var if omitted.
        base_url:    Override base URL (required for ``"local"`` provider).
        timeout:     Per-request timeout in seconds. Default: 60.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider.lower().strip()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class _BaseBackend(ABC):
    """
    Internal contract that every provider backend must satisfy.

    Subclasses receive a frozen ``LLMClientConfig`` at construction time
    and must implement ``call()`` which performs the actual network call.
    """

    def __init__(self, client_config: LLMClientConfig) -> None:
        self._cfg = client_config

    @abstractmethod
    def call(self, prompt: str, request_config: LLMRequestConfig) -> str:
        """
        Send *prompt* to the provider and return the raw response text.

        Args:
            prompt:         The fully-rendered prompt string.
            request_config: Per-call generation parameters.

        Returns:
            Raw response text from the model (stripped, no extra whitespace).

        Raises:
            LLMProviderError:   Provider returned an error.
            LLMTimeoutError:    Request exceeded ``LLMClientConfig.timeout``.
            LLMConfigurationError: Missing credentials or bad config.
        """


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class _OpenAIBackend(_BaseBackend):
    """
    OpenAI Chat Completions backend.

    Reads ``OPENAI_API_KEY`` from the environment when no ``api_key`` is
    supplied in ``LLMClientConfig``.

    Determinism notes:
        * ``temperature=0`` makes generation greedy and reproducible.
        * OpenAI does not expose a ``seed`` parameter on all model versions;
          ``seed=0`` is sent when available and silently ignored otherwise.
    """

    def __init__(self, client_config: LLMClientConfig) -> None:
        super().__init__(client_config)
        self._client = self._build_client()

    def _build_client(self):
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise LLMConfigurationError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        api_key = self._cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key "
                "to LLMClientConfig."
            )

        kwargs = {"api_key": api_key, "timeout": self._cfg.timeout}
        if self._cfg.base_url:
            kwargs["base_url"] = self._cfg.base_url

        return openai.OpenAI(**kwargs)

    def call(self, prompt: str, request_config: LLMRequestConfig) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=request_config.temperature,
                max_tokens=request_config.max_tokens,
                seed=0,  # ignored gracefully on models that don't support it
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            _reraise_as_llm_error(exc, provider="openai")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

class _GeminiBackend(_BaseBackend):
    """
    Google Gemini backend via ``google-generativeai``.

    Reads ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) from the environment
    when no ``api_key`` is supplied.

    Determinism notes:
        * ``temperature=0.0`` makes sampling near-greedy.
        * Gemini exposes ``candidate_count`` — we always request exactly one.
    """

    def __init__(self, client_config: LLMClientConfig) -> None:
        super().__init__(client_config)
        self._model = self._build_model()

    def _build_model(self):
        try:
            import google.generativeai as genai  # type: ignore[import]
        except ImportError as exc:
            raise LLMConfigurationError(
                "google-generativeai package is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        api_key = (
            self._cfg.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise LLMConfigurationError(
                "Gemini API key not found. Set GEMINI_API_KEY or pass api_key "
                "to LLMClientConfig."
            )

        genai.configure(api_key=api_key)
        return genai.GenerativeModel(self._cfg.model)

    def call(self, prompt: str, request_config: LLMRequestConfig) -> str:
        try:
            import google.generativeai as genai  # type: ignore[import]

            generation_config = genai.GenerationConfig(
                temperature=request_config.temperature,
                max_output_tokens=request_config.max_tokens,
                candidate_count=1,
            )
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return response.text.strip()

        except Exception as exc:
            _reraise_as_llm_error(exc, provider="gemini")


# ---------------------------------------------------------------------------
# Local backend  (OpenAI-compatible REST endpoint)
# ---------------------------------------------------------------------------

class _LocalBackend(_BaseBackend):
    """
    Local model backend that speaks the OpenAI Chat Completions protocol.

    Compatible with: Ollama, LM Studio, vLLM, llama.cpp server.

    ``base_url`` is **required** (e.g. ``http://localhost:11434/v1``).
    ``api_key`` defaults to ``"local"`` because most local servers
    do not perform authentication.

    Determinism notes:
        * Pass ``temperature=0`` and a fixed ``seed`` for reproducibility.
        * Actual behaviour depends on the server implementation.
    """

    def __init__(self, client_config: LLMClientConfig) -> None:
        super().__init__(client_config)
        self._client = self._build_client()

    def _build_client(self):
        if not self._cfg.base_url:
            raise LLMConfigurationError(
                "base_url is required for the 'local' provider. "
                "Example: http://localhost:11434/v1"
            )

        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise LLMConfigurationError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        return openai.OpenAI(
            api_key=self._cfg.api_key or "local",
            base_url=self._cfg.base_url,
            timeout=self._cfg.timeout,
        )

    def call(self, prompt: str, request_config: LLMRequestConfig) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=request_config.temperature,
                max_tokens=request_config.max_tokens,
                seed=0,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            _reraise_as_llm_error(exc, provider="local")


# ---------------------------------------------------------------------------
# Error translation helper
# ---------------------------------------------------------------------------

def _reraise_as_llm_error(exc: Exception, provider: str) -> None:
    """
    Translate provider-specific exceptions into the LLM error hierarchy.

    Always raises — the return type is ``None`` only to satisfy type checkers
    at call sites written as ``_reraise_as_llm_error(exc, ...)``.
    """
    name = type(exc).__name__.lower()

    if "timeout" in name or "timeout" in str(exc).lower():
        raise LLMTimeoutError(
            f"[{provider}] Request timed out: {exc}"
        ) from exc

    if any(k in name for k in ("auth", "permission", "apikey", "credential")):
        raise LLMConfigurationError(
            f"[{provider}] Authentication error: {exc}"
        ) from exc

    raise LLMProviderError(
        f"[{provider}] Provider returned an error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Provider-agnostic LLM client.

    Usage::

        config = LLMClientConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)

        request = LLMRequestConfig(temperature=0.0, max_tokens=300)
        text = client.generate("Classify this email: ...", request)

    The client is stateless after construction — the same instance can be
    called concurrently from multiple threads (assuming the underlying SDK
    is thread-safe, which the OpenAI Python SDK is).

    Adding a provider
    -----------------
    Register a ``_BaseBackend`` subclass in ``_REGISTRY``::

        LLMClient._REGISTRY["myprovider"] = _MyBackend
    """

    _REGISTRY: ClassVar[Dict[str, Type[_BaseBackend]]] = {
        "openai": _OpenAIBackend,
        "gemini": _GeminiBackend,
        "local":  _LocalBackend,
    }

    def __init__(self, client_config: LLMClientConfig) -> None:
        """
        Initialise the client and establish the provider backend.

        Args:
            client_config: Provider, model, credentials, and timeout.

        Raises:
            LLMUnsupportedProviderError: If ``provider`` is not registered.
            LLMConfigurationError:       If credentials are missing/invalid.
        """
        provider = client_config.provider
        backend_cls = self._REGISTRY.get(provider)
        if backend_cls is None:
            supported = ", ".join(sorted(self._REGISTRY))
            raise LLMUnsupportedProviderError(
                f"Unknown provider '{provider}'. "
                f"Supported providers: {supported}."
            )
        self._backend: _BaseBackend = backend_cls(client_config)
        self._model = client_config.model
        self._provider = provider

    def generate(
        self,
        prompt: str,
        config: Optional[LLMRequestConfig] = None,
    ) -> str:
        """
        Send *prompt* to the configured LLM and return raw text.

        Args:
            prompt: The fully-rendered prompt string to send.
            config: Per-call generation parameters.  Defaults to
                    ``LLMRequestConfig()`` (temperature=0.1, max_tokens=200).

        Returns:
            Raw response text, stripped of leading/trailing whitespace.
            No parsing, no validation — that is the caller's responsibility.

        Raises:
            LLMProviderError:      Provider call failed.
            LLMTimeoutError:       Request timed out.
            LLMConfigurationError: Bad credentials or config.
        """
        if config is None:
            config = LLMRequestConfig()
        return self._backend.call(prompt, config)

    # ------------------------------------------------------------------
    # Convenience repr for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LLMClient(provider={self._provider!r}, model={self._model!r})"
