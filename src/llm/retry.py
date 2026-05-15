"""
src/llm/retry.py
-----------------
Lightweight retry utilities for LLM API calls.

Public surface
--------------
``RetryConfig``          — dataclass controlling retry behaviour.
``retry_with_backoff()`` — wraps any callable with bounded exponential backoff.
``retryable``            — decorator form of ``retry_with_backoff``.

Backoff schedule
----------------
    delay(attempt) = min(base_delay × multiplier^(attempt - 1), max_delay)

    attempt 1 → base_delay                      (e.g. 1.0 s)
    attempt 2 → base_delay × multiplier         (e.g. 2.0 s)
    attempt 3 → base_delay × multiplier²        (e.g. 4.0 s)
    …         capped at max_delay               (e.g. 30.0 s)

Jitter is opt-in (off by default) so behaviour is deterministic and
reproducible in unit tests without mocking ``random``.

Retryable vs. non-retryable errors
------------------------------------
By default only ``LLMProviderError`` and ``LLMTimeoutError`` are retried.
``LLMConfigurationError`` and ``LLMUnsupportedProviderError`` represent
programmer mistakes (bad key, unknown provider) and are re-raised immediately
regardless of remaining attempts.

Callers may override ``retryable_exceptions`` in ``RetryConfig`` to change
which exception types trigger a retry.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar

from src.llm.client import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMTimeoutError,
    LLMUnsupportedProviderError,
)

logger = logging.getLogger(__name__)

# TypeVar for preserving the return type of the wrapped callable.
_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# Exception raised after all retries are exhausted
# ---------------------------------------------------------------------------

class RetryExhaustedError(Exception):
    """
    Raised when all retry attempts have been consumed.

    Attributes:
        attempts:    Total number of attempts made.
        last_error:  The exception from the final attempt.
    """

    def __init__(self, attempts: int, last_error: Exception) -> None:
        super().__init__(
            f"All {attempts} attempt(s) failed. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

# Exception types that will never be retried, regardless of RetryConfig.
# These indicate configuration issues, not transient failures.
_NON_RETRYABLE: Tuple[Type[Exception], ...] = (
    LLMConfigurationError,
    LLMUnsupportedProviderError,
)

# Default exception types that trigger a retry.
_DEFAULT_RETRYABLE: Tuple[Type[Exception], ...] = (
    LLMProviderError,
    LLMTimeoutError,
)


@dataclass
class RetryConfig:
    """
    Configuration for ``retry_with_backoff``.

    Attributes:
        max_attempts:         Total number of tries (including the first).
                              Must be >= 1.  Default: 3.
        base_delay:           Initial wait in seconds before the second attempt.
                              Default: 1.0 s.
        backoff_multiplier:   Factor by which the delay grows each attempt.
                              Default: 2.0  (doubles each time).
        max_delay:            Upper cap on wait time in seconds.
                              Default: 30.0 s.
        jitter:               If ``True``, adds a uniform random offset in
                              [0, base_delay] to each delay.  Disabled by
                              default for deterministic behaviour.
        retryable_exceptions: Tuple of exception types that should be retried.
                              Defaults to ``(LLMProviderError, LLMTimeoutError)``.

    Example::

        cfg = RetryConfig(max_attempts=4, base_delay=0.5, backoff_multiplier=3)
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: bool = False
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: _DEFAULT_RETRYABLE
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {self.max_attempts}."
            )
        if self.base_delay < 0:
            raise ValueError(
                f"base_delay must be >= 0, got {self.base_delay}."
            )
        if self.backoff_multiplier < 1:
            raise ValueError(
                f"backoff_multiplier must be >= 1, got {self.backoff_multiplier}."
            )

    def compute_delay(self, attempt: int) -> float:
        """
        Compute the sleep duration before *attempt* (1-indexed).

        Args:
            attempt: The attempt number that just failed (1 = first attempt).

        Returns:
            Seconds to wait before the next attempt.
        """
        raw = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        capped = min(raw, self.max_delay)
        if self.jitter:
            capped += random.uniform(0, self.base_delay)  # noqa: S311
        return capped


# ---------------------------------------------------------------------------
# Core retry function
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable[..., _R],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> _R:
    """
    Call *fn* with bounded exponential backoff on transient LLM errors.

    Behaviour
    ---------
    * Attempts ``config.max_attempts`` calls in total.
    * On a retryable exception, waits ``config.compute_delay(attempt)``
      seconds before the next attempt.
    * Non-retryable exceptions (``LLMConfigurationError``,
      ``LLMUnsupportedProviderError``) are re-raised immediately.
    * After exhausting all attempts, raises ``RetryExhaustedError`` which
      wraps the last exception.
    * Logs each failure at WARNING level and the final exhaustion at ERROR.

    Args:
        fn:     The callable to invoke (e.g. ``engine.classify_email``).
        *args:  Positional arguments forwarded to *fn*.
        config: ``RetryConfig`` controlling retry behaviour.
                Defaults to ``RetryConfig()`` (3 attempts, 1 s base delay).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        RetryExhaustedError: All attempts failed with retryable errors.
        Exception:           Non-retryable errors pass through immediately.

    Example::

        config = RetryConfig(max_attempts=4, base_delay=0.5)
        result = retry_with_backoff(engine.classify_email, prompt, config=config)
    """
    cfg = config or RetryConfig()
    last_exc: Optional[Exception] = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn(*args, **kwargs)

        except _NON_RETRYABLE as exc:
            # Configuration / programming errors — fail immediately.
            logger.error(
                "retry | non-retryable error on attempt %d/%d | %s: %s",
                attempt,
                cfg.max_attempts,
                type(exc).__name__,
                exc,
            )
            raise

        except cfg.retryable_exceptions as exc:
            last_exc = exc
            remaining = cfg.max_attempts - attempt

            if remaining == 0:
                # No attempts left — fall through to RetryExhaustedError.
                break

            delay = cfg.compute_delay(attempt)
            logger.warning(
                "retry | attempt %d/%d failed | %s: %s | retrying in %.2f s "
                "(%d attempt(s) remaining)",
                attempt,
                cfg.max_attempts,
                type(exc).__name__,
                exc,
                delay,
                remaining,
            )
            time.sleep(delay)

        except Exception as exc:
            # Unexpected exception type — propagate without retry.
            logger.error(
                "retry | unexpected exception on attempt %d/%d | %s: %s",
                attempt,
                cfg.max_attempts,
                type(exc).__name__,
                exc,
            )
            raise

    logger.error(
        "retry | all %d attempt(s) exhausted | last error: %s: %s",
        cfg.max_attempts,
        type(last_exc).__name__,
        last_exc,
    )
    raise RetryExhaustedError(attempts=cfg.max_attempts, last_error=last_exc)


# ---------------------------------------------------------------------------
# Decorator form
# ---------------------------------------------------------------------------

def retryable(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Decorator that applies ``retry_with_backoff`` to a function.

    Args:
        config: ``RetryConfig`` to use.  Defaults to ``RetryConfig()``.

    Returns:
        A decorator that wraps the target function with retry logic.

    Example::

        @retryable(RetryConfig(max_attempts=5, base_delay=2.0))
        def call_llm(prompt: str) -> str:
            return client.generate(prompt)

        # Also works with default config:
        @retryable()
        def call_llm(prompt: str) -> str:
            return client.generate(prompt)
    """
    def decorator(fn: Callable[..., _R]) -> Callable[..., _R]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> _R:
            return retry_with_backoff(fn, *args, config=config, **kwargs)
        return wrapper
    return decorator
