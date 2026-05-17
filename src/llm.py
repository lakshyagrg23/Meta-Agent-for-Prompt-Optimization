"""
llm.py — LLM client wrapper with retry logic and tiktoken-based token counter.

Provider: OpenAI-compatible API (openai >= 1.0).
Supports any model accessible via the OpenAI SDK (GPT-4o, GPT-4o-mini, etc.).
"""

import os
import time
import json
import tiktoken
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

from src.config import (
    MODEL_NAME,
    LLM_MAX_RETRIES,
    LLM_RETRY_BACKOFF,
    TIKTOKEN_ENCODING,
    OLLAMA_NUM_CTX,
)

# ── Client singleton ───────────────────────────────────────────────────────────
_client: OpenAI | None = None


def init_client(api_key: str | None = None, base_url: str | None = None) -> None:
    """
    Initialise the OpenAI client for local Ollama.
    """
    global _client
    _client = OpenAI(
        api_key=api_key or "ollama",
        base_url=base_url or "http://localhost:11434/v1",
    )


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        init_client()
    return _client


# ── Exceptions ─────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when all retries for an LLM call are exhausted."""
    pass


# ── Core call ──────────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    model: str = MODEL_NAME,
) -> str:
    """
    Call the LLM with retry on transient errors.

    Returns:
        The raw assistant message content string.
    Raises:
        LLMError: if all retries are exhausted.
    """
    client = _get_client()
    delay = LLM_RETRY_BACKOFF

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
            )
            return response.choices[0].message.content or ""

        except RateLimitError as e:
            if attempt == LLM_MAX_RETRIES:
                raise LLMError(f"Rate limit exceeded after {LLM_MAX_RETRIES} retries: {e}") from e
            print(f"  [llm] Rate limit hit (attempt {attempt}/{LLM_MAX_RETRIES}), "
                  f"waiting {delay:.1f}s …")
            time.sleep(delay)
            delay *= 2

        except APIConnectionError as e:
            if attempt == LLM_MAX_RETRIES:
                raise LLMError(f"Connection error after {LLM_MAX_RETRIES} retries: {e}") from e
            print(f"  [llm] Connection error (attempt {attempt}/{LLM_MAX_RETRIES}), "
                  f"waiting {delay:.1f}s …")
            time.sleep(delay)
            delay *= 2

        except APIStatusError as e:
            # 5xx are retriable; 4xx (except 429) are not
            if e.status_code >= 500:
                if attempt == LLM_MAX_RETRIES:
                    raise LLMError(f"Server error {e.status_code} after {LLM_MAX_RETRIES} retries: {e}") from e
                print(f"  [llm] Server error {e.status_code} (attempt {attempt}/{LLM_MAX_RETRIES}), "
                      f"waiting {delay:.1f}s …")
                time.sleep(delay)
                delay *= 2
            else:
                raise LLMError(f"API error {e.status_code}: {e}") from e

    raise LLMError("Unreachable")  # pragma: no cover


# ── Token counting ─────────────────────────────────────────────────────────────

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in `text` using cl100k_base encoding."""
    return len(_get_encoder().encode(text))


# ── JSON extraction helper (shared by agents.py) ──────────────────────────────

def extract_json(raw: str) -> dict:
    """
    Extract a JSON object from a raw LLM response string.
    Handles markdown code fences (```json ... ```) and bare JSON.

    Returns:
        Parsed dict.
    Raises:
        ValueError: if no valid JSON found.
    """
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence line and closing fence
        lines = text.splitlines()
        # find first non-fence line
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines)
        for i in range(len(lines) - 1, start - 1, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response:\n{raw[:300]}")
