from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMClassificationResponse:
    """
    Parsed phishing classification response.
    """

    label: str
    reason: str
    raw_response: str


@dataclass
class LLMRefinementResponse:
    """
    Parsed refinement generation response.
    """

    refined_content: str
    raw_response: str


@dataclass
class LLMRequestConfig:
    """
    Configuration for deterministic LLM calls.
    """

    temperature: float = 0.1
    max_tokens: int = 200