"""
src/llm/parser.py
-----------------
Deterministic parser for phishing classification LLM outputs.

Expected canonical format
--------------------------
    Label: PHISHING
    Reason: Urgency tactic and suspicious link detected.

Tolerances
----------
* Case-insensitive field names and label values.
* Leading / trailing whitespace on any line.
* Extra blank lines, preamble text, or trailing content.
* Alternate label spellings: safe / legitimate / ham / phishing / spam.
* Malformed or missing fields — never raises; always returns a valid
  ``LLMClassificationResponse`` with ``UNKNOWN`` label and the raw text
  as the reason so that the caller can decide how to handle the failure.

No side-effects, no I/O, no randomness.  Same input → same output.
"""

from __future__ import annotations

import re
from typing import Optional

from src.llm.schemas import LLMClassificationResponse


# ---------------------------------------------------------------------------
# Label normalisation map
# ---------------------------------------------------------------------------

# Maps every tolerated spelling (lowercased, stripped) to a canonical label.
_LABEL_NORMALISATION: dict[str, str] = {
    # Phishing variants
    "phishing":      "PHISHING",
    "phish":         "PHISHING",
    "spam":          "PHISHING",
    "malicious":     "PHISHING",
    "suspicious":    "PHISHING",
    "fraud":         "PHISHING",
    "fraudulent":    "PHISHING",

    # Legitimate variants
    "legitimate":    "LEGITIMATE",
    "legit":         "LEGITIMATE",
    "safe":          "LEGITIMATE",
    "ham":           "LEGITIMATE",
    "benign":        "LEGITIMATE",
    "not phishing":  "LEGITIMATE",
    "not_phishing":  "LEGITIMATE",
}

# Sentinel used when parsing fails or label is unrecognisable.
_UNKNOWN_LABEL = "UNKNOWN"

# ---------------------------------------------------------------------------
# Compiled regex patterns
# Re-compiled once at module load for performance across many calls.
# ---------------------------------------------------------------------------

# Matches:  "Label:  PHISHING"  /  "label : phishing"  /  "LABEL:phishing"
_LABEL_RE = re.compile(
    r"^\s*label\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches the first "Reason:" line; captures everything after the colon.
_REASON_FIRST_LINE_RE = re.compile(
    r"^\s*reason\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Used to locate the position of "Reason:" so we can capture multi-line bodies.
_REASON_HEADER_RE = re.compile(
    r"^\s*reason\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Detects lines that look like a new "Key: value" field (to stop multi-line
# reason capture before the next field begins).
_FIELD_LINE_RE = re.compile(
    r"^\s*\w[\w\s]*\s*:\s*\S",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

class ClassificationParser:
    """
    Stateless parser for phishing classification LLM outputs.

    All methods are static — instantiation is only for namespace
    organisation.  Use ``ClassificationParser.parse(raw)`` directly.
    """

    @staticmethod
    def parse(raw_response: str) -> LLMClassificationResponse:
        """
        Parse a raw LLM response into a structured classification result.

        Attempts to extract ``Label`` and ``Reason`` fields using
        progressively more lenient strategies:

        1. Regex extraction of ``Label:`` and ``Reason:`` fields.
        2. Multi-line reason body (captures continuation lines).
        3. Heuristic label scan across the entire response.
        4. Graceful fallback to ``UNKNOWN`` — never raises.

        Args:
            raw_response: Raw string returned by the LLM.

        Returns:
            ``LLMClassificationResponse`` with:
            - ``label``:        Normalised canonical label string.
            - ``reason``:       Extracted reason text (or fallback message).
            - ``raw_response``: The original unmodified response.
        """
        label = ClassificationParser._extract_label(raw_response)
        reason = ClassificationParser._extract_reason(raw_response)
        return LLMClassificationResponse(
            label=label,
            reason=reason,
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Label extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_label(text: str) -> str:
        """
        Extract and normalise the label from *text*.

        Strategy
        --------
        1. Look for an explicit ``Label:`` field.
        2. If absent or unrecognisable, scan every word in the full
           response for a known label keyword.
        3. Return ``UNKNOWN`` if nothing matches.
        """
        # Strategy 1: explicit "Label:" field
        match = _LABEL_RE.search(text)
        if match:
            raw_label = match.group(1).strip()
            normalised = ClassificationParser._normalise_label(raw_label)
            if normalised != _UNKNOWN_LABEL:
                return normalised

        # Strategy 2: heuristic keyword scan over the whole response
        return ClassificationParser._scan_for_label(text)

    @staticmethod
    def _normalise_label(raw: str) -> str:
        """
        Map *raw* label text to a canonical label via the normalisation table.

        Tries exact match first, then substring containment.

        Args:
            raw: Extracted label string (any casing, may contain spaces).

        Returns:
            Canonical label (``PHISHING`` / ``LEGITIMATE``) or ``UNKNOWN``.
        """
        key = raw.strip().lower()

        # Exact match
        if key in _LABEL_NORMALISATION:
            return _LABEL_NORMALISATION[key]

        # Substring containment (e.g. "it is phishing" → PHISHING)
        for token, canonical in _LABEL_NORMALISATION.items():
            if token in key:
                return canonical

        return _UNKNOWN_LABEL

    @staticmethod
    def _scan_for_label(text: str) -> str:
        """
        Scan *text* token by token for a recognisable label keyword.

        Used as a last resort when the ``Label:`` field is absent.
        Returns the first matching canonical label found, or ``UNKNOWN``.
        """
        lowered = text.lower()
        # Check multi-word keys first (e.g. "not phishing") to avoid a
        # shorter key ("phishing") stealing the match prematurely.
        for token in sorted(_LABEL_NORMALISATION, key=len, reverse=True):
            if token in lowered:
                return _LABEL_NORMALISATION[token]
        return _UNKNOWN_LABEL

    # ------------------------------------------------------------------
    # Reason extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reason(text: str) -> str:
        """
        Extract the reason text from *text*.

        Captures the first ``Reason:`` field, including any continuation
        lines that are not themselves new ``Key:`` fields.

        Falls back to ``"No reason provided."`` when the field is absent.
        """
        header_match = _REASON_HEADER_RE.search(text)
        if not header_match:
            return "No reason provided."

        # Slice the text starting from the character after "Reason: "
        body_start = header_match.end()
        remainder = text[body_start:]

        reason_lines: list[str] = []
        for line in remainder.splitlines():
            stripped = line.strip()

            # Stop when we hit the next "Key: value" field
            if reason_lines and _FIELD_LINE_RE.match(line):
                break

            reason_lines.append(stripped)

        reason = " ".join(part for part in reason_lines if part)
        return reason if reason else "No reason provided."


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def parse_classification_response(raw_text: str) -> LLMClassificationResponse:
    """
    Module-level shorthand for ``ClassificationParser.parse()``.

    Equivalent to ``ClassificationParser.parse(raw_text)``.

    Args:
        raw_text: Raw string returned by the LLM.

    Returns:
        ``LLMClassificationResponse`` — never raises.
    """
    return ClassificationParser.parse(raw_text)
