"""
src/critic/error_analysis.py
------------------------------
Deterministic error analysis for phishing prompt optimization.

Analyzes prediction failures to extract false positives, false negatives,
and apply lightweight phishing heuristics without relying on LLMs or
external NLP frameworks.

Behavior is strictly deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, Optional

from src.core.constants import LABEL_PHISHING, LABEL_SAFE
from src.core.prompt_state import EmailInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------

class FailureCategory(Enum):
    """Categories of prediction failures."""
    FALSE_POSITIVE = "false_positive"  # Predicted PHISHING, actually SAFE
    FALSE_NEGATIVE = "false_negative"  # Predicted SAFE, actually PHISHING


@dataclass(frozen=True)
class FailureCase:
    """
    A single failed prediction.

    Attributes:
        email:              The email content that failed.
        true_label:         The ground truth label.
        predicted_label:    The incorrect prediction.
        category:           Whether this was a false positive or negative.
        heuristics_matched: List of heuristic tags (e.g., 'urgency') triggered by the email.
    """
    email: Union[str, EmailInput, Dict[str, str]]
    true_label: str
    predicted_label: str
    category: FailureCategory
    heuristics_matched: List[str]


@dataclass
class FailureAnalysisReport:
    """
    Aggregate summary of all prediction failures in a batch.
    """
    total_failures: int
    false_positives: int
    false_negatives: int
    dominant_category: Optional[FailureCategory]
    cases: List[FailureCase]
    heuristics_summary: Dict[str, int]


# ---------------------------------------------------------------------------
# Heuristic Keyword Sets
# ---------------------------------------------------------------------------

_URGENCY_KEYWORDS = {
    "urgent", "immediately", "action required", "act now", "deadline",
    "within 24 hours", "suspend", "final notice"
}

_CREDENTIAL_KEYWORDS = {
    "password", "login", "credentials", "verify your account", "click here",
    "sign in", "update payment"
}

_IMPERSONATION_KEYWORDS = {
    "account suspended", "security alert", "unusual activity", 
    "customer support", "admin", "administrator"
}


# ---------------------------------------------------------------------------
# ErrorAnalyzer
# ---------------------------------------------------------------------------

class ErrorAnalyzer:
    """
    Analyzes prediction failures deterministically using lightweight heuristics.
    No LLMs, randomness, or external dependencies are used.
    """

    @staticmethod
    def analyze_failures(
        predictions: List[str],
        labels: List[str],
        emails: List[Union[str, EmailInput, Dict[str, str]]]
    ) -> FailureAnalysisReport:
        """
        Analyze a batch of predictions to extract failures and heuristics.

        Args:
            predictions: Model predictions (e.g. ["PHISHING", "SAFE"]).
            labels: Ground truth labels.
            emails: The corresponding email contents.

        Returns:
            A populated FailureAnalysisReport.
        """
        if len(predictions) != len(labels) or len(labels) != len(emails):
            raise ValueError("predictions, labels, and emails must have the same length.")

        cases: List[FailureCase] = []
        fp_count = 0
        fn_count = 0
        heuristics_counts: Dict[str, int] = {
            "urgency": 0,
            "credential_request": 0,
            "impersonation": 0
        }

        for pred, label, email in zip(predictions, labels, emails):
            if pred == label:
                continue

            # Determine category
            if pred == LABEL_PHISHING and label == LABEL_SAFE:
                cat = FailureCategory.FALSE_POSITIVE
                fp_count += 1
            elif pred == LABEL_SAFE and label == LABEL_PHISHING:
                cat = FailureCategory.FALSE_NEGATIVE
                fn_count += 1
            else:
                continue  # Unknown label combinations are ignored

            # Apply heuristics
            matched = ErrorAnalyzer._apply_heuristics(email)
            for h in matched:
                heuristics_counts[h] += 1

            cases.append(FailureCase(
                email=email,
                true_label=label,
                predicted_label=pred,
                category=cat,
                heuristics_matched=matched
            ))

        total = fp_count + fn_count
        dominant = None
        if fp_count > fn_count:
            dominant = FailureCategory.FALSE_POSITIVE
        elif fn_count > fp_count:
            dominant = FailureCategory.FALSE_NEGATIVE
        # Ties leave dominant_category as None

        return FailureAnalysisReport(
            total_failures=total,
            false_positives=fp_count,
            false_negatives=fn_count,
            dominant_category=dominant,
            cases=cases,
            heuristics_summary=heuristics_counts
        )

    @staticmethod
    def get_top_failure_cases(report: FailureAnalysisReport, limit: int = 5) -> List[FailureCase]:
        """
        Return up to `limit` failure cases, prioritizing false negatives, then by number
        of heuristics matched.
        """
        # Sort key:
        # 1. False negatives first (0) vs False positives (1)
        # 2. More heuristics matched (negative len to sort descending)
        def _sort_key(case: FailureCase):
            is_fp = 1 if case.category == FailureCategory.FALSE_POSITIVE else 0
            return (is_fp, -len(case.heuristics_matched))

        sorted_cases = sorted(report.cases, key=_sort_key)
        return sorted_cases[:limit]

    @staticmethod
    def summarize_failures(report: FailureAnalysisReport) -> str:
        """
        Provide a concise, human-readable summary of the failure report.
        """
        if report.total_failures == 0:
            return "No prediction failures."

        dom_str = report.dominant_category.value if report.dominant_category else "none (tie)"
        lines = [
            f"Total Failures: {report.total_failures}",
            f"FN: {report.false_negatives} | FP: {report.false_positives}",
            f"Dominant Category: {dom_str}",
            f"Heuristics Triggered: Urgency ({report.heuristics_summary['urgency']}), "
            f"Credentials ({report.heuristics_summary['credential_request']}), "
            f"Impersonation ({report.heuristics_summary['impersonation']})"
        ]
        return " - ".join(lines)

    @staticmethod
    def _apply_heuristics(email: Union[str, EmailInput, Dict[str, str]]) -> List[str]:
        """
        Check email content against known phishing heuristics.
        """
        text = ""
        if isinstance(email, str):
            text = email.lower()
        elif isinstance(email, EmailInput):
            text = f"{email.subject} {email.body}".lower()
        elif isinstance(email, dict):
            text = f"{email.get('subject', '')} {email.get('body', '')}".lower()

        matched = []
        if any(kw in text for kw in _URGENCY_KEYWORDS):
            matched.append("urgency")
        if any(kw in text for kw in _CREDENTIAL_KEYWORDS):
            matched.append("credential_request")
        if any(kw in text for kw in _IMPERSONATION_KEYWORDS):
            matched.append("impersonation")

        # Deterministic order
        return sorted(matched)
