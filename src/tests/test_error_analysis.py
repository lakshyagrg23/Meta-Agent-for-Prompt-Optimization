"""
Tests for src/critic/error_analysis.py

Covers:
- False negative and false positive extraction
- Dominant category detection
- Heuristics extraction (urgency, credential_request, impersonation)
- Helper methods: get_top_failure_cases, summarize_failures
- Deterministic behavior
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.critic.error_analysis import (
    ErrorAnalyzer,
    FailureCategory,
    FailureCase,
    FailureAnalysisReport
)
from src.core.constants import LABEL_PHISHING, LABEL_SAFE
from src.core.prompt_state import EmailInput

def test_extracts_false_positives_and_negatives():
    predictions = [LABEL_PHISHING, LABEL_SAFE, LABEL_PHISHING, LABEL_SAFE]
    labels =      [LABEL_SAFE,     LABEL_PHISHING, LABEL_PHISHING, LABEL_SAFE]
    emails = ["fp email", "fn email", "tp email", "tn email"]

    report = ErrorAnalyzer.analyze_failures(predictions, labels, emails)

    assert report.total_failures == 2
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.dominant_category is None  # Tie
    assert len(report.cases) == 2

    # Verify cases
    fp_case = next(c for c in report.cases if c.category == FailureCategory.FALSE_POSITIVE)
    assert fp_case.email == "fp email"

    fn_case = next(c for c in report.cases if c.category == FailureCategory.FALSE_NEGATIVE)
    assert fn_case.email == "fn email"
    print("test_extracts_false_positives_and_negatives PASSED")

def test_dominant_category():
    # 2 FPs, 1 FN
    predictions = [LABEL_PHISHING, LABEL_PHISHING, LABEL_SAFE]
    labels =      [LABEL_SAFE,     LABEL_SAFE,     LABEL_PHISHING]
    emails = ["fp1", "fp2", "fn1"]

    report = ErrorAnalyzer.analyze_failures(predictions, labels, emails)
    assert report.dominant_category == FailureCategory.FALSE_POSITIVE
    print("test_dominant_category PASSED")

def test_apply_heuristics_urgency():
    # Email contains 'urgent'
    email = EmailInput(sender="", receiver="", subject="Action Required", body="Please reply urgent.")
    matched = ErrorAnalyzer._apply_heuristics(email)
    assert "urgency" in matched
    assert "credential_request" not in matched
    print("test_apply_heuristics_urgency PASSED")

def test_apply_heuristics_multiple():
    email = "Account suspended! Please verify your account immediately."
    matched = ErrorAnalyzer._apply_heuristics(email)
    assert "urgency" in matched
    assert "credential_request" in matched
    assert "impersonation" in matched
    assert len(matched) == 3
    print("test_apply_heuristics_multiple PASSED")

def test_analyze_failures_heuristic_counts():
    predictions = [LABEL_PHISHING, LABEL_SAFE]
    labels =      [LABEL_SAFE,     LABEL_PHISHING]
    emails = [
        "urgent: verify your account",  # urgency, credential_request
        "normal email"                  # none
    ]
    report = ErrorAnalyzer.analyze_failures(predictions, labels, emails)
    assert report.heuristics_summary["urgency"] == 1
    assert report.heuristics_summary["credential_request"] == 1
    assert report.heuristics_summary["impersonation"] == 0
    print("test_analyze_failures_heuristic_counts PASSED")

def test_get_top_failure_cases_sorting():
    # Should sort FNs first, then by heuristic count
    cases = [
        FailureCase("fp 1 heuristic", LABEL_SAFE, LABEL_PHISHING, FailureCategory.FALSE_POSITIVE, ["urgency"]),
        FailureCase("fn 2 heuristics", LABEL_PHISHING, LABEL_SAFE, FailureCategory.FALSE_NEGATIVE, ["urgency", "credential_request"]),
        FailureCase("fn 0 heuristics", LABEL_PHISHING, LABEL_SAFE, FailureCategory.FALSE_NEGATIVE, []),
    ]
    report = FailureAnalysisReport(3, 1, 2, None, cases, {})
    
    top = ErrorAnalyzer.get_top_failure_cases(report, limit=2)
    assert len(top) == 2
    assert top[0].email == "fn 2 heuristics"  # FN + most heuristics
    assert top[1].email == "fn 0 heuristics"  # FN next
    print("test_get_top_failure_cases_sorting PASSED")

def test_summarize_failures_empty():
    report = FailureAnalysisReport(0, 0, 0, None, [], {"urgency": 0, "credential_request": 0, "impersonation": 0})
    summary = ErrorAnalyzer.summarize_failures(report)
    assert summary == "No prediction failures."
    print("test_summarize_failures_empty PASSED")

def test_summarize_failures_populated():
    report = FailureAnalysisReport(
        total_failures=2,
        false_positives=1,
        false_negatives=1,
        dominant_category=None,
        cases=[],
        heuristics_summary={"urgency": 1, "credential_request": 0, "impersonation": 2}
    )
    summary = ErrorAnalyzer.summarize_failures(report)
    assert "Total Failures: 2" in summary
    assert "Dominant Category: none (tie)" in summary
    assert "Urgency (1)" in summary
    print("test_summarize_failures_populated PASSED")

def test_mismatched_lengths():
    raised = False
    try:
        ErrorAnalyzer.analyze_failures([LABEL_PHISHING], [LABEL_SAFE, LABEL_PHISHING], [])
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for mismatched input lengths"
    print("test_mismatched_lengths PASSED")

if __name__ == "__main__":
    test_extracts_false_positives_and_negatives()
    test_dominant_category()
    test_apply_heuristics_urgency()
    test_apply_heuristics_multiple()
    test_analyze_failures_heuristic_counts()
    test_get_top_failure_cases_sorting()
    test_summarize_failures_empty()
    test_summarize_failures_populated()
    test_mismatched_lengths()
    print("\nAll error_analysis tests passed.")
