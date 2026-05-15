"""
Tests for src/dataset/loader.py

Uses a synthetic in-memory CSV so no real dataset file is required.
Covers: column alias resolution, label normalisation, body filtering,
        optional-field filling, and the get_label_counts helper.
"""

from pathlib import Path
import sys
import textwrap
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.loader import load_dataset, get_label_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(content: str) -> str:
    """Write content to a temp CSV and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    f.write(textwrap.dedent(content).strip())
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_schema():
    """Full-field dataset: all five columns present, mixed label formats."""
    path = _write_csv("""
        sender,receiver,subject,body,label
        evil@phish.com,victim@corp.com,Urgent!,Click http://evil.com NOW,phishing
        hr@corp.com,staff@corp.com,Meeting,Please join at 3pm,safe
        attacker@x.com,user@y.com,Win $1000,You won! Claim here: http://scam.net,Phishing
        noreply@legit.com,me@co.com,Invoice,Your invoice is attached.,ham
    """)
    try:
        records = load_dataset(path)
        assert len(records) == 4, f"Expected 4 records, got {len(records)}"

        # Labels normalised
        assert records[0]["label"] == "PHISHING"
        assert records[1]["label"] == "SAFE"
        assert records[2]["label"] == "PHISHING"   # capitalised variant
        assert records[3]["label"] == "SAFE"        # "ham" -> SAFE

        # URL preserved (not stripped)
        assert "http://evil.com" in records[0]["body"]

        # Structure preserved
        assert records[0]["sender"] == "evil@phish.com"
        assert records[0]["subject"] == "Urgent!"

        print("test_full_schema PASSED")
    finally:
        os.unlink(path)


def test_minimalist_schema():
    """Minimalist dataset: only subject/body/label — sender and receiver added as empty."""
    path = _write_csv("""
        subject,body,label
        Verify now,Click to verify your account,1
        Q3 Report,Please find the report attached,0
    """)
    try:
        records = load_dataset(path)
        assert len(records) == 2

        # Integer labels
        assert records[0]["label"] == "PHISHING"
        assert records[1]["label"] == "SAFE"

        # Optional fields filled with empty strings
        assert records[0]["sender"] == ""
        assert records[0]["receiver"] == ""

        print("test_minimalist_schema PASSED")
    finally:
        os.unlink(path)


def test_column_aliases():
    """Alternative column names (from/to/text/class) are resolved correctly."""
    path = _write_csv("""
        from,to,text,class
        a@b.com,c@d.com,Reset your password immediately,spam
        e@f.com,g@h.com,Your order has shipped,legitimate
    """)
    try:
        records = load_dataset(path)
        assert len(records) == 2
        assert records[0]["sender"] == "a@b.com"
        assert records[0]["receiver"] == "c@d.com"
        assert records[0]["label"] == "PHISHING"
        assert records[1]["label"] == "SAFE"
        print("test_column_aliases PASSED")
    finally:
        os.unlink(path)


def test_empty_body_rows_dropped():
    """Rows with empty or whitespace-only body are dropped."""
    path = _write_csv("""
        body,label
        Valid email body,phishing
        ,safe
           ,phishing
        Another valid body,safe
    """)
    try:
        records = load_dataset(path)
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"
        assert records[0]["body"] == "Valid email body"
        assert records[1]["body"] == "Another valid body"
        print("test_empty_body_rows_dropped PASSED")
    finally:
        os.unlink(path)


def test_unknown_labels_dropped():
    """Rows with unrecognisable label values are silently dropped."""
    path = _write_csv("""
        body,label
        Good body,phishing
        Another body,unknown_label
        Third body,safe
    """)
    try:
        records = load_dataset(path)
        assert len(records) == 2, f"Expected 2, got {len(records)}"
        assert records[0]["label"] == "PHISHING"
        assert records[1]["label"] == "SAFE"
        print("test_unknown_labels_dropped PASSED")
    finally:
        os.unlink(path)


def test_content_preserved():
    """URLs, punctuation, and capitalisation are never modified."""
    raw_body = "URGENT!!! Click http://totally-real-bank.com/login?ref=abc&id=123 NOW!!!"
    path = _write_csv(f"""
        body,label
        "{raw_body}",phishing
    """)
    try:
        records = load_dataset(path)
        assert records[0]["body"] == raw_body, (
            f"Body was mutated:\n  expected: {raw_body}\n  got:      {records[0]['body']}"
        )
        print("test_content_preserved PASSED")
    finally:
        os.unlink(path)


def test_get_label_counts():
    """get_label_counts returns correct frequency dict."""
    records = [
        {"label": "PHISHING"},
        {"label": "SAFE"},
        {"label": "PHISHING"},
        {"label": "PHISHING"},
    ]
    counts = get_label_counts(records)
    assert counts == {"PHISHING": 3, "SAFE": 1}, f"Got: {counts}"
    print("test_get_label_counts PASSED")


def test_missing_body_column_raises():
    """ValueError raised when CSV has no recognisable body column."""
    path = _write_csv("""
        totally_wrong_col,label
        some text,phishing
    """)
    try:
        raised = False
        try:
            load_dataset(path)
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for missing body column"
        print("test_missing_body_column_raises PASSED")
    finally:
        os.unlink(path)


def test_file_not_found_raises():
    """FileNotFoundError raised for non-existent path."""
    raised = False
    try:
        load_dataset("data/raw/does_not_exist.csv")
    except FileNotFoundError:
        raised = True
    assert raised, "Expected FileNotFoundError"
    print("test_file_not_found_raises PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_full_schema()
    test_minimalist_schema()
    test_column_aliases()
    test_empty_body_rows_dropped()
    test_unknown_labels_dropped()
    test_content_preserved()
    test_get_label_counts()
    test_missing_body_column_raises()
    test_file_not_found_raises()
    print("\nAll loader tests passed.")
