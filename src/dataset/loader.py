"""
src/dataset/loader.py
---------------------
Phishing email dataset loader.

Loads CSV-format Kaggle phishing datasets and normalises them into a
consistent list of structured email records, each represented as a
plain ``dict`` with exactly these keys::

    {
        "sender":   str,   # may be empty string if dataset lacks the field
        "receiver": str,   # may be empty string if dataset lacks the field
        "subject":  str,   # may be empty string if dataset lacks the field
        "body":     str,   # always non-empty after cleaning
        "label":    str,   # exactly "SAFE" or "PHISHING"
    }

Design constraints
------------------
* **Deterministic** — same file always produces the same list in the same
  order (rows are never shuffled here; shuffling belongs in the sampler).
* **Non-destructive** — URLs, punctuation, and capitalisation are preserved
  exactly as they appear in the source; no lowercasing or stripping of
  semantic content is performed.
* **Structured** — email fields are never merged into a single text blob.
  The structure is preserved for downstream structured prompt rendering.
* **Pandas-only dependency** — no heavy NLP libraries required at load time.

Supported dataset schemas
--------------------------
The loader uses a column-mapping layer to handle the most common Kaggle
phishing email dataset variants:

1. Full-field datasets (CEAS / AVN / curated collections):
   ``sender``, ``receiver``, ``subject``, ``body``, ``label``

2. Minimalist datasets (Enron / Ling subsets):
   ``subject``, ``body``, ``label``  (sender/receiver absent)

3. Legacy binary-label datasets:
   ``label`` values are integers (0 = safe, 1 = phishing)

Column name aliases are resolved via :data:`COLUMN_ALIASES`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.core.constants import LABEL_PHISHING, LABEL_SAFE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label normalisation map
# ---------------------------------------------------------------------------

#: Maps every known raw label variant to a canonical string.
#: Keys are lower-cased before lookup; add entries here for new datasets.
LABEL_MAP: Dict[str, str] = {
    # String variants
    "phishing":   LABEL_PHISHING,
    "spam":       LABEL_PHISHING,
    "malicious":  LABEL_PHISHING,
    "1":          LABEL_PHISHING,
    # Safe / legitimate variants
    "safe":       LABEL_SAFE,
    "ham":        LABEL_SAFE,
    "legitimate": LABEL_SAFE,
    "benign":     LABEL_SAFE,
    "0":          LABEL_SAFE,
}


# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

#: Maps canonical field names to lists of alternative column names found
#: in the wild.  The first matching alias in the DataFrame is used.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "sender":   ["sender", "from", "from_email", "from_address", "email_from"],
    "receiver": ["receiver", "to", "to_email", "to_address", "email_to", "recipient"],
    "subject":  ["subject", "email_subject", "mail_subject"],
    "body":     ["body", "email_body", "mail_body", "text", "content", "message", "email_text"],
    "label":    ["label", "class", "category", "type", "target", "is_phishing"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str | Path) -> List[Dict[str, str]]:
    """
    Load a phishing email CSV dataset and return a cleaned, normalised list.

    Each record in the returned list is a ``dict`` with keys
    ``sender``, ``receiver``, ``subject``, ``body``, and ``label``.
    The structure is preserved — fields are never merged.

    Cleaning steps applied (in order):

    1. Resolve column aliases to canonical field names.
    2. Fill optional fields (``sender``, ``receiver``, ``subject``) with
       empty strings where absent or null.
    3. Drop rows where ``body`` is null or empty after stripping whitespace.
    4. Drop rows where ``label`` cannot be mapped to ``SAFE`` or ``PHISHING``.
    5. Normalise labels via :data:`LABEL_MAP`.

    Phishing-relevant content (URLs, punctuation, capitalisation) is
    **not** modified during loading.

    Args:
        csv_path: Absolute or relative path to the source CSV file.

    Returns:
        List of normalised email record dicts, one per valid row.
        Order matches the original CSV row order.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If the CSV contains no column that maps to ``body``
                    or ``label`` via :data:`COLUMN_ALIASES`.

    Example::

        records = load_dataset("data/raw/phishing_emails.csv")
        print(records[0])
        # {
        #     "sender":   "attacker@evil.com",
        #     "receiver": "victim@corp.com",
        #     "subject":  "Urgent: verify your account",
        #     "body":     "Click here immediately...",
        #     "label":    "PHISHING",
        # }
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    logger.info("Raw shape: %s rows x %s cols", *df.shape)

    df = _resolve_columns(df, csv_path)
    df = _clean(df)

    records = df[["sender", "receiver", "subject", "body", "label"]].to_dict(
        orient="records"
    )
    logger.info("Loaded %d valid records from %s", len(records), csv_path.name)
    return records


def get_label_counts(records: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Return a frequency count of labels in *records*.

    Useful for a quick sanity-check after loading.

    Args:
        records: List of normalised email record dicts as returned by
                 :func:`load_dataset`.

    Returns:
        Dict mapping each unique label to its row count, e.g.
        ``{"PHISHING": 4825, "SAFE": 3672}``.

    Example::

        counts = get_label_counts(records)
        print(counts)  # {"PHISHING": 4825, "SAFE": 3672}
    """
    counts: Dict[str, int] = {}
    for rec in records:
        lbl = rec["label"]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Rename DataFrame columns to canonical names using :data:`COLUMN_ALIASES`.

    Optional fields (``sender``, ``receiver``, ``subject``) that have no
    matching column in *df* are added as empty-string columns so that
    downstream code always sees all five fields.

    Args:
        df:   Raw DataFrame as loaded from CSV.
        path: Source file path (used only in error messages).

    Returns:
        DataFrame with columns renamed/added to canonical names.

    Raises:
        ValueError: If neither ``body`` nor ``label`` can be resolved.
    """
    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break

    df = df.rename(columns=rename_map)

    # Validate required columns
    for required in ("body", "label"):
        if required not in df.columns:
            available = list(df.columns)
            raise ValueError(
                f"Could not find a column for '{required}' in {path.name}. "
                f"Available columns: {available}. "
                f"Expected one of: {COLUMN_ALIASES[required]}"
            )

    # Add optional columns as empty strings if missing
    for optional in ("sender", "receiver", "subject"):
        if optional not in df.columns:
            logger.debug(
                "Column '%s' not found in %s — filling with empty strings.",
                optional,
                path.name,
            )
            df[optional] = ""

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules to the resolved DataFrame.

    Rules applied in order:

    1. Strip leading/trailing whitespace from all text fields.
    2. Drop rows where ``body`` is empty or whitespace-only.
    3. Normalise ``label`` values via :data:`LABEL_MAP`; drop rows whose
       label cannot be mapped.
    4. Fill null values in optional fields with empty strings.

    Phishing-relevant content (URLs, punctuation, capitalisation) within
    field values is **not** modified.

    Args:
        df: DataFrame with canonical column names.

    Returns:
        Cleaned DataFrame with only valid rows remaining.
    """
    original_count = len(df)

    # 1. Strip whitespace from all string fields (not the body content —
    #    only leading/trailing padding, never internal content).
    for col in ("sender", "receiver", "subject", "body", "label"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    # 2. Drop rows with missing or empty body.
    body_missing = df["body"].eq("") | df["body"].isna()
    if body_missing.any():
        logger.warning(
            "Dropping %d rows with empty/null body.", body_missing.sum()
        )
    df = df[~body_missing].copy()

    # 3. Normalise labels; drop unrecognised values.
    df["label"] = df["label"].str.lower().map(LABEL_MAP)
    unknown_labels = df["label"].isna()
    if unknown_labels.any():
        logger.warning(
            "Dropping %d rows with unrecognised label values.",
            unknown_labels.sum(),
        )
    df = df[~unknown_labels].copy()

    # 4. Fill remaining nulls in optional fields with empty strings.
    for col in ("sender", "receiver", "subject"):
        df[col] = df[col].fillna("")

    logger.info(
        "Cleaning: %d rows in → %d rows out (%d dropped).",
        original_count,
        len(df),
        original_count - len(df),
    )
    return df.reset_index(drop=True)
