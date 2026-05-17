"""
data.py — CSV loader and batch generator.

Handles datasets with optional sender/receiver columns.
All rows are normalised to: id, sender, receiver, subject, body, label
"""

import os
import glob
import pandas as pd
from typing import List, Dict, Generator

from src.config import DATA_DIR, BATCH_SIZE

# ── Constants ──────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"subject", "body", "label"}
OPTIONAL_COLS = {"sender", "receiver"}


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str = DATA_DIR) -> List[Dict]:
    """
    Reads all *.csv files in data_dir, concatenates them into a single
    normalised list of row dicts.

    Returns:
        List of dicts with keys: id, sender, receiver, subject, body, label
    Raises:
        FileNotFoundError: if no CSV files found in data_dir
        ValueError: if a CSV is missing required columns or has bad label values
    """
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}/'")

    frames = []
    for path in sorted(csv_paths):
        df = _load_single_csv(path)
        frames.append(df)
        print(f"  [data] Loaded {len(df):,} rows from {os.path.basename(path)}")

    combined = pd.concat(frames, ignore_index=True)
    combined["id"] = combined.index  # stable global row id

    # reorder columns for clarity
    combined = combined[["id", "sender", "receiver", "subject", "body", "label"]]

    print(f"  [data] Total rows: {len(combined):,}")
    return combined.to_dict(orient="records")


def _load_single_csv(path: str) -> pd.DataFrame:
    """Load one CSV file and normalise its schema."""
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{os.path.basename(path)}' is missing required columns: {missing}"
        )

    # Add optional columns if absent
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""

    # Validate labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    bad_labels = df["label"].isna() | ~df["label"].isin([0, 1])
    if bad_labels.any():
        bad_count = bad_labels.sum()
        print(
            f"  [data] Warning: {bad_count} rows in '{os.path.basename(path)}' "
            f"have invalid labels — dropping them."
        )
        df = df[~bad_labels]

    df["label"] = df["label"].astype(int)

    # Fill NaN in optional text columns
    for col in ["sender", "receiver", "subject", "body"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df[["sender", "receiver", "subject", "body", "label"]]


# ── Batcher ────────────────────────────────────────────────────────────────────

def batch_generator(
    rows: List[Dict],
    batch_size: int = BATCH_SIZE,
) -> Generator[List[Dict], None, None]:
    """
    Yields non-overlapping slices of `batch_size` rows.
    Each row dict is a copy so mutations in the loop don't affect the source.
    """
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def total_batches(rows: List[Dict], batch_size: int = BATCH_SIZE) -> int:
    """Returns the total number of batches for progress display."""
    return (len(rows) + batch_size - 1) // batch_size
