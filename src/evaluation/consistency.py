"""
src/evaluation/consistency.py
------------------------------
Per-sample majority-vote consistency utilities.

Relationship to MetricsEngine.compute_consistency()
----------------------------------------------------
``metrics.py`` provides a **run-major** consistency metric:
  ``runs[i][j]`` = run ``i``'s prediction for sample ``j``.
  It measures how often all runs agree position-by-position across a batch.

This module provides a **sample-major** consistency metric:
  ``repeated_predictions[i][j]`` = the ``j``-th repeated prediction for
  sample ``i``.  It measures how often individual predictions for each
  sample agree with that sample's majority label.

The two metrics capture different failure modes:

* **Run-major** (metrics.py) detects global prompt instability — the
  model flips for whole runs.
* **Sample-major** (this module) detects per-instance instability — the
  model is uncertain about specific emails regardless of run.

Both are needed for a complete picture of LLM inference stability.

Consistency definition (sample-major)
--------------------------------------
For each sample ``i`` with ``k`` repeated predictions:

1. Compute the majority label (most common prediction; ties broken
   deterministically by ``POSITIVE_LABEL`` preference, then alphabetically).
2. Count how many of the ``k`` predictions match the majority label.
3. Sample consistency = (matching count) / k.

Batch consistency = mean of per-sample consistency scores.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from src.core.constants import LABEL_PHISHING

logger = logging.getLogger(__name__)

# Tie-breaking preference: when two labels appear equally often, prefer
# PHISHING (more conservative / lower false-negative risk).
_TIE_PREFERRED: str = LABEL_PHISHING


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """
    Aggregated consistency result for a batch of samples.

    Attributes:
        batch_consistency:   Mean per-sample consistency in [0.0, 1.0].
                             1.0 = every sample's predictions unanimously agree.
        per_sample_scores:   Per-sample consistency score list, one float
                             per sample, in the same order as the input.
        majority_labels:     Majority-vote label for each sample, in the
                             same order as the input.
        sample_count:        Number of samples evaluated.
        repetitions_per_sample: Number of repeated predictions per sample.
                               -1 if samples have varying repetition counts.
    """

    batch_consistency: float
    per_sample_scores: List[float]
    majority_labels: List[str]
    sample_count: int
    repetitions_per_sample: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_consistency(
    repeated_predictions: List[List[str]],
) -> ConsistencyResult:
    """
    Compute majority-vote consistency over a batch of repeated predictions.

    Each inner list holds all repeated predictions for **one sample**.
    The function computes how often each sample's predictions agree with
    its own majority label, then averages across all samples.

    Args:
        repeated_predictions: List of per-sample prediction lists.
            ``repeated_predictions[i][j]`` is the ``j``-th prediction for
            sample ``i``.  All inner lists should be non-empty.

    Returns:
        :class:`ConsistencyResult` with per-sample and batch-level scores.
        Returns an all-zero result if *repeated_predictions* is empty.

    Example::

        rp = [
            ["PHISHING", "PHISHING", "SAFE"],   # sample 0: majority=PHISHING, score=2/3
            ["SAFE", "SAFE", "SAFE"],            # sample 1: majority=SAFE,     score=3/3
        ]
        result = compute_consistency(rp)
        print(result.batch_consistency)   # 0.833
        print(result.majority_labels)     # ["PHISHING", "SAFE"]
    """
    if not repeated_predictions:
        logger.warning("compute_consistency called with empty input — returning zero result.")
        return ConsistencyResult(
            batch_consistency=0.0,
            per_sample_scores=[],
            majority_labels=[],
            sample_count=0,
            repetitions_per_sample=0,
        )

    per_sample_scores: List[float] = []
    majority_labels: List[str] = []
    rep_lengths = set()

    for i, sample_preds in enumerate(repeated_predictions):
        if not sample_preds:
            logger.warning(
                "Sample %d has no predictions — assigning consistency 0.0.", i
            )
            per_sample_scores.append(0.0)
            majority_labels.append("")
            rep_lengths.add(0)
            continue

        majority = get_majority_label(sample_preds)
        score = compute_sample_consistency(sample_preds)
        per_sample_scores.append(score)
        majority_labels.append(majority)
        rep_lengths.add(len(sample_preds))

    batch_consistency = (
        sum(per_sample_scores) / len(per_sample_scores)
        if per_sample_scores
        else 0.0
    )

    reps = rep_lengths.pop() if len(rep_lengths) == 1 else -1

    return ConsistencyResult(
        batch_consistency=batch_consistency,
        per_sample_scores=per_sample_scores,
        majority_labels=majority_labels,
        sample_count=len(repeated_predictions),
        repetitions_per_sample=reps,
    )


def compute_sample_consistency(predictions: List[str]) -> float:
    """
    Compute the consistency score for a single sample's repeated predictions.

    Consistency = (number of predictions matching the majority label) / total.

    Args:
        predictions: Repeated predictions for one sample.  Must be non-empty.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 for empty input.

    Example::

        score = compute_sample_consistency(["PHISHING", "PHISHING", "SAFE"])
        # majority = "PHISHING", 2/3 agree → 0.667
    """
    if not predictions:
        return 0.0

    majority = get_majority_label(predictions)
    matches = sum(1 for p in predictions if p == majority)
    return matches / len(predictions)


def get_majority_label(predictions: List[str]) -> str:
    """
    Return the most common label from *predictions*.

    Tie-breaking rule (deterministic):
    1. Prefer ``"PHISHING"`` — more conservative for phishing detection.
    2. If neither label is ``"PHISHING"``, prefer the alphabetically first
       label among those tied for the highest count.

    This rule ensures the result is always deterministic regardless of
    ``Counter`` iteration order, which can vary across Python versions.

    Args:
        predictions: Non-empty list of label strings.

    Returns:
        The majority label string.

    Raises:
        ValueError: If *predictions* is empty.

    Example::

        get_majority_label(["SAFE", "PHISHING", "SAFE"])  # → "SAFE"  (2 > 1)
        get_majority_label(["SAFE", "PHISHING"])           # → "PHISHING" (tie → prefer PHISHING)
    """
    if not predictions:
        raise ValueError("predictions must be non-empty.")

    counts = Counter(predictions)
    max_count = max(counts.values())

    # Collect all labels tied for the maximum count.
    tied = [label for label, cnt in counts.items() if cnt == max_count]

    if len(tied) == 1:
        return tied[0]

    # Deterministic tie-breaking.
    if _TIE_PREFERRED in tied:
        return _TIE_PREFERRED

    return sorted(tied)[0]  # alphabetical fallback


def consistency_to_signal(
    result: ConsistencyResult,
    threshold: float = 0.80,
) -> bool:
    """
    Convert a :class:`ConsistencyResult` into a boolean instability signal.

    Returns ``True`` (inconsistent) when ``batch_consistency`` falls below
    *threshold*.  This signal is consumed by the critic's
    ``SignalExtractor`` to trigger a ``REFINE_COT`` operator.

    Args:
        result:    :class:`ConsistencyResult` from :func:`compute_consistency`.
        threshold: Minimum acceptable consistency score.  Default ``0.80``
                   means the model must agree with its own majority label
                   at least 80 % of the time across samples.

    Returns:
        ``True`` if the prompt is insufficiently consistent, ``False`` otherwise.

    Example::

        signal = consistency_to_signal(result, threshold=0.85)
        if signal:
            # trigger REFINE_COT
    """
    return result.batch_consistency < threshold
