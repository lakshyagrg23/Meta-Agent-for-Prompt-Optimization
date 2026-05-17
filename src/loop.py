"""
loop.py — Main optimization loop controller.

Orchestrates: data batching → Agent 1 → metrics → consistency (sampled)
              → mutation policy → Agent 2 → re-test → accept/reject → log
"""

import os
import json
import hashlib
import datetime
from typing import List, Dict, Optional

from src.config import (
    MAX_ITERATIONS,
    NO_IMPROVE_PATIENCE,
    IMPROVEMENT_THRESHOLD,
    CONSISTENCY_SAMPLE_EVERY,
    AGENT1_INIT_PROMPT,
    LOGS_DIR,
)
from src.data    import batch_generator, total_batches
from src.agents  import classify_batch, generate_new_prompt
from src.metrics import compute_metrics, compute_consistency
from src.mutation import decide_mutation
from src.llm     import count_tokens


# ── Logging ────────────────────────────────────────────────────────────────────

def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:8]


def _log_iteration(log_path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Print helpers ──────────────────────────────────────────────────────────────

_HEADER = (
    f"{'Iter':>4}  {'Batch':>9}  {'Acc':>6}  {'F1':>6}  "
    f"{'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  "
    f"{'Cons':>6}  {'Tags':<28}  {'Accepted'}"
)
_SEP = "─" * len(_HEADER)


def _print_row(
    iteration:   int,
    batch_start: int,
    batch_end:   int,
    m:           Dict,
    tags:        List[str],
    accepted:    bool,
) -> None:
    cons_str = f"{m['consistency']:.3f}" if m["consistency"] is not None else "  —  "
    tags_str = ",".join(tags) if tags else "(none)"
    print(
        f"{iteration:>4}  {batch_start:>4}-{batch_end:<4}  "
        f"{m['accuracy']:>6.3f}  {m['f1']:>6.3f}  "
        f"{m['tp']:>4}  {m['tn']:>4}  {m['fp']:>4}  {m['fn']:>4}  "
        f"{cons_str:>6}  {tags_str:<28}  {'✓ YES' if accepted else '✗ NO'}"
    )


# ── Prompt I/O ─────────────────────────────────────────────────────────────────

def _load_initial_prompt() -> str:
    try:
        with open(AGENT1_INIT_PROMPT, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Initial Agent 1 prompt not found at '{AGENT1_INIT_PROMPT}'. "
            "Please ensure prompts/agent1_init.txt exists."
        )


def _save_final_prompt(prompt: str, log_dir: str, timestamp: str) -> None:
    path = os.path.join(log_dir, f"final_prompt_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n  [loop] Final prompt saved → {path}")


# ── Main Loop ──────────────────────────────────────────────────────────────────

def run_optimization(
    rows:           List[Dict],
    max_iterations: int = MAX_ITERATIONS,
    log_dir:        str = LOGS_DIR,
) -> Dict:
    """
    Run the prompt optimization loop.

    Args:
        rows:           full dataset as list of dicts.
        max_iterations: hard cap T on number of iterations.
        log_dir:        directory to write jsonl logs and final prompt.

    Returns:
        summary dict with keys:
            final_prompt, history, stop_reason, total_iterations
    """
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path    = os.path.join(log_dir, f"run_{timestamp}.jsonl")
    os.makedirs(log_dir, exist_ok=True)

    current_state    = _load_initial_prompt()
    batches          = list(batch_generator(rows))
    n_batches        = len(batches)

    iteration        = 0
    no_improve_streak= 0
    history          = []
    stop_reason      = "max_iterations"

    print(f"\n{'═'*80}")
    print(f"  Phishing Email Prompt Optimizer")
    print(f"  Dataset: {len(rows):,} rows  |  Batches: {n_batches}  |  Max iters: {max_iterations}")
    print(f"  Log: {log_path}")
    print(f"{'═'*80}\n")
    print(_HEADER)
    print(_SEP)

    for batch in batches:
        if iteration >= max_iterations:
            stop_reason = "max_iterations"
            break

        batch_start = batch[0]["id"]
        batch_end   = batch[-1]["id"]

        # ── Step 1: First-pass classification ─────────────────────────────────
        print(f"\n  [iter {iteration}] Classifying batch {batch_start}-{batch_end} …")
        preds_1  = classify_batch(current_state, batch)
        metrics  = compute_metrics(preds_1, batch)

        # ── Step 2: Consistency (sampled every Nth iteration) ─────────────────
        # if iteration % CONSISTENCY_SAMPLE_EVERY == 0:
        #     consistency = compute_consistency(classify_batch, current_state, batch)
        #     metrics["consistency"]         = consistency
        #     metrics["consistency_sampled"] = True
        # else metrics["consistency"] stays None (set in compute_metrics)

        # ── Step 3: Mutation policy ────────────────────────────────────────────
        policy = decide_mutation(metrics, batch, preds_1)

        # ── Step 4: Agent 2 → new prompt ──────────────────────────────────────
        print(f"  [iter {iteration}] Generating new prompt (tags: {policy['tags']}) …")
        new_state, rationale = generate_new_prompt(current_state, metrics, policy)

        # ── Step 5: Re-test new prompt on same batch ───────────────────────────
        print(f"  [iter {iteration}] Testing new prompt …")
        preds_2     = classify_batch(new_state, batch)
        new_metrics = compute_metrics(preds_2, batch)

        # ── Step 6: Accept / Reject ────────────────────────────────────────────
        f1_delta = new_metrics["f1"] - metrics["f1"]
        accepted = f1_delta > IMPROVEMENT_THRESHOLD

        if accepted:
            current_state     = new_state
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        # ── Step 7: Print row ─────────────────────────────────────────────────
        _print_row(iteration, batch_start, batch_end, metrics, policy["tags"], accepted)

        # ── Step 8: Log ───────────────────────────────────────────────────────
        record = {
            "iteration":          iteration,
            "batch_start":        batch_start,
            "batch_end":          batch_end,
            "current_state_hash": _prompt_hash(current_state),
            "current_tokens":     count_tokens(current_state),
            "new_tokens":         count_tokens(new_state),
            "metrics_before":     metrics,
            "mutation_tags":      policy["tags"],
            "metrics_after":      new_metrics,
            "f1_delta":           round(f1_delta, 4),
            "state_accepted":     accepted,
            "rationale":          rationale,
        }
        _log_iteration(log_path, record)
        history.append(record)

        # ── Step 9: Stopping conditions ───────────────────────────────────────
        if no_improve_streak >= NO_IMPROVE_PATIENCE:
            stop_reason = f"no_improvement_{NO_IMPROVE_PATIENCE}_consecutive"
            print(
                f"\n  [loop] Early stop: no improvement for "
                f"{NO_IMPROVE_PATIENCE} consecutive iterations."
            )
            break

        iteration += 1

    # ── Final summary ──────────────────────────────────────────────────────────
    print(_SEP)
    print(f"\n  Stop reason   : {stop_reason}")
    print(f"  Total iters   : {iteration + 1}")
    if history:
        best = max(history, key=lambda r: r["metrics_before"]["f1"])
        print(f"  Best F1       : {best['metrics_before']['f1']:.4f}  (iter {best['iteration']})")

    _save_final_prompt(current_state, log_dir, timestamp)

    return {
        "final_prompt":      current_state,
        "history":           history,
        "stop_reason":       stop_reason,
        "total_iterations":  iteration + 1,
        "log_path":          log_path,
    }
