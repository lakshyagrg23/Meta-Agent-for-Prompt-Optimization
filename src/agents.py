"""
agents.py — Agent 1 (Classifier) and Agent 2 (Prompt Optimizer).

Agent 1: Given a system prompt (current_state) and a batch of emails,
         returns structured predictions [{id, label, reason}].

Agent 2: Given current_state + metrics + mutation policy,
         returns a new system prompt string (new_state).
"""

import json
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    TEMPERATURE_CLASSIFY,
    TEMPERATURE_OPTIMIZE,
    MAX_TOKENS_CLASSIFY,
    MAX_TOKENS_OPTIMIZE,
    MAX_PROMPT_TOKENS,
    PARSE_MAX_RETRIES,
    AGENT2_SYSTEM_PROMPT,
    PARALLEL_WORKERS,
)
from src.llm import call_llm, extract_json, count_tokens


# ── Exceptions ─────────────────────────────────────────────────────────────────

class ParseError(Exception):
    """Raised when Agent output cannot be parsed after all retries."""
    pass


# ── Email formatter (shared) ───────────────────────────────────────────────────

def _format_email(row: Dict) -> str:
    sender   = row.get("sender")   or "N/A"
    receiver = row.get("receiver") or "N/A"
    subject  = row.get("subject")  or ""
    body     = row.get("body")     or ""
    return (
        f"### EMAIL {row['id']}\n"
        f"FROM: {sender}\n"
        f"TO: {receiver}\n"
        f"SUBJECT: {subject}\n"
        f"BODY:\n{body}"
    )


# ── Agent 1 — Classifier ───────────────────────────────────────────────────────

def _build_classify_user_prompt(row: Dict) -> str:
    email_text = _format_email(row)
    return (
        f"Classify the following email.\n\n"
        f"{email_text}\n\n"
        f"Respond ONLY with valid JSON. No explanation outside the JSON."
    )


def _parse_classify_response(raw: str, row: Dict) -> Dict:
    """
    Parse Agent 1 JSON response for a single email.
    Raises ValueError on schema or label violations.
    """
    data = extract_json(raw)

    label_raw = data.get("label") or data.get("classification") or data.get("prediction") or ""
    label_str = str(label_raw).strip().upper()
    
    reason_raw = data.get("reason") or data.get("justification") or data.get("explanation") or ""
    reason = str(reason_raw)

    if label_str not in ("SAFE", "PHISHING"):
        raise ValueError(f"Invalid label {label_str!r} — must be 'SAFE' or 'PHISHING'.")

    label_int = 1 if label_str == "PHISHING" else 0
    return {"id": row["id"], "label": label_int, "reason": reason[:150]}


def classify_batch(
    system_prompt: str,
    batch: List[Dict],
) -> List[Dict]:
    """
    Run Agent 1 on a batch of emails, one by one.

    Returns:
        List of {id, label, reason} dicts (same order as batch).
    Raises:
        ParseError: if parsing fails after PARSE_MAX_RETRIES attempts.
    """
    results = []
    abort_event = threading.Event()
    
    def process_row(row):
        if abort_event.is_set():
            return False, row['id'], None, Exception("Aborted by another thread")
            
        user_prompt = _build_classify_user_prompt(row)
        last_error  = None
        for attempt in range(1, PARSE_MAX_RETRIES + 2):
            if abort_event.is_set():
                break
                
            try:
                raw = call_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=TEMPERATURE_CLASSIFY,
                    max_tokens=MAX_TOKENS_CLASSIFY,
                )
            except Exception as e:
                abort_event.set()
                return False, row['id'], None, e
                
            try:
                parsed = _parse_classify_response(raw, row)
                return True, row['id'], parsed, None
            except (ValueError, KeyError) as e:
                last_error = e
        return False, row['id'], raw, last_error

    print(f"    [agent1] Predicting {len(batch)} emails using {PARALLEL_WORKERS} workers ...", end="\r")
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row for row in batch}
        completed = 0
        for future in as_completed(futures):
            success, row_id, parsed, error = future.result()
            
            if abort_event.is_set() and error and str(error) != "Aborted by another thread":
                # A critical LLM connection error occurred
                print(f"\n  [agent1] Fatal error on id={row_id}: {error}")
                print(" " * 50, end="\r")
                raise error
                
            completed += 1
            print(f"    [agent1] Predicting {completed}/{len(batch)} emails ...", end="\r")
            
            if success:
                results.append(parsed)
            elif not abort_event.is_set():
                print(f"\n  [agent1] Parse failed on id={row_id}. Reason: {error}")
                print(f"  [agent1] Raw output: {parsed!r}")
                
    print(" " * 50, end="\r") # clear the \r line
    
    # Sort results to match original batch order
    id_to_parsed = {r['id']: r for r in results}
    return [id_to_parsed[row['id']] for row in batch if row['id'] in id_to_parsed]


# ── Agent 2 — Prompt Optimizer ─────────────────────────────────────────────────

def _load_agent2_system() -> str:
    try:
        with open(AGENT2_SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Agent 2 system prompt not found at '{AGENT2_SYSTEM_PROMPT}'. "
            "Please ensure prompts/agent2_system.txt exists."
        )


def _format_examples(examples: List[Dict]) -> str:
    if not examples:
        return "None"
    parts = []
    for row in examples:
        parts.append(
            f"  EMAIL_ID={row['id']} | TRUE_LABEL={row.get('true_label', '?')} "
            f"| PREDICTED={row.get('predicted_label', '?')}\n"
            f"  SUBJECT: {row.get('subject', '')}\n"
            f"  BODY (first 200 chars): {str(row.get('body', ''))[:200]}"
        )
    return "\n\n".join(parts)


def _build_optimize_user_prompt(
    current_prompt:  str,
    metrics:         Dict,
    policy:          Dict,
) -> str:
    m          = metrics
    consistency_str = (
        f"{m['consistency']:.3f} (sampled)"
        if m.get("consistency") is not None
        else "not measured this iteration"
    )
    tags_str   = json.dumps(policy.get("tags", []))
    examples   = _format_examples(policy.get("examples", []))

    return (
        f"CURRENT_PROMPT:\n{current_prompt}\n\n"
        f"PERFORMANCE_METRICS:\n"
        f"  accuracy   = {m.get('accuracy', 0):.4f}\n"
        f"  f1         = {m.get('f1', 0):.4f}\n"
        f"  precision  = {m.get('precision', 0):.4f}\n"
        f"  recall     = {m.get('recall', 0):.4f}\n"
        f"  tp={m.get('tp',0)}  tn={m.get('tn',0)}  fp={m.get('fp',0)}  fn={m.get('fn',0)}\n"
        f"  fn_rate    = {m.get('fn_rate', 0):.4f}\n"
        f"  fp_rate    = {m.get('fp_rate', 0):.4f}\n"
        f"  consistency = {consistency_str}\n\n"
        f"MUTATION_TAGS: {tags_str}\n\n"
        f"FEW_SHOT_EXAMPLES:\n{examples}\n\n"
        f"OUTPUT ONLY valid JSON with no extra text:\n"
        f'{{"new_prompt": "...", "token_count": <int>, "rationale": "..."}}'
    )


def generate_new_prompt(
    current_prompt: str,
    metrics:        Dict,
    policy:         Dict,
) -> tuple[str, str]:
    """
    Run Agent 2 to generate a new system prompt.

    Returns:
        (new_prompt_str, rationale_str)
        Falls back to (current_prompt, "fallback") if:
          - token limit exceeded
          - all parse retries fail
    """
    agent2_system = _load_agent2_system()
    user_prompt   = _build_optimize_user_prompt(current_prompt, metrics, policy)

    last_error = None
    for attempt in range(1, PARSE_MAX_RETRIES + 2):
        raw = call_llm(
            system_prompt=agent2_system,
            user_prompt=user_prompt,
            temperature=TEMPERATURE_OPTIMIZE,
            max_tokens=MAX_TOKENS_OPTIMIZE,
        )
        try:
            data      = extract_json(raw)
            new_p     = data.get("new_prompt", "")
            rationale = str(data.get("rationale", ""))

            if not isinstance(new_p, str) or not new_p.strip():
                raise ValueError("'new_prompt' is empty or not a string.")

            # Hard token-budget check
            tok = count_tokens(new_p)
            if tok > MAX_PROMPT_TOKENS:
                print(
                    f"  [agent2] New prompt exceeds token limit "
                    f"({tok} > {MAX_PROMPT_TOKENS}) — rejecting."
                )
                return current_prompt, "token_limit_exceeded"

            return new_p.strip(), rationale

        except (ValueError, KeyError) as e:
            last_error = e
            if attempt <= PARSE_MAX_RETRIES:
                print(f"  [agent2] Parse error (attempt {attempt}): {e} — retrying …")
            else:
                print(f"  [agent2] Parse failed after {attempt} attempts: {e}")

    print(f"  [agent2] Falling back to current prompt. Last error: {last_error}")
    return current_prompt, "parse_failed_fallback"
