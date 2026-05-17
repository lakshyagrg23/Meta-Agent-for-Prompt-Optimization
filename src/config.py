"""
config.py — All tunable constants for the phishing optimizer.
Override MODEL_NAME and API_KEY via CLI args or environment variables.
"""

import os

# ── Batch / Loop ───────────────────────────────────────────────────────────────
BATCH_SIZE               = 64   # rows per iteration
MAX_ITERATIONS           = 10    # T — hard cap on loop iterations
NO_IMPROVE_PATIENCE      = 3     # consecutive non-improving iterations before early stop

# ── Consistency Sampling ───────────────────────────────────────────────────────
CONSISTENCY_RUNS         = 3     # how many times to re-run Agent 1 for consistency
CONSISTENCY_SAMPLE_EVERY = 3     # measure consistency every Nth iteration (0-indexed)

# ── Acceptance ─────────────────────────────────────────────────────────────────
IMPROVEMENT_THRESHOLD    = 0.005  # min F1 delta for new_state to be accepted

# ── Token Budget ───────────────────────────────────────────────────────────────
MAX_PROMPT_TOKENS        = 800   # hard ceiling on any Agent 1 system prompt
TIKTOKEN_ENCODING        = "cl100k_base"

# ── Mutation Thresholds ────────────────────────────────────────────────────────
HIGH_FN_THRESHOLD         = 0.15  # FN / (FN + TP) > this → few_shot_fn, prompt_enrichment
HIGH_FP_THRESHOLD         = 0.15  # FP / (FP + TN) > this → few_shot_balanced, add_constraints
LOW_ACCURACY_THRESHOLD    = 0.95  # accuracy < this → enrich_role, prompt_enrichment
LOW_CONSISTENCY_THRESHOLD = 0.85  # consistency < this → add_constraints

# ── Few-Shot Example Counts ────────────────────────────────────────────────────
FEW_SHOT_FN_COUNT         = 5    # max FN examples to pass to Agent 2
FEW_SHOT_FP_COUNT         = 3    # max FP examples for balanced set
FEW_SHOT_TN_COUNT         = 3    # max TN examples for balanced set

# ── LLM ───────────────────────────────────────────────────────────────────────
MODEL_NAME               = os.getenv("MODEL_NAME", "gemma4:e2b")
TEMPERATURE_CLASSIFY     = 0.1   # deterministic for classification
TEMPERATURE_OPTIMIZE     = 0.7   # some creativity for prompt generation
MAX_TOKENS_CLASSIFY      = 2048  # response token cap for Agent 1
MAX_TOKENS_OPTIMIZE      = 1200  # response token cap for Agent 2
PARALLEL_WORKERS         = 3     # concurrent threads for Agent 1 classification
OLLAMA_NUM_CTX           = 4096  # context window restriction for Ollama

# ── Retry ──────────────────────────────────────────────────────────────────────
LLM_MAX_RETRIES          = 3     # max retries on API errors
LLM_RETRY_BACKOFF        = 2.0   # seconds — doubled each retry
PARSE_MAX_RETRIES        = 2     # max retries on JSON parse failure

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR                 = "data"
LOGS_DIR                 = "logs"
PROMPTS_DIR              = "prompts"
AGENT1_INIT_PROMPT       = "prompts/agent1_init.txt"
AGENT2_SYSTEM_PROMPT     = "prompts/agent2_system.txt"
