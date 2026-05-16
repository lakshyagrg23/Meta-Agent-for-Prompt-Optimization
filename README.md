# Constrained Adaptive Meta-Agent Framework for Prompt Optimization in Phishing Detection

> A research framework that treats LLM prompts as **structured, mutable memory** and optimizes them iteratively through deterministic governance and bounded semantic refinement, without ever letting the LLM rewrite its own optimization rules.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Design Principles](#key-design-principles)
3. [Framework Architecture](#framework-architecture)
4. [Structured Prompt State](#structured-prompt-state)
5. [Optimization Pipeline](#optimization-pipeline)
6. [Signal-Driven Refinement Policy](#signal-driven-refinement-policy)
7. [Composite Objective Function](#composite-objective-function)
8. [Acceptance & Rollback Mechanism](#acceptance--rollback-mechanism)
9. [Project Structure](#project-structure)
10. [Module Reference](#module-reference)
11. [Getting Started](#getting-started)
12. [Running Tests](#running-tests)
13. [Configuration](#configuration)
14. [Supported Dataset Formats](#supported-dataset-formats)

---

## Overview

This project implements a **constrained adaptive meta-agent** for automatic prompt optimization in phishing email detection. The system iteratively refines LLM prompt components using performance-driven feedback while maintaining:

- **Reproducibility** — deterministic evaluation logic; no stochastic governance decisions.
- **Bounded prompt evolution** — explicit token budgets per component prevent prompt bloat.
- **Interpretable optimization** — every refinement decision is traceable to a measurable signal.

Unlike conventional approaches that use unconstrained reflective agents or free-form LLM feedback loops, this framework separates the *optimization logic* (always deterministic) from *semantic adaptation* (LLM-assisted, but strictly bounded).

---

## Key Design Principles

| Principle | Implementation |
|---|---|
| **Deterministic governance** | Signal extraction and refinement policy are pure Python rule-sets with no LLM involvement |
| **Bounded components** | Each prompt section has a hard token budget; refinement operators cannot exceed it |
| **Controlled same-batch evaluation** | Candidate and current states are compared on the **same** batch, isolating prompt quality from data variance |
| **Rollback safety** | Candidates are only adopted when they improve the composite objective by at least `ε` |
| **Modular refinement** | Only the targeted component changes; the rest of the prompt is preserved unchanged |

---

## Framework Architecture

The optimization loop follows a fixed seven-stage workflow:

```
Prompt State
  → LLM Inference
  → Deterministic Evaluation
  → Signal Extraction
  → Refinement Policy (deterministic)
  → Constrained LLM-Guided Semantic Refinement
  → Candidate Prompt State
  → Acceptance / Rollback
  → (repeat)
```

```
┌─────────────────┐
│  Phishing Dataset│
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Structured Prompt   │◄──────────────────────────────┐
│ State S_t           │                               │
└────────┬────────────┘                               │
         │                                            │
         ▼                                            │
┌─────────────────┐                                   │
│  LLM Inference  │                                   │
└───────┬─────────┘                                   │
        │                                             │
   ┌────┴────┐                                        │
   ▼         ▼                                        │
┌──────┐  ┌─────────┐                                 │
│Deter-│  │Consist- │                                 │
│minist│  │ency     │                                 │
│Eval  │  │Eval     │                                 │
└──┬───┘  └────┬────┘                                 │
   └─────┬─────┘                                      │
         ▼                                            │
┌────────────────┐                                    │
│Signal Extractor│                                    │
└───────┬────────┘                                    │
        ▼                                             │
┌────────────────────┐                                │
│Deterministic       │                                │
│Refinement Policy   │                                │
└───────┬────────────┘                                │
        ▼                                             │
┌────────────────────────┐                            │
│Constrained LLM-Guided  │                            │
│Semantic Refinement     │                            │
└───────┬────────────────┘                            │
        ▼                                             │
┌───────────────────┐                                 │
│Candidate State    │                                 │
│S_{t+1}^{cand}     │                                 │
└────┬──────────────┘                                 │
     │                                                │
 ΔJ ≥ ε?                                             │
 ┌───┴───┐                                            │
 │Accept │  → Updated State S_{t+1} ──────────────────┘
 │       │
 │Rollback│ → Retain S_t ────────────────────────────┘
 └───────┘
```

---

## Structured Prompt State

A prompt is not a plain string — it is a **typed, bounded dataclass** defined in `src/core/prompt_state.py`:

```python
@dataclass
class PromptState:
    base_instruction: str           # Fixed; never mutated
    role: PromptComponent           # Max 20 tokens
    instruction_enrichment: PromptComponent  # Max 50 tokens
    cot: PromptComponent            # Max 25 tokens
    few_shot: FewShotComponent      # Max 120 tokens / n_max examples
    metadata: PromptMetadata        # Iteration, score history, signals
```

### Token Budgets

| Component | Maximum Tokens |
|---|---|
| Role Specification | 20 |
| Instruction Enrichment | 50 |
| Chain-of-Thought Guidance | 25 |
| Few-Shot Memory | 120 |

### Prompt Rendering Order

The `PromptRenderer` assembles sections in a **fixed canonical order** — any deviation is a bug:

```
[ROLE] → [TASK] → [GUIDELINES] → [EXAMPLES] → [REASONING APPROACH] → [EMAIL]
```

Every call with identical inputs produces byte-for-byte identical output (fully deterministic, no timestamps or randomness).

### Few-Shot Memory

```python
F = {e_1, e_2, ..., e_n},   n ≤ n_max
```

Each `FewShotExample` stores:
- **Email content** — structured `EmailInput` (sender, receiver, subject, body) or plain string
- **Label** — `"PHISHING"` or `"SAFE"`
- **Reason** — concise human-readable explanation
- **Relevance score** — used to select examples for replacement at memory capacity

When memory is full, the **lowest-relevance example is replaced** rather than appending, preventing uncontrolled prompt growth.

---

## Optimization Pipeline

### 1. Initialization

```python
S_0 = PromptState(...)   # base instruction fixed; other components optionally seeded
S_best = S_0
```

### 2. Iterative Refinement (per iteration `t`)

```
1. Sample stratified batch B_t
2. Evaluate S_t on B_t  →  EvaluationResult
3. Extract optimization signals from metrics
4. Select refinement operator via deterministic policy
5. Generate candidate S_{t+1}^{cand}  (LLM, bounded)
6. Evaluate S_{t+1}^{cand} on same batch B_t
7. Compute ΔJ = J(candidate) − J(current)
8. Accept if ΔJ ≥ ε, else rollback
9. Periodically evaluate S_best on held-out validation subset
```

### 3. Termination

Optimization stops when any of the following is true:
- Maximum iteration limit reached
- No significant improvement across `k` consecutive iterations (plateau)
- Early stopping threshold triggered

Final output:

```python
S* = argmax_{S ∈ History} J(S)
```

---

## Signal-Driven Refinement Policy

The **critic** converts evaluation metrics into discrete signals, then maps signals to operators — entirely in Python with no LLM involvement:

### Optimization Signals

| Signal | Condition |
|---|---|
| `high_fn` | False-negative rate > threshold `τ_fn` |
| `high_fp` | False-positive rate > threshold `τ_fp` |
| `low_accuracy` | Accuracy < threshold `τ_acc` |
| `inconsistent` | Batch consistency < threshold `τ_cons` (default 0.80) |
| `plateau` | \|J_t − J_{t-k}\| < δ across k iterations |

### Signal → Operator Mapping (prioritized)

| Optimization Signal | Refinement Operator |
|---|---|
| High False Negatives *(highest priority)* | `REFINE_FEWSHOT` |
| High False Positives | `REFINE_ENRICHMENT` |
| Low Overall Accuracy | `REFINE_ROLE` / `REFINE_ENRICHMENT` |
| Optimization Plateau / Inconsistency | `REFINE_COT` |

Phishing recall degradation is treated with **highest priority** given the safety-critical nature of missed phishing attacks.

### Refinement Operators

| Operator | Target Component | Behavior |
|---|---|---|
| `REFINE_ROLE` | `role` | Rewrites lightweight cybersecurity persona framing |
| `REFINE_ENRICHMENT` | `instruction_enrichment` | Updates phishing-specific detection guidance |
| `REFINE_COT` | `cot` | Introduces/refines compact reasoning directives |
| `REFINE_FEWSHOT` | `few_shot` | Inserts or replaces examples aligned with failure patterns |

---

## Composite Objective Function

### Acceptance Objective — `AcceptanceStrategy` (`src/optimization/acceptance.py`)

```
J(S) = 0.4 × F1  +  0.3 × Recall  +  0.2 × Consistency  −  0.1 × PromptCost

PromptCost = clamp(token_count / 2048, 0.0, 1.0)
J(S) ∈ [−0.1, 1.0]
```

| Term | Weight | Rationale |
|---|---|---|
| F1 | 0.4 | Primary signal; balances precision and recall |
| Recall | 0.3 | Missing phishing is operationally worse than false alarms |
| Consistency | 0.2 | Stable prompts → reproducible research results |
| PromptCost | −0.1 | Mild regularizer against prompt bloat |

### Monitoring Objective — `Evaluator` (`src/evaluation/evaluator.py`)

A separate, more granular scoring formula is used for logging and trend tracking:

```
score = 0.30×F1  +  0.25×Recall  +  0.15×Precision
      − 0.20×FNR  − 0.05×FPR  + 0.05×Consistency
```

### Consistency Measurement

Two complementary consistency metrics are computed:

| Mode | Module | Definition |
|---|---|---|
| **Run-major** | `metrics.py` — `MetricsEngine.compute_consistency` | Agreement across full inference runs on the same batch |
| **Sample-major** | `consistency.py` — `compute_consistency` | Per-sample majority-vote agreement across repeated predictions |

Tie-breaking in majority vote always **prefers `PHISHING`** (conservative, lower false-negative risk).

---

## Acceptance & Rollback Mechanism

```python
ΔJ = J(S_{t+1}^cand) − J(S_t)

S_{t+1} = S_{t+1}^cand   if ΔJ ≥ ε
         = S_t             otherwise
```

Acceptance requires **all three conditions**:
1. `ΔJ ≥ ε` (default `ε = 0.01`)
2. Candidate consistency is acceptable
3. Candidate satisfies all token budget constraints

This rollback mechanism ensures:
- Prompt evolution is **monotonic** w.r.t. the objective
- Stochastic LLM semantic errors cannot accumulate
- Optimization trajectories remain stable and auditable

---

## Project Structure

```
Final-Project1/
│
├── main.py                        # Entry point (to be implemented)
├── _smoke_test.py                 # Quick integration sanity check
├── requirements.txt
│
├── configs/
│   ├── experiment_config.yaml     # Optimization hyperparameters
│   ├── model_config.yaml          # LLM client configuration
│   └── token_budgets.yaml         # Per-component token limits
│
├── data/                          # Raw & processed CSV datasets
├── outputs/                       # Experiment results & logs
├── experiments/                   # Experiment scripts / notebooks
├── notebooks/                     # Exploratory analysis
├── prompts/                       # Saved prompt state snapshots
│
└── src/
    ├── core/
    │   ├── prompt_state.py        # PromptState, FewShotExample, PromptComponent
    │   ├── renderer.py            # Deterministic prompt → string renderer
    │   ├── scoring.py             # (stub) Scoring utilities
    │   └── validator.py           # (stub) Prompt structure validation
    │
    ├── critic/
    │   ├── signal_extractor.py    # Metrics → CriticSignals (deterministic)
    │   ├── policy.py              # CriticSignals → RefinementOperator (deterministic)
    │   └── error_analysis.py      # Failure pattern analysis utilities
    │
    ├── evaluation/
    │   ├── metrics.py             # MetricsEngine — accuracy, F1, FPR, FNR, consistency
    │   ├── consistency.py         # Sample-major majority-vote consistency
    │   └── evaluator.py           # Unified evaluation pipeline + composite scoring
    │
    ├── mutations/
    │   ├── base_operator.py       # Abstract base for refinement operators
    │   ├── refine_role.py         # Role specification refinement
    │   ├── refine_enrichment.py   # Instruction enrichment refinement
    │   ├── refine_cot.py          # Chain-of-thought refinement
    │   └── refine_fewshot.py      # Adaptive few-shot memory refinement
    │
    ├── optimization/
    │   ├── acceptance.py          # AcceptanceStrategy — J(S) and ΔJ ≥ ε rule
    │   ├── candidate_generator.py # Candidate state generation (stub)
    │   ├── optimization_loop.py   # Main loop orchestrator (stub)
    │   └── validation_runner.py   # Periodic validation evaluation (stub)
    │
    ├── dataset/
    │   ├── loader.py              # CSV → normalized email record list
    │   ├── sampler.py             # Stratified batch sampler (stub)
    │   └── splitter.py            # Train/validation split (stub)
    │
    ├── llm/
    │   ├── client.py              # LLM API client wrapper (stub)
    │   ├── inference.py           # Batch inference runner (stub)
    │   └── refinement_generation.py  # Constrained semantic refinement via LLM (stub)
    │
    ├── logging/                   # Experiment logging utilities
    └── utils/
        └── token_utils.py         # Token counting utilities
│
└── tests/
    ├── test_metrics.py            # MetricsEngine unit tests
    ├── test_consistency.py        # Consistency module tests
    ├── test_evaluator.py          # Full evaluation pipeline tests
    ├── test_acceptance.py         # AcceptanceStrategy tests
    ├── test_loader.py             # Dataset loader tests
    └── test_renderer.py           # PromptRenderer tests
```

---

## Module Reference

### `src/core/prompt_state.py`
Core dataclasses. Key types:

| Class | Purpose |
|---|---|
| `PromptState` | Top-level structured prompt; `.clone()` for deep copy, `.get_total_token_count()` for budget checks |
| `PromptComponent` | Bounded text component with `content`, `token_budget`, `revision_count` |
| `FewShotComponent` | Ordered list of `FewShotExample` with capacity constraints |
| `FewShotExample` | Single labeled example with relevance score |
| `EmailInput` | Structured email (sender, receiver, subject, body) |
| `PromptMetadata` | Iteration counter, score history, signal log, mutation history |

### `src/core/renderer.py`
`PromptRenderer.render_prompt(state, email) → str`  
Stateless, deterministic. Fixed section order: ROLE → TASK → GUIDELINES → EXAMPLES → REASONING APPROACH → EMAIL.

### `src/evaluation/metrics.py`
`MetricsEngine` — all static methods:

| Method | Description |
|---|---|
| `compute_metrics(predictions, labels)` | Full classification metrics; PHISHING is the positive class |
| `compute_consistency(runs)` | Run-major consistency across repeated inference passes |
| `attach_consistency(metrics, runs)` | Merge consistency into an existing `EvaluationMetrics` object |

### `src/evaluation/consistency.py`
Sample-major majority-vote consistency:

| Function | Description |
|---|---|
| `compute_consistency(repeated_predictions)` | Returns `ConsistencyResult` with per-sample and batch scores |
| `compute_sample_consistency(predictions)` | Single-sample score |
| `get_majority_label(predictions)` | Tie-breaking prefers `"PHISHING"` |
| `consistency_to_signal(result, threshold)` | Boolean instability signal for the critic |

### `src/evaluation/evaluator.py`
`Evaluator.evaluate(predictions, labels, repeated_predictions, prompt_token_count) → EvaluationResult`  
Full pipeline: metrics → consistency → composite score.

### `src/optimization/acceptance.py`
`AcceptanceStrategy` — all static methods:

| Method | Description |
|---|---|
| `compute_score(metrics, token_count)` | Compute J(S) = 0.4·F1 + 0.3·Recall + 0.2·Consistency − 0.1·Cost |
| `should_accept(current, candidate, epsilon)` | Returns True if ΔJ ≥ ε |
| `score_delta(current, candidate)` | Returns raw improvement Δ |

### `src/dataset/loader.py`
`load_dataset(csv_path) → List[Dict[str, str]]`  
Normalizes Kaggle-format phishing CSVs to records with keys `sender`, `receiver`, `subject`, `body`, `label`. Labels are mapped to exactly `"PHISHING"` or `"SAFE"`.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An LLM API key (e.g. OpenAI, Anthropic, or a compatible local model)

### Installation

```bash
git clone <repo-url>
cd Final-Project1
pip install -r requirements.txt
```

> **Note:** `requirements.txt` currently lists core scientific dependencies. Install additional packages as needed for the LLM client (`openai`, `anthropic`, etc.).

### Quick Smoke Test

Verifies that the core data structures, token utilities, and renderer work correctly without requiring any LLM or dataset:

```bash
python _smoke_test.py
# Expected: All checks passed.
```

---

## Running Tests

```bash
# Run all unit tests
python -m pytest src/tests/ -v

# Run a specific suite
python -m pytest src/tests/test_metrics.py -v
python -m pytest src/tests/test_consistency.py -v
python -m pytest src/tests/test_acceptance.py -v
python -m pytest src/tests/test_evaluator.py -v
python -m pytest src/tests/test_loader.py -v
python -m pytest src/tests/test_renderer.py -v
```

---

## Configuration

Configuration files live in `configs/`. All are YAML format.

| File | Purpose |
|---|---|
| `experiment_config.yaml` | Iteration limits, batch size, epsilon threshold, plateau window, validation frequency |
| `model_config.yaml` | LLM provider, model name, temperature, max tokens for refinement calls |
| `token_budgets.yaml` | Per-component token ceilings (role, enrichment, CoT, few-shot) |

---

## Supported Dataset Formats

The dataset loader handles the most common Kaggle phishing email dataset schemas automatically:

| Schema | Required Columns | Notes |
|---|---|---|
| Full-field | `sender`, `receiver`, `subject`, `body`, `label` | CEAS, AVN, curated sets |
| Minimalist | `body`, `label` | Enron, Ling subsets; sender/receiver auto-filled as `""` |
| Binary-label | Any of the above with `label` as `0`/`1` | `0 = SAFE`, `1 = PHISHING` |

**Recognized label aliases**: `phishing`, `spam`, `malicious`, `1` → `PHISHING`;  `safe`, `ham`, `legitimate`, `benign`, `0` → `SAFE`.

Column name aliases are resolved automatically — see `COLUMN_ALIASES` in `src/dataset/loader.py` for the full list.

