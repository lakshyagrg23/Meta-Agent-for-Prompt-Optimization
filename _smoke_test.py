import sys
sys.path.insert(0, ".")

from src.core.prompt_state import (
    FewShotExample, FewShotComponent, PromptComponent, PromptMetadata, PromptState,
)
from src.utils.token_utils import (
    estimate_token_count,
    count_example_tokens,
    count_fewshot_tokens,
    count_component_tokens,
    count_total_prompt_tokens,
)
from src.core.renderer import PromptRenderer

# ===========================================================================
# token_utils checks
# ===========================================================================

assert estimate_token_count("hello world foo") == 3
assert estimate_token_count("") == 0

ex1 = FewShotExample(email="click here now", label="phishing", reason="urgency tactic")
# ex1: email=3, label=1, reason=2  -> 6
assert count_example_tokens(ex1) == 6, f"ex1={count_example_tokens(ex1)}"

ex2 = FewShotExample(email="quarterly report attached", label="legitimate", reason="normal business")
# ex2: email=3, label=1, reason=2  -> 6
assert count_example_tokens(ex2) == 6, f"ex2={count_example_tokens(ex2)}"

fsc = FewShotComponent(examples=[ex1, ex2], token_budget=200, max_examples=5)
assert count_fewshot_tokens(fsc) == 12, f"fsc={count_fewshot_tokens(fsc)}"

comp = PromptComponent(content="You are a security analyst", token_budget=50)
assert count_component_tokens(comp) == 5, f"comp={count_component_tokens(comp)}"

# ===========================================================================
# PromptState checks
# ===========================================================================

state = PromptState(
    base_instruction="Classify the email",            # 3
    role=PromptComponent(
        content="You are a security analyst",         # 5
        token_budget=50,
    ),
    instruction_enrichment=PromptComponent(
        content="Focus on phishing signals",          # 4
        token_budget=100,
    ),
    cot=PromptComponent(
        content="Think step by step",                 # 4
        token_budget=150,
    ),
    few_shot=FewShotComponent(
        examples=[ex1, ex2],                          # 12
        token_budget=300,
        max_examples=5,
    ),
    metadata=PromptMetadata(),
)
expected_total = 3 + 5 + 4 + 4 + 12   # = 28
got = state.get_total_token_count()
assert got == expected_total, f"total: got {got}, expected {expected_total}"

# clone isolation
clone = state.clone()
clone.role.content = "MUTATED"
assert state.role.content == "You are a security analyst", "clone leaked into original!"
clone.few_shot.examples.append(ex1)
assert len(state.few_shot.examples) == 2, "clone list leaked into original!"

# update_component
state.update_component("role", "New role text")
assert state.role.content == "New role text"

# increment_revision
state.increment_revision("role")
assert state.role.revision_count == 1
state.increment_revision("few_shot")
assert state.few_shot.revision_count == 1

# ===========================================================================
# PromptRenderer checks
# ===========================================================================

raw_email = "  Dear user, click here to claim your prize!  "
prompt = PromptRenderer.render_prompt(state, raw_email)

# Determinism: same inputs → same output
assert PromptRenderer.render_prompt(state, raw_email) == prompt

# Section order
role_pos        = prompt.index("[ROLE]")
task_pos        = prompt.index("[TASK]")
guidelines_pos  = prompt.index("[GUIDELINES]")
examples_pos    = prompt.index("[EXAMPLES]")
reasoning_pos   = prompt.index("[REASONING APPROACH]")
email_pos       = prompt.index("[EMAIL TO CLASSIFY]")

assert role_pos < task_pos < guidelines_pos < examples_pos < reasoning_pos < email_pos, \
    "Section order violated"

# Few-shot examples rendered in order
assert "Example 1:" in prompt
assert "Example 2:" in prompt
ex1_pos = prompt.index("Example 1:")
ex2_pos = prompt.index("Example 2:")
assert ex1_pos < ex2_pos

# Email content is stripped
assert "Dear user, click here to claim your prize!" in prompt
assert "  Dear user" not in prompt

# Labels and reasons present
assert "phishing" in prompt
assert "urgency tactic" in prompt

# Empty few-shot still renders section header
state_no_shots = state.clone()
state_no_shots.few_shot.examples = []
prompt_no_shots = PromptRenderer.render_prompt(state_no_shots, "test email")
assert "[EXAMPLES]" in prompt_no_shots

print("All checks passed.")
