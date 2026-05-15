from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.prompt_state import (
    PromptState,
    PromptComponent,
    FewShotComponent,
    FewShotExample,
    PromptMetadata
)

from src.core.renderer import PromptRenderer


def test_render():

    prompt_state = PromptState(
        base_instruction="Classify the email as SAFE or PHISHING.",

        role=PromptComponent(
            content="You are a phishing detection expert.",
            token_budget=20
        ),

        instruction_enrichment=PromptComponent(
            content="Analyze urgency and suspicious links.",
            token_budget=50
        ),

        cot=PromptComponent(
            content="Briefly analyze before classification.",
            token_budget=25
        ),

        few_shot=FewShotComponent(
            examples=[
                FewShotExample(
                    email="Verify your account immediately.",
                    label="PHISHING",
                    reason="Urgency and credential theft attempt."
                )
            ],
            token_budget=120,
            max_examples=3
        ),

        metadata=PromptMetadata()
    )

    email_data = {
    "sender": "support-paypal-security@gmail.com",
    "receiver": "user@gmail.com",
    "subject": "Urgent Account Verification Required",
    "body": "Your account will be suspended unless verified immediately."
    }

    rendered = PromptRenderer.render_prompt(
        prompt_state,
        email_data
    )

    print(rendered)

    print("\nTotal Tokens:")
    print(prompt_state.get_total_token_count())


if __name__ == "__main__":
    test_render()