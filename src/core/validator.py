"""
src/core/validator.py
---------------------
Deterministic validation for PromptState.

Ensures all constraints are met:
- token budgets
- valid labels
- few-shot capacity
- required types
- non-empty base instruction

Validation strictly REPORTS errors and NEVER mutates the state.
"""

from dataclasses import dataclass
from typing import List

from src.core.prompt_state import PromptState, PromptComponent, FewShotComponent, FewShotExample, EmailInput
from src.core.constants import LABEL_PHISHING, LABEL_SAFE
from src.utils.token_utils import count_component_tokens, count_fewshot_tokens

@dataclass
class ValidationError:
    """Represents a single validation failure."""
    component: str
    message: str

@dataclass
class ValidationResult:
    """Aggregate result of validating a PromptState."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]


class PromptValidator:
    """
    Static validator for PromptState objects.
    """

    @staticmethod
    def validate_state(state: PromptState) -> ValidationResult:
        """
        Validates the entire PromptState against all constraints.
        
        Does not mutate the state. Returns a ValidationResult with errors/warnings.
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        if not hasattr(state, 'base_instruction') or not isinstance(state.base_instruction, str):
            errors.append(ValidationError("base_instruction", "base_instruction must be a string."))
        elif not state.base_instruction.strip():
            errors.append(ValidationError("base_instruction", "base_instruction cannot be empty or whitespace only."))

        # Role
        if not hasattr(state, 'role') or not isinstance(state.role, PromptComponent):
            errors.append(ValidationError("role", "role must be a PromptComponent instance."))
        else:
            errors.extend(PromptValidator.validate_component("role", state.role))

        # Instruction Enrichment
        if not hasattr(state, 'instruction_enrichment') or not isinstance(state.instruction_enrichment, PromptComponent):
            errors.append(ValidationError("instruction_enrichment", "instruction_enrichment must be a PromptComponent instance."))
        else:
            errors.extend(PromptValidator.validate_component("instruction_enrichment", state.instruction_enrichment))

        # CoT
        if not hasattr(state, 'cot') or not isinstance(state.cot, PromptComponent):
            errors.append(ValidationError("cot", "cot must be a PromptComponent instance."))
        else:
            errors.extend(PromptValidator.validate_component("cot", state.cot))

        # Few Shot
        if not hasattr(state, 'few_shot') or not isinstance(state.few_shot, FewShotComponent):
            errors.append(ValidationError("few_shot", "few_shot must be a FewShotComponent instance."))
        else:
            errors.extend(PromptValidator.validate_fewshot(state.few_shot))

        # Renderability requirements
        # e.g., missing metadata is technically an error for a full valid state
        if not hasattr(state, 'metadata'):
            errors.append(ValidationError("metadata", "metadata must be present on the PromptState."))

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def validate_component(name: str, component: PromptComponent) -> List[ValidationError]:
        """
        Validates a single PromptComponent.
        """
        errors = []
        
        if not isinstance(component.content, str):
            errors.append(ValidationError(name, f"content must be a string, got {type(component.content).__name__}."))
        
        if not isinstance(component.token_budget, int) or component.token_budget < 0:
            errors.append(ValidationError(name, f"token_budget must be a non-negative integer (got {component.token_budget})."))
            
        # If token_budget is valid and content is string, check budget
        if isinstance(component.content, str) and isinstance(component.token_budget, int) and component.token_budget >= 0:
            used_tokens = count_component_tokens(component)
            if used_tokens > component.token_budget:
                errors.append(ValidationError(name, f"Token budget exceeded: {used_tokens} > {component.token_budget}."))
                
        return errors

    @staticmethod
    def validate_fewshot(few_shot: FewShotComponent) -> List[ValidationError]:
        """
        Validates the FewShotComponent and its examples.
        """
        errors = []
        
        if not isinstance(few_shot.examples, list):
            errors.append(ValidationError("few_shot", "examples must be a list."))
            return errors  # Cannot proceed safely
            
        if not isinstance(few_shot.token_budget, int) or few_shot.token_budget < 0:
            errors.append(ValidationError("few_shot", f"token_budget must be a non-negative integer (got {few_shot.token_budget})."))
            
        if not isinstance(few_shot.max_examples, int) or few_shot.max_examples < 0:
            errors.append(ValidationError("few_shot", f"max_examples must be a non-negative integer (got {few_shot.max_examples})."))
            
        if isinstance(few_shot.max_examples, int) and few_shot.max_examples >= 0:
            if len(few_shot.examples) > few_shot.max_examples:
                errors.append(ValidationError("few_shot", f"Capacity exceeded: {len(few_shot.examples)} examples > {few_shot.max_examples} max_examples."))
                
        valid_labels = {LABEL_PHISHING, LABEL_SAFE}
        
        for idx, ex in enumerate(few_shot.examples):
            if not isinstance(ex, FewShotExample):
                errors.append(ValidationError("few_shot", f"Example {idx} is not a FewShotExample instance."))
                continue

            if ex.label not in valid_labels:
                errors.append(ValidationError("few_shot", f"Example {idx} has invalid label: '{ex.label}'."))
                
            # Verify Email Input Renderability
            if isinstance(ex.email, EmailInput):
                # An EmailInput should be strings for its attributes
                if not isinstance(ex.email.subject, str) or not isinstance(ex.email.body, str):
                    errors.append(ValidationError("few_shot", f"Example {idx} EmailInput must have string subject and body."))
            elif isinstance(ex.email, str):
                if not ex.email.strip():
                    errors.append(ValidationError("few_shot", f"Example {idx} has an empty email string."))
            else:
                errors.append(ValidationError("few_shot", f"Example {idx} has invalid email type: {type(ex.email).__name__}."))
                
        # Token Budget
        if isinstance(few_shot.token_budget, int) and few_shot.token_budget >= 0:
            try:
                used_tokens = count_fewshot_tokens(few_shot)
                if used_tokens > few_shot.token_budget:
                    errors.append(ValidationError("few_shot", f"Token budget exceeded: {used_tokens} > {few_shot.token_budget}."))
            except Exception as e:
                errors.append(ValidationError("few_shot", f"Failed to count tokens: {str(e)}"))

        return errors
