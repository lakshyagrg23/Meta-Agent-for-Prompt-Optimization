class OptimizationLoop:
    """
    Main orchestration loop for adaptive prompt refinement.

    Workflow:
    1. Evaluate current PromptState
    2. Extract deterministic signals
    3. Select refinement operator
    4. Generate refined candidate
    5. Validate candidate
    6. Evaluate candidate on SAME batch
    7. Accept/reject candidate
    8. Log iteration history
    """

    def run(self):
        """
        Execute iterative optimization process.
        """