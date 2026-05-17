"""
main.py — CLI Entry Point for the Phishing Email Prompt Optimizer.
"""

import os
import argparse
from dotenv import load_dotenv

from src.config import DATA_DIR, LOGS_DIR, MODEL_NAME, MAX_ITERATIONS
from src.llm import init_client
from src.data import load_dataset
from src.loop import run_optimization


def main():
    parser = argparse.ArgumentParser(description="Phishing Email Prompt Optimizer")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory containing CSV files")
    parser.add_argument("--log-dir", type=str, default=LOGS_DIR,
                        help="Directory to save logs and final prompt")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                        help="Maximum number of loop iterations")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="OpenAI model name to use")
    
    args = parser.parse_args()

    # Load env variables (like OPENAI_API_KEY)
    load_dotenv()

    # API key not needed for local Ollama

    # Initialize LLM client
    init_client()

    print(f"Loading data from {args.data_dir} ...")
    try:
        rows = load_dataset(args.data_dir)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Run the optimization loop
    try:
        run_optimization(
            rows=rows,
            max_iterations=args.max_iterations,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nOptimization failed: {e}")


if __name__ == "__main__":
    main()
