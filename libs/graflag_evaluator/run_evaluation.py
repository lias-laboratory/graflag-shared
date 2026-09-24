#!/usr/bin/env python3
"""Standalone script to run evaluation on an experiment."""

import logging
import sys
from pathlib import Path

# Add graflag_evaluator to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graflag_evaluator import Evaluator

def main():
    # Without this every logger.info/warning in the package is discarded,
    # which is why silent metric failures were invisible in container logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if len(sys.argv) < 2:
        print("Usage: run_evaluation.py <experiment_directory>")
        print("Example: run_evaluation.py /shared/experiments/exp_name")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    if not (exp_dir / "results.json").exists():
        print(f"Error: results.json not found in {exp_dir}")
        sys.exit(1)
    
    # Run evaluation
    print(f"[INFO] Loading experiment from: {exp_dir}")
    evaluator = Evaluator(exp_dir)
    eval_path = evaluator.evaluate()

    # A non-zero exit is what lets the orchestrator tell a crashed evaluation
    # from a clean one; previously every outcome exited 0.
    errors = getattr(evaluator, "errors", [])
    if errors:
        print(f"\n[FAIL] Evaluation finished with {len(errors)} problem(s); "
              f"see {eval_path}")
        sys.exit(1)

    print(f"\n[OK] Evaluation complete! Results saved to: {eval_path}")

if __name__ == "__main__":
    main()
