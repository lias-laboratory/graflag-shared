#!/usr/bin/env python3
"""Standalone script to run evaluation on an experiment."""

import sys
from pathlib import Path

# Add graflag_evaluator to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graflag_evaluator import Evaluator

def main():
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
    
    print(f"\n[OK] Evaluation complete! Results saved to: {eval_path}")

if __name__ == "__main__":
    main()
