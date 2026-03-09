"""Main evaluator orchestrator."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .metrics import MetricCalculator
from .plots import PlotGenerator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Main evaluation orchestrator for GraFlag experiments.
    
    Automatically:
    1. Loads results.json from experiment directory
    2. Detects result type and loads appropriate data
    3. Computes all relevant metrics
    4. Generates evaluation plots (ROC, PR, spot curves)
    5. Saves evaluation.json with all metrics and metadata
    """
    
    def __init__(self, experiment_path: Path):
        """
        Initialize evaluator for an experiment.
        
        Args:
            experiment_path: Path to experiment directory
        """
        self.experiment_path = Path(experiment_path)
        self.results_path = self.experiment_path / "results.json"
        self.eval_dir = self.experiment_path / "eval"
        
        if not self.results_path.exists():
            raise FileNotFoundError(f"results.json not found in {self.experiment_path}")
        
        # Create eval directory
        self.eval_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        self.result_type = self.results.get("result_type")
        if not self.result_type:
            raise ValueError("result_type not found in results.json")
        
        logger.info(f"[INFO] Evaluating experiment: {self.experiment_path.name}")
        logger.info(f"   Result type: {self.result_type}")
    
    def _load_scores_and_ground_truth(self) -> tuple:
        """Load scores and ground truth from results."""
        scores_raw = self.results.get("scores", [])
        ground_truth_raw = self.results.get("ground_truth", [])
        
        if len(scores_raw) == 0:
            raise ValueError("No scores found in results.json")
        if len(ground_truth_raw) == 0:
            raise ValueError("No ground_truth found in results.json")
        
        # Handle ragged arrays (e.g., TEMPORAL_EDGE_ANOMALY_SCORES where each
        # snapshot has different number of edges). Use dtype=object for ragged.
        try:
            scores = np.array(scores_raw)
        except ValueError:
            # Ragged array - use object dtype
            scores = np.array(scores_raw, dtype=object)
        
        try:
            ground_truth = np.array(ground_truth_raw)
        except ValueError:
            # Ragged array - use object dtype
            ground_truth = np.array(ground_truth_raw, dtype=object)
        
        logger.info(f"   Scores shape: {scores.shape}, dtype: {scores.dtype}")
        logger.info(f"   Ground truth shape: {ground_truth.shape}, dtype: {ground_truth.dtype}")
        
        return scores, ground_truth
    
    def _find_spot_files(self) -> Dict[str, Path]:
        """Find all spot CSV files in experiment directory."""
        spot_files = {}
        for csv_file in self.experiment_path.glob("*.csv"):
            metric_key = csv_file.stem  # filename without extension
            spot_files[metric_key] = csv_file
        
        if spot_files:
            logger.info(f"   Found {len(spot_files)} spot files: {list(spot_files.keys())}")
        
        return spot_files
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics for the experiment.
        
        Returns:
            Dictionary of computed metrics
        """
        scores, ground_truth = self._load_scores_and_ground_truth()
        
        # Get additional data (timestamps, edges, etc.)
        kwargs = {
            "timestamps": self.results.get("timestamps"),
            "edges": self.results.get("edges"),
            "node_ids": self.results.get("node_ids"),
            "graph_ids": self.results.get("graph_ids"),
        }
        
        # Compute metrics
        logger.info("[INFO] Computing metrics...")
        metrics = MetricCalculator.calculate_metrics(
            self.result_type, scores, ground_truth, **kwargs
        )
        
        logger.info(f"[OK] Computed {len(metrics)} metrics")
        return metrics
    
    def generate_plots(self) -> list:
        """Generate all evaluation plots.

        Returns:
            List of generated spot curve plot filenames
        """
        logger.info("[INFO] Generating plots...")

        scores, ground_truth = self._load_scores_and_ground_truth()

        # ROC curve
        roc_path = self.eval_dir / "roc_curve.png"
        PlotGenerator.plot_roc_curve(scores, ground_truth, roc_path,
                                     title=f"ROC Curve - {self.experiment_path.name}")

        # PR curve
        pr_path = self.eval_dir / "pr_curve.png"
        PlotGenerator.plot_pr_curve(scores, ground_truth, pr_path,
                                   title=f"PR Curve - {self.experiment_path.name}")

        # Score distribution
        dist_path = self.eval_dir / "score_distribution.png"
        PlotGenerator.plot_score_distribution(scores, ground_truth, dist_path,
                                             title=f"Score Distribution - {self.experiment_path.name}")

        # Spot curves from spot files (generates separate files)
        spot_files = self._find_spot_files()
        spot_plot_files = []
        if spot_files:
            spot_plot_files = PlotGenerator.plot_spot_curves(
                spot_files, self.eval_dir,
                title=f"Spot Curves - {self.experiment_path.name}"
            )

        logger.info(f"[OK] Plots saved to {self.eval_dir}")
        return spot_plot_files
    
    def evaluate(self) -> Path:
        """
        Run full evaluation: compute metrics and generate plots.

        Returns:
            Path to evaluation.json
        """
        # Compute metrics
        computed_metrics = self.compute_metrics()

        # Generate plots (returns list of spot curve plot filenames)
        spot_plot_files = self.generate_plots()

        # Build evaluation results
        evaluation = {
            "experiment_name": self.experiment_path.name,
            "result_type": self.result_type,
            "metrics": computed_metrics,
            "metadata": self.results.get("metadata", {}),
            "plots": {
                "roc_curve": "roc_curve.png",
                "pr_curve": "pr_curve.png",
                "score_distribution": "score_distribution.png",
            },
        }

        # Add spot curve plots if available
        spot_files = self._find_spot_files()
        if spot_files:
            evaluation["spot_files"] = list(spot_files.keys())
            # Add each spot curve plot
            for plot_file in spot_plot_files:
                plot_key = plot_file.replace('.png', '')
                evaluation["plots"][plot_key] = plot_file
        
        # Save evaluation.json
        eval_json_path = self.eval_dir / "evaluation.json"
        with open(eval_json_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"[OK] Evaluation complete!")
        logger.info(f"   Results: {eval_json_path}")
        logger.info(f"   Plots: {self.eval_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"[INFO] Evaluation Summary: {self.experiment_path.name}")
        print("="*60)
        print(f"Result Type: {self.result_type}")
        print(f"\nKey Metrics:")
        for key, value in computed_metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                print(f"  {key}: {value}")
        print(f"\nPlots saved to: {self.eval_dir}")
        print("="*60 + "\n")
        
        return eval_json_path
