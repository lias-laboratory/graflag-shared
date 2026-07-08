"""Plot generation utilities for evaluation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn import metrics
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _flatten_ragged(arr: np.ndarray) -> np.ndarray:
    """Flatten array, handling ragged/object arrays properly."""
    if arr.dtype == object or (arr.ndim == 1 and len(arr) > 0 and isinstance(arr[0], (list, np.ndarray))):
        # Ragged array - concatenate all elements
        return np.concatenate([np.asarray(x).flatten() for x in arr])
    return arr.flatten()


class PlotGenerator:
    """Generate evaluation plots."""
    
    @staticmethod
    def plot_roc_curve(scores: np.ndarray, ground_truth: np.ndarray, 
                       output_path: Path, title: str = "ROC Curve"):
        """
        Generate ROC curve plot.
        
        Args:
            scores: Anomaly scores
            ground_truth: Ground truth labels
            output_path: Path to save plot
            title: Plot title
        """
        scores_flat = _flatten_ragged(scores)
        gt_flat = _flatten_ragged(ground_truth)
        
        # Remove invalid scores
        valid_mask = scores_flat > -2
        scores_valid = scores_flat[valid_mask]
        gt_valid = gt_flat[valid_mask]
        
        if len(np.unique(gt_valid)) < 2:
            logger.warning("Cannot plot ROC: only one class present")
            return
        
        fpr, tpr, thresholds = metrics.roc_curve(gt_valid, scores_valid)
        auc_score = metrics.auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"[OK] ROC curve saved to {output_path}")
    
    @staticmethod
    def plot_pr_curve(scores: np.ndarray, ground_truth: np.ndarray,
                     output_path: Path, title: str = "Precision-Recall Curve"):
        """
        Generate Precision-Recall curve plot.
        
        Args:
            scores: Anomaly scores
            ground_truth: Ground truth labels
            output_path: Path to save plot
            title: Plot title
        """
        scores_flat = _flatten_ragged(scores)
        gt_flat = _flatten_ragged(ground_truth)
        
        valid_mask = scores_flat > -2
        scores_valid = scores_flat[valid_mask]
        gt_valid = gt_flat[valid_mask]
        
        if len(np.unique(gt_valid)) < 2:
            logger.warning("Cannot plot PR: only one class present")
            return
        
        precision, recall, thresholds = metrics.precision_recall_curve(gt_valid, scores_valid)
        auc_score = metrics.auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AUC-PR = {auc_score:.4f}', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"[OK] PR curve saved to {output_path}")
    
    @staticmethod
    def plot_score_distribution(scores: np.ndarray, ground_truth: np.ndarray,
                                output_path: Path, title: str = "Score Distribution"):
        """
        Generate score distribution plot (histogram for anomalies vs normal).
        
        Args:
            scores: Anomaly scores
            ground_truth: Ground truth labels
            output_path: Path to save plot
            title: Plot title
        """
        scores_flat = _flatten_ragged(scores)
        gt_flat = _flatten_ragged(ground_truth)
        
        valid_mask = scores_flat > -2
        scores_valid = scores_flat[valid_mask]
        gt_valid = gt_flat[valid_mask]
        
        normal_scores = scores_valid[gt_valid == 0]
        anomaly_scores = scores_valid[gt_valid == 1]
        
        plt.figure(figsize=(8, 6))
        plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', color='red')
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"[OK] Score distribution saved to {output_path}")
    
    @staticmethod
    def plot_spot_curves(spot_files: Dict[str, Path], output_dir: Path,
                        title: str = "Spot Curves") -> List[str]:
        """
        Generate separate spot curve plots from spot CSV files.

        Args:
            spot_files: Dictionary mapping metric_key to CSV path
            output_dir: Directory to save plots
            title: Plot title prefix

        Returns:
            List of generated plot filenames
        """
        if not spot_files:
            logger.warning("No spot files to plot")
            return []

        generated_plots = []

        for metric_key, csv_path in spot_files.items():
            try:
                df = pd.read_csv(csv_path)

                # Get columns to plot (exclude timestamp and epoch)
                plot_cols = [col for col in df.columns if col not in ('timestamp', 'epoch')]

                if not plot_cols:
                    continue

                # Create a separate plot for this spot file
                plt.figure(figsize=(10, 6))

                for col in plot_cols:
                    plt.plot(df.index, df[col], label=col, marker='o', markersize=3, linewidth=1.5)

                plt.xlabel('Step/Epoch', fontsize=12)
                plt.ylabel('Value', fontsize=12)
                plt.title(f"{metric_key.replace('_', ' ').title()} Curves", fontsize=14)
                plt.legend(fontsize=10, loc='best')
                plt.grid(alpha=0.3)
                plt.tight_layout()

                # Save with metric_key name
                output_filename = f"{metric_key}_curves.png"
                output_path = output_dir / output_filename
                plt.savefig(output_path, dpi=150)
                plt.close()

                generated_plots.append(output_filename)
                logger.info(f"[OK] {metric_key} curves saved to {output_path}")

            except Exception as e:
                logger.error(f"Error plotting {metric_key}: {e}")

        return generated_plots
