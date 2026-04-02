"""Metric calculators for different result types."""

import importlib.util
import numpy as np
from pathlib import Path
from sklearn import metrics
from typing import Dict, List, Any, Callable
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """
    Base class for metric calculation.
    
    Supports plugin-based architecture for adding new metrics.
    """
    
    # Registry of metric functions by result type
    _METRIC_REGISTRY: Dict[str, List[Callable]] = {}
    
    @classmethod
    def register_metric(cls, result_type: str, metric_func: Callable):
        """
        Register a new metric function for a result type.
        
        Args:
            result_type: Result type (e.g., "EDGE_STREAM_ANOMALY_SCORES")
            metric_func: Function that takes (scores, ground_truth, **kwargs) 
                        and returns Dict[str, float]
        """
        if result_type not in cls._METRIC_REGISTRY:
            cls._METRIC_REGISTRY[result_type] = []
        cls._METRIC_REGISTRY[result_type].append(metric_func)
        logger.debug(f"Registered metric {metric_func.__name__} for {result_type}")
    
    @classmethod
    def load_plugins(cls, *plugin_dirs: Path):
        """Load custom metric plugins from directories.

        Each ``.py`` file in the given directories is imported.  The file is
        expected to call ``MetricCalculator.register_metric()`` at import time
        to register its metrics.

        Non-existent directories are silently skipped.

        Args:
            *plugin_dirs: Paths to directories containing plugin ``.py`` files.
        """
        for plugin_dir in plugin_dirs:
            plugin_dir = Path(plugin_dir)
            if not plugin_dir.is_dir():
                continue
            for py_file in sorted(plugin_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                module_name = f"graflag_plugin_{py_file.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    logger.info(f"[INFO] Loaded plugin: {py_file.name}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load plugin {py_file.name}: {e}")

    @classmethod
    def calculate_metrics(cls, result_type: str, scores: np.ndarray,
                         ground_truth: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Calculate all registered metrics for a result type.
        
        Args:
            result_type: Type of anomaly detection result
            scores: Anomaly scores
            ground_truth: Ground truth labels
            **kwargs: Additional parameters (timestamps, edges, etc.)
        
        Returns:
            Dictionary of computed metrics
        """
        if result_type not in cls._METRIC_REGISTRY:
            logger.warning(f"No metrics registered for {result_type}")
            return {}
        
        all_metrics = {}
        for metric_func in cls._METRIC_REGISTRY[result_type]:
            try:
                result = metric_func(scores, ground_truth, **kwargs)
                all_metrics.update(result)
            except Exception as e:
                logger.error(f"Error in {metric_func.__name__}: {e}")
        
        return all_metrics


# ============================================================================
# Standard Metrics for Binary Anomaly Detection
# ============================================================================

def compute_classification_metrics(scores: np.ndarray, ground_truth: np.ndarray, 
                                   **kwargs) -> Dict[str, float]:
    """
    Compute standard classification metrics (works for all types).
    
    Metrics:
    - AUC-ROC: Area under ROC curve
    - AUC-PR: Area under Precision-Recall curve
    - Precision@K: Precision in top K predictions
    - Recall@K: Recall in top K predictions
    - F1@K: F1 score in top K predictions
    - Best F1: Best F1 score across all thresholds
    """
    # Handle nested lists (e.g., TEMPORAL_EDGE_ANOMALY_SCORES where each snapshot
    # has different number of edges). np.array() creates object array for ragged lists.
    if scores.dtype == object or (scores.ndim == 1 and isinstance(scores[0], (list, np.ndarray))):
        # Flatten nested structure
        scores_flat = np.concatenate([np.asarray(s).flatten() for s in scores])
        gt_flat = np.concatenate([np.asarray(g).flatten() for g in ground_truth])
    else:
        scores_flat = scores.flatten()
        gt_flat = ground_truth.flatten()
    
    # Remove invalid scores (-2, -1) if present
    valid_mask = (scores_flat >= 0) & (scores_flat <= 1) if np.max(scores_flat) <= 1 else scores_flat > -2
    scores_valid = scores_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]
    
    if len(np.unique(gt_valid)) < 2:
        logger.warning("Ground truth has only one class, skipping some metrics")
        return {"auc_roc": None, "auc_pr": None}
    
    # AUC-ROC
    auc_roc = metrics.roc_auc_score(gt_valid, scores_valid)
    
    # AUC-PR
    precision, recall, _ = metrics.precision_recall_curve(gt_valid, scores_valid)
    auc_pr = metrics.auc(recall, precision)
    
    # Precision/Recall/F1 at K (K = number of anomalies)
    k = int(np.sum(gt_valid))
    top_k_indices = np.argsort(scores_valid)[-k:]
    predictions_at_k = np.zeros_like(gt_valid)
    predictions_at_k[top_k_indices] = 1
    
    precision_at_k = metrics.precision_score(gt_valid, predictions_at_k, zero_division=0)
    recall_at_k = metrics.recall_score(gt_valid, predictions_at_k, zero_division=0)
    f1_at_k = metrics.f1_score(gt_valid, predictions_at_k, zero_division=0)
    
    # Best F1 across all thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1 = np.max(f1_scores)
    best_f1_threshold = _[np.argmax(f1_scores)] if len(_) > 0 else None
    
    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "precision_at_k": round(float(precision_at_k), 4),
        "recall_at_k": round(float(recall_at_k), 4),
        "f1_at_k": round(float(f1_at_k), 4),
        "best_f1": round(float(best_f1), 4),
        "best_f1_threshold": round(float(best_f1_threshold), 4) if best_f1_threshold else None,
        "num_anomalies": int(k),
        "num_samples": int(len(gt_valid)),
        "anomaly_ratio": round(float(k / len(gt_valid)), 4),
    }


def compute_temporal_metrics(scores: np.ndarray, ground_truth: np.ndarray, 
                            timestamps: List[int] = None, **kwargs) -> Dict[str, float]:
    """
    Compute temporal-specific metrics.
    
    Metrics:
    - Early detection rate: How early anomalies are detected
    - Temporal consistency: How consistent scores are over time
    """
    if timestamps is None:
        return {}
    
    # Early detection: average time between first high score and actual anomaly
    # (This is a placeholder - implement based on your specific needs)
    
    return {
        "temporal_span": int(max(timestamps) - min(timestamps)) if timestamps else 0,
        "num_timestamps": len(set(timestamps)) if timestamps else 0,
    }


def compute_edge_metrics(scores: np.ndarray, ground_truth: np.ndarray,
                        edges: List[List[int]] = None, **kwargs) -> Dict[str, float]:
    """
    Compute edge-specific metrics.
    
    Metrics:
    - Number of unique edges
    - Edge degree distribution stats
    """
    if edges is None:
        return {}
    
    # Count unique edges
    unique_edges = len(set(tuple(e) for e in edges))
    
    # Node degree stats (how many times each node appears)
    nodes = [n for edge in edges for n in edge]
    unique_nodes = len(set(nodes))
    
    return {
        "num_unique_edges": int(unique_edges),
        "num_unique_nodes": int(unique_nodes),
        "total_edge_occurrences": int(len(edges)),
    }


# ============================================================================
# Register Default Metrics
# ============================================================================

# Register for all result types
for result_type in [
    "NODE_ANOMALY_SCORES",
    "EDGE_ANOMALY_SCORES", 
    "GRAPH_ANOMALY_SCORES",
    "TEMPORAL_NODE_ANOMALY_SCORES",
    "TEMPORAL_EDGE_ANOMALY_SCORES",
    "TEMPORAL_GRAPH_ANOMALY_SCORES",
    "NODE_STREAM_ANOMALY_SCORES",
    "EDGE_STREAM_ANOMALY_SCORES",
    "GRAPH_STREAM_ANOMALY_SCORES",
]:
    MetricCalculator.register_metric(result_type, compute_classification_metrics)

# Register temporal metrics for temporal and stream types
for result_type in [
    "TEMPORAL_NODE_ANOMALY_SCORES",
    "TEMPORAL_EDGE_ANOMALY_SCORES",
    "TEMPORAL_GRAPH_ANOMALY_SCORES",
    "NODE_STREAM_ANOMALY_SCORES",
    "EDGE_STREAM_ANOMALY_SCORES",
    "GRAPH_STREAM_ANOMALY_SCORES",
]:
    MetricCalculator.register_metric(result_type, compute_temporal_metrics)

# Register edge metrics for edge types
for result_type in [
    "EDGE_ANOMALY_SCORES",
    "TEMPORAL_EDGE_ANOMALY_SCORES",
    "EDGE_STREAM_ANOMALY_SCORES",
]:
    MetricCalculator.register_metric(result_type, compute_edge_metrics)


def get_metrics_for_type(result_type: str) -> List[str]:
    """
    Get list of available metrics for a result type.
    
    Args:
        result_type: Result type string
    
    Returns:
        List of metric names
    """
    if result_type not in MetricCalculator._METRIC_REGISTRY:
        return []
    
    # Extract metric names from registered functions
    metric_names = []
    for func in MetricCalculator._METRIC_REGISTRY[result_type]:
        metric_names.append(func.__name__)
    
    return metric_names
