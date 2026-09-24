"""Metric calculators for different result types."""

import importlib.util
import sys
import numpy as np
from pathlib import Path
from sklearn import metrics
from typing import Dict, List, Any, Callable

from .preprocessing import prepare_pairs
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """
    Base class for metric calculation.
    
    Supports plugin-based architecture for adding new metrics.
    """
    
    # Registry of metric functions by result type
    _METRIC_REGISTRY: Dict[str, List[Callable]] = {}

    # Plugin files already imported, so repeated Evaluator() construction
    # does not re-register the same metrics.
    _LOADED_PLUGINS: set = set()
    
    @classmethod
    def register_metric(cls, result_type: str, metric_func: Callable):
        """
        Register a new metric function for a result type.
        
        Args:
            result_type: Result type (e.g., "EDGE_STREAM_ANOMALY_SCORES")
            metric_func: Function that takes (scores, ground_truth, **kwargs) 
                        and returns Dict[str, float]
        """
        registered = cls._METRIC_REGISTRY.setdefault(result_type, [])
        # _METRIC_REGISTRY is class-level and load_plugins runs from
        # Evaluator.__init__, so batch-evaluating N experiments in one process
        # re-imported every plugin N times and appended N copies of each
        # function. all_metrics.update() masked it, leaving only an N-times
        # slowdown and duplicated names from get_metrics_for_type().
        if any(getattr(f, "__name__", None) == metric_func.__name__
               and getattr(f, "__module__", None) == metric_func.__module__
               for f in registered):
            logger.debug(
                f"Metric {metric_func.__name__} already registered for {result_type}"
            )
            return
        registered.append(metric_func)
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
                resolved = py_file.resolve()
                if resolved in cls._LOADED_PLUGINS:
                    continue
                # Include the directory in the module name: plugins/x.py and
                # custom_metrics/x.py produced the same name, so an
                # experiment-local plugin meant to replace a global one
                # registered a second copy of both instead.
                module_name = f"graflag_plugin_{plugin_dir.name}_{py_file.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    mod = importlib.util.module_from_spec(spec)
                    # Register before exec: @dataclass, pickle and
                    # typing.get_type_hints look the module up in sys.modules.
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                    cls._LOADED_PLUGINS.add(resolved)
                    logger.info(f"[INFO] Loaded plugin: {py_file.name}")
                except Exception as e:
                    sys.modules.pop(module_name, None)
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

def _select_top_k(scores: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Expected true positives in the top-`k` under random tie-breaking.

    `np.argsort` breaks ties by array position, so a detector emitting a
    constant score got "the last k rows of the file" and could appear to beat
    its own base rate. When ties straddle the k-th boundary the honest answer
    is the expectation over which tied items are picked, which this computes
    exactly: every item strictly above the cut is counted, and the tied block
    contributes its positive rate times the number of slots left.
    """
    if k <= 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    cutoff = scores[order[k - 1]]

    above = scores > cutoff
    tied = scores == cutoff

    tp_strict = int(np.count_nonzero(gt[above] == 1))
    slots_left = k - int(np.count_nonzero(above))
    if slots_left <= 0:
        return float(tp_strict)

    n_tied = int(np.count_nonzero(tied))
    if n_tied == 0:
        return float(tp_strict)
    pos_tied = int(np.count_nonzero(gt[tied] == 1))
    return float(tp_strict + slots_left * (pos_tied / n_tied))


def _empty_classification_metrics(**known) -> Dict[str, Any]:
    """The full key set with `None` for anything not computable.

    The early-return used to emit 2 keys where the success path emits 11,
    forcing every consumer to handle two incompatible shapes -- and it omitted
    exactly the counts that would let a human diagnose the condition.
    """
    result = {
        "auc_roc": None,
        "auc_pr": None,
        "precision_at_k": None,
        "recall_at_k": None,
        "f1_at_k": None,
        "k": None,
        "best_f1": None,
        "best_f1_threshold": None,
        "num_anomalies": None,
        "num_samples": None,
        "anomaly_ratio": None,
    }
    result.update(known)
    return result


def compute_classification_metrics(scores: np.ndarray, ground_truth: np.ndarray,
                                   k: int = None, **kwargs) -> Dict[str, Any]:
    """
    Compute standard classification metrics (works for all types).

    Args:
        scores: Anomaly scores, flat or nested per snapshot.
        ground_truth: Labels, same total length as scores after flattening.
        k: Cut-off for the @k metrics. Defaults to the number of positives, at
           which precision@k and recall@k coincide by construction; pass an
           explicit k to get distinct operating-point numbers.

    Metrics:
    - AUC-ROC: Area under ROC curve
    - AUC-PR: Average precision (the correct PR-curve estimator)
    - Precision@K / Recall@K / F1@K at the chosen K
    - Best F1: Best F1 score across all thresholds
    """
    scores_valid, gt_valid, report = prepare_pairs(scores, ground_truth)

    filtering = {"filtering": report.to_dict()}

    if report.kept == 0:
        logger.warning(
            "No samples left to evaluate: all %d were sentinel or non-finite values",
            report.total,
        )
        return _empty_classification_metrics(num_samples=0, **filtering)

    n_positive = int(np.count_nonzero(gt_valid == 1))
    if len(np.unique(gt_valid)) < 2:
        logger.warning(
            "Ground truth has only one class (%d samples, %d positive); "
            "ranking metrics are undefined",
            report.kept, n_positive,
        )
        return _empty_classification_metrics(
            num_anomalies=n_positive,
            num_samples=int(report.kept),
            anomaly_ratio=round(float(n_positive / report.kept), 4),
            **filtering,
        )

    # AUC-ROC
    auc_roc = metrics.roc_auc_score(gt_valid, scores_valid)

    # AUC-PR. average_precision_score, not auc(recall, precision): the
    # trapezoidal rule linearly interpolates between PR operating points, which
    # is not valid because precision does not vary linearly with recall. The
    # error is largest where consecutive points are far apart -- the low base
    # rates typical of anomaly detection.
    auc_pr = metrics.average_precision_score(gt_valid, scores_valid)

    # Precision/Recall/F1 at K.
    k_used = n_positive if k is None else int(k)
    k_used = max(0, min(k_used, int(report.kept)))
    expected_tp = _select_top_k(scores_valid, gt_valid, k_used)

    precision_at_k = expected_tp / k_used if k_used > 0 else 0.0
    recall_at_k = expected_tp / n_positive if n_positive > 0 else 0.0
    denom = precision_at_k + recall_at_k
    f1_at_k = (2 * precision_at_k * recall_at_k / denom) if denom > 0 else 0.0

    # Best F1 across all thresholds
    precision, recall, thresholds = metrics.precision_recall_curve(gt_valid, scores_valid)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    # precision/recall carry a trailing sentinel point with no threshold.
    best_f1_threshold = (
        float(thresholds[best_idx]) if best_idx < len(thresholds) else None
    )

    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "precision_at_k": round(float(precision_at_k), 4),
        "recall_at_k": round(float(recall_at_k), 4),
        "f1_at_k": round(float(f1_at_k), 4),
        "k": int(k_used),
        "best_f1": round(best_f1, 4),
        # `is not None`, not truthiness: a threshold of 0.0 is legitimate and
        # is exactly what min-max normalised scores produce.
        "best_f1_threshold": (
            round(best_f1_threshold, 4) if best_f1_threshold is not None else None
        ),
        "num_anomalies": int(n_positive),
        "num_samples": int(report.kept),
        "anomaly_ratio": round(float(n_positive / report.kept), 4),
        **filtering,
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
