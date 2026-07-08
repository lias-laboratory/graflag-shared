"""GraFlag Evaluator - Modular evaluation system for graph anomaly detection."""

from .metrics import MetricCalculator, get_metrics_for_type
from .evaluator import Evaluator

__all__ = ["MetricCalculator", "get_metrics_for_type", "Evaluator"]

__version__ = "1.1.0"
