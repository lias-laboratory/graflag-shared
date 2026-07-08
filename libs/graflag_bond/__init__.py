"""
GraFlag Bond - Generic PyGOD Detector Wrapper

This library provides a unified interface for running PyGOD anomaly detection
methods through the GraFlag framework.
"""

from .detectors import BondDetector
from .utils import get_all_parameters

__version__ = "1.0.0"
__all__ = [
    "BondDetector",
    "get_all_parameters"
]
