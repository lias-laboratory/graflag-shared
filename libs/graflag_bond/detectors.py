"""
PyGOD Detector Enumeration

Dynamically discovers and maps all PyGOD detector classes.
"""

import inspect
import pygod.detector


class BondDetector:
    """Dynamic PyGOD detector registry."""
    
    _detectors = None
    
    @classmethod
    def _load_detectors(cls):
        """Load all detector classes from pygod.detector module."""
        if cls._detectors is not None:
            return
        
        cls._detectors = {}
        
        # Inspect pygod.detector module for all classes
        for name, obj in inspect.getmembers(pygod.detector, inspect.isclass):
            # Filter to only include classes defined in pygod.detector
            if obj.__module__.startswith('pygod.detector'):
                # Store with lowercase name as key
                cls._detectors[name.lower()] = obj
    
    @classmethod
    def from_method_name(cls, method_name: str):
        """
        Get detector class from method name.
        
        Args:
            method_name: Method name (e.g., 'bond_dominant', 'dominant', 'DOMINANT')
            
        Returns:
            Detector class name (lowercase)
            
        Raises:
            ValueError: If method name is not supported
        """
        cls._load_detectors()
        
        # Remove bond_ prefix if present and convert to lowercase
        name = method_name.lower().replace("bond_", "")
        
        if name not in cls._detectors:
            supported = ", ".join(sorted(cls._detectors.keys()))
            raise ValueError(f"Unsupported detector: {name}. Supported: {supported}")
        
        return name
    
    @classmethod
    def get_detector_class(cls, detector_name: str):
        """
        Get the PyGOD detector class by name.
        
        Args:
            detector_name: Detector name (e.g., 'dominant', 'adone')
            
        Returns:
            PyGOD detector class
            
        Raises:
            ValueError: If detector name is not found
        """
        cls._load_detectors()
        
        name = detector_name.lower()
        if name not in cls._detectors:
            supported = ", ".join(sorted(cls._detectors.keys()))
            raise ValueError(f"Detector not found: {name}. Available: {supported}")
        
        return cls._detectors[name]
    
    @classmethod
    def list_detectors(cls):
        """
        List all available detector names.
        
        Returns:
            List of detector names (lowercase)
        """
        cls._load_detectors()
        return sorted(cls._detectors.keys())
