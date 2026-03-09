"""
Utility functions for graflag_bond.

Dynamically handles parameter extraction from environment variables.
Converts values to appropriate Python types based on parameter names and values.
"""

import os
from typing import Dict, Any
import torch.nn.functional as F


def str_to_bool(value: str) -> bool:
    """Convert string to boolean."""
    return value.lower() in ('true', '1', 'yes')


def get_activation_function(activation_value: str):
    """
    Convert activation function path/name to PyTorch activation function.
    Handles any torch.nn.functional activation function dynamically.
    
    Args:
        activation_value: Activation function path (e.g., 'torch.nn.functional.relu')
        
    Returns:
        PyTorch activation function
    """
    # Extract function name from full path
    if 'torch.nn.functional.' in activation_value:
        func_name = activation_value.split('.')[-1]
    else:
        func_name = activation_value
    
    # Get the function from torch.nn.functional
    if hasattr(F, func_name):
        return getattr(F, func_name)
    else:
        # Default to relu if function not found
        return F.relu


def get_backbone_class(backbone_value: str):
    """
    Convert backbone path to PyTorch Geometric class.
    Handles any torch_geometric.nn class dynamically.
    
    Args:
        backbone_value: Backbone class path (e.g., 'torch_geometric.nn.GCN')
        
    Returns:
        PyTorch Geometric class or None
    """
    if backbone_value.lower() == 'none':
        return None
    
    try:
        # Extract class name from full path
        if 'torch_geometric.nn.' in backbone_value:
            class_name = backbone_value.split('.')[-1]
        else:
            class_name = backbone_value
        
        # Import torch_geometric.nn
        import torch_geometric.nn as pyg_nn
        
        # Get the class dynamically
        if hasattr(pyg_nn, class_name):
            return getattr(pyg_nn, class_name)
        else:
            return None
    except ImportError:
        return None


def convert_env_value(env_name: str, env_value: str, expected_type: type = None) -> Any:
    """
    Convert environment variable value to appropriate Python type.
    
    Args:
        env_name: Name of environment variable (uppercase)
        env_value: String value from environment
        expected_type: Expected type from function signature (if available)
        
    Returns:
        Converted value with appropriate type
    """
    # Handle activation functions (callable)
    if 'torch.nn.functional' in env_value:
        return get_activation_function(env_value)
    
    # Handle backbone classes (torch.nn.Module)
    if 'torch_geometric.nn' in env_value:
        return get_backbone_class(env_value)
    
    # Handle None
    if env_value.lower() == 'none':
        return None
    
    # Handle boolean values
    if env_value.lower() in ['true', 'false']:
        return str_to_bool(env_value)
    
    # If we have expected type from signature, use it
    if expected_type is not None:
        try:
            if expected_type == float:
                return float(env_value)
            elif expected_type == int:
                return int(env_value)
            elif expected_type == bool:
                return str_to_bool(env_value)
            elif expected_type == str:
                return env_value
        except (ValueError, TypeError):
            pass
    
    # Fallback: Try to detect type from value
    try:
        # Try int first (if no decimal point)
        if '.' not in env_value:
            return int(env_value)
        
        # Has decimal point, convert to float
        return float(env_value)
    except (ValueError, AttributeError):
        pass
    
    # Return as string if conversion fails
    return env_value


def get_all_parameters(detector_class=None) -> Dict[str, Any]:
    """
    Get all parameters from environment variables.
    Only reads environment variables prefixed with underscore (_PARAM_NAME).
    Automatically converts parameter names from _UPPER_CASE to lower_case
    and values to appropriate Python types based on detector signature.
    
    Args:
        detector_class: Optional detector class to inspect for parameter types
    
    Returns:
        Dictionary of all parameters with correct types
    """
    import inspect
    
    params = {}
    
    # Get parameter types from detector signature if available
    param_types = {}
    if detector_class is not None:
        try:
            sig = inspect.signature(detector_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'args', 'kwargs']:
                    continue
                
                # First try to get type from annotation
                if param.annotation != inspect.Parameter.empty:
                    param_types[param_name] = param.annotation
                # If no annotation, get type from default value
                elif param.default != inspect.Parameter.empty and param.default is not None:
                    param_types[param_name] = type(param.default)
        except (ValueError, TypeError):
            pass
    
    # Iterate through all environment variables
    for env_name, env_value in os.environ.items():
        # Only process variables that start with underscore
        if not env_name.startswith('_'):
            continue
        
        # Remove underscore prefix and convert to lowercase
        param_name = env_name[1:].lower()
        
        # Get expected type from signature
        expected_type = param_types.get(param_name)
        
        # Convert value to appropriate type
        param_value = convert_env_value(env_name, env_value, expected_type)
        
        # Add to parameters
        params[param_name] = param_value
    
    return params
