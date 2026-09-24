"""
Utility functions for graflag_bond.

Dynamically handles parameter extraction from environment variables.
Converts values to appropriate Python types based on parameter names and values.
"""

from typing import Dict, Any

import torch.nn.functional as F
from graflag_runner import params


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
    func = getattr(F, func_name, None)
    if callable(func):
        return func
    # Silently falling back to relu would benchmark a different model than the
    # one requested, which invalidates the comparison the run exists to make.
    available = sorted(n for n in dir(F) if not n.startswith("_") and callable(getattr(F, n)))
    raise ValueError(
        f"Unknown activation '{activation_value}'. "
        f"Expected a torch.nn.functional name, e.g. one of: "
        f"{', '.join(available[:8])}, ..."
    )


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
        cls = getattr(pyg_nn, class_name, None)
        if cls is not None:
            return cls
        raise ValueError(
            f"Unknown backbone '{backbone_value}': torch_geometric.nn has no "
            f"'{class_name}'. Returning None here surfaced later as "
            f"\"'NoneType' object is not callable\" from inside PyGOD."
        )
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve backbone '{backbone_value}': torch_geometric is "
            f"not installed in this image."
        ) from exc


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
    # Handle activation functions (callable). Dispatch on the parameter's
    # expected type as well as the string: gating only on the fully-qualified
    # name made the short forms ('relu', 'GCN') that both resolvers explicitly
    # support unreachable, so they were passed through as raw strings and blew
    # up mid-forward-pass with "'str' object is not callable".
    if 'torch.nn.functional' in env_value or env_name.upper() == '_ACT':
        return get_activation_function(env_value)

    # Handle backbone classes (torch.nn.Module)
    if 'torch_geometric.nn' in env_value or env_name.upper() == '_BACKBONE':
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
    """Read the detector's keyword arguments from the environment.

    The generic half of this -- scanning `_FOO` variables, stripping the
    prefix, coercing to the signature's types and dropping what the callee
    does not accept -- now lives in graflag_runner.method.params(), where
    every method can use it. What stays here is what is specific to PyGOD:
    resolving `torch.nn.functional.relu` and `torch_geometric.nn.GCN` from
    their names.

    Args:
        detector_class: the PyGOD detector whose __init__ the result must fit.

    Returns:
        Keyword arguments for the detector's constructor.
    """
    return params(detector_class, convert=convert_env_value)
