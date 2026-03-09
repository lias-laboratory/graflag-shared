"""Streaming utilities for handling large result data."""

import json
from typing import Iterator, Any, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StreamableArray:
    """
    Wrapper for large arrays that can be generated on-demand.
    
    This allows methods to avoid loading entire large matrices into memory
    by generating rows/values lazily as needed during JSON serialization.
    
    Usage:
        # Instead of building a huge list:
        # scores = [[...], [...], ...]  # 10GB in memory
        
        # Use a generator:
        def generate_rows():
            for i in range(1000000):
                yield compute_row(i)
        
        scores = StreamableArray(generate_rows())
    """
    
    def __init__(self, generator: Iterator):
        """
        Initialize with a generator or iterator.
        
        Args:
            generator: Iterator that yields array elements (rows, values, etc.)
        """
        self.generator = generator
    
    def __iter__(self):
        """Make this object iterable."""
        return self.generator


def stream_write_json(data: dict, output_path: Path, streamable_keys: List[str] = None):
    """
    Write JSON to file with support for streaming large arrays.
    
    This function writes JSON incrementally, streaming any StreamableArray objects
    row-by-row instead of loading them entirely into memory. Regular data is
    serialized normally.
    
    Args:
        data: Dictionary to serialize to JSON
        output_path: Path to output JSON file
        streamable_keys: Keys in data that contain StreamableArray objects
                        (auto-detected if None)
    
    Example:
        data = {
            "result_type": "TEMPORAL_EDGE_ANOMALY_SCORES",
            "scores": StreamableArray(generate_scores()),  # Large!
            "timestamps": [0, 1, 2, ...],  # Regular list
            "metadata": {...}
        }
        stream_write_json(data, Path("results.json"))
    """
    if streamable_keys is None:
        # Auto-detect StreamableArray objects
        streamable_keys = [
            key for key, value in data.items() 
            if isinstance(value, StreamableArray)
        ]
    
    logger.debug(f"Stream writing JSON with streamable keys: {streamable_keys}")
    
    with open(output_path, 'w') as f:
        f.write('{\n')
        
        keys = list(data.keys())
        for idx, key in enumerate(keys):
            value = data[key]
            is_last = (idx == len(keys) - 1)
            
            # Write key
            f.write(f'  {json.dumps(key)}: ')
            
            # Handle streamable arrays
            if isinstance(value, StreamableArray):
                _stream_write_array(f, value, indent=2)
            else:
                # Regular JSON serialization with proper indentation
                serialized = json.dumps(value, indent=2)
                # Indent all lines after the first
                lines = serialized.split('\n')
                f.write(lines[0])
                for line in lines[1:]:
                    f.write('\n  ' + line)
            
            # Comma between fields
            if not is_last:
                f.write(',')
            f.write('\n')
        
        f.write('}\n')
    
    logger.info(f"[OK] Streamed JSON written to: {output_path}")


def _stream_write_array(f, streamable: StreamableArray, indent: int = 0):
    """
    Internal function to stream write an array element by element.
    
    Args:
        f: File handle to write to
        streamable: StreamableArray to serialize
        indent: Indentation level (number of spaces)
    """
    indent_str = ' ' * indent
    f.write('[\n')
    
    first = True
    row_count = 0
    
    for element in streamable:
        if not first:
            f.write(',\n')
        
        # Write element with indentation
        f.write(f'{indent_str}  ')
        json.dump(element, f)
        
        first = False
        row_count += 1
        
        # Progress logging for very large arrays (every 1000 rows)
        if row_count % 1000 == 0:
            logger.debug(f"  ... streamed {row_count} rows")
    
    f.write(f'\n{indent_str}]')
    
    if row_count >= 1000:
        logger.info(f"  Streamed total: {row_count} rows")


def is_streamable(obj: Any) -> bool:
    """
    Check if an object is streamable (generator/iterator).
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is a generator or iterator (excluding strings/lists)
    """
    return (
        isinstance(obj, StreamableArray) or
        (hasattr(obj, '__iter__') and 
         hasattr(obj, '__next__') and 
         not isinstance(obj, (str, bytes, list, tuple, dict)))
    )
