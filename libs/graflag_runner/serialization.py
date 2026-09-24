"""One place that decides how a Python value becomes JSON.

Both write paths in this package -- the standard `json.dump` in
`results.py` and the incremental writer in `streaming.py` -- have to make the
same two decisions, and they used to make them in separate copies of the same
logic:

* **numpy types.** `ResultWriter.save_scores` stores its keyword arguments
  verbatim, so `np.ndarray`, `np.float32` and friends reach the encoder
  unconverted. `methods/taddy` really does pass a raw ndarray.
* **non-finite floats.** Python writes bare ``NaN`` and ``Infinity`` and reads
  them back again, but they are not valid JSON per RFC 8259, so
  ``JSON.parse``, ``jq`` and Go/Rust parsers all reject them. A diverged run
  (loss → inf) produced scores the dashboard could not load.
"""

import math
from pathlib import Path
from typing import Any, Tuple

try:
    import numpy as np
except ImportError:
    # graflag_runner declares no numpy dependency, and numpy is used here only
    # to recognise numpy values: without it there are none to convert. 1.1.0
    # imported it unconditionally, and the package failed to import at all.
    np = None


def json_default(obj: Any) -> Any:
    """`default=` hook for `json.dump`: serialize what the stdlib cannot.

    Only called for values the encoder does not already understand, so the
    common path pays nothing.
    """
    if np is not None:
        if isinstance(obj, np.generic):
            return to_jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_jsonable(value: Any) -> Any:
    """Return `value` with non-finite floats replaced by None.

    Used where the encoder cannot intervene: `default=` is never consulted for
    a float, so NaN has to be removed before `json.dump` sees it.
    """
    return _Sanitizer()(value)


def sanitize(value: Any) -> Tuple[Any, int]:
    """Like :func:`to_jsonable`, but also report how many values were replaced.

    Callers use the count to warn, so the substitution is visible rather than
    silent.
    """
    sanitizer = _Sanitizer()
    return sanitizer(value), sanitizer.replaced


class _Sanitizer:
    """Recursive non-finite replacement that keeps its own tally."""

    def __init__(self) -> None:
        self.replaced = 0

    def __call__(self, value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                self.replaced += 1
                return None
            return value
        if np is not None:
            if isinstance(value, np.generic):
                return self(value.item())
            if isinstance(value, np.ndarray):
                return self(value.tolist())
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        return value
