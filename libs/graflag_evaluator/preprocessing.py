"""Shared score/ground-truth preparation for metrics and plots.

Metrics and plots used to flatten and filter independently, with different
rules, so the AUC printed on a plot could disagree with the AUC written to
``evaluation.json`` for the same input. Both now go through :func:`prepare_pairs`.

The filtering is explicit rather than inferred. It removes exactly the sentinel
values defined in ``RESULTS_STANDARD.md`` plus anything non-finite, and reports
what it dropped so the evaluation can record it instead of silently shrinking
the sample.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Special score values defined in RESULTS_STANDARD.md.
SENTINEL_UNKNOWN = -1.0   # Unknown / unassigned
SENTINEL_INACTIVE = -2.0  # Inactive / unseen at this time step
SENTINELS = (SENTINEL_UNKNOWN, SENTINEL_INACTIVE)


@dataclass
class FilterReport:
    """What :func:`prepare_pairs` removed, so callers can record it."""

    total: int = 0
    kept: int = 0
    dropped_unknown: int = 0
    dropped_inactive: int = 0
    dropped_non_finite: int = 0

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    def to_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped_unknown": self.dropped_unknown,
            "dropped_inactive": self.dropped_inactive,
            "dropped_non_finite": self.dropped_non_finite,
        }


def flatten_ragged(arr: Any) -> np.ndarray:
    """Flatten an array, handling ragged/object arrays.

    The decision is made per array. Deciding it from one array and applying it
    to another silently corrupts the pairing when the two have different
    nesting -- which is exactly what the ``-2`` padding convention invites,
    since scores end up rectangular while ground truth stays ragged.
    """
    arr = np.asarray(arr, dtype=object) if _is_ragged_sequence(arr) else np.asarray(arr)

    if arr.dtype == object or (
        arr.ndim == 1 and len(arr) > 0 and isinstance(arr[0], (list, np.ndarray))
    ):
        if len(arr) == 0:
            return np.asarray([], dtype=float)
        return np.concatenate([np.asarray(x).flatten() for x in arr])
    return arr.flatten()


def _is_ragged_sequence(arr: Any) -> bool:
    """True for a list of unequal-length sequences, which numpy cannot stack."""
    if isinstance(arr, np.ndarray) or not isinstance(arr, (list, tuple)):
        return False
    lengths = {len(x) for x in arr if isinstance(x, (list, tuple, np.ndarray))}
    return len(lengths) > 1


def prepare_pairs(scores: Any, ground_truth: Any):
    """Flatten both arrays and drop sentinel / non-finite entries.

    Returns ``(scores_valid, gt_valid, report)`` as float and int arrays plus a
    :class:`FilterReport`.

    Raises:
        ValueError: if the two arrays do not flatten to the same length. That
            mismatch used to surface as an opaque numpy boolean-index error
            attributed to the evaluator rather than to the method that produced
            the mismatched output.
    """
    scores_flat = flatten_ragged(scores).astype(float, copy=False)
    gt_flat = flatten_ragged(ground_truth)

    if scores_flat.shape[0] != gt_flat.shape[0]:
        raise ValueError(
            f"scores and ground_truth flatten to different lengths "
            f"({scores_flat.shape[0]} vs {gt_flat.shape[0]}). The method's "
            f"scores and labels are not aligned; check their grouping."
        )

    report = FilterReport(total=int(scores_flat.shape[0]))

    is_unknown = scores_flat == SENTINEL_UNKNOWN
    is_inactive = scores_flat == SENTINEL_INACTIVE
    is_non_finite = ~np.isfinite(scores_flat)

    report.dropped_unknown = int(np.count_nonzero(is_unknown))
    report.dropped_inactive = int(np.count_nonzero(is_inactive))
    report.dropped_non_finite = int(np.count_nonzero(is_non_finite))

    keep = ~(is_unknown | is_inactive | is_non_finite)
    scores_valid = scores_flat[keep]
    gt_valid = np.asarray(gt_flat[keep]).astype(int, copy=False)
    report.kept = int(scores_valid.shape[0])

    if report.dropped:
        logger.info(
            "[INFO] Excluded %d/%d samples (unknown=%d, inactive=%d, non-finite=%d)",
            report.dropped, report.total, report.dropped_unknown,
            report.dropped_inactive, report.dropped_non_finite,
        )

    return scores_valid, gt_valid, report
