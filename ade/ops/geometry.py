from __future__ import annotations

import numpy as np


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+) with XYZ in the first three columns")
    return arr


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 3x3, 3x4, or 4x4 transform to XYZ columns."""

    arr = _as_points(points)
    matrix = np.asarray(transform, dtype=np.float64)

    if matrix.shape == (3, 3):
        xyz = arr[:, :3] @ matrix.T
    elif matrix.shape == (3, 4):
        xyz = arr[:, :3] @ matrix[:, :3].T + matrix[:, 3]
    elif matrix.shape == (4, 4):
        homogeneous = np.c_[arr[:, :3], np.ones(arr.shape[0], dtype=np.float64)]
        xyz = (homogeneous @ matrix.T)[:, :3]
    else:
        raise ValueError("transform must have shape (3, 3), (3, 4), or (4, 4)")

    result = arr.copy()
    result[:, :3] = xyz
    return result


def crop_bounds(points: np.ndarray, min_bound=None, max_bound=None, return_mask: bool = False):
    """Select points inside axis-aligned XYZ bounds."""

    arr = _as_points(points)
    mask = np.ones(arr.shape[0], dtype=bool)
    if min_bound is not None:
        mask &= np.all(arr[:, :3] >= np.asarray(min_bound, dtype=np.float64), axis=1)
    if max_bound is not None:
        mask &= np.all(arr[:, :3] <= np.asarray(max_bound, dtype=np.float64), axis=1)
    cropped = arr[mask].copy()
    return (cropped, mask) if return_mask else cropped
