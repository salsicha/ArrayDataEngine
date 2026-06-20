from __future__ import annotations

import numpy as np


def normalize_image(image: np.ndarray, min_value=None, max_value=None, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(image)
    min_value = arr.min() if min_value is None else min_value
    max_value = arr.max() if max_value is None else max_value
    span = max_value - min_value
    if span == 0:
        return np.zeros(arr.shape, dtype=dtype)
    return ((arr - min_value) / span).astype(dtype)


def pad_image(image: np.ndarray, pad_width, value=0) -> np.ndarray:
    if np.isscalar(pad_width):
        pad = ((int(pad_width), int(pad_width)), (int(pad_width), int(pad_width)))
    elif len(pad_width) == 2 and all(np.isscalar(v) for v in pad_width):
        pad = ((int(pad_width[0]), int(pad_width[0])), (int(pad_width[1]), int(pad_width[1])))
    else:
        pad = pad_width
    if np.asarray(image).ndim == 3 and len(pad) == 2:
        pad = tuple(pad) + ((0, 0),)
    return np.pad(image, pad, mode="constant", constant_values=value)


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    out_h, out_w = shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError("shape must contain positive height and width")
    row_idx = np.linspace(0, arr.shape[0] - 1, out_h).round().astype(int)
    col_idx = np.linspace(0, arr.shape[1] - 1, out_w).round().astype(int)
    return arr[row_idx][:, col_idx]


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("image must have shape (H, W, 3+)")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return arr[..., :3] @ weights


def valid_depth_mask(depth: np.ndarray, min_depth: float = 0.0, max_depth: float | None = None) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float64)
    mask = np.isfinite(arr) & (arr > min_depth)
    if max_depth is not None:
        mask &= arr <= max_depth
    return mask


def depth_to_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if fx == 0 or fy == 0:
        raise ValueError("fx and fy must be non-zero")
    z = np.asarray(depth, dtype=np.float64) / scale
    valid = valid_depth_mask(z) if mask is None else np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(valid)
    z_valid = z[rows, cols]
    x = (cols - cx) * z_valid / fx
    y = (rows - cy) * z_valid / fy
    return np.column_stack((x, y, z_valid))
