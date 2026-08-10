"""Helpers shared by the persistent buffer backends (TileDB, Arrow)."""

from __future__ import annotations

import numpy as np

SPATIAL_INDEX_DIMS = 3


def encode_name(name) -> bytes:
    if isinstance(name, bytes):
        return name[:256]
    return str(name).encode()[:256]


def decode_frame_id(value) -> str | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    return None


def encode_frame_id(value) -> bytes:
    decoded = decode_frame_id(value)
    if decoded is None:
        return b""
    return decoded.encode()[:256]


def spatial_bounds_for_data(data) -> tuple[bool, np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of message data usable as a spatial index."""

    mins = np.full(SPATIAL_INDEX_DIMS, np.nan, dtype=np.float64)
    maxs = np.full(SPATIAL_INDEX_DIMS, np.nan, dtype=np.float64)
    values = np.asarray(data)
    if values.size == 0 or values.ndim == 0 or values.ndim > 2 or values.shape[-1] < 1:
        return False, mins, maxs

    dims = min(int(values.shape[-1]), SPATIAL_INDEX_DIMS)
    try:
        coords = values.astype(np.float64, copy=False).reshape((-1, int(values.shape[-1])))[:, :dims]
    except (TypeError, ValueError):
        return False, mins, maxs

    finite = np.isfinite(coords).all(axis=1)
    if not finite.any():
        return False, mins, maxs

    valid_coords = coords[finite]
    mins[:dims] = np.min(valid_coords, axis=0)
    maxs[:dims] = np.max(valid_coords, axis=0)
    return True, mins, maxs


def slice_contains(index: int, start: int | None, stop: int | None, step: int | None) -> bool:
    start = 0 if start is None else start
    step = 1 if step is None else step
    if index < start:
        return False
    if stop is not None and index >= stop:
        return False
    return (index - start) % step == 0
