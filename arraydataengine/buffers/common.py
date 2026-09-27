"""Helpers shared by the persistent buffer backends (TileDB, Arrow)."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from urllib.parse import quote

SPATIAL_INDEX_DIMS = 3


def resolve_topic_path(group_uri, topic: str, paths: dict[str, str]) -> str:
    """Allocate an escaped topic path, retaining hydrated legacy locations.

    The suffix keeps data arrays separate from TileDB timestamp sidecars.
    Existing directories are reserved even when their legacy name happens
    to match the new encoding.
    """
    if topic not in paths:
        base = Path(group_uri) / ("topic-" + quote(topic, safe="") + ".data")
        candidate = base
        suffix = 0
        while str(candidate) in paths.values() or candidate.exists():
            suffix += 1
            candidate = Path(str(base) + f".{suffix}")
        paths[topic] = str(candidate)
    return paths[topic]


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
    """Axis-aligned bounding box of message data usable as a spatial index.

    The last axis holds coordinates, so organized clouds (``H x W x 3``) and
    unorganized clouds (``N x 3``) are indexed alike. Invalid bounds mean
    "unknown": spatial pushdown must keep such rows for the exact filter.
    """

    mins = np.full(SPATIAL_INDEX_DIMS, np.nan, dtype=np.float64)
    maxs = np.full(SPATIAL_INDEX_DIMS, np.nan, dtype=np.float64)
    values = np.asarray(data)
    if values.size == 0 or values.ndim == 0 or values.shape[-1] < 1:
        return False, mins, maxs

    dims = min(int(values.shape[-1]), SPATIAL_INDEX_DIMS)
    coords = values.reshape((-1, int(values.shape[-1])))[:, :dims]
    if coords.dtype.kind in "iu":
        # Integer payloads (e.g. images) are always finite; skip the float copy.
        mins[:dims] = np.min(coords, axis=0)
        maxs[:dims] = np.max(coords, axis=0)
        return True, mins, maxs
    if coords.dtype.kind not in "fb":
        # Complex, datetime, and string payloads have no spatial meaning.
        return False, mins, maxs
    coords = coords.astype(np.float64, copy=False)

    finite = np.isfinite(coords).all(axis=1)
    if not finite.any():
        return False, mins, maxs

    valid_coords = coords if finite.all() else coords[finite]
    mins[:dims] = np.min(valid_coords, axis=0)
    maxs[:dims] = np.max(valid_coords, axis=0)
    return True, mins, maxs


def spatial_overlap_mask(valid, spatial_min, spatial_max, min_bound, max_bound) -> np.ndarray:
    """Rows whose stored bounds may intersect the query box.

    ``spatial_min``/``spatial_max`` are sequences of per-column arrays in
    query-column order. Rows without valid bounds cannot be pruned.
    """
    valid = np.asarray(valid, dtype=bool)
    overlap = np.ones(valid.shape[0], dtype=bool)
    for bound_index, (lower, upper) in enumerate(zip(spatial_min, spatial_max)):
        overlap &= np.asarray(lower, dtype=np.float64) <= max_bound[bound_index]
        overlap &= np.asarray(upper, dtype=np.float64) >= min_bound[bound_index]
    return ~valid | overlap


def slice_contains(index: int, start: int | None, stop: int | None, step: int | None) -> bool:
    start = 0 if start is None else start
    step = 1 if step is None else step
    if index < start:
        return False
    if stop is not None and index >= stop:
        return False
    return (index - start) % step == 0


def apply_index_range(keep: np.ndarray, counter: int, start, stop, step) -> tuple[np.ndarray, int]:
    """Vectorized lazy ``index_range`` over the rows still kept in a chunk.

    ``counter`` is the number of rows that reached the operation in earlier
    chunks. Returns the narrowed mask and the updated counter.
    """
    kept = np.flatnonzero(keep)
    positions = counter + np.arange(kept.size, dtype=np.int64)
    start = 0 if start is None else int(start)
    step = 1 if step is None else int(step)
    contains = positions >= start
    if stop is not None:
        contains &= positions < int(stop)
    contains &= (positions - start) % step == 0
    narrowed = np.zeros_like(keep)
    narrowed[kept[contains]] = True
    return narrowed, counter + kept.size


def index_ranges_exhausted(operations, counters) -> bool:
    """True once any lazy ``index_range`` has passed its stop: no later row can match."""
    return any(
        operation.kind == "index_range"
        and operation.args[1] is not None
        and counters[index] >= operation.args[1]
        for index, operation in enumerate(operations)
    )


def native_message_data(data) -> np.ndarray:
    """Message payload as an array in native byte order."""
    values = np.asarray(data)
    if values.dtype.byteorder not in ("=", "|"):
        values = values.astype(values.dtype.newbyteorder("="))
    return values


def check_message_schema(topic: str, data: np.ndarray, shape: tuple, dtype) -> None:
    """Reject payloads that do not match a topic's fixed shape and dtype."""
    if tuple(data.shape) != tuple(shape):
        raise ValueError(
            f"topic {topic} messages must keep shape {tuple(shape)}, got {tuple(data.shape)}; "
            "pad variable-size messages to a fixed shape before appending"
        )
    if data.dtype != np.dtype(dtype):
        raise ValueError(f"topic {topic} messages must keep dtype {np.dtype(dtype)}, got {data.dtype}")


def raise_collected(errors: list[BaseException], what: str) -> None:
    """Raise errors gathered while closing every topic: one as-is, several as a group."""
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(f"failed to close {len(errors)} {what} topics", errors)


def check_persistable_data(topic: str, data: np.ndarray) -> None:
    """Reject payloads a persistent (fixed-schema) store cannot represent."""
    if data.dtype.hasobject or data.dtype.fields is not None or data.dtype.kind == "V":
        raise ValueError(f"topic {topic} messages must have a plain numeric, bool, or string dtype, got {data.dtype}")
    if data.size == 0:
        raise ValueError(
            f"topic {topic} messages must hold at least one element, got shape {tuple(data.shape)}; "
            "pad variable-size messages to a fixed shape before appending"
        )
