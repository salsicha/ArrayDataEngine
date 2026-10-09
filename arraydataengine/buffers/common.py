"""Helpers shared by the persistent buffer backends (TileDB, Arrow)."""

from __future__ import annotations

import json
import os
import re

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


def is_flexible_string(dtype) -> bool:
    """A string dtype without a bound itemsize (TileDB var-length strings)."""
    dtype = np.dtype(dtype)
    return dtype.kind in "SU" and dtype.itemsize == 0


def conform_dtype(topic: str, data: np.ndarray, dtype) -> np.ndarray:
    """`data` in the topic's stored dtype.

    Identical dtypes pass unchanged; safe widening casts (``int16`` into an
    ``int32`` topic, ``float32`` into ``float64``, ``<U3`` into ``<U5``) are
    applied; anything that could lose information is rejected.
    """
    dtype = np.dtype(dtype)
    if data.dtype == dtype:
        return data
    if is_flexible_string(dtype):
        # Var-length string storage keeps any length of the same kind.
        if data.dtype.kind == dtype.kind:
            return data
    elif np.can_cast(data.dtype, dtype, "safe") and (
        data.dtype.kind == dtype.kind or (data.dtype.kind in "biufc" and dtype.kind in "biufc")
    ):
        return data.astype(dtype)
    raise ValueError(f"topic {topic} messages must keep dtype {dtype}, got {data.dtype}")


def check_message_schema(topic: str, data: np.ndarray, shape: tuple, dtype) -> np.ndarray:
    """Reject payloads that do not match a topic's fixed shape and dtype.

    Returns `data`, safely widened to the stored dtype when needed.
    """
    if tuple(data.shape) != tuple(shape):
        raise ValueError(
            f"topic {topic} messages must keep shape {tuple(shape)}, got {tuple(data.shape)}; "
            "pad variable-size messages to a fixed shape before appending"
        )
    return conform_dtype(topic, data, dtype)


# -- variable-size (ragged) topics ---------------------------------------------


def validate_capacity(value, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return int(value)


def normalize_ragged_topics(ragged_topics) -> dict:
    """``ragged_topics`` as ``{topic: capacity or None}``.

    Accepts a mapping, or an iterable of topic names (capacity from
    ``max_points``/the default).
    """
    if not ragged_topics:
        return {}
    if isinstance(ragged_topics, str):
        ragged_topics = [ragged_topics]
    if not hasattr(ragged_topics, "items"):
        ragged_topics = {topic: None for topic in ragged_topics}
    return {
        str(topic): validate_capacity(capacity, f"ragged_topics[{topic!r}]")
        for topic, capacity in ragged_topics.items()
    }


def ragged_capacity(topic: str, ragged_topics: dict, max_points: int | None) -> int:
    """Rows per message of a ragged topic: ``ragged_topics[topic]`` >
    ``max_points`` (DataBuffer, else the source's) > ``DEFAULT_MAX_POINTS``."""
    capacity = (ragged_topics or {}).get(topic)
    if capacity is not None:
        return int(capacity)
    if max_points is not None:
        return int(max_points)
    from ..sensors.pointcloud2_sensor import DEFAULT_MAX_POINTS

    return int(DEFAULT_MAX_POINTS)


def message_is_ragged(msg: dict, ragged_topics: dict) -> bool:
    return "point_count" in msg or msg["topic"] in (ragged_topics or {})


def _message_rows(topic: str, data: np.ndarray, msg: dict) -> np.ndarray:
    """The real rows of a variable-size message (rows past ``point_count`` are padding)."""
    if data.ndim == 0:
        raise ValueError(f"topic {topic} is variable-size, so its messages need at least one axis, got a scalar")
    count = msg.get("point_count")
    if count is not None and not isinstance(count, (bool, np.bool_)):
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = None
        if count is not None and 0 <= count < data.shape[0]:
            return data[:count]
    return data


def new_ragged_schema(topic: str, data: np.ndarray, msg: dict, capacity: int) -> tuple[tuple, np.dtype]:
    """Stored ``(shape, dtype)`` of a new ragged topic: ``(capacity, *trailing)``."""
    rows = _message_rows(topic, data, msg)
    shape = (int(capacity), *rows.shape[1:])
    if int(np.prod(shape, dtype=np.int64)) == 0:
        raise ValueError(f"topic {topic} messages must hold at least one element per row, got shape {tuple(data.shape)}")
    return shape, rows.dtype


def conform_ragged(topic: str, data: np.ndarray, msg: dict, shape: tuple, dtype) -> tuple[np.ndarray, int]:
    """Validate a ragged message: returns its real rows (unpadded) and their count."""
    rows = _message_rows(topic, data, msg)
    capacity, trailing = int(shape[0]), tuple(shape[1:])
    if tuple(rows.shape[1:]) != trailing:
        raise ValueError(
            f"topic {topic} messages must keep trailing shape {trailing} (rows, *{trailing}), "
            f"got {tuple(rows.shape)}"
        )
    length = int(rows.shape[0])
    if length > capacity:
        raise ValueError(
            f"topic {topic} message has {length} rows, more than the topic's capacity of {capacity}; "
            f"raise it with DataBuffer(ragged_topics={{{topic!r}: {length}}}) or DataBuffer(max_points=...) "
            "before the topic is first stored (an existing store keeps its capacity)"
        )
    return conform_dtype(topic, rows, dtype), length


def conform_message(topic: str, data: np.ndarray, msg: dict, shape: tuple, dtype, ragged: bool):
    """Validate a message for an existing topic.

    Returns ``(rows, length)``: for ragged topics the unpadded rows and their
    count; otherwise the payload in the stored dtype and None. Point clouds
    (messages with ``point_count``) appended to a fixed-shape topic written
    by older versions are zero-padded to its row count, as those versions did.
    """
    if ragged:
        return conform_ragged(topic, data, msg, shape, dtype)
    shape = tuple(shape)
    if (
        "point_count" in msg and tuple(data.shape) != shape and data.ndim == len(shape) >= 1
        and tuple(data.shape[1:]) == shape[1:] and data.shape[0] <= shape[0]
    ):
        padded = np.zeros(shape, dtype=np.dtype(dtype) if not is_flexible_string(dtype) else data.dtype)
        padded[: data.shape[0]] = conform_dtype(topic, data, dtype)
        return padded, None
    return check_message_schema(topic, data, shape, dtype), None


def pad_rows(rows_list, lengths, shape, dtype) -> np.ndarray:
    """Stack unpadded ragged rows into a zero-padded ``(n, *shape)`` array."""
    out = np.zeros((len(rows_list), *shape), dtype=dtype)
    for index, (rows, length) in enumerate(zip(rows_list, lengths)):
        if length:
            out[index, :length] = rows
    return out


def trim_rows(data: np.ndarray, lengths) -> list:
    """Unpadded rows of padded ragged ``data`` (one array per message)."""
    if lengths is None:
        return list(data)
    return [row[: int(length)] for row, length in zip(data, lengths)]


def source_topic_count(source, topic: str, cache: dict) -> int | None:
    """The source's total message count for `topic`, or None when unknown.

    Topics the source does not list (direct appends of derived topics) and
    non-positive counts (e.g. ``BaseSource.get_count`` for unknown topics)
    are unknown, so they never cap a topic.
    """
    if topic in cache:
        return cache[topic]
    count = None
    get_count = getattr(source, "get_count", None)
    if callable(get_count):
        try:
            count = int(get_count(topic))
        except Exception:
            count = None
        if count is not None and count <= 0:
            count = None
        if count is not None:
            get_topics = getattr(source, "get_topics", None)
            if callable(get_topics):
                try:
                    listed = cache.get(_LISTED_TOPICS)
                    if listed is None:
                        listed = cache[_LISTED_TOPICS] = frozenset(get_topics())
                    if topic not in listed:
                        count = None
                except Exception:
                    pass
    cache[topic] = count
    return count


_LISTED_TOPICS = object()


ARROW_MANIFEST_KEYS = frozenset({"topic", "count", "shape", "dtype"})
_ARROW_TOPIC_DIR = re.compile(r"topic-.+\.data(?:\.\d+)?")


def is_arrow_topic_dir_name(name: str) -> bool:
    return _ARROW_TOPIC_DIR.fullmatch(name) is not None


def parse_arrow_manifest(text: str) -> dict | None:
    """A manifest dict when `text` is an Arrow topic manifest, else None."""
    try:
        manifest = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(manifest, dict) or not ARROW_MANIFEST_KEYS <= manifest.keys():
        return None
    if not isinstance(manifest.get("count"), int) or not isinstance(manifest.get("topic"), str):
        return None
    fragments = manifest.get("fragments")
    if fragments is not None and not isinstance(fragments, list):
        return None
    return manifest


def holds_arrow_store(path) -> bool:
    """True when `path` contains Arrow topic directories with manifests."""
    try:
        entries = list(os.scandir(path))
    except OSError:
        return False
    for entry in entries:
        if not entry.is_dir():
            continue
        manifest = os.path.join(entry.path, "manifest.json")
        if not os.path.isfile(manifest):
            continue
        if is_arrow_topic_dir_name(entry.name):
            return True
        try:
            with open(manifest, encoding="utf-8") as handle:
                if parse_arrow_manifest(handle.read(1 << 20)) is not None:
                    return True
        except OSError:
            continue
    return False


TILEDB_GROUP_ENTRIES = frozenset({"__tiledb_group.tdb", "__group", "__meta"})


def is_tiledb_group_dir(path) -> bool:
    try:
        return bool(set(os.listdir(path)) & TILEDB_GROUP_ENTRIES)
    except OSError:
        return False


def holds_tiledb_arrays(path) -> bool:
    """True when `path` contains TileDB array directories."""
    try:
        entries = list(os.scandir(path))
    except OSError:
        return False
    return any(
        entry.is_dir() and (
            os.path.isdir(os.path.join(entry.path, "__schema"))
            or os.path.isfile(os.path.join(entry.path, "__array_schema.tdb"))
        )
        for entry in entries
    )


def is_tiledb_store(path) -> bool:
    """Whether `path` is a TileDB store (rather than an Arrow one).

    Older versions created an empty TileDB group inside Arrow stores opened
    with ``backend="tiledb"``; such a directory is still an Arrow store.
    """
    if not is_tiledb_group_dir(path):
        return False
    return holds_tiledb_arrays(path) or not holds_arrow_store(path)


def raise_collected(errors: list[BaseException], what: str) -> None:
    """Raise errors gathered while closing every topic: one as-is, several as a group."""
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(f"failed to close {len(errors)} {what} topics", errors)


def check_persistable_data(topic: str, data: np.ndarray, shape: tuple | None = None) -> None:
    """Reject payloads a persistent (fixed-schema) store cannot represent.

    `shape` is the stored row shape when it differs from the payload's
    (zero-padded variable-size topics).
    """
    if data.dtype.hasobject or data.dtype.fields is not None or data.dtype.kind == "V":
        raise ValueError(f"topic {topic} messages must have a plain numeric, bool, or string dtype, got {data.dtype}")
    if data.dtype.kind in "SU" and data.dtype.itemsize == 0:
        raise ValueError(f"topic {topic} messages must have a sized string dtype, got {data.dtype}")
    if (data.size if shape is None else int(np.prod(shape, dtype=np.int64))) == 0:
        raise ValueError(
            f"topic {topic} messages must hold at least one element, got shape {tuple(data.shape)}; "
            "pad variable-size messages to a fixed shape before appending"
        )
