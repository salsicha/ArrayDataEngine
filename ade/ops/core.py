from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def topic_parts(topic_data: dict | np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Return `(ids, timestamps, data)` from ADE topic data."""

    if isinstance(topic_data, np.ndarray) and topic_data.dtype.fields:
        ids = topic_data["id"] if "id" in topic_data.dtype.fields else None
        return ids, np.asarray(topic_data["ts"], dtype=np.float64), np.asarray(topic_data["data"])

    if not isinstance(topic_data, dict) or "ts" not in topic_data or "data" not in topic_data:
        raise TypeError("topic_data must be a structured topic array or a dict with 'ts' and 'data'")

    ts = np.asarray(topic_data["ts"], dtype=np.float64)
    data = np.asarray(topic_data["data"])
    ids = topic_data.get("id")
    if ids is None:
        return None, ts, data

    ids_array = np.asarray(ids)
    if ids_array.ndim == 0 or ids_array.shape[:1] != ts.shape[:1]:
        ids_array = np.full(ts.shape, ids, dtype=object)
    return ids_array, ts, data


def _make_topic(ids: np.ndarray | None, ts: np.ndarray, data: np.ndarray) -> dict:
    result = {"ts": np.asarray(ts, dtype=np.float64), "data": np.asarray(data)}
    if ids is not None:
        result["id"] = np.asarray(ids)
    return result


def _call_with_metadata(fn: Callable, data: np.ndarray, ts: float, message_id: Any) -> Any:
    try:
        return fn(data, ts, message_id)
    except TypeError:
        try:
            return fn(data, ts)
        except TypeError:
            return fn(data)


def select_indices(topic_data: dict | np.ndarray, start: int | None = None, stop: int | None = None, step: int | None = None) -> dict:
    ids, ts, data = topic_parts(topic_data)
    selection = slice(start, stop, step)
    selected_ids = None if ids is None else ids[selection].copy()
    return _make_topic(selected_ids, ts[selection].copy(), data[selection].copy())


def select_time_range(topic_data: dict | np.ndarray, start: float, end: float, inclusive: bool = True) -> dict:
    if start > end:
        raise ValueError("start must be less than or equal to end")

    ids, ts, data = topic_parts(topic_data)
    if inclusive:
        mask = (ts >= start) & (ts <= end)
    else:
        mask = (ts > start) & (ts < end)
    selected_ids = None if ids is None else ids[mask].copy()
    return _make_topic(selected_ids, ts[mask].copy(), data[mask].copy())


def map_topic(topic_data: dict | np.ndarray, fn: Callable, copy: bool = True) -> dict:
    ids, ts, data = topic_parts(topic_data)
    mapped = [
        _call_with_metadata(fn, value.copy() if copy else value, float(timestamp), None if ids is None else ids[i])
        for i, (timestamp, value) in enumerate(zip(ts, data))
    ]
    return _make_topic(None if ids is None else ids.copy(), ts.copy(), np.asarray(mapped))


def filter_topic(topic_data: dict | np.ndarray, predicate: Callable) -> dict:
    ids, ts, data = topic_parts(topic_data)
    mask = np.asarray([
        bool(_call_with_metadata(predicate, value, float(timestamp), None if ids is None else ids[i]))
        for i, (timestamp, value) in enumerate(zip(ts, data))
    ])
    selected_ids = None if ids is None else ids[mask].copy()
    return _make_topic(selected_ids, ts[mask].copy(), data[mask].copy())


def reduce_topic(topic_data: dict | np.ndarray, fn: Callable, initial: Any | None = None) -> Any:
    ids, ts, data = topic_parts(topic_data)
    iterator = enumerate(zip(ts, data))
    if initial is None:
        try:
            i, (timestamp, value) = next(iterator)
        except StopIteration as exc:
            raise ValueError("cannot reduce an empty topic without an initial value") from exc
        acc = value
    else:
        acc = initial

    for i, (timestamp, value) in iterator:
        try:
            acc = fn(acc, value, float(timestamp), None if ids is None else ids[i])
        except TypeError:
            acc = fn(acc, value)
    return acc


def window_topic(topic_data: dict | np.ndarray, size: int | None = None, seconds: float | None = None) -> Iterable[dict]:
    ids, ts, data = topic_parts(topic_data)
    if size is None and seconds is None:
        raise ValueError("size or seconds must be provided")
    if size is not None and size < 1:
        raise ValueError("size must be at least 1")
    if seconds is not None and seconds < 0:
        raise ValueError("seconds must be non-negative")

    for end_index in range(ts.size):
        start_index = 0
        if size is not None:
            start_index = max(start_index, end_index - size + 1)
        if seconds is not None:
            start_index = max(start_index, int(np.searchsorted(ts, ts[end_index] - seconds, side="left")))

        window_slice = slice(start_index, end_index + 1)
        selected_ids = None if ids is None else ids[window_slice].copy()
        yield _make_topic(selected_ids, ts[window_slice].copy(), data[window_slice].copy())


def nearest_time_index(timestamps: np.ndarray, query_time: float, tolerance: float | None = None) -> int | None:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size == 0:
        return None

    insert_at = int(np.searchsorted(ts, query_time))
    candidates = []
    if insert_at < ts.size:
        candidates.append(insert_at)
    if insert_at > 0:
        candidates.append(insert_at - 1)

    best = min(candidates, key=lambda i: abs(ts[i] - query_time))
    if tolerance is not None and abs(ts[best] - query_time) > tolerance:
        return None
    return int(best)


def align_nearest(reference_topic: dict | np.ndarray, target_topic: dict | np.ndarray, tolerance: float | None = None) -> dict:
    _, ref_ts, _ = topic_parts(reference_topic)
    target_ids, target_ts, target_data = topic_parts(target_topic)

    indices = np.array([
        -1 if (idx := nearest_time_index(target_ts, float(timestamp), tolerance)) is None else idx
        for timestamp in ref_ts
    ], dtype=np.int64)
    valid = indices >= 0
    safe_indices = np.where(valid, indices, 0)

    aligned = {
        "reference_ts": ref_ts.copy(),
        "target_ts": np.where(valid, target_ts[safe_indices], np.nan),
        "target_index": indices,
        "valid": valid,
    }
    aligned_data = np.full((ref_ts.size,) + target_data.shape[1:], np.nan, dtype=np.float64)
    if valid.any():
        aligned_data[valid] = target_data[indices[valid]]
    aligned["data"] = aligned_data
    if target_ids is not None:
        aligned["id"] = np.asarray([target_ids[i] if ok else None for i, ok in zip(safe_indices, valid)], dtype=object)
    return aligned
