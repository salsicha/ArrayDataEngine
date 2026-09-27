from __future__ import annotations

import numpy as np

from .core import topic_view


def describe_topic(topic_data, name: str | None = None) -> dict:
    """Summary statistics for one buffered topic.

    Returns a plain dict with message counts, time range, rate and
    inter-message jitter, the data schema, and (for numeric data) value
    statistics. Works on topic dicts, structured arrays, and TopicViews.

    Time statistics use only finite timestamps, taken in time order, so
    out-of-order or NaN stamps do not produce negative durations or
    intervals. ``non_monotonic_count`` counts backwards steps between
    consecutive finite timestamps as recorded; ``nonfinite_ts_count``
    counts NaN/inf timestamps.
    """

    view = topic_view(topic_data, copy=False)
    ts = np.asarray(view.timestamps, dtype=np.float64)
    data = np.asarray(view.data)
    count = int(ts.shape[0])
    recorded = ts[np.isfinite(ts)]
    finite_ts = np.sort(recorded)
    finite_count = int(finite_ts.shape[0])

    result: dict = {
        "topic": name if name is not None else view.metadata.topic,
        "count": count,
        "start_time": None if finite_count == 0 else float(finite_ts[0]),
        "end_time": None if finite_count == 0 else float(finite_ts[-1]),
        "duration": None if finite_count < 2 else float(finite_ts[-1] - finite_ts[0]),
        "rate_hz": None,
        "dt_mean": None,
        "dt_std": None,
        "dt_min": None,
        "dt_max": None,
        "non_monotonic_count": int(np.count_nonzero(np.diff(recorded) < 0)),
        "nonfinite_ts_count": count - finite_count,
        "dtype": str(data.dtype),
        "shape": tuple(data.shape[1:]),
        "frame_id": view.metadata.frame_id,
    }

    if finite_count >= 2:
        dt = np.diff(finite_ts)
        result["dt_mean"] = float(dt.mean())
        result["dt_std"] = float(dt.std())
        result["dt_min"] = float(dt.min())
        result["dt_max"] = float(dt.max())
        if result["duration"] and result["duration"] > 0:
            result["rate_hz"] = float((finite_count - 1) / result["duration"])

    if count and np.issubdtype(data.dtype, np.number):
        values = data.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        finite_count = int(finite.sum())
        result["nan_fraction"] = float(1.0 - finite_count / values.size) if values.size else 0.0
        if finite_count:
            finite_values = values[finite]
            result["data_min"] = float(finite_values.min())
            result["data_max"] = float(finite_values.max())
            result["data_mean"] = float(finite_values.mean())

    return result


def describe_dataset(dataset) -> dict[str, dict]:
    """Per-topic `describe_topic` summaries.

    Accepts a mapping of ``{topic: topic_data}`` or a DataBuffer (its valid
    buffered rows are summarized per topic).
    """

    if hasattr(dataset, "get_topics") and hasattr(dataset, "topic_view"):
        # DataBuffer: topic_view attaches frame/source metadata
        return {
            topic: describe_topic(dataset.topic_view(topic, copy=False), name=topic)
            for topic in dataset.get_topics()
        }
    return {topic: describe_topic(data, name=topic) for topic, data in dict(dataset).items()}


def format_describe(summaries: dict[str, dict]) -> str:
    """Human-readable table for `describe_dataset` output."""

    lines = []
    for topic, info in summaries.items():
        rate = "-" if info["rate_hz"] is None else f"{info['rate_hz']:.1f} Hz"
        duration = "-" if info["duration"] is None else f"{info['duration']:.2f} s"
        jitter = "-" if info["dt_std"] is None else f"{info['dt_std'] * 1e3:.2f} ms"
        frame = info.get("frame_id") or "-"
        line = (
            f"{topic}: {info['count']} msgs | {rate} | {duration} | "
            f"jitter {jitter} | {info['dtype']} {info['shape']} | frame {frame}"
        )
        if info.get("non_monotonic_count"):
            line += f" | {info['non_monotonic_count']} out-of-order"
        if info.get("nonfinite_ts_count"):
            line += f" | {info['nonfinite_ts_count']} non-finite ts"
        lines.append(line)
    return "\n".join(lines)
