from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import topic_view


def _text_array(values) -> np.ndarray:
    """Unicode array of `values`; None becomes "" and bytes are decoded."""

    return np.asarray([
        "" if value is None else (value.decode(errors="replace") if isinstance(value, bytes) else str(value))
        for value in values
    ], dtype=str)


def _object_data_payload(data: np.ndarray) -> dict:
    """Pickle-free encoding of object-dtype topic data.

    Ragged per-row arrays (e.g. variable-size point clouds) are stored as
    their concatenation plus row offsets; per-row strings as a unicode
    array. Anything else is rejected, since storing it would need pickle.
    """

    rows = list(data.ravel()) if data.ndim != 1 else list(data)
    if data.ndim == 1 and rows and all(isinstance(row, np.ndarray) and row.ndim >= 1 for row in rows):
        dtypes = {row.dtype for row in rows}
        trailing = {row.shape[1:] for row in rows}
        if len(dtypes) == 1 and len(trailing) == 1 and next(iter(dtypes)) != object:
            lengths = np.asarray([row.shape[0] for row in rows], dtype=np.int64)
            return {
                "data_ragged_values": np.concatenate(rows, axis=0),
                "data_ragged_offsets": np.concatenate(([0], np.cumsum(lengths))),
                "data_kind": np.asarray("ragged"),
            }
    if all(isinstance(row, (str, bytes)) for row in rows):
        return {
            "data": _text_array(rows).reshape(data.shape),
            "data_kind": np.asarray("text"),
        }
    raise TypeError(
        "save_topic_npz cannot store this object-dtype data without pickle; "
        "supported object payloads are 1-D arrays of per-row NumPy arrays "
        "sharing a dtype and trailing shape (ragged rows) or of strings"
    )


def _npz_path(path) -> Path:
    # np.savez appends ".npz" to any other name; return the file it writes.
    path = Path(path)
    if not path.name.endswith(".npz"):
        path = path.with_name(path.name + ".npz")
    return path


def save_topic_npz(path, topic_data, compressed: bool = True) -> Path:
    """Save a buffered topic to a portable `.npz` file.

    Stores timestamps, data, message ids (as unicode; missing ids become
    empty strings), per-row frame ids, and topic metadata. The file loads
    without pickle. Object-dtype data is supported for ragged per-row
    arrays (stored as values plus offsets) and strings; other object data
    raises TypeError. ``.npz`` is appended to `path` when missing, and the
    returned path is the file actually written.
    """

    view = topic_view(topic_data, copy=False)
    data = np.asarray(view.data)
    payload: dict = {"ts": np.asarray(view.timestamps, dtype=np.float64)}
    if data.dtype == object:
        payload.update(_object_data_payload(data))
    else:
        payload["data"] = data
    if view.ids is not None:
        payload["id"] = _text_array(view.ids)
    if view.frame_ids is not None:
        payload["frame_ids"] = _text_array(view.frame_ids)
    metadata = {
        "topic": view.metadata.topic,
        "source_uri": view.metadata.source_uri,
        "frame_id": view.metadata.frame_id,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata))

    path = _npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **payload)
    return path


def load_topic_npz(path) -> dict:
    """Load a topic saved by `save_topic_npz` back into a topic dict.

    Ragged and string object data come back as 1-D object arrays; per-row
    frame ids come back under ``"frame_ids"`` (empty entries as None).
    """

    with np.load(Path(path), allow_pickle=False) as archive:
        kind = str(archive["data_kind"].item()) if "data_kind" in archive else "array"
        if kind == "ragged":
            values = archive["data_ragged_values"]
            offsets = archive["data_ragged_offsets"]
            data = np.empty(offsets.shape[0] - 1, dtype=object)
            for index in range(data.shape[0]):
                data[index] = values[offsets[index]:offsets[index + 1]].copy()
        elif kind == "text":
            data = archive["data"].astype(object)
        else:
            data = archive["data"].copy()
        result: dict = {
            "ts": archive["ts"].copy(),
            "data": data,
        }
        if "id" in archive:
            ids = archive["id"].astype(object)
            result["id"] = ids
            result["name"] = ids.copy()
        if "frame_ids" in archive:
            frames = archive["frame_ids"].astype(object)
            frames[frames == ""] = None
            result["frame_ids"] = frames
        metadata = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(archive["metadata_json"].item()))

    for key in ("topic", "source_uri", "frame_id"):
        if metadata.get(key) is not None:
            result[key] = metadata[key]
    return result
