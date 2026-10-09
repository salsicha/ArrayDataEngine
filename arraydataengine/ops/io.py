from __future__ import annotations

import json
import numbers
from pathlib import Path

import numpy as np

from .core import topic_view


def _text_array(values) -> np.ndarray:
    """Unicode array of `values`; None becomes "" and bytes are decoded."""

    return np.asarray([
        "" if value is None else (value.decode(errors="replace") if isinstance(value, bytes) else str(value))
        for value in values
    ], dtype=str)


def _json_text(value):
    """Metadata value as JSON-safe text (bytes decoded), or None."""

    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _ids_payload(ids: np.ndarray) -> np.ndarray:
    """Message ids for the archive: integer ids stay integers, anything
    else is stored as unicode (missing ids become empty strings)."""

    ids = np.asarray(ids)
    if ids.dtype.kind in "iu":
        return ids
    if ids.dtype == object and ids.size and all(
        isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)) for value in ids.ravel()
    ):
        return np.asarray([int(value) for value in ids.ravel()], dtype=np.int64).reshape(ids.shape)
    return _text_array(ids)


def _bytes_rows(rows: list) -> bool:
    return bool(rows) and all(isinstance(row, (bytes, bytearray, np.bytes_)) for row in rows)


def _ragged_rows(rows: list) -> list | None:
    """`rows` as arrays sharing one dtype and trailing shape (empty rows,
    e.g. a cloud with no points, adopt the others' dtype and shape), or
    None when they cannot be stored as one ragged block."""

    if not rows or not all(isinstance(row, np.ndarray) and row.ndim >= 1 for row in rows):
        return None
    reference = [row for row in rows if row.shape[0] > 0] or rows
    dtypes = {row.dtype for row in reference}
    trailing = {row.shape[1:] for row in reference}
    if len(dtypes) != 1 or len(trailing) != 1:
        return None
    dtype = next(iter(dtypes))
    tail = next(iter(trailing))
    if dtype == object:
        return None
    return [row if row.shape[0] > 0 else np.empty((0,) + tail, dtype=dtype) for row in rows]


def _object_data_payload(data: np.ndarray) -> dict:
    """Pickle-free encoding of object-dtype topic data.

    Ragged per-row arrays (e.g. variable-size point clouds) are stored as
    their concatenation plus row offsets; per-row bytes (e.g. encoded
    images) as their concatenated uint8 values plus offsets, so they come
    back byte-for-byte; per-row strings as a unicode array. Anything else
    is rejected, since storing it would need pickle.
    """

    rows = list(data.ravel()) if data.ndim != 1 else list(data)
    if not rows:
        # Nothing to encode; it loads back as an empty object array.
        return {"data": np.zeros(data.shape, dtype=str), "data_kind": np.asarray("text")}
    if data.ndim == 1:
        ragged = _ragged_rows(rows)
        if ragged is not None:
            lengths = np.asarray([row.shape[0] for row in ragged], dtype=np.int64)
            return {
                "data_ragged_values": np.concatenate(ragged, axis=0),
                "data_ragged_offsets": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
                "data_kind": np.asarray("ragged"),
            }
    if _bytes_rows(rows):
        blobs = [bytes(row) for row in rows]
        sizes = np.asarray([len(blob) for blob in blobs], dtype=np.int64)
        return {
            "data_bytes_values": np.frombuffer(b"".join(blobs), dtype=np.uint8).copy(),
            "data_bytes_offsets": np.concatenate(([0], np.cumsum(sizes))).astype(np.int64),
            "data_shape": np.asarray(data.shape, dtype=np.int64),
            "data_kind": np.asarray("bytes"),
        }
    if rows and all(isinstance(row, (str, bytes, bytearray, np.bytes_)) for row in rows):
        texts = []
        for row in rows:
            if isinstance(row, (bytes, bytearray, np.bytes_)):
                try:
                    row = bytes(row).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise TypeError(
                        "save_topic_npz cannot store rows mixing text and non-UTF-8 bytes without loss"
                    ) from exc
            texts.append(row)
        return {
            "data": np.asarray(texts, dtype=str).reshape(data.shape),
            "data_kind": np.asarray("text"),
        }
    raise TypeError(
        "save_topic_npz cannot store this object-dtype data without pickle; "
        "supported object payloads are 1-D arrays of per-row NumPy arrays "
        "sharing a dtype and trailing shape (ragged rows), of bytes, or of strings"
    )


def _npz_path(path) -> Path:
    # np.savez appends ".npz" to any other name; return the file it writes.
    path = Path(path)
    if not path.name.endswith(".npz"):
        path = path.with_name(path.name + ".npz")
    return path


def save_topic_npz(path, topic_data, compressed: bool = True) -> Path:
    """Save a buffered topic to a portable `.npz` file.

    Stores timestamps, data, message ids (integers as integers, anything
    else as unicode; missing ids become empty strings), per-row frame ids,
    ragged ``lengths`` (padded topics) and topic metadata. The file loads
    without pickle. Object-dtype data is supported for ragged per-row
    arrays (stored as values plus offsets), bytes (round-trip exactly) and
    strings; other object data raises TypeError. ``.npz`` is appended to
    `path` when missing, and the returned path is the file actually
    written.
    """

    view = topic_view(topic_data, copy=False)
    data = np.asarray(view.data)
    payload: dict = {"ts": np.asarray(view.timestamps, dtype=np.float64)}
    lengths = None if view.lengths is None else np.asarray(view.lengths, dtype=np.int64)
    if lengths is not None and data.dtype != object and data.ndim >= 2:
        # Storage pads to capacity; keep only up to the longest real row.
        data = data[:, : int(lengths.max(initial=0))]
    if data.dtype == object:
        payload.update(_object_data_payload(data))
    else:
        payload["data"] = data
    if lengths is not None:
        payload["lengths"] = lengths
    if view.ids is not None:
        payload["id"] = _ids_payload(view.ids)
    if view.frame_ids is not None:
        payload["frame_ids"] = _text_array(view.frame_ids)
    metadata = {
        "topic": _json_text(view.metadata.topic),
        "source_uri": _json_text(view.metadata.source_uri),
        "frame_id": _json_text(view.metadata.frame_id),
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata))

    path = _npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **payload)
    return path


def load_topic_npz(path) -> dict:
    """Load a topic saved by `save_topic_npz` back into a topic dict.

    Ragged, bytes and string object data come back as 1-D object arrays;
    padded ragged topics come back padded with their ``"lengths"``; per-row
    frame ids come back under ``"frame_ids"`` (empty entries as None);
    integer ids stay integers.
    """

    with np.load(Path(path), allow_pickle=False) as archive:
        kind = str(archive["data_kind"].item()) if "data_kind" in archive else "array"
        if kind == "ragged":
            values = archive["data_ragged_values"]
            offsets = archive["data_ragged_offsets"]
            data = np.empty(offsets.shape[0] - 1, dtype=object)
            for index in range(data.shape[0]):
                data[index] = values[offsets[index]:offsets[index + 1]].copy()
        elif kind == "bytes":
            raw = archive["data_bytes_values"].tobytes()
            offsets = archive["data_bytes_offsets"]
            data = np.empty(offsets.shape[0] - 1, dtype=object)
            for index in range(data.shape[0]):
                data[index] = raw[int(offsets[index]):int(offsets[index + 1])]
            if "data_shape" in archive:
                data = data.reshape(tuple(int(size) for size in archive["data_shape"]))
        elif kind == "text":
            data = archive["data"].astype(object)
        else:
            data = archive["data"].copy()
        result: dict = {
            "ts": archive["ts"].copy(),
            "data": data,
        }
        if "lengths" in archive:
            result["lengths"] = archive["lengths"].astype(np.int64)
        if "id" in archive:
            stored_ids = archive["id"]
            ids = stored_ids.copy() if stored_ids.dtype.kind in "iu" else stored_ids.astype(object)
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
