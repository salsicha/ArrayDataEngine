from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import topic_view


def save_topic_npz(path, topic_data, compressed: bool = True) -> Path:
    """Save a buffered topic to a portable `.npz` file.

    Stores timestamps, data, message ids (as unicode; missing ids become
    empty strings), and topic metadata. The file loads without pickle.
    """

    view = topic_view(topic_data, copy=False)
    payload: dict = {
        "ts": np.asarray(view.timestamps, dtype=np.float64),
        "data": np.asarray(view.data),
    }
    if view.ids is not None:
        payload["id"] = np.asarray([
            "" if value is None else (value.decode(errors="replace") if isinstance(value, bytes) else str(value))
            for value in view.ids
        ])
    metadata = {
        "topic": view.metadata.topic,
        "source_uri": view.metadata.source_uri,
        "frame_id": view.metadata.frame_id,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **payload)
    return path


def load_topic_npz(path) -> dict:
    """Load a topic saved by `save_topic_npz` back into a topic dict."""

    with np.load(Path(path), allow_pickle=False) as archive:
        result: dict = {
            "ts": archive["ts"].copy(),
            "data": archive["data"].copy(),
        }
        if "id" in archive:
            ids = archive["id"].astype(object)
            result["id"] = ids
            result["name"] = ids.copy()
        metadata = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(archive["metadata_json"].item()))

    for key in ("topic", "source_uri", "frame_id"):
        if metadata.get(key) is not None:
            result[key] = metadata[key]
    return result
