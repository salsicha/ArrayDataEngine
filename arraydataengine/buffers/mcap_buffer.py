"""Append-only MCAP segments with an atomic commit manifest.

Reads stream messages without a Parquet dependency. Eager array APIs still
materialize their selected result; iter_topic_chunks bounds payload batches.
Only one process may append at a time. An interrupted, uncommitted segment is
ignored and source replay resumes from the last manifest commit.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from ..recording import McapRecorder, _require_mcap
from ..sources.mcap_source import McapSource
from .common import decode_frame_id, encode_name

MANIFEST = "mcap_store.json"
FORMAT = "ade-mcap-store-v1"


def read_manifest(uri):
    path = Path(uri) / MANIFEST
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("format") != FORMAT or not isinstance(value.get("segments"), list):
        raise ValueError(f"Invalid MCAP store manifest: {path}")
    for name in value["segments"]:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".mcap"):
            raise ValueError("Invalid MCAP segment path")
    return value


def is_mcap_store(uri):
    return (Path(uri) / MANIFEST).is_file()


def _atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class McapBuffer:
    def __init__(self, data_source, init_source, group_uri, axis="", topics=None,
                 chunk_size=1024**2, segment_bytes=64*1024**2, memory_budget_bytes=64*1024**2,
                 compression="zstd", ragged_topics=None, max_points=None):
        _require_mcap()
        for name, value in (("chunk_size", chunk_size), ("segment_bytes", segment_bytes),
                            ("memory_budget_bytes", memory_budget_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if compression not in ("zstd", "lz4", "none"):
            raise ValueError("compression must be zstd, lz4, or none")
        self.data_source, self.init_source = data_source, init_source
        self.group_uri, self._axis = str(group_uri), axis
        self.topics = list(topics or [])
        self.chunk_size = min(chunk_size, memory_budget_bytes)
        self.segment_bytes = segment_bytes
        self.memory_budget_bytes = memory_budget_bytes
        self.compression = compression
        self._writer = self._lock = None
        self._segment_size = 0
        self._resume_seen = {}
        self._manifest = read_manifest(group_uri)
        self._segments = list((self._manifest or {}).get("segments", []))
        self.counters = dict((self._manifest or {}).get("counts", {}))
        self.frame_ids = dict((self._manifest or {}).get("frame_ids", {}))
        self.closed_topics = dict((self._manifest or {}).get("closed", {}))
        self.names, self.timestamps = {}, {}
        for topic in self.counters:
            if topic not in self.topics:
                self.topics.append(topic)
        if init_source is not None:
            self._acquire()

    def _check_location(self):
        root = Path(self.group_uri)
        placeholders = {".gitkeep", ".keep", "lost+found", ".DS_Store"}
        if not is_mcap_store(root) and root.exists() and any(p.name not in placeholders for p in root.iterdir()):
            raise ValueError(f"{root} is not an MCAP store; choose an empty directory or its existing backend")

    def _acquire(self):
        if self._lock is not None:
            return
        self._check_location()
        root = Path(self.group_uri)
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".mcap-write.lock").open("a")
        try:
            if os.name == "nt":
                import msvcrt
                lock.write("0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read_manifest(root) != self._manifest:
                raise ValueError("MCAP store changed since this handle opened; reopen before appending")
        except BaseException:
            lock.close()
            raise
        self._lock = lock
        if self._manifest is None:
            self._commit()

    def _commit(self):
        value = {"format": FORMAT, "segments": self._segments, "counts": self.counters,
                 "frame_ids": self.frame_ids, "closed": self.closed_topics}
        _atomic_json(Path(self.group_uri) / MANIFEST, value)
        self._manifest = json.loads(json.dumps(value))

    @property
    def msg_len(self):
        return dict(self.counters)

    def get_size(self, topic):
        return self.counters.get(topic, 0)

    def append_buffer(self, msg):
        # Encode/validate before creating a store or changing its metadata.
        from ..recording import _encode_message
        topic, log_time, publish_time, sequence, payload = _encode_message(msg)
        if len(payload) > self.memory_budget_bytes:
            raise ValueError("MCAP message exceeds memory_budget_bytes; increase the budget")
        self._acquire()
        if self._writer is None:
            self._pending = Path(self.group_uri) / f".pending-{uuid.uuid4().hex}"
            self._writer = McapRecorder(self._pending, chunk_size=self.chunk_size, compression=self.compression)
        from ..recording import ENCODING
        self._writer.write_encoded(topic, payload, timestamp_ns=log_time,
                                   publish_time_ns=publish_time, sequence=sequence, message_encoding=ENCODING)
        frame = decode_frame_id(msg.get("frame_id"))
        if self.counters.get(topic, 0) == 0:
            self.frame_ids[topic] = frame
        elif self.frame_ids.get(topic) != frame:
            self.frame_ids[topic] = None
        self.counters[topic] = self.counters.get(topic, 0) + 1
        self.closed_topics[topic] = False
        if topic not in self.topics:
            self.topics.append(topic)
        self._segment_size += len(payload)
        if self._segment_size >= self.segment_bytes:
            self.flush()

    def flush(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            name = f"part-{uuid.uuid4().hex}.mcap"
            os.replace(self._pending, Path(self.group_uri) / name)
            self._segments.append(name)
            self._segment_size = 0
            self._commit()

    def close_topic(self, topic, closed=None):
        if closed is not None and self._lock is not None:
            self.closed_topics[topic] = bool(closed)
        self.flush()
        if self._lock is not None:
            self._commit()

    def close(self, closed=None):
        try:
            self.flush()
            if self._lock is not None:
                if closed is not None:
                    self.closed_topics.update({topic: bool(closed) for topic in self.counters})
                self._commit()
        finally:
            if self._lock is not None:
                self._lock.close()
                self._lock = None

    def close_completed(self, source_exhausted=False):
        if source_exhausted and self._lock is not None:
            for topic in self._resume_seen:
                if topic in self.counters:
                    self.closed_topics[topic] = True
        self.close()

    def roll_buffer(self, axis):
        self._axis = axis
        while True:
            msg = next(self.data_source)
            topic = msg["topic"]
            seen = self._resume_seen.get(topic, 0)
            self._resume_seen[topic] = seen + 1
            if seen >= self.counters.get(topic, 0):
                self.append_buffer(msg)
            if topic == axis:
                return

    def messages(self, topic=None):
        self.flush()
        for name in list(self._segments):
            yield from McapSource(Path(self.group_uri) / name, topics=topic).messages()

    def _result(self, topic, messages):
        rows = list(messages)
        values = [np.asarray(msg["data"]) for msg in rows]
        lengths = None
        if not values:
            data = np.empty((0,))
        elif all(value.shape == values[0].shape and value.dtype == values[0].dtype for value in values):
            data = np.stack(values)
        elif all(value.ndim == values[0].ndim and value.ndim > 0 and
                 value.shape[1:] == values[0].shape[1:] and value.dtype == values[0].dtype for value in values):
            lengths = np.array([len(value) for value in values])
            data = np.zeros((len(values), max(lengths), *values[0].shape[1:]), dtype=values[0].dtype)
            for i, value in enumerate(values):
                data[i, :len(value)] = value
        else:
            data = np.empty(len(values), dtype=object)
            data[:] = values
        names = np.array([encode_name(msg.get("name", topic)) for msg in rows], dtype=object)
        result = {"data": data, "ts": np.array([m["timestamp"] for m in rows], dtype=float),
                  "id": names, "name": names, "frame_ids": np.array([m.get("frame_id") for m in rows], dtype=object),
                  "topic": topic, "source_uri": self.group_uri}
        if lengths is not None or any("point_count" in m for m in rows):
            result["lengths"] = np.array([m.get("point_count", len(m["data"])) for m in rows], dtype=np.int64)
        if self.frame_ids.get(topic) is not None:
            result["frame_id"] = self.frame_ids[topic]
        return result

    def get_index_range(self, axis, start=None, stop=None, step=None, copy=True):
        start, stop, step = slice(start, stop, step).indices(self.get_size(axis))
        if step < 1:
            raise ValueError("step must be positive")
        return self._result(axis, itertools.islice(self.messages(axis), start, stop, step))

    def get_time_range(self, axis, start, end):
        return self._result(axis, (m for m in self.messages(axis) if start <= m["timestamp"] <= end))

    def get_last_seconds(self, axis, seconds):
        last = self.get_index_range(axis, -1)
        if not len(last["ts"]):
            return last
        end = last["ts"][-1]
        return self.get_time_range(axis, end - seconds, end)

    def get_topic(self, topic, copy=True):
        return self.get_index_range(topic, copy=copy)

    def get_buffer(self, copy=True):
        return {topic: self.get_topic(topic, copy) for topic in self.counters}

    def iter_topic_chunks(self, axis, chunk_size, copy=False, operations=()):
        from ..ops.core import _apply_pushdown_to_view, topic_view
        operations = tuple(operations)
        counters = [0] * len(operations)
        batch, size = [], 0
        for msg in self.messages(axis):
            keep = True
            for i, operation in enumerate(operations):
                if operation.kind == "index_range":
                    start, stop, step = operation.args
                    start, step = start or 0, step or 1
                    if start < 0 or stop is not None and stop < 0:
                        raise ValueError("negative streaming indices require get_index_range")
                    pos = counters[i]
                    counters[i] += 1
                    keep = pos >= start and (stop is None or pos < stop) and (pos-start) % step == 0
                else:
                    keep = len(_apply_pushdown_to_view(topic_view(self._result(axis, [msg])), [operation])) > 0
                if not keep:
                    break
            if not keep:
                continue
            nbytes = np.asarray(msg["data"]).nbytes
            if nbytes > self.memory_budget_bytes:
                raise ValueError("MCAP message exceeds memory_budget_bytes")
            shape_changed = batch and (np.asarray(batch[0]["data"]).shape != np.asarray(msg["data"]).shape or
                                       np.asarray(batch[0]["data"]).dtype != np.asarray(msg["data"]).dtype)
            if batch and (shape_changed or len(batch) >= chunk_size or size + nbytes > self.memory_budget_bytes):
                yield self._result(axis, batch)
                batch, size = [], 0
            batch.append(msg)
            size += nbytes
        if batch:
            yield self._result(axis, batch)

    def __getitem__(self, subscript):
        if isinstance(subscript, (int, np.integer)):
            result = self.get_index_range(self._axis, int(subscript), None if subscript == -1 else int(subscript)+1)
            if not len(result["data"]):
                raise IndexError(subscript)
            return result["data"][0]
        if isinstance(subscript, slice):
            return self.get_index_range(self._axis, subscript.start, subscript.stop, subscript.step)["data"]
        raise TypeError("MCAP indices must be integers or slices")

    def __setitem__(self, subscript, value):
        raise TypeError("MCAP recordings are append-only; use tiledb for in-place updates")


class McapStoreSource:
    supports_topic_filter = True

    def __init__(self, path, topics=None, max_points=None):
        self.data_path = str(path)
        self._buffer = McapBuffer(iter(()), None, path)
        self._topics = None if topics is None else set([topics] if isinstance(topics, str) else topics)
        self.max_points = max_points

    def data_exists(self):
        return is_mcap_store(self.data_path)

    def get_data_path(self):
        return self.data_path

    def get_topics(self):
        return [t for t in self._buffer.topics if self._topics is None or t in self._topics]

    def get_topic_types(self):
        return {t: "ade-numpy-v1" for t in self._buffer.topics}

    def get_count(self, topic):
        if topic not in self.get_topics():
            return 0
        if self.max_points is None:
            return self._buffer.counters.get(topic, 0)
        return sum(m["topic"] == topic for m in self.messages())

    def messages(self):
        for message in self._buffer.messages():
            if self._topics is not None and message["topic"] not in self._topics:
                continue
            if self.max_points is not None and message.get("point_count", 0) > self.max_points:
                continue
            yield message

    def get_message(self):
        yield from self.messages()
