"""Apache Arrow / Parquet persistent buffer backend.

Each topic is a directory of Parquet fragment files plus a ``manifest.json``:

    <group_uri>/<topic>/part-00000.parquet
    <group_uri>/<topic>/part-00001.parquet
    <group_uri>/<topic>/manifest.json

Appends stage in memory and flush a fragment when the staged payload reaches
``flush_bytes`` (or on close), so ingest memory stays bounded regardless of
dataset size. Reads stream through ``pyarrow.dataset`` with bounded readahead,
so scans of larger-than-memory topics stay bounded too. Row-group size is
derived from the first message's byte size (targeting ``row_group_bytes``),
which keeps single-message random access cheap for large payloads.

Tuning knobs (all exposed through ``DataBuffer(backend_options=...)``):

- ``flush_bytes`` (default 32 MB): staged bytes per topic before a fragment
  is written. Larger values mean fewer fragments and faster scans at the cost
  of ingest memory.
- ``row_group_bytes`` (default 16 MB): target Parquet row-group size. Smaller
  groups make random access finer-grained; larger groups scan faster.
- ``compression`` (default ``"zstd"``): Parquet codec (``"zstd"``,
  ``"snappy"``, ``"lz4"``, ``"none"``...).
- ``batch_readahead`` / ``fragment_readahead`` (default 1 each): scanner
  prefetch depth. Raise for throughput at the cost of scan memory.
- ``use_threads`` (default False): parallel scanning. Off by default so
  streamed batches preserve append order and memory stays bounded.

Unlike the TileDB backend, fragments are immutable: ``__setitem__`` is not
supported. Use ``backend="tiledb"`` when in-place cell updates are needed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .common import (
    decode_frame_id,
    encode_frame_id,
    encode_name,
    slice_contains,
    spatial_bounds_for_data,
)

_logger = logging.getLogger(__name__)

DEFAULT_FLUSH_BYTES = 32 * 1024 * 1024
DEFAULT_ROW_GROUP_BYTES = 16 * 1024 * 1024
MANIFEST_NAME = "manifest.json"


class ArrowBuffer:
    def __init__(
        self,
        data_source,
        init_source,
        group_uri,
        axis: str = "",
        topics=None,
        flush_bytes: int = DEFAULT_FLUSH_BYTES,
        row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
        compression: str = "zstd",
        batch_readahead: int = 1,
        fragment_readahead: int = 1,
        use_threads: bool = False,
    ):
        import pyarrow  # noqa: F401  (fail fast with a clear error)

        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = str(group_uri)
        self._axis = axis
        self.topics = [] if topics is None else list(topics)

        self.flush_bytes = int(flush_bytes)
        self.row_group_bytes = int(row_group_bytes)
        self.compression = None if str(compression).lower() == "none" else compression
        self.batch_readahead = int(batch_readahead)
        self.fragment_readahead = int(fragment_readahead)
        self.use_threads = bool(use_threads)

        self.counters: dict[str, int] = {}
        self.frame_ids: dict[str, str | None] = {}
        self.names: dict[str, bytes] = {}
        self.closed_topics: dict[str, bool] = {}
        self.timestamps: dict = {}
        self._resume_seen: dict[str, int] = {}
        self._persisted: dict[str, int] = {}
        self._fragments: dict[str, list[Path]] = {}
        self._staged: dict[str, dict] = {}
        self._schemas: dict[str, tuple[tuple[int, ...], str]] = {}

        self.read_only = data_source is None or init_source is None
        self._hydrate_existing_topics()

    # -- properties ----------------------------------------------------------

    @property
    def msg_len(self) -> dict:
        """Messages appended per topic (persisted + staged)."""
        return dict(self.counters)

    # -- topic layout --------------------------------------------------------

    def _topic_dir(self, topic: str) -> Path:
        return Path(self.group_uri) / topic.replace("/", "_")

    def _manifest_path(self, topic: str) -> Path:
        return self._topic_dir(topic) / MANIFEST_NAME

    # -- hydration / resume --------------------------------------------------

    def _hydrate_existing_topics(self) -> None:
        root = Path(self.group_uri)
        if not root.is_dir():
            return
        for entry in sorted(root.iterdir()):
            manifest_path = entry / MANIFEST_NAME
            if not entry.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _logger.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
                continue
            topic = manifest.get("topic", entry.name)
            fragments = sorted(entry.glob("part-*.parquet"))
            count = int(manifest.get("count", 0))
            self.counters[topic] = count
            self._persisted[topic] = count
            self._fragments[topic] = fragments
            self.closed_topics[topic] = bool(manifest.get("closed", False))
            self.frame_ids[topic] = manifest.get("frame_id")
            name = manifest.get("name")
            if name is not None:
                self.names[topic] = name.encode() if isinstance(name, str) else bytes(name)
            shape = manifest.get("shape")
            dtype = manifest.get("dtype")
            if shape is not None and dtype is not None:
                self._schemas[topic] = (tuple(int(v) for v in shape), str(dtype))
            if topic not in self.topics:
                self.topics.append(topic)

    def _should_skip_replayed_message(self, topic: str) -> bool:
        existing_count = self.counters.get(topic, 0)
        seen = self._resume_seen.get(topic, 0)
        self._resume_seen[topic] = seen + 1
        return seen < existing_count

    def reset(self) -> None:
        self.close()
        self.counters = {}
        self.frame_ids = {}
        self.names = {}
        self.closed_topics = {}
        self.timestamps = {}
        self._resume_seen = {}
        self._persisted = {}
        self._fragments = {}
        self._staged = {}
        self._schemas = {}
        self._hydrate_existing_topics()

    # -- write path ----------------------------------------------------------

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            topic = msg["topic"]

            if topic not in self.counters:
                self.counters[topic] = 0
                self._persisted.setdefault(topic, 0)
                self._fragments.setdefault(topic, [])
                if topic not in self.topics:
                    self.topics.append(topic)

            if self._should_skip_replayed_message(topic):
                if topic == self._axis:
                    break
                continue

            if self.closed_topics.get(topic, False):
                if topic == self._axis:
                    break
                continue

            self.append_buffer(msg)

            if topic == self._axis:
                break

    def append_buffer(self, msg: dict) -> None:
        # An explicit append is a write intent (mirrors the TileDB backend).
        self.read_only = False
        topic = msg["topic"]
        data = np.asarray(msg["data"])

        if topic not in self.counters:
            self.counters[topic] = 0
        self._persisted.setdefault(topic, 0)
        self._fragments.setdefault(topic, [])
        if topic not in self.topics:
            self.topics.append(topic)

        schema = self._schemas.get(topic)
        if schema is None:
            self._schemas[topic] = (tuple(data.shape), str(data.dtype))
        elif tuple(data.shape) != schema[0]:
            raise ValueError(
                f"topic {topic} messages must keep shape {schema[0]}, got {tuple(data.shape)}; "
                "pad variable-size messages to a fixed shape before appending"
            )

        self.names[topic] = encode_name(msg.get("name", topic))
        self._record_frame_id(msg)

        staged = self._staged.setdefault(
            topic, {"ts": [], "name": [], "frame_id": [], "data": [], "bytes": 0}
        )
        staged["ts"].append(float(msg["timestamp"]))
        staged["name"].append(encode_name(msg.get("name", topic)))
        staged["frame_id"].append(encode_frame_id(msg.get("frame_id")))
        staged["data"].append(data)
        staged["bytes"] += data.nbytes
        self.counters[topic] += 1

        if staged["bytes"] >= self.flush_bytes:
            self._flush_topic(topic)

    def _record_frame_id(self, msg: dict) -> None:
        if "frame_id" not in msg or msg["frame_id"] is None:
            return
        topic = msg["topic"]
        frame_id = decode_frame_id(msg["frame_id"])
        if frame_id is None:
            return
        if topic not in self.frame_ids:
            self.frame_ids[topic] = frame_id
        elif self.frame_ids[topic] != frame_id:
            self.frame_ids[topic] = None

    def _flush_topic(self, topic: str) -> None:
        staged = self._staged.get(topic)
        if not staged or not staged["ts"]:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        data = np.ascontiguousarray(np.asarray(staged["data"]))
        shape, dtype = self._schemas[topic]
        tensor_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(np.dtype(dtype)), shape)
        spatial = [spatial_bounds_for_data(value) for value in staged["data"]]

        columns: dict = {
            "ts": pa.array(staged["ts"], type=pa.float64()),
            "name": pa.array(staged["name"], type=pa.binary()),
            "frame_id": pa.array(staged["frame_id"], type=pa.binary()),
            "spatial_valid": pa.array([valid for valid, _, _ in spatial], type=pa.bool_()),
        }
        for dim in range(3):
            columns[f"spatial_min_{dim}"] = pa.array(
                [float(mins[dim]) for _, mins, _ in spatial], type=pa.float64()
            )
            columns[f"spatial_max_{dim}"] = pa.array(
                [float(maxs[dim]) for _, _, maxs in spatial], type=pa.float64()
            )
        columns["data"] = pa.FixedShapeTensorArray.from_numpy_ndarray(data)
        table = pa.table(columns)

        message_bytes = max(1, int(np.prod(shape)) * np.dtype(dtype).itemsize)
        rows_per_group = max(1, self.row_group_bytes // message_bytes)

        topic_dir = self._topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        fragment_path = topic_dir / f"part-{len(self._fragments[topic]):05d}.parquet"
        pq.write_table(
            table,
            fragment_path,
            row_group_size=int(rows_per_group),
            compression=self.compression or "none",
        )
        self._fragments[topic].append(fragment_path)
        self._persisted[topic] += len(staged["ts"])
        self._staged[topic] = {"ts": [], "name": [], "frame_id": [], "data": [], "bytes": 0}
        self._write_manifest(topic)

    def _write_manifest(self, topic: str, closed: bool | None = None) -> None:
        if closed is None:
            closed = self.closed_topics.get(topic, False)
        else:
            self.closed_topics[topic] = closed
        shape, dtype = self._schemas.get(topic, ((), "float64"))
        manifest = {
            "topic": topic,
            "count": self._persisted.get(topic, 0),
            "closed": bool(closed),
            "name": self.names.get(topic, b"").decode(errors="replace"),
            "frame_id": self.frame_ids.get(topic),
            "shape": list(shape),
            "dtype": dtype,
        }
        topic_dir = self._topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path(topic).write_text(json.dumps(manifest), encoding="utf-8")

    def close_topic(self, topic: str, closed: bool | None = None) -> None:
        if self.read_only:
            return
        if topic not in self.counters:
            return
        self._flush_topic(topic)
        self._write_manifest(topic, closed=closed)

    def close(self, closed: bool | None = None) -> None:
        for topic in list(self.counters):
            self.close_topic(topic, closed)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- read path -----------------------------------------------------------

    def _ensure_readable(self, topic: str) -> None:
        if not self.read_only:
            self._flush_topic(topic)
            self._write_manifest(topic)

    def _dataset(self, topic: str):
        import pyarrow.dataset as ds

        fragments = self._fragments.get(topic, [])
        if not fragments:
            return None
        return ds.dataset([str(path) for path in fragments], format="parquet")

    def _scanner(self, dataset, **kwargs):
        kwargs.setdefault("batch_readahead", self.batch_readahead)
        kwargs.setdefault("fragment_readahead", self.fragment_readahead)
        kwargs.setdefault("use_threads", self.use_threads)
        return dataset.scanner(**kwargs)

    def _empty_topic_result(self, topic: str) -> dict:
        shape, dtype = self._schemas.get(topic, ((), "float64"))
        result = {
            "id": np.array([], dtype=object),
            "name": np.array([], dtype=object),
            "ts": np.array([], dtype=np.float64),
            "data": np.empty((0, *shape), dtype=np.dtype(dtype)),
            "topic": topic,
            "source_uri": str(self._topic_dir(topic)),
        }
        if self.frame_ids.get(topic) is not None:
            result["frame_id"] = self.frame_ids[topic]
        return result

    def _table_to_result(self, topic: str, table, copy: bool = True) -> dict:
        names = np.array(table["name"].combine_chunks().to_pylist(), dtype=object)
        data = table["data"].combine_chunks().to_numpy_ndarray()
        if copy or not data.flags.writeable:
            data = data.copy()
        result = {
            "id": names,
            "name": names.copy() if copy else names,
            "ts": np.asarray(table["ts"].combine_chunks(), dtype=np.float64).copy(),
            "data": data,
            "topic": topic,
            "source_uri": str(self._topic_dir(topic)),
        }
        if self.frame_ids.get(topic) is not None:
            result["frame_id"] = self.frame_ids[topic]
        return result

    def get_index_range(
        self,
        axis: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        copy: bool = True,
    ) -> dict:
        if axis not in self.counters:
            return self._empty_topic_result(axis)
        self._ensure_readable(axis)

        count = self._persisted.get(axis, 0)
        range_start, range_stop, range_step = slice(start, stop, step).indices(count)
        if range_step < 1:
            raise ValueError("step must be positive")
        indices = np.arange(range_start, range_stop, range_step, dtype=np.int64)
        dataset = self._dataset(axis)
        if dataset is None or indices.size == 0:
            return self._empty_topic_result(axis)
        return self._table_to_result(axis, dataset.take(indices), copy=copy)

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        import pyarrow.dataset as ds

        if axis not in self.counters:
            return self._empty_topic_result(axis)
        self._ensure_readable(axis)
        dataset = self._dataset(axis)
        if dataset is None:
            return self._empty_topic_result(axis)
        scanner = self._scanner(
            dataset, filter=(ds.field("ts") >= float(start)) & (ds.field("ts") <= float(end))
        )
        return self._table_to_result(axis, scanner.to_table())

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if axis not in self.counters or self._persisted_after_flush(axis) == 0:
            return self._empty_topic_result(axis)
        dataset = self._dataset(axis)
        last = dataset.take([self._persisted[axis] - 1])
        end = float(last["ts"][0].as_py())
        return self.get_time_range(axis, end - float(seconds), end)

    def _persisted_after_flush(self, axis: str) -> int:
        self._ensure_readable(axis)
        return self._persisted.get(axis, 0)

    def get_buffer(self, copy: bool = True) -> dict:
        return {
            topic: self.get_index_range(topic, 0, self.counters.get(topic, 0), copy=copy)
            for topic in self.counters
        }

    def get_topic(self, topic: str, copy: bool = True) -> dict:
        return self.get_index_range(topic, 0, self.counters.get(topic, 0), copy=copy)

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False, operations=()):
        import pyarrow.dataset as ds

        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if axis not in self.counters:
            return
        self._ensure_readable(axis)
        dataset = self._dataset(axis)
        if dataset is None:
            return

        operations = tuple(operations or ())
        scan_filter = None
        if operations and operations[0].kind == "time_range":
            start, end = operations[0].args
            if operations[0].kwargs.get("inclusive", True):
                scan_filter = (ds.field("ts") >= float(start)) & (ds.field("ts") <= float(end))
            else:
                scan_filter = (ds.field("ts") > float(start)) & (ds.field("ts") < float(end))
            operations = operations[1:]

        counters = [0] * len(operations)
        scanner = self._scanner(dataset, batch_size=int(chunk_size), filter=scan_filter)
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            table = batch
            ts = np.asarray(table["ts"], dtype=np.float64)
            keep = np.ones(ts.shape[0], dtype=bool)

            for operation_index, operation in enumerate(operations):
                if not keep.any():
                    break
                if operation.kind == "time_range":
                    op_start, op_end = operation.args
                    if operation.kwargs.get("inclusive", True):
                        keep &= (ts >= op_start) & (ts <= op_end)
                    else:
                        keep &= (ts > op_start) & (ts < op_end)
                elif operation.kind == "index_range":
                    for row in range(ts.shape[0]):
                        if not keep[row]:
                            continue
                        position = counters[operation_index]
                        counters[operation_index] += 1
                        keep[row] = slice_contains(position, *operation.args)
                elif operation.kind == "frame_id":
                    targets = operation.args[0]
                    topic_frame = decode_frame_id(self.frame_ids.get(axis))
                    if topic_frame is not None:
                        if topic_frame not in targets:
                            keep[:] = False
                    else:
                        frames = table["frame_id"].to_pylist()
                        for row, frame in enumerate(frames):
                            if keep[row] and decode_frame_id(frame) not in targets:
                                keep[row] = False
                elif operation.kind == "spatial_bounds":
                    min_bound, max_bound = operation.args
                    columns = operation.kwargs["columns"]
                    valid = np.asarray(table["spatial_valid"], dtype=bool)
                    mask = valid.copy()
                    usable = True
                    for bound_index, column in enumerate(columns):
                        if column >= 3:
                            usable = False
                            break
                        spatial_min = np.asarray(table[f"spatial_min_{column}"], dtype=np.float64)
                        spatial_max = np.asarray(table[f"spatial_max_{column}"], dtype=np.float64)
                        mask &= spatial_min <= max_bound[bound_index]
                        mask &= spatial_max >= min_bound[bound_index]
                    if usable:
                        keep &= mask
                else:
                    raise ValueError(f"unsupported pushdown operation: {operation.kind}")

            if not keep.any():
                continue
            selected = table.filter(np.asarray(keep)) if not keep.all() else table
            names = np.array(selected["name"].to_pylist(), dtype=object)
            data = selected["data"].to_numpy_ndarray()
            if copy or not data.flags.writeable:
                data = data.copy()
            chunk = {
                "id": names,
                "name": names.copy() if copy else names,
                "ts": np.asarray(selected["ts"], dtype=np.float64).copy(),
                "data": data,
                "topic": axis,
                "source_uri": str(self._topic_dir(axis)),
            }
            if self.frame_ids.get(axis) is not None:
                chunk["frame_id"] = self.frame_ids[axis]
            yield chunk

    # -- subscripts ----------------------------------------------------------

    def __getitem__(self, subscript):
        topic = self._axis
        if topic not in self.counters:
            return None
        self._ensure_readable(topic)
        dataset = self._dataset(topic)
        if dataset is None:
            return None
        count = self._persisted.get(topic, 0)

        if isinstance(subscript, slice):
            indices = np.arange(*subscript.indices(count), dtype=np.int64)
        elif isinstance(subscript, (int, np.integer)):
            index = int(subscript)
            if index < 0:
                index += count
            indices = np.array([index], dtype=np.int64)
        else:
            raise TypeError(f"unsupported subscript type: {type(subscript).__name__}")

        data = dataset.take(indices)["data"].combine_chunks().to_numpy_ndarray()
        return np.squeeze(data.copy())

    def __setitem__(self, subscript, newval):
        raise NotImplementedError(
            "the arrow backend stores immutable Parquet fragments and does not "
            "support in-place writes; use DataBuffer(backend='tiledb') for that"
        )
