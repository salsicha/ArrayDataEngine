from __future__ import annotations
import logging
import os
import shutil
from contextlib import suppress
import numpy as np
import tiledb

_logger = logging.getLogger(__name__)


def _encode_name(name) -> bytes:
    if isinstance(name, bytes):
        return name[:256]
    return str(name).encode()[:256]


def _decode_frame_id(value) -> str | None:
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


def _slice_contains(index: int, start: int | None, stop: int | None, step: int | None) -> bool:
    start = 0 if start is None else start
    step = 1 if step is None else step
    if index < start:
        return False
    if stop is not None and index >= stop:
        return False
    return (index - start) % step == 0


def _concat_topic_parts(parts: list[dict]) -> dict:
    if len(parts) == 1:
        return parts[0]

    first = parts[0]
    result = {
        "id": np.concatenate([part["id"] for part in parts]),
        "name": np.concatenate([part["name"] for part in parts]),
        "ts": np.concatenate([part["ts"] for part in parts]),
        "data": np.concatenate([part["data"] for part in parts], axis=0),
        "topic": first["topic"],
        "source_uri": first["source_uri"],
    }
    if "frame_id" in first:
        result["frame_id"] = first["frame_id"]
    return result


class TileDBBuffer:
    def __init__(self, data_source, init_source, group_uri, axis="", topics=None):
        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = group_uri
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self.counters = {}
        self.msg_len = {}
        self.names = {}
        self.frame_ids = {}
        self.closed_topics = {}
        self._resume_seen = {}
        self._open_arrays = {}
        self._open_timestamp_arrays = {}
        self.timestamps = {}
        self.read_only = data_source is None
        self._hydrate_existing_topics()

    def _write_metadata(self, topic: str, tiledb_array, closed: bool | None = None) -> None:
        if topic not in self.counters:
            return

        tiledb_array.meta["name"] = self.names.get(topic, "")
        tiledb_array.meta["topic"] = topic
        tiledb_array.meta["count"] = self.counters[topic]
        frame_id = self.frame_ids.get(topic)
        if frame_id is not None:
            tiledb_array.meta["frame_id"] = frame_id
        else:
            with suppress(Exception):
                del tiledb_array.meta["frame_id"]
        if closed is not None:
            tiledb_array.meta["closed"] = closed

    def _write_timestamp_metadata(self, topic: str, tiledb_array, closed: bool | None = None) -> None:
        if topic not in self.counters:
            return

        tiledb_array.meta["topic"] = topic
        tiledb_array.meta["count"] = self.counters[topic]
        if closed is not None:
            tiledb_array.meta["closed"] = closed

    def close_topic(self, topic: str, closed: bool | None = None) -> None:
        arr = self._open_arrays.pop(topic, None)
        if arr is not None:
            try:
                if not self.read_only:
                    self._write_metadata(topic, arr, closed)
            finally:
                arr.close()
        elif not self.read_only:
            uri = self._get_array_uri(topic)
            if os.path.exists(uri) and topic in self.counters:
                with tiledb.open(uri, "w") as tiledb_array:
                    self._write_metadata(topic, tiledb_array, closed)

        timestamp_arr = self._open_timestamp_arrays.pop(topic, None)
        if timestamp_arr is not None:
            try:
                if not self.read_only:
                    self._write_timestamp_metadata(topic, timestamp_arr, closed)
            finally:
                timestamp_arr.close()
        elif not self.read_only:
            timestamp_uri = self._get_timestamp_array_uri(topic)
            if os.path.exists(timestamp_uri) and topic in self.counters:
                with tiledb.open(timestamp_uri, "w") as tiledb_array:
                    self._write_timestamp_metadata(topic, tiledb_array, closed)

    def close(self, closed: bool | None = None):
        topics = set(self.counters) | set(self._open_arrays) | set(self._open_timestamp_arrays)
        for topic in topics:
            self.close_topic(topic, closed)

    def __del__(self):
        with suppress(Exception):
            self.close()

    def reset(self) -> None:
        self.close()
        self.counters = {}
        self.msg_len = {}
        self.names = {}
        self.frame_ids = {}
        self.closed_topics = {}
        self._resume_seen = {}
        self.timestamps = {}
        self._hydrate_existing_topics()

    def _get_array_uri(self, topic: str) -> str:
        return os.path.join(self.group_uri, topic.replace("/", "_"))

    def _get_timestamp_array_uri(self, topic: str) -> str:
        return self._get_array_uri(topic) + "__timestamps"

    def _add_group_member(self, uri: str, name: str) -> None:
        with tiledb.Group(self.group_uri, "w") as group:
            with suppress(Exception):
                group.add(uri, name)

    def _hydrate_existing_topics(self) -> None:
        if not os.path.isdir(self.group_uri):
            return

        for entry in os.listdir(self.group_uri):
            if entry.endswith("__timestamps"):
                continue
            uri = os.path.join(self.group_uri, entry)
            if not os.path.isdir(uri):
                continue
            with suppress(Exception):
                self._hydrate_topic(uri, fallback_topic=entry)

    def _hydrate_topic(self, uri: str, fallback_topic: str | None = None) -> None:
        with tiledb.open(uri, "r") as tiledb_array:
            topic = tiledb_array.meta.get("topic", fallback_topic)
            if topic is None:
                return

            count = int(tiledb_array.meta.get("count", 0))
            timestamp_uri = self._get_timestamp_array_uri(topic)
            if os.path.exists(timestamp_uri):
                with tiledb.open(timestamp_uri, "r") as timestamp_array:
                    count = int(timestamp_array.meta.get("count", count))
                    self.closed_topics[topic] = bool(timestamp_array.meta.get("closed", tiledb_array.meta.get("closed", False)))
            else:
                self.closed_topics[topic] = bool(tiledb_array.meta.get("closed", False))

            domain = tiledb_array.schema.domain.dim(0).domain
            self.counters[topic] = count
            self.msg_len[topic] = int(domain[1]) + 1
            self.names[topic] = tiledb_array.meta.get("name", "")
            frame_id = _decode_frame_id(tiledb_array.meta.get("frame_id"))
            if frame_id is not None:
                self.frame_ids[topic] = frame_id
            if topic not in self.topics:
                self.topics.append(topic)

    def _init_tdb(self, msg: dict) -> None:
        data_len = max(self.msg_len[msg['topic']], 1)
        uri = self._get_array_uri(msg['topic'])
        timestamp_uri = self._get_timestamp_array_uri(msg['topic'])

        if os.path.exists(uri):
            arr = self._open_arrays.pop(msg['topic'], None)
            if arr is not None:
                with suppress(Exception):
                    arr.close()
            timestamp_arr = self._open_timestamp_arrays.pop(msg['topic'], None)
            if timestamp_arr is not None:
                with suppress(Exception):
                    timestamp_arr.close()

            if os.path.exists(timestamp_uri):
                self._hydrate_topic(uri, fallback_topic=msg['topic'])
                return

            shutil.rmtree(uri)

        dims = [
            tiledb.Dim(
                name="images" if dim == 0 else "dim_" + str(dim - 1),
                domain=(0, data_len - 1 if dim == 0 else (msg['data'].shape[dim - 1] - 1)),
                tile=1 if dim == 0 else msg['data'].shape[dim - 1],
                dtype=np.int32,
            )
            for dim in range(msg['data'].ndim + 1)
        ]
        
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(*dims),
            sparse=False,
            attrs=[tiledb.Attr(name="features", dtype=msg['data'].dtype)],
        )
        os.makedirs(uri, exist_ok=True)
        tiledb.Array.create(uri, schema)
        self._add_group_member(uri, msg['topic'])

        timestamp_schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="message", domain=(0, data_len - 1), tile=min(data_len, 1024), dtype=np.int32)
            ),
            sparse=False,
            attrs=[
                tiledb.Attr(name="timestamp", dtype=np.float64),
                tiledb.Attr(name="name", dtype="S256"),
            ],
        )
        os.makedirs(timestamp_uri, exist_ok=True)
        tiledb.Array.create(timestamp_uri, timestamp_schema)
        self._add_group_member(timestamp_uri, msg['topic'] + "__timestamps")

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            
            if msg['topic'] not in self.counters:
                self.counters[msg['topic']] = 0
                self.msg_len[msg['topic']] = self.init_source.get_count(msg['topic'])
                self._init_tdb(msg)

            if self._should_skip_replayed_message(msg['topic']):
                if msg['topic'] == self._axis:
                    break
                continue

            if self.closed_topics.get(msg['topic'], False) and self.counters[msg['topic']] >= self.msg_len[msg['topic']]:
                if msg['topic'] == self._axis:
                    break
                continue

            self.append_buffer(msg)

            if msg['topic'] == self._axis:
                break

    def _should_skip_replayed_message(self, topic: str) -> bool:
        existing_count = self.counters.get(topic, 0)
        seen = self._resume_seen.get(topic, 0)
        if seen < existing_count:
            self._resume_seen[topic] = seen + 1
            return True
        self._resume_seen[topic] = seen + 1
        return False

    def append_buffer(self, msg: dict) -> None:
        topic = msg['topic']
        array_uri = self._get_array_uri(topic)

        if not os.path.exists(array_uri):
            self._init_tdb(msg)

        if topic not in self._open_arrays:
            self._open_arrays[topic] = tiledb.open(array_uri, "w")
        if topic not in self._open_timestamp_arrays:
            self._open_timestamp_arrays[topic] = tiledb.open(self._get_timestamp_array_uri(topic), "w")

        tiledb_array = self._open_arrays[topic]
        timestamp_array = self._open_timestamp_arrays[topic]
        counter = self.counters[topic]
        if counter >= self.msg_len[topic]:
            raise ValueError(f"topic {topic} is already full at {counter} messages")
        self.names[topic] = msg["name"]
        self._record_frame_id(msg)

        key = (counter,) + tuple(slice(None) for _ in msg['data'].shape)
        tiledb_array[key] = msg['data']
        timestamp_array[counter] = {
            "timestamp": np.array(msg['timestamp'], dtype=np.float64),
            "name": np.array(_encode_name(msg.get("name", topic)), dtype="S256"),
        }
        self.counters[topic] = counter + 1

    def _record_frame_id(self, msg: dict) -> None:
        if "frame_id" not in msg or msg["frame_id"] is None:
            return

        topic = msg["topic"]
        frame_id = _decode_frame_id(msg["frame_id"])
        if frame_id is None:
            return
        if topic not in self.frame_ids:
            self.frame_ids[topic] = frame_id
        elif self.frame_ids[topic] != frame_id:
            self.frame_ids[topic] = None

    def _metadata_for_topic(self, topic: str) -> dict:
        frame_id = self.frame_ids.get(topic)
        return {} if frame_id is None else {"frame_id": frame_id}

    def _get_timestamps(self, topic: str, count: int, start: int = 0, stop: int | None = None) -> np.ndarray:
        if count == 0:
            return np.array([], dtype=np.float64)

        stop = count if stop is None else min(stop, count)
        if start >= stop:
            return np.array([], dtype=np.float64)
        timestamp_uri = self._get_timestamp_array_uri(topic)
        if not os.path.exists(timestamp_uri):
            return np.array([], dtype=np.float64)

        with tiledb.DenseArray(timestamp_uri) as tiledb_array:
            return np.asarray(tiledb_array[start:stop]["timestamp"], dtype=np.float64)

    def _get_names(self, topic: str, count: int, start: int = 0, stop: int | None = None) -> np.ndarray:
        stop = count if stop is None else min(stop, count)
        if start >= stop:
            return np.array([], dtype="S256")

        timestamp_uri = self._get_timestamp_array_uri(topic)
        if not os.path.exists(timestamp_uri):
            return np.full(stop - start, topic, dtype=object)

        with tiledb.DenseArray(timestamp_uri) as tiledb_array:
            if "name" not in tiledb_array.schema.attr_names:
                return np.full(stop - start, topic, dtype=object)
            return np.asarray(tiledb_array[start:stop]["name"])

    def _read_timestamp_scalar(self, tiledb_array, index: int) -> float:
        return float(np.asarray(tiledb_array[index:index + 1]["timestamp"], dtype=np.float64)[0])

    def _timestamp_search(self, tiledb_array, count: int, value: float, side: str = "left") -> int:
        left = 0
        right = count
        while left < right:
            mid = (left + right) // 2
            timestamp = self._read_timestamp_scalar(tiledb_array, mid)
            if timestamp < value or (side == "right" and timestamp <= value):
                left = mid + 1
            else:
                right = mid
        return left

    def _time_range_to_index_range(
        self,
        tiledb_array,
        count: int,
        start: float,
        end: float,
        inclusive: bool = True,
    ) -> tuple[int, int]:
        if count == 0:
            return 0, 0

        lower_side = "left" if inclusive else "right"
        upper_side = "right" if inclusive else "left"
        first = self._timestamp_search(tiledb_array, count, start, side=lower_side)
        last = self._timestamp_search(tiledb_array, count, end, side=upper_side)
        return first, max(first, last)

    def _normalize_index_range(
        self,
        count: int,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> tuple[int, int, int]:
        range_start, range_stop, range_step = slice(start, stop, step).indices(count)
        if range_step < 1:
            raise ValueError("step must be positive")
        return range_start, range_stop, range_step

    def _read_index_attr(self, tiledb_array, attr: str, indices: np.ndarray, fallback: str | None = None) -> np.ndarray:
        if indices.size == 0:
            return np.array([])
        if attr not in tiledb_array.schema.attr_names:
            if fallback is None:
                return np.array([])
            return np.full(indices.size, fallback, dtype=object)

        first = int(indices[0])
        last = int(indices[-1]) + 1
        values = np.asarray(tiledb_array[first:last][attr])
        if indices.size != last - first or not np.array_equal(indices, np.arange(first, last)):
            values = values[indices - first]
        return values

    def _read_data_attr(self, tiledb_array, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return tiledb_array[0:0]["features"]

        first = int(indices[0])
        last = int(indices[-1]) + 1
        values = tiledb_array[first:last]["features"]
        if indices.size != last - first or not np.array_equal(indices, np.arange(first, last)):
            values = values[indices - first]
        return values

    def _iter_range_indices(self, start: int, stop: int, step: int, chunk_size: int):
        current = start
        while current < stop:
            remaining = ((stop - current - 1) // step) + 1
            size = min(chunk_size, remaining)
            indices = current + step * np.arange(size, dtype=np.int64)
            yield indices
            current = int(indices[-1]) + step

    def _iter_selected_indices(self, index_array, count: int, chunk_size: int, operations):
        operations = tuple(operations or ())

        scan_start = 0
        scan_stop = count
        if operations and operations[0].kind == "time_range":
            start, end = operations[0].args
            scan_start, scan_stop = self._time_range_to_index_range(
                index_array,
                count,
                start,
                end,
                inclusive=operations[0].kwargs.get("inclusive", True),
            )
            operations = operations[1:]

        counters = [0] * len(operations)
        for indices in self._iter_range_indices(scan_start, scan_stop, 1, chunk_size):
            selected = indices
            for operation_index, operation in enumerate(operations):
                if selected.size == 0:
                    break
                if operation.kind == "time_range":
                    start, end = operation.args
                    timestamps = self._read_index_attr(index_array, "timestamp", selected)
                    if operation.kwargs.get("inclusive", True):
                        mask = (timestamps >= start) & (timestamps <= end)
                    else:
                        mask = (timestamps > start) & (timestamps < end)
                    selected = selected[mask]
                elif operation.kind == "index_range":
                    keep = np.zeros(selected.size, dtype=bool)
                    for i in range(selected.size):
                        current = counters[operation_index]
                        counters[operation_index] += 1
                        keep[i] = _slice_contains(current, *operation.args)
                    selected = selected[keep]
                else:
                    raise ValueError(f"unsupported pushdown operation: {operation.kind}")

            if selected.size:
                yield selected

    def _read_topic_indices(self, axis: str, data_array, index_array, indices: np.ndarray, copy: bool = False) -> dict:
        uri = self._get_array_uri(axis)
        data = self._read_data_attr(data_array, indices)
        timestamps = self._read_index_attr(index_array, "timestamp", indices)
        names = self._read_index_attr(index_array, "name", indices, fallback=axis)
        result = {
            "id": names.copy() if copy else names,
            "name": names.copy() if copy else names,
            "ts": timestamps.copy() if copy else timestamps,
            "data": data.copy() if copy else data,
            "topic": axis,
            "source_uri": uri,
        }
        if self.frame_ids.get(axis) is not None:
            result["frame_id"] = self.frame_ids[axis]
        return result

    def get_buffer(self, copy: bool = True) -> dict:
        buffer = {}
        for topic, count in self.counters.items():
            buffer[topic] = self.get_index_range(topic, 0, count, copy=copy)
        return buffer

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False, operations=()):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if axis not in self.counters:
            return

        self.close_topic(axis)
        count = self.counters[axis]
        uri = self._get_array_uri(axis)
        timestamp_uri = self._get_timestamp_array_uri(axis)
        with tiledb.DenseArray(uri) as data_array, tiledb.DenseArray(timestamp_uri) as index_array:
            for indices in self._iter_selected_indices(index_array, count, chunk_size, operations):
                yield self._read_topic_indices(axis, data_array, index_array, indices, copy=copy)

    def get_index_range(
        self,
        axis: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        copy: bool = True,
    ) -> dict:
        if axis not in self.counters:
            return {
                "id": np.array([], dtype=object),
                "name": np.array([], dtype=object),
                "ts": np.array([], dtype=np.float64),
                "data": np.array([]),
                "topic": axis,
                "source_uri": self._get_array_uri(axis),
                **self._metadata_for_topic(axis),
            }

        self.close_topic(axis)
        count = self.counters[axis]
        range_start, range_stop, range_step = self._normalize_index_range(count, start, stop, step)
        uri = self._get_array_uri(axis)
        timestamp_uri = self._get_timestamp_array_uri(axis)
        parts = []
        with tiledb.DenseArray(uri) as data_array, tiledb.DenseArray(timestamp_uri) as index_array:
            for indices in self._iter_range_indices(range_start, range_stop, range_step, 1024):
                parts.append(self._read_topic_indices(axis, data_array, index_array, indices, copy=copy))

            if not parts:
                empty_data = data_array[0:0]["features"]
                return {
                    "id": np.array([], dtype=object),
                    "name": np.array([], dtype=object),
                    "ts": np.array([], dtype=np.float64),
                    "data": empty_data.copy() if copy else empty_data,
                    "topic": axis,
                    "source_uri": uri,
                    **self._metadata_for_topic(axis),
                }

        return _concat_topic_parts(parts)

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        if axis not in self.counters:
            return {
                "id": np.array([], dtype=object),
                "name": np.array([], dtype=object),
                "ts": np.array([], dtype=np.float64),
                "data": np.array([]),
                "topic": axis,
                "source_uri": self._get_array_uri(axis),
                **self._metadata_for_topic(axis),
            }

        self.close_topic(axis)
        timestamp_uri = self._get_timestamp_array_uri(axis)
        with tiledb.DenseArray(timestamp_uri) as index_array:
            first, last = self._time_range_to_index_range(index_array, self.counters[axis], start, end)
        return self.get_index_range(axis, first, last)

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if axis not in self.counters or self.counters[axis] == 0:
            return {
                "id": np.array([], dtype=object),
                "name": np.array([], dtype=object),
                "ts": np.array([], dtype=np.float64),
                "data": np.array([]),
                "topic": axis,
                "source_uri": self._get_array_uri(axis),
                **self._metadata_for_topic(axis),
            }

        self.close_topic(axis)
        timestamp_uri = self._get_timestamp_array_uri(axis)
        with tiledb.DenseArray(timestamp_uri) as index_array:
            end = self._read_timestamp_scalar(index_array, self.counters[axis] - 1)
        return self.get_time_range(axis, end - seconds, end)

    def __getitem__(self, subscript):
        topic = self._axis
        uri = self._get_array_uri(topic)
        if not os.path.exists(uri):
            return None

        self.close_topic(topic)

        if isinstance(subscript, slice):
            with tiledb.DenseArray(uri) as A:
                return A[subscript]["features"]

        elif isinstance(subscript, int):
            if subscript < 0:
                subscript = self.counters[self._axis] + subscript

            with tiledb.DenseArray(uri) as A:
                return A[subscript]["features"]

    def __setitem__(self, subscript, newval) -> bool | None:
        topic = self._axis
        uri = self._get_array_uri(topic)
        if not os.path.exists(uri):
            return None

        self.close_topic(topic)

        if isinstance(subscript, slice):
            with tiledb.open(uri, "w") as A:
                A[subscript] = newval
                return True

        elif isinstance(subscript, int):
            if subscript < 0:
                subscript = self.counters[self._axis] + subscript

            with tiledb.open(uri, "w") as A:
                A[subscript] = newval
                return True
