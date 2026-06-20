from __future__ import annotations
import logging
import os
import shutil
from contextlib import suppress
import numpy as np
import tiledb

_logger = logging.getLogger(__name__)

class TileDBBuffer:
    def __init__(self, data_source, init_source, group_uri, axis="", topics=None):
        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = group_uri
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self.counters = {}
        self.timestamps = {}
        self.msg_len = {}
        self.names = {}
        self._open_arrays = {}
        self._open_timestamp_arrays = {}

    def _write_metadata(self, topic: str, tiledb_array, closed: bool | None = None) -> None:
        if topic not in self.counters:
            return

        tiledb_array.meta["name"] = self.names.get(topic, "")
        tiledb_array.meta["topic"] = topic
        tiledb_array.meta["count"] = self.counters[topic]
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
                self._write_metadata(topic, arr, closed)
            finally:
                arr.close()
        else:
            uri = self._get_array_uri(topic)
            if os.path.exists(uri) and topic in self.counters:
                with tiledb.open(uri, "w") as tiledb_array:
                    self._write_metadata(topic, tiledb_array, closed)

        timestamp_arr = self._open_timestamp_arrays.pop(topic, None)
        if timestamp_arr is not None:
            try:
                self._write_timestamp_metadata(topic, timestamp_arr, closed)
            finally:
                timestamp_arr.close()
        else:
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
        self.timestamps = {}
        self.msg_len = {}
        self.names = {}

    def _get_array_uri(self, topic: str) -> str:
        return self.group_uri + topic.replace("/", "_")

    def _get_timestamp_array_uri(self, topic: str) -> str:
        return self._get_array_uri(topic) + "__timestamps"

    def _add_group_member(self, uri: str, name: str) -> None:
        with tiledb.Group(self.group_uri, "w") as group:
            with suppress(Exception):
                group.add(uri, name)

    def _init_tdb(self, msg: dict) -> None:
        data_len = max(self.msg_len[msg['topic']], 1)
        uri = self._get_array_uri(msg['topic'])
        timestamp_uri = self._get_timestamp_array_uri(msg['topic'])

        if os.path.exists(uri):
            # If open, close it first
            arr = self._open_arrays.pop(msg['topic'], None)
            if arr is not None:
                with suppress(Exception):
                    arr.close()
            timestamp_arr = self._open_timestamp_arrays.pop(msg['topic'], None)
            if timestamp_arr is not None:
                with suppress(Exception):
                    timestamp_arr.close()

            with tiledb.open(uri, "r") as tiledb_array:
                if tiledb_array.meta.get("closed", False):
                    _logger.info("Full data set exists: %s", uri)
                    return
            shutil.rmtree(uri)
            if os.path.exists(timestamp_uri):
                shutil.rmtree(timestamp_uri)

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
            attrs=[tiledb.Attr(name="timestamp", dtype=np.float64)],
        )
        os.makedirs(timestamp_uri, exist_ok=True)
        tiledb.Array.create(timestamp_uri, timestamp_schema)
        self._add_group_member(timestamp_uri, msg['topic'] + "__timestamps")

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            
            if not msg['topic'] in self.timestamps:
                self.counters[msg['topic']] = 0
                self.msg_len[msg['topic']] = self.init_source.get_count(msg['topic'])
                self.timestamps[msg['topic']] = np.empty(max(self.msg_len[msg['topic']], 1), dtype=np.float64)
                self._init_tdb(msg)

            self.append_buffer(msg)

            if msg['topic'] == self._axis:
                break

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
        self.timestamps[topic][counter] = msg['timestamp']
        self.names[topic] = msg["name"]

        key = (counter,) + tuple(slice(None) for _ in msg['data'].shape)
        tiledb_array[key] = msg['data']
        timestamp_array[counter] = msg['timestamp']
        self.counters[topic] = counter + 1

    def _get_timestamps(self, topic: str, count: int) -> np.ndarray:
        if count == 0:
            return np.array([], dtype=np.float64)

        timestamps = self.timestamps.get(topic)
        if timestamps is not None:
            return np.asarray(timestamps[:count], dtype=np.float64).copy()

        timestamp_uri = self._get_timestamp_array_uri(topic)
        if not os.path.exists(timestamp_uri):
            return np.array([], dtype=np.float64)

        with tiledb.DenseArray(timestamp_uri) as tiledb_array:
            return np.asarray(tiledb_array[0:count]["timestamp"], dtype=np.float64)

    def get_buffer(self, copy: bool = True) -> dict:
        buffer = {}
        for topic, count in self.counters.items():
            self.close_topic(topic)
            uri = self._get_array_uri(topic)
            with tiledb.DenseArray(uri) as A:
                buffer[topic] = {}
                buffer[topic]['id'] = np.full(count, topic, dtype=object)
                buffer[topic]['name'] = buffer[topic]['id']
                buffer[topic]['ts'] = self._get_timestamps(topic, count)
                data = A[0:count]["features"]
                buffer[topic]['data'] = data.copy() if copy else data
                buffer[topic]['topic'] = topic
                buffer[topic]['source_uri'] = uri
        return buffer

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if axis not in self.counters:
            return

        self.close_topic(axis)
        count = self.counters[axis]
        uri = self._get_array_uri(axis)
        timestamps = self._get_timestamps(axis, count)
        with tiledb.DenseArray(uri) as A:
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                data = A[start:stop]["features"]
                yield {
                    "id": np.full(stop - start, axis, dtype=object),
                    "name": np.full(stop - start, axis, dtype=object),
                    "ts": timestamps[start:stop].copy() if copy else timestamps[start:stop],
                    "data": data.copy() if copy else data,
                    "topic": axis,
                    "source_uri": uri,
                }

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        if axis not in self.counters:
            return {
                "id": np.array([], dtype=object),
                "name": np.array([], dtype=object),
                "ts": np.array([], dtype=np.float64),
                "data": np.array([]),
                "topic": axis,
                "source_uri": self._get_array_uri(axis),
            }

        self.close_topic(axis)
        timestamps = self._get_timestamps(axis, self.counters[axis])
        mask = (timestamps >= start) & (timestamps <= end)
        indices = np.flatnonzero(mask)
        uri = self._get_array_uri(axis)

        with tiledb.DenseArray(uri) as A:
            if indices.size == 0:
                data = A[0:0]["features"]
            else:
                first = int(indices[0])
                last = int(indices[-1]) + 1
                data = A[first:last]["features"][mask[first:last]]

        return {
            "id": np.full(indices.size, axis, dtype=object),
            "name": np.full(indices.size, axis, dtype=object),
            "ts": timestamps[mask],
            "data": data,
            "topic": axis,
            "source_uri": uri,
        }

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if axis not in self.counters or self.counters[axis] == 0:
            return {
                "id": np.array([], dtype=object),
                "name": np.array([], dtype=object),
                "ts": np.array([], dtype=np.float64),
                "data": np.array([]),
                "topic": axis,
                "source_uri": self._get_array_uri(axis),
            }

        timestamps = self._get_timestamps(axis, self.counters[axis])
        end = timestamps[-1]
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
