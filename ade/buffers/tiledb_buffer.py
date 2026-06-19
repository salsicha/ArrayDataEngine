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

    def _write_metadata(self, topic: str, tiledb_array, closed: bool | None = None) -> None:
        if topic not in self.counters:
            return

        tiledb_array.meta["timestamp"] = np.asarray(self.timestamps.get(topic, []), dtype=np.float64)
        tiledb_array.meta["name"] = self.names.get(topic, "")
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
            return

        uri = self._get_array_uri(topic)
        if os.path.exists(uri) and topic in self.counters:
            with tiledb.open(uri, "w") as tiledb_array:
                self._write_metadata(topic, tiledb_array, closed)

    def close(self, closed: bool | None = None):
        for topic in list(self._open_arrays):
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

    def _init_tdb(self, msg: dict) -> None:
        data_len = max(self.msg_len[msg['topic']], 1)
        uri = self._get_array_uri(msg['topic'])

        if os.path.exists(uri):
            # If open, close it first
            arr = self._open_arrays.pop(msg['topic'], None)
            if arr is not None:
                with suppress(Exception):
                    arr.close()

            with tiledb.open(uri, "r") as tiledb_array:
                if tiledb_array.meta.get("closed", False):
                    _logger.info("Full data set exists: %s", uri)
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
    
        with tiledb.Group(self.group_uri, "w") as g:
            g.add(uri, msg['topic'])

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            
            if not msg['topic'] in self.timestamps:
                self.timestamps[msg['topic']] = []
                self.counters[msg['topic']] = 0
                self.msg_len[msg['topic']] = self.init_source.get_count(msg['topic'])
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

        tiledb_array = self._open_arrays[topic]
        self.timestamps[topic].append(msg['timestamp'])
        self.names[topic] = msg["name"]

        key = (self.counters[topic],) + tuple(slice(None) for _ in msg['data'].shape)
        tiledb_array[key] = msg['data']
        self.counters[topic] += 1

    def get_buffer(self) -> dict:
        buffer = {}
        for topic, count in self.counters.items():
            self.close_topic(topic)
            uri = self._get_array_uri(topic)
            with tiledb.DenseArray(uri) as A:
                buffer[topic] = {}
                buffer[topic]['id'] = topic
                buffer[topic]['ts'] = np.asarray(self.timestamps.get(topic, A.meta.get("timestamp", [])))
                buffer[topic]['data'] = A[0:count]["features"]
        return buffer

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        if axis not in self.counters:
            return {"id": axis, "ts": np.array([], dtype=np.float64), "data": np.array([])}

        self.close_topic(axis)
        timestamps = np.asarray(self.timestamps.get(axis, []), dtype=np.float64)
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

        return {"id": axis, "ts": timestamps[mask], "data": data}

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if axis not in self.counters or len(self.timestamps.get(axis, [])) == 0:
            return {"id": axis, "ts": np.array([], dtype=np.float64), "data": np.array([])}

        end = self.timestamps[axis][-1]
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
