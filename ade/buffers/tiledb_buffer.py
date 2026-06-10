from __future__ import annotations
import os
import shutil
import numpy as np
import tiledb

class TileDBBuffer:
    def __init__(self, data_source, init_source, group_uri, axis="", topics=[]):
        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = group_uri
        self._axis = axis
        self.topics = topics
        self.counters = {}
        self.timestamps = {}
        self.msg_len = {}
        self._open_arrays = {}

    def close(self):
        for topic, arr in list(self._open_arrays.items()):
            try:
                arr.close()
            except Exception:
                pass
        self._open_arrays.clear()

    def __del__(self):
        self.close()

    def reset(self) -> None:
        self.close()
        self.counters = {}
        self.timestamps = {}
        self.msg_len = {}

    def _get_array_uri(self, topic: str) -> str:
        return self.group_uri + topic.replace("/", "_")

    def _init_tdb(self, msg: dict) -> None:
        data_len = self.msg_len[msg['topic']]
        uri = self._get_array_uri(msg['topic'])

        if os.path.exists(uri):
            # If open, close it first
            if msg['topic'] in self._open_arrays:
                try:
                    self._open_arrays[msg['topic']].close()
                except Exception:
                    pass
                del self._open_arrays[msg['topic']]

            with tiledb.open(uri, "r") as tiledb_array:
                try:
                    if tiledb_array.meta["closed"]:
                        print(f"Full data set exists! {uri}")
                        return
                except Exception:
                    print("Closed tag not present.")
            shutil.rmtree(uri)

        dims = [
            tiledb.Dim(
                name="images" if dim == 0 else "dim_" + str(dim - 1),
                domain=(0, data_len if dim == 0 else (msg['data'].shape[dim - 1] - 1)),
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
        tiledb_array[self.counters[topic], :] = msg['data']
        self.counters[topic] += 1

        tiledb_array.meta["timestamp"] = np.array(self.timestamps[topic])
        tiledb_array.meta["name"] = msg["name"]
        tiledb_array.meta["topic"] = msg["topic"]
        tiledb_array.meta["count"] = self.counters[topic]

    def get_buffer(self) -> dict:
        buffer = {}
        with tiledb.Group(self.group_uri) as g:
            for a in g:
                # Flush the open array to disk so reader can read the latest data
                if a.name in self._open_arrays:
                    try:
                        self._open_arrays[a.name].close()
                        del self._open_arrays[a.name]
                    except Exception:
                        pass
                
                with tiledb.DenseArray(a.uri) as A:
                    if A.meta.get("topic") and A.meta.get("topic") in self.counters:
                        topic = A.meta.get("topic")
                        buffer[topic] = {}
                        buffer[topic]['id'] = topic
                        buffer[topic]['ts'] = A.meta.get("timestamp")
                        buffer[topic]['data'] = A[0:self.counters[topic]]["features"]
        return buffer

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            with tiledb.Group(self.group_uri) as g:
                for a in g:
                    if a.name == self._axis:
                        # Flush open writer first
                        if a.name in self._open_arrays:
                            try:
                                self._open_arrays[a.name].close()
                                del self._open_arrays[a.name]
                            except Exception:
                                pass
                        with tiledb.DenseArray(a.uri) as A:
                            return A[subscript.start:subscript.stop]["features"]

        elif isinstance(subscript, int):
            if subscript < 0:
                subscript = self.counters[self._axis] + subscript

            with tiledb.Group(self.group_uri) as g:
                for a in g:
                    if a.name == self._axis:
                        # Flush open writer first
                        if a.name in self._open_arrays:
                            try:
                                self._open_arrays[a.name].close()
                                del self._open_arrays[a.name]
                            except Exception:
                                pass
                        with tiledb.DenseArray(a.uri) as A:
                            return A[subscript]["features"]

    def __setitem__(self, subscript, newval) -> bool | None:
        if isinstance(subscript, slice):
            with tiledb.Group(self.group_uri) as g:
                for a in g:
                    if a.name == self._axis:
                        # Close open writer before opening in standard write mode
                        if a.name in self._open_arrays:
                            try:
                                self._open_arrays[a.name].close()
                                del self._open_arrays[a.name]
                            except Exception:
                                pass
                        with tiledb.open(a.uri, "w") as A:
                            A[subscript.start:subscript.stop] = newval
                            return True

        elif isinstance(subscript, int):
            with tiledb.Group(self.group_uri) as g:
                for a in g:
                    if a.name == self._axis:
                        # Close open writer before opening in standard write mode
                        if a.name in self._open_arrays:
                            try:
                                self._open_arrays[a.name].close()
                                del self._open_arrays[a.name]
                            except Exception:
                                pass
                        with tiledb.open(a.uri, "w") as A:
                            A[subscript] = newval
                            return True
