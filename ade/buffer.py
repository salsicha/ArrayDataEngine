from __future__ import annotations

import os
import tiledb
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source import DataSources

from .buffers.numpy_buffer import NumpyBuffer
from .buffers.tiledb_buffer import TileDBBuffer


class DataBuffer:
    """Buffer Class
    Attributes:
    Args:
    Returns:
    """

    def __init__(self, data_source: DataSources, buffer_depth=1, data_uri="/tmp/tiledb/my_group/", topics=[], axis="", use_db=False):
        """Constructor
        
        """
        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = topics
        self.use_db = use_db
        self._init_source = data_source
        self.group_uri = data_uri

        if self.use_db:
            if not os.path.exists(self.group_uri):
                os.makedirs(self.group_uri, exist_ok=True)
                tiledb.group_create(self.group_uri)

        self.set_methods()
        self.reset()

    @property
    def counters(self):
        return getattr(self.buffer_impl, 'counters', {})

    @counters.setter
    def counters(self, val):
        if hasattr(self, 'buffer_impl'):
            self.buffer_impl.counters = val

    @property
    def timestamps(self):
        return getattr(self.buffer_impl, 'timestamps', {})

    @timestamps.setter
    def timestamps(self, val):
        if hasattr(self, 'buffer_impl'):
            self.buffer_impl.timestamps = val

    @property
    def msg_len(self):
        return getattr(self.buffer_impl, 'msg_len', {})

    @msg_len.setter
    def msg_len(self, val):
        if hasattr(self, 'buffer_impl'):
            self.buffer_impl.msg_len = val

    @property
    def _data_buffer(self):
        return getattr(self.buffer_impl, '_data_buffer', {})

    @_data_buffer.setter
    def _data_buffer(self, val):
        if hasattr(self, 'buffer_impl'):
            self.buffer_impl._data_buffer = val

    def get_group_uri(self) -> str:
        return self.group_uri

    def set_methods(self) -> None:
        pass

    def reset(self) -> None:
        data_source = self._init_source

        # If data_source is a function instead of a class, call it directly
        try:
            self.topics = data_source.get_topics()
            self.data_source = data_source.get_message()
        except Exception:
            self.data_source = self._init_source()

        if not self.use_db:
            self.buffer_impl = NumpyBuffer(self.data_source, self.buffer_depth, self._axis, self.topics)
        else:
            self.buffer_impl = TileDBBuffer(self.data_source, self._init_source, self.group_uri, self._axis, self.topics)

        if not self._axis in self.topics:
            print(f"{self._axis} is not in {self.topics}")
        else:
            for i in range(self.buffer_depth):
                self.roll_buffer(self._axis)

    def reset_buffer(self):
        self.reset()
        self.roll_buffer(self._axis)

    def set_axis(self, axis: str) -> None:
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis
        self.buffer_impl._axis = axis

    def set_topics(self, topics):
        self.topics = topics
        self.buffer_impl.topics = topics

    def get_topics(self):
        return self.topics

    def get_size(self):
        return self.msg_len[self._axis]

    def load_data_db(self, axis: str) -> None:
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis
        while True:
            try:
                self.roll_buffer(self._axis)
            except Exception as e:
                print("Finished loading: ", str(e))

                if self.use_db:
                    for topic in self.topics:
                        uri = self.buffer_impl._get_array_uri(topic)
                        if topic in self.buffer_impl._open_arrays:
                            try:
                                self.buffer_impl._open_arrays[topic].close()
                                del self.buffer_impl._open_arrays[topic]
                            except Exception:
                                pass
                        with tiledb.open(uri, "w") as tiledb_array:
                            tiledb_array.meta["closed"] = True
                return

    def get_data(self, axis):
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis

        counter = 0
        while True:
            counter += 1
            try:
                self.roll_buffer(self._axis)
                yield self.get_buffer(), counter
            except StopIteration:
                print("End of source")
                self.reset_buffer()
                self.roll_buffer(self._axis)
                return

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        self.buffer_impl.roll_buffer(axis)

    def append_buffer(self, msg: dict) -> None:
        self.buffer_impl.append_buffer(msg)

    def get_buffer(self) -> dict:
        return self.buffer_impl.get_buffer()

    def __getitem__(self, subscript: slice | int) -> np.ndarray | float | int:
        return self.buffer_impl[subscript]

    def __setitem__(self, subscript: slice | int, newval: np.ndarray) -> bool | None:
        return self.buffer_impl.__setitem__(subscript, newval)
