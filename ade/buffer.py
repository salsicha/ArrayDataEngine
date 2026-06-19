from __future__ import annotations

import logging
import os
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source import DataSources

from .buffers.numpy_buffer import NumpyBuffer

_logger = logging.getLogger(__name__)


class DataBuffer:
    """Buffer Class
    Attributes:
    Args:
    Returns:
    """

    def __init__(
        self,
        data_source: DataSources,
        buffer_depth=1,
        data_uri="/tmp/tiledb/my_group/",
        topics=None,
        axis="",
        use_db=False,
        preload=1,
    ):
        """Constructor
        
        """
        if buffer_depth < 1:
            raise ValueError("buffer_depth must be at least 1")

        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self.use_db = use_db
        self._init_source = data_source
        self.group_uri = data_uri
        self.preload = preload

        if self.use_db:
            import tiledb

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

    def _get_data_source(self):
        data_source = self._init_source

        if hasattr(data_source, "get_topics") and hasattr(data_source, "get_message"):
            self.topics = data_source.get_topics()
            return data_source.get_message()

        if callable(data_source):
            return data_source()

        raise TypeError("data_source must expose get_message() or be a callable generator factory")

    def _get_preload_count(self, preload) -> int:
        if preload is None:
            preload = self.preload
        if preload is True:
            preload = self.buffer_depth
        elif preload is False:
            preload = 0
        return max(0, min(int(preload), self.buffer_depth))

    def reset(self, preload=None) -> None:
        self.data_source = self._get_data_source()

        if not self.use_db:
            self.buffer_impl = NumpyBuffer(self.data_source, self.buffer_depth, self._axis, self.topics)
        else:
            from .buffers.tiledb_buffer import TileDBBuffer

            self.buffer_impl = TileDBBuffer(self.data_source, self._init_source, self.group_uri, self._axis, self.topics)

        if not self._axis:
            return
        if self._axis not in self.topics:
            raise ValueError(f"Axis: {self._axis} not in topics: {self.topics}")

        for i in range(self._get_preload_count(preload)):
            self.roll_buffer(self._axis)

    def reset_buffer(self):
        self.reset(preload=0)
        self.roll_buffer(self._axis)

    def set_axis(self, axis: str) -> None:
        if not axis in self.topics:
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")
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
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis
        while True:
            try:
                self.roll_buffer(self._axis)
            except StopIteration as e:
                _logger.info("Finished loading data source: %s", e)

                if self.use_db:
                    for topic in self.topics:
                        self.buffer_impl.close_topic(topic, closed=True)
                return

    def get_data(self, axis):
        if not axis in self.topics:
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis

        counter = 0
        while True:
            counter += 1
            try:
                self.roll_buffer(self._axis)
                yield self.get_buffer(), counter
            except StopIteration:
                _logger.info("End of source")
                self.reset_buffer()
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
