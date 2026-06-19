from __future__ import annotations
import numpy as np

class NumpyBuffer:
    def __init__(self, data_source, buffer_depth=1, axis="", topics=None):
        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self._data_buffer = {}
        self._write_indices = {}
        self._counts = {}
        self.data_source = data_source

    def reset(self) -> None:
        self._data_buffer = {}
        self._write_indices = {}
        self._counts = {}

    def _init_topic(self, msg: dict) -> None:
        topic = msg['topic']
        self._data_buffer[topic] = np.zeros(
            self.buffer_depth,
            dtype=[('ts', '<f8'), ('id', 'S'), ('data', msg['data'].dtype, msg['data'].shape)]
        )
        self._write_indices[topic] = 0
        self._counts[topic] = 0

    def _logical_indices(self, topic: str) -> np.ndarray:
        depth = self.buffer_depth
        count = self._counts.get(topic, 0)
        write_index = self._write_indices.get(topic, 0)

        if count < depth:
            return np.concatenate((np.arange(count, depth), np.arange(0, count)))

        return np.concatenate((np.arange(write_index, depth), np.arange(0, write_index)))

    def _ordered_topic(self, topic: str) -> np.ndarray:
        return self._data_buffer[topic][self._logical_indices(topic)]

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            if msg['topic'] not in self._data_buffer:
                self._init_topic(msg)
            self.append_buffer(msg)
            if msg['topic'] == self._axis:
                break

    def append_buffer(self, msg: dict) -> None:
        topic = msg['topic']
        if topic not in self._data_buffer:
            self._init_topic(msg)

        write_index = self._write_indices[topic]
        self._data_buffer[topic]['data'][write_index] = msg['data']
        self._data_buffer[topic]['ts'][write_index] = msg['timestamp']
        self._data_buffer[topic]['id'][write_index] = msg['name']

        self._write_indices[topic] = (write_index + 1) % self.buffer_depth
        self._counts[topic] = min(self._counts[topic] + 1, self.buffer_depth)

    def get_buffer(self) -> dict:
        return {topic: self._ordered_topic(topic).copy() for topic in self._data_buffer}

    def __getitem__(self, subscript):
        ordered_data = self._ordered_topic(self._axis)['data']
        if isinstance(subscript, slice):
            return np.squeeze(ordered_data[subscript])
        elif isinstance(subscript, (int, np.integer)):
            return np.squeeze(ordered_data[subscript])

    def __setitem__(self, subscript, newval):
        logical_indices = self._logical_indices(self._axis)
        if isinstance(subscript, slice):
            self._data_buffer[self._axis]['data'][logical_indices[subscript]] = newval
        elif isinstance(subscript, (int, np.integer)):
            self._data_buffer[self._axis]['data'][logical_indices[subscript]] = newval
        elif isinstance(subscript, float):
            count = self._counts.get(self._axis, 0)
            if count == 0:
                return
            valid_indices = logical_indices[-count:]
            timestamps = self._data_buffer[self._axis]['ts'][valid_indices]
            newest_ts = timestamps[-1]
            self._data_buffer[self._axis]['data'][valid_indices[timestamps > newest_ts - subscript]] = newval
        else:
            self._data_buffer[self._axis]['data'][logical_indices[subscript]] = newval
