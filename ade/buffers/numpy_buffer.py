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
            dtype=[('ts', '<f8'), ('id', 'S256'), ('data', msg['data'].dtype, msg['data'].shape)]
        )
        self._write_indices[topic] = 0
        self._counts[topic] = 0

    def _logical_indices(self, topic: str) -> np.ndarray:
        depth = self.buffer_depth
        count = self._counts.get(topic, 0)
        write_index = self._write_indices.get(topic, 0)

        if count == 0:
            return np.arange(depth)

        if count < depth:
            return np.concatenate((np.arange(count, depth), np.arange(0, count)))

        return np.concatenate((np.arange(write_index, depth), np.arange(0, write_index)))

    def _ordered_topic(self, topic: str, copy: bool = False) -> np.ndarray:
        logical_indices = self._logical_indices(topic)
        if logical_indices.size == self.buffer_depth and np.array_equal(logical_indices, np.arange(self.buffer_depth)):
            ordered = self._data_buffer[topic]
        else:
            ordered = self._data_buffer[topic][logical_indices]
        return ordered.copy() if copy else ordered

    def _valid_ordered_topic(self, topic: str) -> np.ndarray:
        count = self._counts.get(topic, 0)
        ordered = self._ordered_topic(topic)
        if count == 0:
            return ordered[:0]
        return ordered[-count:]

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

    def get_buffer(self, copy: bool = True) -> dict:
        return {topic: self._ordered_topic(topic, copy=copy) for topic in self._data_buffer}

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        topic = self._valid_ordered_topic(axis)
        mask = (topic['ts'] >= start) & (topic['ts'] <= end)
        selected = topic[mask]
        return {
            "id": axis,
            "ts": selected['ts'].copy(),
            "data": selected['data'].copy(),
        }

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        topic = self._valid_ordered_topic(axis)
        if topic.size == 0:
            return {
                "id": axis,
                "ts": np.array([], dtype=np.float64),
                "data": topic['data'].copy(),
            }
        end = topic['ts'][-1]
        return self.get_time_range(axis, end - seconds, end)

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
