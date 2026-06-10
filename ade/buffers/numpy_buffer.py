from __future__ import annotations
import numpy as np

class NumpyBuffer:
    def __init__(self, data_source, buffer_depth=1, axis="", topics=[]):
        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = topics
        self._data_buffer = {}
        self.data_source = data_source

    def reset(self) -> None:
        self._data_buffer = {}

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            if msg['topic'] not in self._data_buffer:
                self._data_buffer[msg['topic']] = np.zeros(
                    self.buffer_depth, 
                    dtype=[('ts', '<f8'), ('id', 'S'), ('data', msg['data'].dtype, msg['data'].shape)]
                )
            self.append_buffer(msg)
            if msg['topic'] == self._axis:
                break

    def append_buffer(self, msg: dict) -> None:
        topic = msg['topic']
        self._data_buffer[topic]['data'][:-1] = self._data_buffer[topic]['data'][1:]
        self._data_buffer[topic]['data'][-1] = msg['data']

        self._data_buffer[topic]['ts'][:-1] = self._data_buffer[topic]['ts'][1:]
        self._data_buffer[topic]['ts'][-1] = msg['timestamp']

        self._data_buffer[topic]['id'][:-1] = self._data_buffer[topic]['id'][1:]
        self._data_buffer[topic]['id'][-1] = msg['name']

    def get_buffer(self) -> dict:
        return self._data_buffer.copy()

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return np.squeeze(self._data_buffer[self._axis]['data'][subscript.start, subscript.stop, subscript.step])
        elif isinstance(subscript, (int, float)):
            return np.squeeze(self._data_buffer[self._axis]['data'][subscript])

    def __setitem__(self, subscript, newval):
        if isinstance(subscript, slice):
            self._data_buffer[self._axis]['data'][subscript.start, subscript.stop, subscript.step] = newval
        elif isinstance(subscript, (int, float)):
            self._data_buffer[self._axis]['data'][self._data_buffer[self._axis]['ts'] > \
                                                self._data_buffer[self._axis]['ts'][-1] - subscript] = newval
        else:
            self._data_buffer[self._axis]['data'][subscript] = newval
