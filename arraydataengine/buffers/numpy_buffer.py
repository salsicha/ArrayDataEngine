from __future__ import annotations
import numpy as np

from ..ops.core import _row_frame_id, _spatial_value_in_bounds
from .common import (
    conform_message,
    decode_frame_id as _decode_frame_id,
    encode_name,
    message_is_ragged,
    native_message_data,
    new_ragged_schema,
    normalize_ragged_topics,
    ragged_capacity,
)

# Row layout used for topics that have not received a message yet.
_EMPTY_TOPIC_DTYPE = np.dtype([('ts', '<f8'), ('id', 'S256'), ('frame_id', object), ('data', '<f8')])
# Rows allocated when a topic receives its first message; storage doubles as
# messages arrive until it reaches `buffer_depth`, so deep rings of large
# messages only cost memory for messages actually received.
_INITIAL_ROWS = 8


class NumpyBuffer:
    """Rolling in-memory ring of the newest `buffer_depth` messages per topic.

    Each topic is a structured array with ``ts``, ``id``, ``frame_id``, and
    ``data`` fields (plus ``length`` for variable-size topics, whose ``data``
    rows are zero-padded to the topic's capacity). Reads return only rows
    holding real messages, oldest first.
    """

    def __init__(self, data_source, buffer_depth=1, axis="", topics=None, ragged_topics=None, max_points=None):
        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self.ragged_topics = normalize_ragged_topics(ragged_topics)
        self.max_points = max_points
        self._data_buffer = {}
        self._write_indices = {}
        self._counts = {}
        self.frame_ids = {}
        self._frame_counts = {}
        self.data_source = data_source

    def reset(self) -> None:
        self._data_buffer = {}
        self._write_indices = {}
        self._counts = {}
        self.frame_ids = {}
        self._frame_counts = {}

    def _is_ragged(self, topic: str) -> bool:
        storage = self._data_buffer.get(topic)
        return storage is not None and "length" in storage.dtype.names

    def _init_topic(self, msg: dict) -> None:
        topic = msg['topic']
        data = native_message_data(msg['data'])
        fields = [('ts', '<f8'), ('id', 'S256'), ('frame_id', object)]
        if message_is_ragged(msg, self.ragged_topics):
            capacity = ragged_capacity(topic, self.ragged_topics, self.max_points)
            shape, dtype = new_ragged_schema(topic, data, msg, capacity)
            fields += [('data', dtype, shape), ('length', '<i8')]
        else:
            fields.append(('data', data.dtype, data.shape))
        self._data_buffer[topic] = self._allocate(np.dtype(fields), min(self.buffer_depth, _INITIAL_ROWS))
        self._write_indices[topic] = 0
        self._counts[topic] = 0
        self._frame_counts[topic] = {}

    @staticmethod
    def _allocate(dtype: np.dtype, rows: int) -> np.ndarray:
        storage = np.zeros(rows, dtype=dtype)
        storage['frame_id'] = None
        return storage

    def _grow(self, topic: str) -> None:
        """Double a topic's storage (up to `buffer_depth`) before it wraps."""
        storage = self._data_buffer[topic]
        rows = min(self.buffer_depth, max(1, 2 * storage.shape[0]))
        grown = self._allocate(storage.dtype, rows)
        grown[:storage.shape[0]] = storage
        self._data_buffer[topic] = grown

    def _logical_indices(self, topic: str) -> np.ndarray:
        """Storage rows holding real messages, oldest first."""
        count = self._counts.get(topic, 0)
        write_index = self._write_indices.get(topic, 0)
        rows = self._data_buffer[topic].shape[0] if topic in self._data_buffer else 0
        if count < rows or write_index == 0 or write_index >= rows:
            # Not wrapped yet (storage may still be growing): append order.
            return np.arange(count)
        return np.concatenate((np.arange(write_index, rows), np.arange(0, write_index)))

    def _valid_ordered_topic(self, topic: str) -> np.ndarray:
        if topic not in self._data_buffer:
            # A listed topic that has not received a message yet is empty.
            return np.zeros(0, dtype=_EMPTY_TOPIC_DTYPE)
        storage = self._data_buffer[topic]
        count = self._counts.get(topic, 0)
        write_index = self._write_indices.get(topic, 0)
        if count < storage.shape[0] or write_index == 0 or write_index >= storage.shape[0]:
            return storage[:count]
        return storage[self._logical_indices(topic)]

    @property
    def msg_len(self) -> dict:
        return dict(self._counts)

    def get_size(self, topic: str) -> int:
        """Messages currently buffered for `topic`."""
        return int(self._counts.get(topic, 0))

    def get_topic(self, topic: str, copy: bool = True) -> np.ndarray:
        """Return only the rows that hold real messages, oldest first."""
        valid = self._valid_ordered_topic(topic)
        return valid.copy() if copy else valid

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            self.append_buffer(msg)
            if msg['topic'] == self._axis:
                break

    def append_buffer(self, msg: dict) -> None:
        topic = msg['topic']
        data = native_message_data(msg['data'])
        new_topic = topic not in self._data_buffer
        if new_topic:
            self._init_topic(msg)
        try:
            # Validate before writing so a rejected message leaves no partial
            # row; numpy assignment would otherwise broadcast or cast silently.
            field = self._data_buffer[topic].dtype.fields['data'][0]
            rows, length = conform_message(topic, data, msg, field.shape, field.base, self._is_ragged(topic))
            name = encode_name(msg.get('name', topic))
            timestamp = float(msg['timestamp'])
        except BaseException:
            if new_topic:
                self._drop_topic(topic)
            raise

        storage = self._data_buffer[topic]
        if self._counts[topic] == storage.shape[0] < self.buffer_depth:
            self._grow(topic)
            storage = self._data_buffer[topic]
        write_index = self._write_indices[topic]
        if length is None:
            storage['data'][write_index] = rows
        else:
            row = storage['data'][write_index]
            row[:length] = rows
            row[length:] = 0
            storage['length'][write_index] = length
        storage['ts'][write_index] = timestamp
        storage['id'][write_index] = name
        self._record_frame_id(msg)

        self._write_indices[topic] = (write_index + 1) % self.buffer_depth
        self._counts[topic] = min(self._counts[topic] + 1, self.buffer_depth)

    def _drop_topic(self, topic: str) -> None:
        for state in (self._data_buffer, self._write_indices, self._counts, self._frame_counts, self.frame_ids):
            state.pop(topic, None)

    def get_buffer(self, copy: bool = True) -> dict:
        """Per topic, the rows holding real messages in chronological order."""
        return {topic: self.get_topic(topic, copy=copy) for topic in self._data_buffer}

    def _result(self, axis: str, selected: np.ndarray, copy: bool = True) -> dict:
        def own(values):
            return values.copy() if copy else values

        result = {
            "id": own(selected["id"]),
            "name": own(selected["id"]),
            "ts": own(selected["ts"]),
            "data": own(selected["data"]),
            "topic": axis,
            "frame_ids": own(selected["frame_id"]),
            **self._metadata_for_topic(axis),
        }
        if selected.dtype.names and "length" in selected.dtype.names:
            result["lengths"] = np.asarray(selected["length"], dtype=np.int64).copy()
        return result

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False, operations=()):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        topic = self._apply_operations(axis, self._valid_ordered_topic(axis), operations)
        for start in range(0, topic.shape[0], chunk_size):
            yield self._result(axis, topic[start:start + chunk_size], copy=copy)

    def get_index_range(self, axis: str, start: int | None = None, stop: int | None = None, step: int | None = None) -> dict:
        topic = self._valid_ordered_topic(axis)
        return self._result(axis, topic[slice(start, stop, step)])

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        topic = self._valid_ordered_topic(axis)
        mask = (topic['ts'] >= start) & (topic['ts'] <= end)
        return self._result(axis, topic[mask])

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        topic = self._valid_ordered_topic(axis)
        if topic.size == 0:
            return self._result(axis, topic)
        end = topic['ts'][-1]
        return self.get_time_range(axis, end - seconds, end)

    def _record_frame_id(self, msg: dict) -> None:
        topic = msg["topic"]
        index = self._write_indices[topic]
        frames = self._data_buffer[topic]["frame_id"]
        counts = self._frame_counts[topic]
        if self._counts[topic] == self.buffer_depth:
            previous = frames[index]
            counts[previous] -= 1
            if counts[previous] == 0:
                del counts[previous]
        frame = _decode_frame_id(msg.get("frame_id"))
        if frame is None:
            frame = _row_frame_id(msg["data"], msg.get("name"))
        frames[index] = frame
        counts[frame] = counts.get(frame, 0) + 1
        self.frame_ids[topic] = next(iter(counts)) if len(counts) == 1 else None

    def _metadata_for_topic(self, topic: str) -> dict:
        frame_id = self.frame_ids.get(topic)
        return {} if frame_id is None else {"frame_id": frame_id}

    def _apply_operations(self, axis: str, topic: np.ndarray, operations) -> np.ndarray:
        selected = topic
        for operation in operations:
            if operation.kind == "time_range":
                start, end = operation.args
                if operation.kwargs.get("inclusive", True):
                    mask = (selected['ts'] >= start) & (selected['ts'] <= end)
                else:
                    mask = (selected['ts'] > start) & (selected['ts'] < end)
                selected = selected[mask]
            elif operation.kind == "index_range":
                selected = selected[slice(*operation.args)]
            elif operation.kind == "frame_id":
                targets = operation.args[0]
                selected = selected[np.array([
                    frame in targets for frame in selected["frame_id"]
                ], dtype=bool)]
            elif operation.kind == "spatial_bounds":
                min_bound, max_bound = operation.args
                columns = operation.kwargs["columns"]
                ragged = selected.dtype.names is not None and "length" in selected.dtype.names
                # Padding rows of variable-size messages are not points.
                values = (
                    (value[:length] for value, length in zip(selected["data"], selected["length"]))
                    if ragged else selected["data"]
                )
                mask = np.array([
                    _spatial_value_in_bounds(value, min_bound=min_bound, max_bound=max_bound, columns=columns)
                    for value in values
                ], dtype=bool)
                selected = selected[mask]
            else:
                raise ValueError(f"unsupported pushdown operation: {operation.kind}")
        return selected

    def __getitem__(self, subscript):
        if not isinstance(subscript, (slice, int, np.integer)):
            raise TypeError(f"unsupported subscript type: {type(subscript).__name__}")
        topic = self._valid_ordered_topic(self._axis)
        if isinstance(subscript, slice) or "length" not in (topic.dtype.names or ()):
            return np.squeeze(topic['data'][subscript])
        # One message of a variable-size topic: its real rows only.
        row = topic[subscript]
        return row['data'][:int(row['length'])].copy()

    def __setitem__(self, subscript, newval):
        if self._axis not in self._data_buffer:
            return
        storage = self._data_buffer[self._axis]
        logical_indices = self._logical_indices(self._axis)
        if isinstance(subscript, float):
            if logical_indices.size == 0:
                return
            timestamps = storage['ts'][logical_indices]
            newest_ts = timestamps[-1]
            storage['data'][logical_indices[timestamps > newest_ts - subscript]] = newval
            return
        targets = logical_indices[subscript]
        if isinstance(subscript, (int, np.integer)) and self._is_ragged(self._axis):
            field = storage.dtype.fields['data'][0]
            values = np.asarray(newval)
            if values.ndim == len(field.shape) and values.shape[1:] == field.shape[1:] and values.shape[0] <= field.shape[0]:
                # Replace one variable-size message (rows up to the capacity).
                storage['data'][targets] = 0
                storage['data'][targets][:values.shape[0]] = values
                storage['length'][targets] = values.shape[0]
                return
        storage['data'][targets] = newval
