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
        self.close()
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

    def close(self, closed: bool | None = None) -> None:
        buffer_impl = getattr(self, "buffer_impl", None)
        if buffer_impl is not None and hasattr(buffer_impl, "close"):
            buffer_impl.close(closed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(closed=exc_type is None)
        return False

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

    def get_buffer(self, copy: bool = True) -> dict:
        return self.buffer_impl.get_buffer(copy=copy)

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        if axis not in self.topics:
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")
        if start > end:
            raise ValueError("start must be less than or equal to end")
        return self.buffer_impl.get_time_range(axis, start, end)

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if axis not in self.topics:
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        return self.buffer_impl.get_last_seconds(axis, seconds)

    def _validate_topic_axis(self, axis: str) -> None:
        if axis not in self.topics:
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")

    def _source_uri(self) -> str | None:
        if self.use_db:
            return self.group_uri

        source = self._init_source
        for candidate in (source, getattr(source, "source", None)):
            if candidate is None:
                continue
            get_data_path = getattr(candidate, "get_data_path", None)
            if callable(get_data_path):
                return get_data_path()
            data_path = getattr(candidate, "data_path", None)
            if data_path is not None:
                return data_path
        return None

    def topic_view(self, axis: str, copy: bool = True, metadata=None):
        from .ops import topic_view

        self._validate_topic_axis(axis)
        return topic_view(
            self.get_buffer(copy=copy)[axis],
            topic=axis,
            source_uri=self._source_uri(),
            metadata=metadata,
            copy=False,
        )

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False):
        from .ops import topic_view

        self._validate_topic_axis(axis)
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        if hasattr(self.buffer_impl, "iter_topic_chunks"):
            for chunk in self.buffer_impl.iter_topic_chunks(axis, chunk_size, copy=copy):
                yield topic_view(
                    chunk,
                    topic=axis,
                    source_uri=chunk.get("source_uri", self._source_uri()),
                    copy=False,
                )
            return

        yield from self.topic_view(axis, copy=copy).iter_chunks(chunk_size, copy=False)

    def _combine_topic_views(self, axis: str, views, data=None) -> dict:
        from .ops import TopicView

        ids_parts = []
        ts_parts = []
        data_parts = []
        for view in views:
            if view.ids is not None:
                ids_parts.append(view.ids)
            ts_parts.append(view.timestamps)
            if data is None:
                data_parts.append(view.data)

        ids = np.concatenate(ids_parts) if ids_parts else None
        timestamps = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.float64)
        if data is None:
            data = np.concatenate(data_parts, axis=0) if data_parts else np.array([])
        return TopicView(ids, timestamps, data, topic=axis, source_uri=self._source_uri()).as_dict()

    def map_topic(
        self,
        axis: str,
        fn,
        copy: bool = True,
        out: np.ndarray | None = None,
        chunk_size: int | None = None,
    ) -> dict:
        self._validate_topic_axis(axis)
        if chunk_size is None:
            return self.topic_view(axis, copy=False).map(fn, copy=copy, out=out).as_dict()

        output = None if out is None else np.asarray(out)
        views = []
        offset = 0
        for chunk in self.iter_topic_chunks(axis, chunk_size, copy=False):
            chunk_out = None if output is None else output[offset:offset + len(chunk)]
            mapped = chunk.map(fn, copy=copy, out=chunk_out)
            views.append(mapped)
            offset += len(chunk)

        if output is not None and offset != output.shape[0]:
            raise ValueError("out must have the same leading dimension as topic data")
        return self._combine_topic_views(axis, views, data=output)

    def filter_topic(
        self,
        axis: str,
        predicate,
        copy: bool = True,
        chunk_size: int | None = None,
    ) -> dict:
        self._validate_topic_axis(axis)
        if chunk_size is None:
            return self.topic_view(axis, copy=False).filter(predicate, copy=copy).as_dict()

        views = [
            chunk.filter(predicate, copy=copy)
            for chunk in self.iter_topic_chunks(axis, chunk_size, copy=False)
        ]
        return self._combine_topic_views(axis, views)

    def reduce_topic(
        self,
        axis: str,
        fn,
        initial=None,
        copy: bool = True,
        chunk_size: int | None = None,
    ):
        self._validate_topic_axis(axis)
        if chunk_size is None:
            return self.topic_view(axis, copy=False).reduce(fn, initial=initial, copy=copy)

        has_acc = initial is not None
        acc = initial
        for chunk in self.iter_topic_chunks(axis, chunk_size, copy=False):
            if len(chunk) == 0:
                continue
            acc = chunk.reduce(fn, initial=acc if has_acc else None, copy=copy)
            has_acc = True

        if not has_acc:
            raise ValueError("cannot reduce an empty topic without an initial value")
        return acc

    def window_topic(self, axis: str, size: int | None = None, seconds: float | None = None, copy: bool = True):
        self._validate_topic_axis(axis)
        for window in self.topic_view(axis, copy=False).window(size=size, seconds=seconds, copy=copy):
            yield window.as_dict()

    def __getitem__(self, subscript: slice | int) -> np.ndarray | float | int:
        return self.buffer_impl[subscript]

    def __setitem__(self, subscript: slice | int, newval: np.ndarray) -> bool | None:
        return self.buffer_impl.__setitem__(subscript, newval)
