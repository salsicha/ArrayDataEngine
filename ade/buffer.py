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

    BACKENDS = ("memory", "arrow", "tiledb")

    def __init__(
        self,
        data_source: DataSources,
        buffer_depth=1,
        data_uri="/tmp/tiledb/my_group/",
        topics=None,
        axis="",
        use_db=False,
        preload=1,
        backend: str | None = None,
        backend_options: dict | None = None,
    ):
        """Constructor

        `backend` selects the storage engine:

        - ``"memory"``: rolling in-memory NumPy ring buffer (no persistence).
        - ``"arrow"``: persistent Apache Arrow / Parquet fragments at
          `data_uri` — the default for new persistent stores.
        - ``"tiledb"``: persistent TileDB dense arrays at `data_uri`
          (supports in-place `__setitem__` writes, unlike arrow).

        When `backend` is omitted, `use_db=False` means ``"memory"`` and
        `use_db=True` picks ``"arrow"`` — unless `data_uri` already holds a
        TileDB store, which keeps opening as ``"tiledb"`` for compatibility.

        `backend_options` is passed through to the backend constructor. The
        arrow backend accepts `flush_bytes`, `row_group_bytes`,
        `compression`, `batch_readahead`, `fragment_readahead`, and
        `use_threads` (see `ade.buffers.arrow_buffer` for defaults).
        """
        if buffer_depth < 1:
            raise ValueError("buffer_depth must be at least 1")

        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self._init_source = data_source
        self.group_uri = data_uri
        self.preload = preload
        self.backend = self._resolve_backend(backend, use_db, data_uri)
        self.use_db = self.backend != "memory"
        self._backend_options = dict(backend_options or {})
        if self._backend_options and self.backend != "arrow":
            raise ValueError(f"backend_options are only supported by the arrow backend, not {self.backend!r}")

        if self.backend == "tiledb":
            import tiledb

            if not os.path.exists(self.group_uri):
                os.makedirs(self.group_uri, exist_ok=True)
                tiledb.group_create(self.group_uri)

        self.set_methods()
        self.reset()

    @classmethod
    def _resolve_backend(cls, backend: str | None, use_db: bool, data_uri) -> str:
        if backend is not None:
            if backend not in cls.BACKENDS:
                raise ValueError(f"backend must be one of {cls.BACKENDS}, got {backend!r}")
            return backend
        if not use_db:
            return "memory"
        return "tiledb" if cls._is_tiledb_store(data_uri) else "arrow"

    @staticmethod
    def _is_tiledb_store(data_uri) -> bool:
        try:
            entries = set(os.listdir(data_uri))
        except OSError:
            return False
        return bool(entries & {"__tiledb_group.tdb", "__group", "__meta"})

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

        if data_source is None and self.use_db:
            return iter(())

        if hasattr(data_source, "get_topics") and hasattr(data_source, "get_message"):
            self.topics = data_source.get_topics()
            return data_source.get_message()

        if callable(data_source):
            return data_source()

        raise TypeError("data_source must expose get_message() or be a callable generator factory")

    def _get_preload_count(self, preload) -> int:
        if self._init_source is None:
            return 0
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

        if self.backend == "memory":
            self.buffer_impl = NumpyBuffer(self.data_source, self.buffer_depth, self._axis, self.topics)
        elif self.backend == "arrow":
            from .buffers.arrow_buffer import ArrowBuffer

            self.buffer_impl = ArrowBuffer(
                self.data_source,
                self._init_source,
                self.group_uri,
                self._axis,
                self.topics,
                **self._backend_options,
            )
            self.topics = self.buffer_impl.topics
        else:
            from .buffers.tiledb_buffer import TileDBBuffer

            self.buffer_impl = TileDBBuffer(self.data_source, self._init_source, self.group_uri, self._axis, self.topics)
            self.topics = self.buffer_impl.topics

        if not self._axis:
            return
        if self._axis not in self.topics:
            raise ValueError(f"Axis: {self._axis} not in topics: {self.topics}")

        for _ in range(self._get_preload_count(preload)):
            try:
                self.roll_buffer(self._axis)
            except StopIteration:
                # Sources with fewer messages than preload should not leak a
                # bare StopIteration out of the constructor.
                _logger.info("Data source exhausted during preload")
                break

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
        """Message count for the current axis: the number of buffered
        messages on the numpy backend, or the topic's expected total count
        (equal to buffered messages after a full ingest) on the TileDB
        backend."""
        return self.msg_len.get(self._axis, 0)

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
                try:
                    self.reset_buffer()
                except StopIteration:
                    # A restarted source can be empty; a bare StopIteration
                    # here would become a RuntimeError under PEP 479.
                    _logger.info("Source is empty after reset")
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

    def get_index_range(
        self,
        axis: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> dict:
        if axis not in self.topics:
            raise ValueError(f"Axis: {axis} not in topics: {self.topics}")
        if step is not None and step < 1:
            raise ValueError("step must be positive")
        return self.buffer_impl.get_index_range(axis, start, stop, step)

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

    def _topic_frame_id(self, axis: str) -> str | None:
        return getattr(self.buffer_impl, "frame_ids", {}).get(axis)

    def _topic_data(self, axis: str, copy: bool = True):
        # Prefer get_topic, which returns only rows holding real messages;
        # get_buffer includes the zero-filled slots of a partially-full buffer.
        get_topic = getattr(self.buffer_impl, "get_topic", None)
        if callable(get_topic):
            return get_topic(axis, copy=copy)
        return self.get_buffer(copy=copy)[axis]

    def topic_view(self, axis: str, copy: bool = True, metadata=None):
        from .ops import topic_view

        self._validate_topic_axis(axis)
        return topic_view(
            self._topic_data(axis, copy=copy),
            topic=axis,
            source_uri=self._source_uri(),
            frame_id=self._topic_frame_id(axis),
            metadata=metadata,
            copy=False,
        )

    def topic(self, axis: str):
        from .ops import TopicPipeline

        self._validate_topic_axis(axis)
        return TopicPipeline(
            lambda chunk_size, copy, operations=(): self.iter_topic_chunks(
                axis,
                chunk_size,
                copy=copy,
                operations=operations,
            ),
            topic=axis,
            source_uri=self._source_uri(),
            frame_id=self._topic_frame_id(axis),
        )

    def query_topic(self, axis: str):
        return self.topic(axis)

    def dataset(self, topics=None):
        from .ops import DatasetQuery

        if topics is None:
            selected_topics = list(self.topics)
        elif isinstance(topics, (str, bytes)):
            selected_topics = [topics]
        else:
            selected_topics = list(topics)

        for topic in selected_topics:
            self._validate_topic_axis(topic)
        return DatasetQuery({topic: self.topic(topic) for topic in selected_topics})

    def query(self, topics=None):
        return self.dataset(topics=topics)

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False, operations=()):
        from .ops import topic_view

        self._validate_topic_axis(axis)
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        if hasattr(self.buffer_impl, "iter_topic_chunks"):
            for chunk in self.buffer_impl.iter_topic_chunks(axis, chunk_size, copy=copy, operations=operations):
                yield topic_view(
                    chunk,
                    topic=axis,
                    source_uri=chunk.get("source_uri", self._source_uri()),
                    copy=False,
                )
            return

        from .ops.core import _apply_pushdown_to_view

        # _apply_pushdown_to_view handles every operation kind (including
        # frame_id and spatial_bounds) and raises on unknown kinds instead of
        # silently dropping them.
        view = _apply_pushdown_to_view(self.topic_view(axis, copy=copy), operations)
        yield from view.iter_chunks(chunk_size, copy=False)

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
        return self.topic(axis).map(fn, copy=copy).collect(chunk_size=chunk_size, out=out)

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
        return self.topic(axis).filter(predicate, copy=copy).collect(chunk_size=chunk_size)

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
        return self.topic(axis).reduce(fn, initial=initial, copy=copy, chunk_size=chunk_size)

    def window_topic(self, axis: str, size: int | None = None, seconds: float | None = None, copy: bool = True):
        self._validate_topic_axis(axis)
        for window in self.topic_view(axis, copy=False).window(size=size, seconds=seconds, copy=copy):
            yield window.as_dict()

    def __getitem__(self, subscript: slice | int) -> np.ndarray | float | int:
        return self.buffer_impl[subscript]

    def __setitem__(self, subscript: slice | int, newval: np.ndarray) -> bool | None:
        return self.buffer_impl.__setitem__(subscript, newval)
