from __future__ import annotations

import logging
import os
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source import DataSources

from .buffers.common import holds_arrow_store, is_tiledb_store, normalize_ragged_topics, validate_capacity
from .buffers.numpy_buffer import NumpyBuffer

_logger = logging.getLogger(__name__)


class DataBuffer:
    """Buffer Class
    Attributes:
    Args:
    Returns:
    """

    BACKENDS = ("memory", "arrow", "tiledb")
    BACKEND_OPTIONS = {
        "memory": frozenset(),
        "arrow": frozenset({
            "flush_bytes", "row_group_bytes", "compression", "batch_readahead", "fragment_readahead",
            "use_threads", "row_overhead_bytes",
        }),
        "tiledb": frozenset({"flush_bytes", "consolidate_max_bytes", "row_overhead_bytes"}),
    }

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
        ragged_topics=None,
        max_points: int | None = None,
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
        arrow backend accepts `flush_bytes`, `row_overhead_bytes`,
        `row_group_bytes`, `compression`, `batch_readahead`,
        `fragment_readahead`, and `use_threads` (see
        `arraydataengine.buffers.arrow_buffer` for defaults); the tiledb
        backend accepts `flush_bytes`, `row_overhead_bytes`, and
        `consolidate_max_bytes`.

        `topics` limits buffering to those topics (plus `axis`); by default
        every topic the source lists is buffered.

        Variable-size topics (point clouds): a topic whose first message has
        a `point_count` key, or that is listed in `ragged_topics` (a mapping
        `{topic: capacity}` or a list of topics), stores each message's rows
        zero-padded to the topic's capacity together with its real row count.
        Reads then include a `lengths` array (structured `get_buffer()` arrays
        a `length` field). The capacity is `ragged_topics[topic]`, else
        `max_points`, else the source's `max_points`, else
        `DEFAULT_MAX_POINTS`; a message with more rows raises ValueError.
        """
        if buffer_depth < 1:
            raise ValueError("buffer_depth must be at least 1")

        self.buffer_depth = buffer_depth
        self._source_exhausted = False
        self._axis = axis
        self._requested_topics = None if topics is None else list(dict.fromkeys(topics))
        self.topics = [] if topics is None else list(self._requested_topics)
        self._init_source = data_source
        self.group_uri = data_uri
        self.preload = preload
        self.ragged_topics = normalize_ragged_topics(ragged_topics)
        self.max_points = validate_capacity(max_points, "max_points")
        self.backend = self._resolve_backend(backend, use_db, data_uri)
        self.use_db = self.backend != "memory"
        self._backend_options = dict(backend_options or {})
        unsupported = set(self._backend_options) - self.BACKEND_OPTIONS[self.backend]
        if unsupported:
            supported = ", ".join(sorted(self.BACKEND_OPTIONS[self.backend])) or "none"
            raise ValueError(
                f"backend_options {sorted(unsupported)} are not supported by the {self.backend!r} backend "
                f"(supported: {supported})"
            )

        if self.backend == "tiledb":
            self._prepare_tiledb_location()

        self.set_methods()
        self.reset()

    def _prepare_tiledb_location(self) -> None:
        """Create the TileDB group for a writer opening a missing or empty directory.

        Read-only opens never write; directories holding an Arrow store are
        refused; other non-empty directories get their group (if they hold
        TileDB arrays) when the first topic is created.
        """
        import tiledb

        uri = str(self.group_uri)
        if holds_arrow_store(uri) and not is_tiledb_store(uri):
            raise ValueError(f"{uri} holds an Arrow store; open it with backend='arrow' (or no backend)")
        if self._init_source is None or tiledb.object_type(uri) == "group":
            return
        try:
            empty = not os.listdir(uri)
        except FileNotFoundError:
            empty = True
        if empty:
            os.makedirs(uri, exist_ok=True)
            tiledb.group_create(uri)

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
        return is_tiledb_store(data_uri)

    def _source_max_points(self) -> int | None:
        """The configured `max_points` of the source (or the source it wraps)."""
        source = self._init_source
        for candidate in (source, getattr(source, "source", None)):
            value = getattr(candidate, "max_points", None) if candidate is not None else None
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0:
                return int(value)
        return None

    def _ragged_options(self) -> dict:
        max_points = self.max_points if self.max_points is not None else self._source_max_points()
        return {"ragged_topics": self.ragged_topics, "max_points": max_points}

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
            source_topics = list(data_source.get_topics())
            if self._requested_topics is None:
                self.topics = source_topics
            else:
                self.topics = list(self._requested_topics)
                if self._axis and self._axis not in self.topics and self._axis in source_topics:
                    self.topics.append(self._axis)
            messages = data_source.get_message()
        elif callable(data_source):
            if self._requested_topics is not None:
                self.topics = list(dict.fromkeys([*self._requested_topics, *self.topics]))
            messages = data_source()
        else:
            raise TypeError("data_source must expose get_message() or be a callable generator factory")

        if self._requested_topics is None:
            return messages
        return self._selected_messages(messages)

    def _selected_messages(self, messages):
        """Only messages of the requested topics (and the current axis)."""
        wanted = frozenset(self._requested_topics)
        for msg in messages:
            topic = msg.get("topic") if isinstance(msg, dict) else None
            if topic in wanted or topic == self._axis:
                yield msg

    def _get_preload_count(self, preload) -> int:
        """Number of axis messages to roll during `reset()`.

        `preload=True` means `buffer_depth`; integer values are capped at
        `buffer_depth` (so the default depth of 1 preloads at most one axis
        message). Use `load_data_db()` or `roll_buffer()` to ingest more.
        """
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
        self._source_exhausted = False
        self.data_source = self._get_data_source()

        if self.backend == "memory":
            self.buffer_impl = NumpyBuffer(
                self.data_source, self.buffer_depth, self._axis, self.topics, **self._ragged_options()
            )
        elif self.backend == "arrow":
            from .buffers.arrow_buffer import ArrowBuffer

            self.buffer_impl = ArrowBuffer(
                self.data_source,
                self._init_source,
                self.group_uri,
                self._axis,
                self.topics,
                **self._ragged_options(),
                **self._backend_options,
            )
            self.topics = self.buffer_impl.topics
        else:
            from .buffers.tiledb_buffer import TileDBBuffer

            self.buffer_impl = TileDBBuffer(
                self.data_source,
                self._init_source,
                self.group_uri,
                self._axis,
                self.topics,
                **self._ragged_options(),
                **self._backend_options,
            )
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
        if exc_type is None:
            self.close_completed()
            return False
        # Leaving on an exception persists staged rows but keeps every
        # topic's closed flag (partial topics stay resumable, complete ones
        # stay closed), and never replaces the original exception.
        try:
            self.close()
        except Exception as close_error:
            _logger.warning("Closing the buffer after an error also failed: %r", close_error)
            if isinstance(exc, BaseException):
                exc.add_note(f"closing the buffer also failed: {close_error!r}")
        return False

    def close_completed(self) -> None:
        """Close the store, marking only topics known to be complete as closed.

        A topic is complete once it holds the source's `get_count(topic)`
        messages, or once the source was exhausted after delivering it in this
        session. Other topics keep their closed flag so a later run with the
        same source resumes them.
        """
        buffer_impl = getattr(self, "buffer_impl", None)
        if buffer_impl is None:
            return
        if hasattr(buffer_impl, "close_completed"):
            buffer_impl.close_completed(source_exhausted=self._source_exhausted)
        else:
            self.close()

    def compact(self, **kwargs) -> None:
        """Merge fragments and vacuum superseded files (TileDB backend).

        Run only while no other process reads the store; see
        `TileDBBuffer.compact`. Other backends need no compaction.
        """
        compact = getattr(getattr(self, "buffer_impl", None), "compact", None)
        if callable(compact):
            compact(**kwargs)

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
        """Messages available for the current axis on every backend: rows
        buffered in the memory ring, or rows stored (persisted and staged) in
        a persistent store. `buffer[get_size() - 1]` is the newest message."""
        buffer_impl = getattr(self, "buffer_impl", None)
        get_size = getattr(buffer_impl, "get_size", None)
        if callable(get_size):
            return get_size(self._axis)
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
                    # Mark complete only topics this source delivered; other
                    # stored topics (e.g. derived ones) keep their flag.
                    seen = getattr(self.buffer_impl, "_resume_seen", None)
                    for topic in list(self.topics):
                        self.buffer_impl.close_topic(topic, closed=True if seen is None or topic in seen else None)
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
        try:
            self.buffer_impl.roll_buffer(axis)
        except StopIteration:
            self._source_exhausted = True
            raise

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
        # Prefer get_topic: one topic's rows holding real messages, oldest first.
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
