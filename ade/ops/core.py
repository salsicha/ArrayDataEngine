from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TopicMetadata:
    """Metadata carried with a buffered topic operation."""

    topic: str | None = None
    source_uri: str | None = None
    frame_id: str | None = None
    dtype: np.dtype | None = None
    shape: tuple[int, ...] = ()
    count: int = 0
    start_time: float | None = None
    end_time: float | None = None
    names: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        timestamps: np.ndarray,
        data: np.ndarray,
        ids: np.ndarray | None = None,
        metadata: "TopicMetadata | dict | None" = None,
        topic: str | None = None,
        source_uri: str | None = None,
        frame_id: str | None = None,
    ) -> "TopicMetadata":
        base = _coerce_metadata(metadata)
        ts = np.asarray(timestamps, dtype=np.float64)
        values = np.asarray(data)
        names = None if ids is None else np.asarray(ids).copy()

        if topic is None:
            topic = base.topic if base is not None else None
        if source_uri is None:
            source_uri = base.source_uri if base is not None else None
        if frame_id is None:
            frame_id = base.frame_id if base is not None else None

        count = int(ts.shape[0])
        return cls(
            topic=topic,
            source_uri=source_uri,
            frame_id=frame_id,
            dtype=values.dtype,
            shape=tuple(values.shape[1:]),
            count=count,
            start_time=None if count == 0 else float(ts[0]),
            end_time=None if count == 0 else float(ts[-1]),
            names=names,
        )


class TopicView:
    """Common operation interface for buffered topic arrays."""

    def __init__(
        self,
        ids: np.ndarray | None,
        timestamps: np.ndarray,
        data: np.ndarray,
        metadata: TopicMetadata | dict | None = None,
        topic: str | None = None,
        source_uri: str | None = None,
        frame_id: str | None = None,
        copy: bool = False,
    ):
        ts = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))
        values = np.asarray(data)
        if values.ndim == 0 and ts.size == 1:
            values = values.reshape((1,))
        if values.shape[:1] != ts.shape[:1]:
            raise ValueError("timestamps and data must have the same leading dimension")

        normalized_ids = _normalize_ids(ids, ts.size)
        if copy:
            ts = ts.copy()
            values = values.copy()
            normalized_ids = None if normalized_ids is None else normalized_ids.copy()

        self.ids = normalized_ids
        self.timestamps = ts
        self.data = values
        self.metadata = TopicMetadata.from_arrays(
            self.timestamps,
            self.data,
            self.ids,
            metadata=metadata,
            topic=topic,
            source_uri=source_uri,
            frame_id=frame_id,
        )

    @property
    def ts(self) -> np.ndarray:
        return self.timestamps

    @property
    def names(self) -> np.ndarray | None:
        return self.ids

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def as_dict(self, copy: bool = False, include_metadata: bool = True) -> dict:
        result = {
            "ts": self.timestamps.copy() if copy else self.timestamps,
            "data": self.data.copy() if copy else self.data,
        }
        if self.ids is not None:
            result["id"] = self.ids.copy() if copy else self.ids
            result["name"] = self.ids.copy() if copy else self.ids
        if self.metadata.topic is not None:
            result["topic"] = self.metadata.topic
        if self.metadata.source_uri is not None:
            result["source_uri"] = self.metadata.source_uri
        if self.metadata.frame_id is not None:
            result["frame_id"] = self.metadata.frame_id
        if include_metadata:
            result["metadata"] = self.metadata
        return result

    def select_indices(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        copy: bool = True,
    ) -> "TopicView":
        return self._select(slice(start, stop, step), copy=copy)

    def select_time_range(self, start: float, end: float, inclusive: bool = True, copy: bool = True) -> "TopicView":
        if start > end:
            raise ValueError("start must be less than or equal to end")

        if inclusive:
            mask = (self.timestamps >= start) & (self.timestamps <= end)
        else:
            mask = (self.timestamps > start) & (self.timestamps < end)
        return self._select(mask, copy=copy)

    def iter_chunks(self, chunk_size: int, copy: bool = False) -> Iterable["TopicView"]:
        chunk_size = _validated_chunk_size(chunk_size)
        for start in range(0, len(self), chunk_size):
            yield self._select(slice(start, start + chunk_size), copy=copy)

    def map(
        self,
        fn: Callable,
        copy: bool = True,
        out: np.ndarray | None = None,
        chunk_size: int | None = None,
    ) -> "TopicView":
        chunk_size = _validated_chunk_size(chunk_size) if chunk_size is not None else None
        ids = None if self.ids is None else self.ids.copy()
        ts = self.timestamps.copy()

        if out is not None:
            mapped = np.asarray(out)
            if mapped.shape[:1] != self.data.shape[:1]:
                raise ValueError("out must have the same leading dimension as topic data")
            for index, timestamp, value, message_id in self._iter_rows(chunk_size):
                mapped[index] = _call_with_metadata(
                    fn,
                    value.copy() if copy else value,
                    float(timestamp),
                    message_id,
                )
            return TopicView(ids, ts, mapped, metadata=self.metadata)

        mapped_values = [
            _call_with_metadata(fn, value.copy() if copy else value, float(timestamp), message_id)
            for _, timestamp, value, message_id in self._iter_rows(chunk_size)
        ]
        return TopicView(ids, ts, np.asarray(mapped_values), metadata=self.metadata)

    def filter(self, predicate: Callable, copy: bool = True, chunk_size: int | None = None) -> "TopicView":
        chunk_size = _validated_chunk_size(chunk_size) if chunk_size is not None else None
        mask = np.zeros(len(self), dtype=bool)
        for index, timestamp, value, message_id in self._iter_rows(chunk_size):
            mask[index] = bool(
                _call_with_metadata(
                    predicate,
                    value.copy() if copy else value,
                    float(timestamp),
                    message_id,
                )
            )
        return self._select(mask, copy=copy)

    def reduce(
        self,
        fn: Callable,
        initial: Any | None = None,
        copy: bool = True,
        chunk_size: int | None = None,
    ) -> Any:
        iterator = self._iter_rows(chunk_size)
        if initial is None:
            try:
                _, _, value, _ = next(iterator)
            except StopIteration as exc:
                raise ValueError("cannot reduce an empty topic without an initial value") from exc
            acc = value.copy() if copy else value
        else:
            acc = initial

        for _, timestamp, value, message_id in iterator:
            try:
                acc = fn(acc, value.copy() if copy else value, float(timestamp), message_id)
            except TypeError:
                acc = fn(acc, value.copy() if copy else value)
        return acc

    def window(
        self,
        size: int | None = None,
        seconds: float | None = None,
        copy: bool = True,
    ) -> Iterable["TopicView"]:
        if size is None and seconds is None:
            raise ValueError("size or seconds must be provided")
        if size is not None and size < 1:
            raise ValueError("size must be at least 1")
        if seconds is not None and seconds < 0:
            raise ValueError("seconds must be non-negative")

        for end_index in range(len(self)):
            start_index = 0
            if size is not None:
                start_index = max(start_index, end_index - size + 1)
            if seconds is not None:
                start_index = max(
                    start_index,
                    int(np.searchsorted(self.timestamps, self.timestamps[end_index] - seconds, side="left")),
                )
            yield self._select(slice(start_index, end_index + 1), copy=copy)

    def _select(self, selection: slice | np.ndarray, copy: bool) -> "TopicView":
        ids = None if self.ids is None else self.ids[selection]
        ts = self.timestamps[selection]
        data = self.data[selection]
        return TopicView(ids, ts, data, metadata=self.metadata, copy=copy)

    def _iter_rows(self, chunk_size: int | None = None):
        if chunk_size is None:
            for index, (timestamp, value) in enumerate(zip(self.timestamps, self.data)):
                yield index, timestamp, value, None if self.ids is None else self.ids[index]
            return

        for chunk_start in range(0, len(self), chunk_size):
            chunk_stop = min(chunk_start + chunk_size, len(self))
            for index in range(chunk_start, chunk_stop):
                yield index, self.timestamps[index], self.data[index], None if self.ids is None else self.ids[index]


@dataclass(frozen=True)
class _PipelineOperation:
    kind: str
    args: tuple
    kwargs: dict


class TopicPipeline:
    """Lazy operation pipeline for buffered topic arrays."""

    def __init__(
        self,
        chunk_source: Callable[..., Iterable[TopicView]],
        operations: Iterable[_PipelineOperation] = (),
        metadata: TopicMetadata | dict | None = None,
        topic: str | None = None,
        source_uri: str | None = None,
        frame_id: str | None = None,
    ):
        self._chunk_source = chunk_source
        self._operations = tuple(operations)
        self.metadata = _coerce_metadata(metadata) or TopicMetadata(
            topic=topic,
            source_uri=source_uri,
            frame_id=frame_id,
        )

    def map(self, fn: Callable, copy: bool = True) -> "TopicPipeline":
        return self._with_operation("map", fn, copy=copy)

    def filter(self, predicate: Callable, copy: bool = True) -> "TopicPipeline":
        return self._with_operation("filter", predicate, copy=copy)

    def time_range(self, start: float, end: float, inclusive: bool = True) -> "TopicPipeline":
        if start > end:
            raise ValueError("start must be less than or equal to end")
        return self._with_operation("time_range", float(start), float(end), inclusive=inclusive)

    def select_time_range(self, start: float, end: float, inclusive: bool = True) -> "TopicPipeline":
        return self.time_range(start, end, inclusive=inclusive)

    def index_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "TopicPipeline":
        if start is not None and start < 0:
            raise ValueError("negative lazy index ranges require collect() first")
        if stop is not None and stop < 0:
            raise ValueError("negative lazy index ranges require collect() first")
        if step is not None and step < 1:
            raise ValueError("step must be at least 1")
        return self._with_operation("index_range", start, stop, step)

    def select_indices(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "TopicPipeline":
        return self.index_range(start, stop, step)

    def iter_rows(self, chunk_size: int = 1024, copy: bool = False) -> Iterable[dict]:
        for message_id, timestamp, value in self._iter_processed_rows(chunk_size=chunk_size, copy=copy):
            yield {
                "id": message_id,
                "name": message_id,
                "ts": timestamp,
                "data": value,
            }

    def iter_chunks(self, chunk_size: int = 1024, copy: bool = False) -> Iterable[TopicView]:
        chunk_size = _validated_chunk_size(chunk_size)
        ids: list[Any] = []
        timestamps: list[float] = []
        values: list[np.ndarray] = []

        for message_id, timestamp, value in self._iter_processed_rows(chunk_size=chunk_size, copy=copy):
            ids.append(message_id)
            timestamps.append(timestamp)
            values.append(value.copy() if copy else value)
            if len(values) == chunk_size:
                yield self._make_chunk(ids, timestamps, values, copy=copy)
                ids, timestamps, values = [], [], []

        if values:
            yield self._make_chunk(ids, timestamps, values, copy=copy)

    def reduce(
        self,
        fn: Callable,
        initial: Any | None = None,
        chunk_size: int = 1024,
        copy: bool = True,
    ) -> Any:
        iterator = self._iter_processed_rows(chunk_size=chunk_size, copy=copy)
        if initial is None:
            try:
                _, _, value = next(iterator)
            except StopIteration as exc:
                raise ValueError("cannot reduce an empty topic without an initial value") from exc
            acc = value.copy() if copy else value
        else:
            acc = initial

        for message_id, timestamp, value in iterator:
            try:
                acc = fn(acc, value.copy() if copy else value, float(timestamp), message_id)
            except TypeError:
                acc = fn(acc, value.copy() if copy else value)
        return acc

    def collect(
        self,
        chunk_size: int = 1024,
        copy: bool = True,
        out: np.ndarray | None = None,
    ) -> dict:
        chunk_size = _validated_chunk_size(chunk_size)
        ids_parts = []
        ts_parts = []
        data_parts = []
        output = None if out is None else np.asarray(out)
        offset = 0

        for chunk in self.iter_chunks(chunk_size=chunk_size, copy=copy):
            if chunk.ids is not None:
                ids_parts.append(chunk.ids.copy() if copy else chunk.ids)
            ts_parts.append(chunk.timestamps.copy() if copy else chunk.timestamps)
            if output is None:
                data_parts.append(chunk.data.copy() if copy else chunk.data)
            else:
                next_offset = offset + len(chunk)
                if next_offset > output.shape[0]:
                    raise ValueError("out is too small for collected pipeline output")
                output[offset:next_offset] = chunk.data
                offset = next_offset

        ids = np.concatenate(ids_parts) if ids_parts else None
        timestamps = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.float64)
        if output is None:
            data = np.concatenate(data_parts, axis=0) if data_parts else np.array([])
        else:
            data = output if offset == output.shape[0] else output[:offset]
        return TopicView(ids, timestamps, data, metadata=self.metadata).as_dict(copy=False)

    def window(
        self,
        size: int | None = None,
        seconds: float | None = None,
        copy: bool = True,
    ) -> "TopicWindowPipeline":
        return TopicWindowPipeline(self, size=size, seconds=seconds, copy=copy)

    def _with_operation(self, kind: str, *args, **kwargs) -> "TopicPipeline":
        return TopicPipeline(
            self._chunk_source,
            operations=(*self._operations, _PipelineOperation(kind, args, kwargs)),
            metadata=self.metadata,
        )

    def _source_chunks(self, chunk_size: int, copy: bool) -> Iterable[TopicView]:
        pushdown_operations, _ = self._split_pushdown_operations()
        yield from self._chunk_source(chunk_size, copy, pushdown_operations)

    def _iter_processed_rows(self, chunk_size: int, copy: bool):
        chunk_size = _validated_chunk_size(chunk_size)
        _, operations = self._split_pushdown_operations()
        index_counters = [0] * len(operations)

        for chunk in self._source_chunks(chunk_size=chunk_size, copy=copy):
            for _, timestamp, value, message_id in chunk._iter_rows():
                current_value = value.copy() if copy else value
                current_timestamp = float(timestamp)
                current_id = message_id
                keep = True

                for op_index, operation in enumerate(operations):
                    if operation.kind == "map":
                        fn = operation.args[0]
                        op_copy = operation.kwargs.get("copy", True)
                        current_value = _call_with_metadata(
                            fn,
                            current_value.copy() if op_copy else current_value,
                            current_timestamp,
                            current_id,
                        )
                    elif operation.kind == "filter":
                        predicate = operation.args[0]
                        op_copy = operation.kwargs.get("copy", True)
                        keep = bool(
                            _call_with_metadata(
                                predicate,
                                current_value.copy() if op_copy else current_value,
                                current_timestamp,
                                current_id,
                            )
                        )
                    elif operation.kind == "time_range":
                        start, end = operation.args
                        if operation.kwargs.get("inclusive", True):
                            keep = current_timestamp >= start and current_timestamp <= end
                        else:
                            keep = current_timestamp > start and current_timestamp < end
                    elif operation.kind == "index_range":
                        current_index = index_counters[op_index]
                        index_counters[op_index] += 1
                        keep = _slice_contains(current_index, *operation.args)
                    else:
                        raise ValueError(f"unsupported pipeline operation: {operation.kind}")

                    if not keep:
                        break

                if keep:
                    yield current_id, current_timestamp, current_value

    def _split_pushdown_operations(self) -> tuple[tuple[_PipelineOperation, ...], tuple[_PipelineOperation, ...]]:
        pushdown_operations = []
        for operation in self._operations:
            if operation.kind not in {"time_range", "index_range"}:
                break
            pushdown_operations.append(operation)

        pushed = tuple(pushdown_operations)
        return pushed, self._operations[len(pushed):]

    def _make_chunk(self, ids: list[Any], timestamps: list[float], values: list[np.ndarray], copy: bool) -> TopicView:
        ids_array = np.asarray(ids, dtype=object)
        ts_array = np.asarray(timestamps, dtype=np.float64)
        data_array = np.asarray(values)
        return TopicView(ids_array, ts_array, data_array, metadata=self.metadata, copy=copy)


class TopicWindowPipeline:
    """Lazy sliding-window view over a topic pipeline."""

    def __init__(
        self,
        pipeline: TopicPipeline,
        size: int | None = None,
        seconds: float | None = None,
        copy: bool = True,
    ):
        if size is None and seconds is None:
            raise ValueError("size or seconds must be provided")
        if size is not None and size < 1:
            raise ValueError("size must be at least 1")
        if seconds is not None and seconds < 0:
            raise ValueError("seconds must be non-negative")

        self.pipeline = pipeline
        self.size = size
        self.seconds = seconds
        self.copy = copy

    def iter_windows(self, chunk_size: int = 1024) -> Iterable[TopicView]:
        ids = deque()
        timestamps = deque()
        values = deque()

        for message_id, timestamp, value in self.pipeline._iter_processed_rows(chunk_size=chunk_size, copy=self.copy):
            ids.append(message_id)
            timestamps.append(timestamp)
            values.append(value.copy() if self.copy else value)

            if self.seconds is not None:
                cutoff = timestamp - self.seconds
                while timestamps and timestamps[0] < cutoff:
                    ids.popleft()
                    timestamps.popleft()
                    values.popleft()

            if self.size is not None:
                while len(values) > self.size:
                    ids.popleft()
                    timestamps.popleft()
                    values.popleft()

            yield TopicView(
                np.asarray(ids, dtype=object),
                np.asarray(timestamps, dtype=np.float64),
                np.asarray(values),
                metadata=self.pipeline.metadata,
                copy=self.copy,
            )

    def collect(self, chunk_size: int = 1024) -> list[TopicView]:
        return list(self.iter_windows(chunk_size=chunk_size))


def topic_view(
    topic_data: dict | np.ndarray | TopicView,
    topic: str | None = None,
    source_uri: str | None = None,
    frame_id: str | None = None,
    metadata: TopicMetadata | dict | None = None,
    copy: bool = False,
) -> TopicView:
    """Return a metadata-preserving view over a buffered topic."""

    if isinstance(topic_data, TopicView):
        return TopicView(
            topic_data.ids,
            topic_data.timestamps,
            topic_data.data,
            metadata=topic_data.metadata if metadata is None else metadata,
            topic=topic,
            source_uri=source_uri,
            frame_id=frame_id,
            copy=copy,
        )

    inferred_metadata = None
    if isinstance(topic_data, dict):
        inferred_metadata = topic_data.get("metadata")
        topic = topic if topic is not None else topic_data.get("topic")
        source_uri = source_uri if source_uri is not None else topic_data.get("source_uri")
        frame_id = frame_id if frame_id is not None else topic_data.get("frame_id")
        if topic is None and "id" in topic_data and np.asarray(topic_data["id"]).ndim == 0:
            topic = _decode_scalar(topic_data["id"])

    ids, ts, data = topic_parts(topic_data)
    return TopicView(
        ids,
        ts,
        data,
        metadata=metadata if metadata is not None else inferred_metadata,
        topic=topic,
        source_uri=source_uri,
        frame_id=frame_id,
        copy=copy,
    )


def topic_pipeline(
    topic_data: dict | np.ndarray | TopicView,
    topic: str | None = None,
    source_uri: str | None = None,
    frame_id: str | None = None,
    metadata: TopicMetadata | dict | None = None,
) -> TopicPipeline:
    """Return a lazy operation pipeline over topic data."""

    view = topic_view(
        topic_data,
        topic=topic,
        source_uri=source_uri,
        frame_id=frame_id,
        metadata=metadata,
        copy=False,
    )

    def source(chunk_size: int, copy: bool, operations=()):
        selected = _apply_pushdown_to_view(view, operations)
        yield from selected.iter_chunks(chunk_size=chunk_size, copy=copy)

    return TopicPipeline(source, metadata=view.metadata)


def _apply_pushdown_to_view(view: TopicView, operations: Iterable[_PipelineOperation]) -> TopicView:
    selected = view
    for operation in operations:
        if operation.kind == "time_range":
            start, end = operation.args
            selected = selected.select_time_range(
                start,
                end,
                inclusive=operation.kwargs.get("inclusive", True),
                copy=False,
            )
        elif operation.kind == "index_range":
            selected = selected.select_indices(*operation.args, copy=False)
        else:
            raise ValueError(f"unsupported pushdown operation: {operation.kind}")
    return selected


def topic_parts(topic_data: dict | np.ndarray | TopicView) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Return `(ids, timestamps, data)` from ADE topic data."""

    if isinstance(topic_data, TopicView):
        return topic_data.ids, topic_data.timestamps, topic_data.data

    if isinstance(topic_data, np.ndarray) and topic_data.dtype.fields:
        ids = topic_data["id"] if "id" in topic_data.dtype.fields else None
        return ids, np.asarray(topic_data["ts"], dtype=np.float64), np.asarray(topic_data["data"])

    if not isinstance(topic_data, dict) or "ts" not in topic_data or "data" not in topic_data:
        raise TypeError("topic_data must be a structured topic array or a dict with 'ts' and 'data'")

    ts = np.asarray(topic_data["ts"], dtype=np.float64)
    data = np.asarray(topic_data["data"])
    ids = topic_data.get("id", topic_data.get("name"))
    if ids is None:
        return None, ts, data

    return _normalize_ids(ids, ts.shape[0]), ts, data


def _coerce_metadata(metadata: TopicMetadata | dict | None) -> TopicMetadata | None:
    if metadata is None or isinstance(metadata, TopicMetadata):
        return metadata
    return TopicMetadata(
        topic=metadata.get("topic"),
        source_uri=metadata.get("source_uri"),
        frame_id=metadata.get("frame_id"),
        dtype=np.dtype(metadata["dtype"]) if metadata.get("dtype") is not None else None,
        shape=tuple(metadata.get("shape", ())),
        count=int(metadata.get("count", 0)),
        start_time=metadata.get("start_time"),
        end_time=metadata.get("end_time"),
        names=None if metadata.get("names") is None else np.asarray(metadata["names"]),
    )


def _decode_scalar(value: Any) -> str | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    return None


def _normalize_ids(ids: Any, count: int) -> np.ndarray | None:
    if ids is None:
        return None

    ids_array = np.asarray(ids)
    if ids_array.ndim == 0 or ids_array.shape[:1] != (count,):
        ids_array = np.full((count,), ids, dtype=object)
    return ids_array


def _validated_chunk_size(chunk_size: int) -> int:
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    return chunk_size


def _slice_contains(index: int, start: int | None, stop: int | None, step: int | None) -> bool:
    start = 0 if start is None else start
    step = 1 if step is None else step
    if index < start:
        return False
    if stop is not None and index >= stop:
        return False
    return (index - start) % step == 0


def _call_with_metadata(fn: Callable, data: np.ndarray, ts: float, message_id: Any) -> Any:
    try:
        return fn(data, ts, message_id)
    except TypeError:
        try:
            return fn(data, ts)
        except TypeError:
            return fn(data)


def select_indices(topic_data: dict | np.ndarray, start: int | None = None, stop: int | None = None, step: int | None = None) -> dict:
    return topic_view(topic_data).select_indices(start, stop, step).as_dict()


def select_time_range(topic_data: dict | np.ndarray, start: float, end: float, inclusive: bool = True) -> dict:
    return topic_view(topic_data).select_time_range(start, end, inclusive=inclusive).as_dict()


def map_topic(
    topic_data: dict | np.ndarray,
    fn: Callable,
    copy: bool = True,
    out: np.ndarray | None = None,
    chunk_size: int | None = None,
) -> dict:
    return topic_view(topic_data).map(fn, copy=copy, out=out, chunk_size=chunk_size).as_dict()


def filter_topic(
    topic_data: dict | np.ndarray,
    predicate: Callable,
    copy: bool = True,
    chunk_size: int | None = None,
) -> dict:
    return topic_view(topic_data).filter(predicate, copy=copy, chunk_size=chunk_size).as_dict()


def reduce_topic(
    topic_data: dict | np.ndarray,
    fn: Callable,
    initial: Any | None = None,
    copy: bool = True,
    chunk_size: int | None = None,
) -> Any:
    return topic_view(topic_data).reduce(fn, initial=initial, copy=copy, chunk_size=chunk_size)


def window_topic(
    topic_data: dict | np.ndarray,
    size: int | None = None,
    seconds: float | None = None,
    copy: bool = True,
) -> Iterable[dict]:
    for window in topic_view(topic_data).window(size=size, seconds=seconds, copy=copy):
        yield window.as_dict()


def iter_chunks(topic_data: dict | np.ndarray | TopicView, chunk_size: int, copy: bool = False) -> Iterable[TopicView]:
    yield from topic_view(topic_data).iter_chunks(chunk_size, copy=copy)


def nearest_time_index(timestamps: np.ndarray, query_time: float, tolerance: float | None = None) -> int | None:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size == 0:
        return None

    insert_at = int(np.searchsorted(ts, query_time))
    candidates = []
    if insert_at < ts.size:
        candidates.append(insert_at)
    if insert_at > 0:
        candidates.append(insert_at - 1)

    best = min(candidates, key=lambda i: abs(ts[i] - query_time))
    if tolerance is not None and abs(ts[best] - query_time) > tolerance:
        return None
    return int(best)


def align_nearest(reference_topic: dict | np.ndarray, target_topic: dict | np.ndarray, tolerance: float | None = None) -> dict:
    _, ref_ts, _ = topic_parts(reference_topic)
    target_ids, target_ts, target_data = topic_parts(target_topic)

    indices = np.array([
        -1 if (idx := nearest_time_index(target_ts, float(timestamp), tolerance)) is None else idx
        for timestamp in ref_ts
    ], dtype=np.int64)
    valid = indices >= 0
    safe_indices = np.where(valid, indices, 0)

    aligned = {
        "reference_ts": ref_ts.copy(),
        "target_ts": np.where(valid, target_ts[safe_indices], np.nan),
        "target_index": indices,
        "valid": valid,
    }
    aligned_data = np.full((ref_ts.size,) + target_data.shape[1:], np.nan, dtype=np.float64)
    if valid.any():
        aligned_data[valid] = target_data[indices[valid]]
    aligned["data"] = aligned_data
    if target_ids is not None:
        aligned["id"] = np.asarray([target_ids[i] if ok else None for i, ok in zip(safe_indices, valid)], dtype=object)
    return aligned
