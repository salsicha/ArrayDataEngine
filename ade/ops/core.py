from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_COLLECT_MAX_BYTES = 512 * 1024 * 1024


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


@dataclass(frozen=True)
class PipelineProgress:
    """Progress snapshot for long-running source and topic pipelines."""

    processed: int
    emitted: int
    skipped: int
    topic: str | None = None
    message_id: Any | None = None
    timestamp: float | None = None
    done: bool = False
    cancelled: bool = False
    checkpoint: dict[str, Any] | None = None


class PipelineCancelled(RuntimeError):
    """Raised when a pipeline cancellation token is set."""


class CancellationToken:
    """Mutable cancellation token shared with pipeline execution."""

    def __init__(self, cancelled: bool = False):
        self.cancelled = bool(cancelled)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False


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


@dataclass(frozen=True)
class _ProcessedChunk:
    rows: tuple[tuple[Any, float, np.ndarray], ...]
    processed: int
    emitted: int
    skipped: int
    message_id: Any | None = None
    timestamp: float | None = None


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

    def frame_id(self, *frame_ids: str | Iterable[str]) -> "TopicPipeline":
        targets = _normalize_text_selection(frame_ids, "frame_id")
        return self._with_operation("frame_id", targets)

    def spatial_bounds(
        self,
        min_bound,
        max_bound,
        columns: tuple[int, ...] | None = None,
    ) -> "TopicPipeline":
        min_array, max_array, columns = _normalize_bounds(min_bound, max_bound, columns)
        return self._with_operation("spatial_bounds", min_array, max_array, columns=columns)

    def iter_rows(
        self,
        chunk_size: int = 1024,
        copy: bool = False,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
        max_workers: int | None = 1,
    ) -> Iterable[dict]:
        if _validated_max_workers(max_workers) > 1:
            for chunk in self.iter_chunks(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
                max_workers=max_workers,
            ):
                for _, timestamp, value, message_id in chunk._iter_rows():
                    yield {
                        "id": message_id,
                        "name": message_id,
                        "ts": float(timestamp),
                        "data": value.copy() if copy else value,
                    }
            return

        for message_id, timestamp, value in self._iter_processed_rows(
            chunk_size=chunk_size,
            copy=copy,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        ):
            yield {
                "id": message_id,
                "name": message_id,
                "ts": timestamp,
                "data": value,
            }

    def iter_chunks(
        self,
        chunk_size: int = 1024,
        copy: bool = False,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
        max_workers: int | None = 1,
    ) -> Iterable[TopicView]:
        chunk_size = _validated_chunk_size(chunk_size)
        max_workers = _validated_max_workers(max_workers)
        ids: list[Any] = []
        timestamps: list[float] = []
        values: list[np.ndarray] = []

        if max_workers == 1:
            processed_rows = self._iter_processed_rows(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
            )
        else:
            processed_rows = self._iter_processed_rows_parallel(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
                max_workers=max_workers,
            )

        for message_id, timestamp, value in processed_rows:
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
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ) -> Any:
        iterator = self._iter_processed_rows(
            chunk_size=chunk_size,
            copy=copy,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        )
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
        max_rows: int | None = None,
        max_bytes: int | None = DEFAULT_COLLECT_MAX_BYTES,
        allow_large: bool = False,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
        max_workers: int | None = 1,
    ) -> dict:
        chunk_size = _validated_chunk_size(chunk_size)
        ids_parts = []
        ts_parts = []
        data_parts = []
        output = None if out is None else np.asarray(out)
        offset = 0
        collected_bytes = 0

        for chunk in self.iter_chunks(
            chunk_size=chunk_size,
            copy=copy,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
            max_workers=max_workers,
        ):
            offset += len(chunk)
            collected_bytes += _topic_view_nbytes(chunk, include_data=output is None)
            _check_collect_limits(offset, collected_bytes, max_rows, max_bytes, allow_large)

            if chunk.ids is not None:
                ids_parts.append(chunk.ids.copy() if copy else chunk.ids)
            ts_parts.append(chunk.timestamps.copy() if copy else chunk.timestamps)
            if output is None:
                data_parts.append(chunk.data.copy() if copy else chunk.data)
            else:
                if offset > output.shape[0]:
                    raise ValueError("out is too small for collected pipeline output")
                output[offset - len(chunk):offset] = chunk.data

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

    def _iter_processed_rows(
        self,
        chunk_size: int,
        copy: bool,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ):
        chunk_size = _validated_chunk_size(chunk_size)
        _, operations = self._split_pushdown_operations()
        index_counters = _checkpoint_operation_counters(checkpoint, len(operations), kind="topic")
        resume_processed = _checkpoint_processed(checkpoint)
        processed = 0
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        progress_interval = _validated_progress_interval(progress_interval)
        last_progress: PipelineProgress | None = None

        for chunk in self._source_chunks(chunk_size=chunk_size, copy=copy):
            for _, timestamp, value, message_id in chunk._iter_rows():
                processed += 1
                if processed <= resume_processed:
                    continue
                _raise_if_cancelled(
                    cancel_token,
                    checkpoint,
                    PipelineProgress(
                        processed=processed - 1,
                        emitted=emitted,
                        skipped=skipped,
                        topic=self.metadata.topic,
                        checkpoint=_checkpoint_snapshot(checkpoint),
                    ),
                    operation_counters=index_counters,
                )
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
                    elif operation.kind == "frame_id":
                        targets = operation.args[0]
                        metadata_frame_id = _decode_text(self.metadata.frame_id)
                        if metadata_frame_id is not None:
                            keep = metadata_frame_id in targets
                        else:
                            keep = _row_frame_id(current_value, current_id) in targets
                    elif operation.kind == "spatial_bounds":
                        min_bound, max_bound = operation.args
                        keep = _spatial_value_in_bounds(
                            current_value,
                            min_bound=min_bound,
                            max_bound=max_bound,
                            columns=operation.kwargs["columns"],
                        )
                    else:
                        raise ValueError(f"unsupported pipeline operation: {operation.kind}")

                    if not keep:
                        break

                if keep:
                    emitted += 1
                else:
                    skipped += 1

                last_progress = PipelineProgress(
                    processed=processed,
                    emitted=emitted,
                    skipped=skipped,
                    topic=self.metadata.topic,
                    message_id=current_id,
                    timestamp=current_timestamp,
                    checkpoint=_checkpoint_snapshot(checkpoint),
                )
                _update_checkpoint(checkpoint, last_progress, operation_counters=index_counters)
                _notify_progress(progress_callback, last_progress, progress_interval)

                if keep:
                    yield current_id, current_timestamp, current_value

        done_progress = PipelineProgress(
            processed=max(processed, resume_processed),
            emitted=emitted,
            skipped=skipped,
            topic=self.metadata.topic,
            message_id=None if last_progress is None else last_progress.message_id,
            timestamp=None if last_progress is None else last_progress.timestamp,
            done=True,
            checkpoint=_checkpoint_snapshot(checkpoint),
        )
        _update_checkpoint(checkpoint, done_progress, operation_counters=index_counters)
        _notify_progress(progress_callback, done_progress, progress_interval, force=True)

    def _iter_processed_rows_parallel(
        self,
        chunk_size: int,
        copy: bool,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
        max_workers: int | None = 1,
    ):
        chunk_size = _validated_chunk_size(chunk_size)
        max_workers = _validated_max_workers(max_workers)
        _, operations = self._split_pushdown_operations()
        _validate_parallel_operations(operations)

        resume_processed = _checkpoint_processed(checkpoint)
        processed = resume_processed
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        progress_interval = _validated_progress_interval(progress_interval)
        operation_counters = _checkpoint_operation_counters(checkpoint, len(operations), kind="topic")
        source_seen = 0
        next_sequence = 0
        next_yield = 0
        pending = {}
        last_progress: PipelineProgress | None = None

        def submit_ready(executor, source_iter):
            nonlocal next_sequence, source_seen
            while len(pending) < max_workers * 2:
                _raise_if_cancelled(
                    cancel_token,
                    checkpoint,
                    PipelineProgress(
                        processed=processed,
                        emitted=emitted,
                        skipped=skipped,
                        topic=self.metadata.topic,
                        checkpoint=_checkpoint_snapshot(checkpoint),
                    ),
                    operation_counters=operation_counters,
                )
                try:
                    chunk = next(source_iter)
                except StopIteration:
                    return

                chunk_length = len(chunk)
                if source_seen + chunk_length <= resume_processed:
                    source_seen += chunk_length
                    continue
                if source_seen < resume_processed:
                    chunk = chunk._select(slice(resume_processed - source_seen, None), copy=copy)
                    source_seen = resume_processed

                source_seen += len(chunk)
                pending[next_sequence] = executor.submit(
                    _process_pipeline_chunk,
                    chunk,
                    operations,
                    self.metadata,
                    copy,
                )
                next_sequence += 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            source_iter = iter(self._source_chunks(chunk_size=chunk_size, copy=copy))
            submit_ready(executor, source_iter)
            while pending:
                future = pending.pop(next_yield)
                result = future.result()
                next_yield += 1

                processed += result.processed
                emitted += result.emitted
                skipped += result.skipped
                last_progress = PipelineProgress(
                    processed=processed,
                    emitted=emitted,
                    skipped=skipped,
                    topic=self.metadata.topic,
                    message_id=result.message_id,
                    timestamp=result.timestamp,
                    checkpoint=_checkpoint_snapshot(checkpoint),
                )
                _update_checkpoint(checkpoint, last_progress, operation_counters=operation_counters)
                _notify_progress(progress_callback, last_progress, progress_interval)
                _raise_if_cancelled(
                    cancel_token,
                    checkpoint,
                    last_progress,
                    operation_counters=operation_counters,
                )

                for row in result.rows:
                    yield row

                submit_ready(executor, source_iter)

        done_progress = PipelineProgress(
            processed=processed,
            emitted=emitted,
            skipped=skipped,
            topic=self.metadata.topic,
            message_id=None if last_progress is None else last_progress.message_id,
            timestamp=None if last_progress is None else last_progress.timestamp,
            done=True,
            checkpoint=_checkpoint_snapshot(checkpoint),
        )
        _update_checkpoint(checkpoint, done_progress, operation_counters=operation_counters)
        _notify_progress(progress_callback, done_progress, progress_interval, force=True)

    def _split_pushdown_operations(self) -> tuple[tuple[_PipelineOperation, ...], tuple[_PipelineOperation, ...]]:
        pushdown_operations = []
        remaining_operations = []
        pushing = True
        for operation in self._operations:
            if pushing and operation.kind in {"time_range", "index_range", "frame_id", "spatial_bounds"}:
                pushdown_operations.append(operation)
                if operation.kind == "spatial_bounds":
                    remaining_operations.append(operation)
                    pushing = False
                continue
            pushing = False
            remaining_operations.append(operation)

        return tuple(pushdown_operations), tuple(remaining_operations)

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

    def iter_windows(
        self,
        chunk_size: int = 1024,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ) -> Iterable[TopicView]:
        ids = deque()
        timestamps = deque()
        values = deque()

        for message_id, timestamp, value in self.pipeline._iter_processed_rows(
            chunk_size=chunk_size,
            copy=self.copy,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        ):
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

    def collect(
        self,
        chunk_size: int = 1024,
        max_windows: int | None = None,
        max_bytes: int | None = DEFAULT_COLLECT_MAX_BYTES,
        allow_large: bool = False,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ) -> list[TopicView]:
        windows = []
        collected_bytes = 0
        for window in self.iter_windows(
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        ):
            windows.append(window)
            collected_bytes += _topic_view_nbytes(window)
            _check_collect_limits(len(windows), collected_bytes, max_windows, max_bytes, allow_large)
        return windows


class DatasetQuery:
    """Lazy selection interface over multiple topic pipelines."""

    def __init__(self, topics: Mapping[str, TopicPipeline | dict | np.ndarray | TopicView]):
        self._pipelines = {
            topic: pipeline if isinstance(pipeline, TopicPipeline) else topic_pipeline(pipeline, topic=topic)
            for topic, pipeline in topics.items()
        }

    @property
    def topics(self) -> tuple[str, ...]:
        return tuple(self._pipelines)

    def select_topics(self, *topics: str | Iterable[str]) -> "DatasetQuery":
        names = _normalize_topic_selection(topics)
        missing = [topic for topic in names if topic not in self._pipelines]
        if missing:
            raise ValueError(f"topics not found: {missing}")
        return DatasetQuery({topic: self._pipelines[topic] for topic in names})

    def select_topic(self, topic: str) -> "DatasetQuery":
        return self.select_topics(topic)

    def time_range(self, start: float, end: float, inclusive: bool = True) -> "DatasetQuery":
        if start > end:
            raise ValueError("start must be less than or equal to end")
        return self._map_pipelines(lambda pipeline: pipeline.time_range(start, end, inclusive=inclusive))

    def select_time_range(self, start: float, end: float, inclusive: bool = True) -> "DatasetQuery":
        return self.time_range(start, end, inclusive=inclusive)

    def index_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "DatasetQuery":
        return self._map_pipelines(lambda pipeline: pipeline.index_range(start, stop, step))

    def select_indices(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "DatasetQuery":
        return self.index_range(start, stop, step)

    def frame_id(self, *frame_ids: str | Iterable[str]) -> "DatasetQuery":
        targets = _normalize_text_selection(frame_ids, "frame_id")
        selected = {}
        for topic, pipeline in self._pipelines.items():
            metadata_frame_id = pipeline.metadata.frame_id
            if metadata_frame_id is not None:
                if _decode_text(metadata_frame_id) in targets:
                    selected[topic] = pipeline
                continue

            selected[topic] = pipeline.frame_id(targets)
        return DatasetQuery(selected)

    def geographic_bounds(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        columns: tuple[int, int] = (0, 1),
    ) -> "DatasetQuery":
        if min_lat > max_lat:
            raise ValueError("min_lat must be less than or equal to max_lat")
        if min_lon > max_lon:
            raise ValueError("min_lon must be less than or equal to max_lon")
        return self.filter(
            lambda data, ts, name: _geographic_value_in_bounds(
                data,
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
                columns=columns,
            ),
            copy=False,
        )

    def geo_bounds(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        columns: tuple[int, int] = (0, 1),
    ) -> "DatasetQuery":
        return self.geographic_bounds(min_lat, min_lon, max_lat, max_lon, columns=columns)

    def spatial_bounds(
        self,
        min_bound,
        max_bound,
        columns: tuple[int, ...] | None = None,
    ) -> "DatasetQuery":
        return self._map_pipelines(lambda pipeline: pipeline.spatial_bounds(min_bound, max_bound, columns=columns))

    def map(self, fn: Callable, copy: bool = True) -> "DatasetQuery":
        return self._map_pipelines(lambda pipeline: pipeline.map(fn, copy=copy))

    def filter(self, predicate: Callable, copy: bool = True) -> "DatasetQuery":
        return self._map_pipelines(lambda pipeline: pipeline.filter(predicate, copy=copy))

    def iter_topics(self) -> Iterable[tuple[str, TopicPipeline]]:
        yield from self._pipelines.items()

    def iter_chunks(
        self,
        chunk_size: int = 1024,
        copy: bool = False,
        max_workers: int | None = 1,
    ) -> Iterable[tuple[str, TopicView]]:
        for topic, pipeline in self._pipelines.items():
            for chunk in pipeline.iter_chunks(chunk_size=chunk_size, copy=copy, max_workers=max_workers):
                yield topic, chunk

    def iter_rows(
        self,
        chunk_size: int = 1024,
        copy: bool = False,
        max_workers: int | None = 1,
    ) -> Iterable[dict]:
        for topic, pipeline in self._pipelines.items():
            for row in pipeline.iter_rows(chunk_size=chunk_size, copy=copy, max_workers=max_workers):
                row["topic"] = topic
                yield row

    def collect(
        self,
        chunk_size: int = 1024,
        copy: bool = True,
        max_rows: int | None = None,
        max_bytes: int | None = DEFAULT_COLLECT_MAX_BYTES,
        allow_large: bool = False,
        max_workers: int | None = 1,
        topic_workers: int | None = 1,
    ) -> dict[str, dict]:
        chunk_size = _validated_chunk_size(chunk_size)
        max_workers = _validated_max_workers(max_workers)
        topic_workers = _validated_max_workers(topic_workers)
        if topic_workers > 1:
            return self._collect_parallel_topics(
                chunk_size=chunk_size,
                copy=copy,
                max_rows=max_rows,
                max_bytes=max_bytes,
                allow_large=allow_large,
                max_workers=max_workers,
                topic_workers=topic_workers,
            )

        collected = {}
        total_rows = 0
        total_bytes = 0

        for topic, pipeline in self._pipelines.items():
            ids_parts = []
            ts_parts = []
            data_parts = []
            for chunk in pipeline.iter_chunks(chunk_size=chunk_size, copy=copy, max_workers=max_workers):
                total_rows += len(chunk)
                total_bytes += _topic_view_nbytes(chunk)
                _check_collect_limits(total_rows, total_bytes, max_rows, max_bytes, allow_large)

                if chunk.ids is not None:
                    ids_parts.append(chunk.ids.copy() if copy else chunk.ids)
                ts_parts.append(chunk.timestamps.copy() if copy else chunk.timestamps)
                data_parts.append(chunk.data.copy() if copy else chunk.data)

            ids = np.concatenate(ids_parts) if ids_parts else None
            timestamps = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.float64)
            data = np.concatenate(data_parts, axis=0) if data_parts else np.array([])
            collected[topic] = TopicView(ids, timestamps, data, metadata=pipeline.metadata).as_dict(copy=False)

        return collected

    def as_pipelines(self) -> dict[str, TopicPipeline]:
        return dict(self._pipelines)

    def _map_pipelines(self, fn: Callable[[TopicPipeline], TopicPipeline]) -> "DatasetQuery":
        return DatasetQuery({topic: fn(pipeline) for topic, pipeline in self._pipelines.items()})

    def _collect_parallel_topics(
        self,
        chunk_size: int,
        copy: bool,
        max_rows: int | None,
        max_bytes: int | None,
        allow_large: bool,
        max_workers: int,
        topic_workers: int,
    ) -> dict[str, dict]:
        collected = {}
        total_rows = 0
        total_bytes = 0
        items = list(self._pipelines.items())

        def collect_topic(item):
            topic, pipeline = item
            return topic, pipeline.collect(
                chunk_size=chunk_size,
                copy=copy,
                max_rows=max_rows,
                max_bytes=max_bytes,
                allow_large=allow_large,
                max_workers=max_workers,
            )

        with ThreadPoolExecutor(max_workers=topic_workers) as executor:
            futures = [executor.submit(collect_topic, item) for item in items]
            for future in futures:
                topic, result = future.result()
                rows = int(np.asarray(result["ts"]).shape[0])
                total_rows += rows
                total_bytes += _topic_result_nbytes(result)
                _check_collect_limits(total_rows, total_bytes, max_rows, max_bytes, allow_large)
                collected[topic] = result

        return collected


class SourcePipeline:
    """Streaming operation pipeline over `DataSources` messages."""

    def __init__(
        self,
        data_source,
        operations: Iterable[_PipelineOperation] = (),
        topics: Iterable[str] | None = None,
    ):
        self.data_source = data_source
        self._operations = tuple(operations)
        self.topics = tuple(_source_topics(data_source, topics))

    def select_topics(self, *topics: str | Iterable[str]) -> "SourcePipeline":
        names = _normalize_topic_selection(topics)
        missing = [topic for topic in names if self.topics and topic not in self.topics]
        if missing:
            raise ValueError(f"topics not found: {missing}")
        return self._with_operation("source_topics", frozenset(names), topics=names)

    def select_topic(self, topic: str) -> "SourcePipeline":
        return self.select_topics(topic)

    def map(self, fn: Callable, copy: bool = True) -> "SourcePipeline":
        return self._with_operation("source_map", fn, copy=copy)

    def filter(self, predicate: Callable, copy: bool = True) -> "SourcePipeline":
        return self._with_operation("source_filter", predicate, copy=copy)

    def time_range(self, start: float, end: float, inclusive: bool = True) -> "SourcePipeline":
        if start > end:
            raise ValueError("start must be less than or equal to end")
        return self._with_operation("source_time_range", float(start), float(end), inclusive=inclusive)

    def select_time_range(self, start: float, end: float, inclusive: bool = True) -> "SourcePipeline":
        return self.time_range(start, end, inclusive=inclusive)

    def index_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "SourcePipeline":
        if start is not None and start < 0:
            raise ValueError("negative source index ranges are not supported")
        if stop is not None and stop < 0:
            raise ValueError("negative source index ranges are not supported")
        if step is not None and step < 1:
            raise ValueError("step must be at least 1")
        return self._with_operation("source_index_range", start, stop, step)

    def select_indices(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "SourcePipeline":
        return self.index_range(start, stop, step)

    def iter_messages(
        self,
        copy: bool = True,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ) -> Iterable[dict]:
        index_counters = _checkpoint_operation_counters(checkpoint, len(self._operations), kind="source")
        resume_processed = _checkpoint_processed(checkpoint)
        processed = 0
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        progress_interval = _validated_progress_interval(progress_interval)
        last_progress: PipelineProgress | None = None
        for raw_message in _source_messages(self.data_source):
            processed += 1
            if processed <= resume_processed:
                continue
            _raise_if_cancelled(
                cancel_token,
                checkpoint,
                PipelineProgress(
                    processed=processed - 1,
                    emitted=emitted,
                    skipped=skipped,
                    checkpoint=_checkpoint_snapshot(checkpoint),
                ),
                operation_counters=index_counters,
            )
            message = _normalize_source_message(raw_message, copy=copy)
            keep = True

            for operation_index, operation in enumerate(self._operations):
                if operation.kind == "source_topics":
                    keep = message["topic"] in operation.args[0]
                elif operation.kind == "source_map":
                    mapped = _call_source_callable(
                        operation.args[0],
                        _copy_source_message(message) if operation.kwargs.get("copy", True) else message,
                    )
                    message = _mapped_source_message(message, mapped, copy=copy)
                elif operation.kind == "source_filter":
                    keep = bool(
                        _call_source_callable(
                            operation.args[0],
                            _copy_source_message(message) if operation.kwargs.get("copy", True) else message,
                        )
                    )
                elif operation.kind == "source_time_range":
                    start, end = operation.args
                    timestamp = float(message["timestamp"])
                    if operation.kwargs.get("inclusive", True):
                        keep = timestamp >= start and timestamp <= end
                    else:
                        keep = timestamp > start and timestamp < end
                elif operation.kind == "source_index_range":
                    topic = message["topic"]
                    counters = index_counters[operation_index]
                    current_index = counters.get(topic, 0)
                    counters[topic] = current_index + 1
                    keep = _slice_contains(current_index, *operation.args)
                else:
                    raise ValueError(f"unsupported source pipeline operation: {operation.kind}")

                if not keep:
                    break

            if keep:
                emitted += 1
            else:
                skipped += 1

            last_progress = PipelineProgress(
                processed=processed,
                emitted=emitted,
                skipped=skipped,
                topic=message["topic"],
                message_id=message.get("name"),
                timestamp=float(message["timestamp"]),
                checkpoint=_checkpoint_snapshot(checkpoint),
            )
            _update_checkpoint(checkpoint, last_progress, operation_counters=index_counters)
            _notify_progress(progress_callback, last_progress, progress_interval)

            if keep:
                yield _copy_source_message(message) if copy else message

        done_progress = PipelineProgress(
            processed=max(processed, resume_processed),
            emitted=emitted,
            skipped=skipped,
            topic=None if last_progress is None else last_progress.topic,
            message_id=None if last_progress is None else last_progress.message_id,
            timestamp=None if last_progress is None else last_progress.timestamp,
            done=True,
            checkpoint=_checkpoint_snapshot(checkpoint),
        )
        _update_checkpoint(checkpoint, done_progress, operation_counters=index_counters)
        _notify_progress(progress_callback, done_progress, progress_interval, force=True)

    def to_buffer(
        self,
        buffer_depth: int = 1,
        data_uri: str = "/tmp/tiledb/my_group/",
        use_db: bool = False,
        axis: str | None = None,
        buffer=None,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ):
        """Stream processed messages into a `DataBuffer`."""

        if buffer is not None:
            return self._append_to_buffer(
                buffer,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
            )

        if not self.topics:
            raise ValueError("source pipeline has no topics")
        if use_db:
            self._validate_countable(self.topics)

        from ..buffer import DataBuffer

        selected_axis = axis if axis is not None else self.topics[0]
        pipeline_source = _PipelineSource(
            self,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        )
        result = DataBuffer(
            data_source=pipeline_source,
            buffer_depth=buffer_depth,
            data_uri=data_uri,
            topics=list(self.topics),
            axis=selected_axis,
            use_db=use_db,
            preload=0,
        )
        try:
            result.load_data_db(selected_axis)
        except PipelineCancelled:
            result.close(closed=False)
            raise
        return result

    def write_to_buffer(self, buffer=None, **kwargs):
        """Alias for `to_buffer()`."""

        return self.to_buffer(buffer=buffer, **kwargs)

    def persist_to_tiledb(
        self,
        data_uri: str,
        buffer_depth: int = 1,
        axis: str | None = None,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ):
        """Stream processed messages into a TileDB-backed `DataBuffer`."""

        return self.to_buffer(
            buffer_depth=buffer_depth,
            data_uri=data_uri,
            use_db=True,
            axis=axis,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        )

    def _with_operation(self, kind: str, *args, topics: Iterable[str] | None = None, **kwargs) -> "SourcePipeline":
        return SourcePipeline(
            self.data_source,
            operations=(*self._operations, _PipelineOperation(kind, args, kwargs)),
            topics=self.topics if topics is None else topics,
        )

    def _topic_capacity(self, topic: str) -> int:
        get_count = getattr(self.data_source, "get_count", None)
        if not callable(get_count):
            raise ValueError("TileDB source pipeline output requires data_source.get_count(topic)")
        return int(get_count(topic))

    def _validate_countable(self, topics: Iterable[str]) -> None:
        for topic in topics:
            self._topic_capacity(topic)

    def _append_to_buffer(
        self,
        buffer,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ):
        try:
            for message in self.iter_messages(
                copy=True,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
            ):
                topic = message["topic"]
                if topic not in buffer.topics:
                    buffer.topics.append(topic)
                if hasattr(buffer.buffer_impl, "topics") and topic not in buffer.buffer_impl.topics:
                    buffer.buffer_impl.topics.append(topic)
                if getattr(buffer, "use_db", False):
                    self._prepare_tiledb_append(buffer, message)
                buffer.append_buffer(message)
        except PipelineCancelled:
            if getattr(buffer, "use_db", False):
                buffer.close(closed=False)
            raise

        if getattr(buffer, "use_db", False):
            for topic in buffer.topics:
                buffer.buffer_impl.close_topic(topic, closed=True)
        return buffer

    def _prepare_tiledb_append(self, buffer, message: Mapping[str, Any]) -> None:
        impl = buffer.buffer_impl
        topic = message["topic"]
        if topic in impl.counters:
            return
        impl.counters[topic] = 0
        impl.msg_len[topic] = max(self._topic_capacity(topic), 1)
        impl._init_tdb(message)


def source_pipeline(data_source, topics: Iterable[str] | None = None) -> SourcePipeline:
    """Return a streaming operation pipeline over source messages."""

    return SourcePipeline(data_source, topics=topics)


class _PipelineSource:
    def __init__(
        self,
        pipeline: SourcePipeline,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ):
        self.pipeline = pipeline
        self.progress_callback = progress_callback
        self.cancel_token = cancel_token
        self.checkpoint = checkpoint
        self.progress_interval = progress_interval

    def get_topics(self):
        return list(self.pipeline.topics)

    def get_count(self, topic):
        return max(self.pipeline._topic_capacity(topic), 1)

    def get_message(self):
        yield from self.pipeline.iter_messages(
            copy=True,
            progress_callback=self.progress_callback,
            cancel_token=self.cancel_token,
            checkpoint=self.checkpoint,
            progress_interval=self.progress_interval,
        )


def dataset_query(topics: Mapping[str, TopicPipeline | dict | np.ndarray | TopicView]) -> DatasetQuery:
    """Return a lazy dataset-level query over one or more topics."""

    return DatasetQuery(topics)


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
        elif operation.kind == "frame_id":
            targets = operation.args[0]
            metadata_frame_id = _decode_text(selected.metadata.frame_id)
            if metadata_frame_id is not None:
                if metadata_frame_id not in targets:
                    selected = selected._select(np.zeros(len(selected), dtype=bool), copy=False)
            else:
                selected = selected.filter(
                    lambda data, ts, name, targets=targets: _row_frame_id(data, name) in targets,
                    copy=False,
                )
        elif operation.kind == "spatial_bounds":
            min_bound, max_bound = operation.args
            selected = selected.filter(
                lambda data, ts, name, min_bound=min_bound, max_bound=max_bound, operation=operation: (
                    _spatial_value_in_bounds(
                        data,
                        min_bound=min_bound,
                        max_bound=max_bound,
                        columns=operation.kwargs["columns"],
                    )
                ),
                copy=False,
            )
        else:
            raise ValueError(f"unsupported pushdown operation: {operation.kind}")
    return selected


def _process_pipeline_chunk(
    chunk: TopicView,
    operations: tuple[_PipelineOperation, ...],
    metadata: TopicMetadata,
    copy: bool,
) -> _ProcessedChunk:
    rows = []
    processed = 0
    emitted = 0
    skipped = 0
    last_id = None
    last_timestamp = None
    metadata_frame_id = _decode_text(metadata.frame_id)

    for _, timestamp, value, message_id in chunk._iter_rows():
        processed += 1
        current_value = value.copy() if copy else value
        current_timestamp = float(timestamp)
        current_id = message_id
        keep = True
        last_id = current_id
        last_timestamp = current_timestamp

        for operation in operations:
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
            elif operation.kind == "frame_id":
                targets = operation.args[0]
                if metadata_frame_id is not None:
                    keep = metadata_frame_id in targets
                else:
                    keep = _row_frame_id(current_value, current_id) in targets
            elif operation.kind == "spatial_bounds":
                min_bound, max_bound = operation.args
                keep = _spatial_value_in_bounds(
                    current_value,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    columns=operation.kwargs["columns"],
                )
            elif operation.kind == "index_range":
                raise ValueError("parallel topic execution does not support non-leading index_range operations")
            else:
                raise ValueError(f"unsupported pipeline operation: {operation.kind}")

            if not keep:
                break

        if keep:
            emitted += 1
            rows.append((current_id, current_timestamp, current_value))
        else:
            skipped += 1

    return _ProcessedChunk(
        rows=tuple(rows),
        processed=processed,
        emitted=emitted,
        skipped=skipped,
        message_id=last_id,
        timestamp=last_timestamp,
    )


def _validate_parallel_operations(operations: Iterable[_PipelineOperation]) -> None:
    if any(operation.kind == "index_range" for operation in operations):
        raise ValueError(
            "parallel topic execution requires index_range operations to appear before map/filter operations "
            "so they can be pushed down before chunks are processed"
        )


def _source_topics(data_source, topics: Iterable[str] | None = None) -> list[str]:
    if topics is not None:
        return [str(topic) for topic in topics]
    get_topics = getattr(data_source, "get_topics", None)
    if callable(get_topics):
        return [str(topic) for topic in get_topics()]
    return []


def _source_messages(data_source):
    get_message = getattr(data_source, "get_message", None)
    if callable(get_message):
        yield from get_message()
        return
    if callable(data_source):
        yield from data_source()
        return
    raise TypeError("data_source must expose get_message() or be a callable generator factory")


def _normalize_source_message(message: Mapping[str, Any], copy: bool = True) -> dict:
    if not isinstance(message, Mapping):
        raise TypeError("source messages must be mappings")
    if "topic" not in message:
        raise ValueError("source message missing 'topic'")
    if "data" not in message:
        raise ValueError("source message missing 'data'")

    result = dict(message)
    result["topic"] = str(result["topic"])
    if "timestamp" not in result:
        if "ts" not in result:
            raise ValueError("source message missing 'timestamp'")
        result["timestamp"] = result["ts"]
    if "name" not in result:
        result["name"] = result.get("id", result["topic"])

    result["timestamp"] = float(result["timestamp"])
    data = np.asarray(result["data"])
    result["data"] = data.copy() if copy else data
    return result


def _copy_source_message(message: Mapping[str, Any]) -> dict:
    result = dict(message)
    data = result.get("data")
    if isinstance(data, np.ndarray):
        result["data"] = data.copy()
    return result


def _call_source_callable(fn: Callable, message: Mapping[str, Any]):
    try:
        return fn(message)
    except TypeError:
        return _call_with_metadata(
            fn,
            message["data"],
            float(message["timestamp"]),
            message.get("name"),
        )


def _mapped_source_message(previous: Mapping[str, Any], mapped, copy: bool) -> dict:
    if mapped is None:
        raise ValueError("source map functions must return a message mapping or replacement data")
    if isinstance(mapped, Mapping):
        return _normalize_source_message(mapped, copy=copy)

    result = dict(previous)
    data = np.asarray(mapped)
    result["data"] = data.copy() if copy else data
    return result


def _validated_progress_interval(progress_interval: int) -> int:
    progress_interval = int(progress_interval)
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1")
    return progress_interval


def _checkpoint_processed(checkpoint: Mapping[str, Any] | None) -> int:
    return 0 if checkpoint is None else int(checkpoint.get("processed", 0))


def _checkpoint_emitted(checkpoint: Mapping[str, Any] | None) -> int:
    return 0 if checkpoint is None else int(checkpoint.get("emitted", 0))


def _checkpoint_skipped(checkpoint: Mapping[str, Any] | None) -> int:
    return 0 if checkpoint is None else int(checkpoint.get("skipped", 0))


def _checkpoint_operation_counters(
    checkpoint: Mapping[str, Any] | None,
    count: int,
    kind: str,
):
    if checkpoint is None:
        saved = None
    else:
        saved = checkpoint.get("operation_counters")
    if kind == "source":
        counters = [dict() for _ in range(count)]
        if isinstance(saved, list):
            for index, values in enumerate(saved[:count]):
                if isinstance(values, Mapping):
                    counters[index] = {str(key): int(value) for key, value in values.items()}
        return counters

    counters = [0] * count
    if isinstance(saved, list):
        for index, value in enumerate(saved[:count]):
            try:
                counters[index] = int(value)
            except (TypeError, ValueError):
                counters[index] = 0
    return counters


def _checkpoint_snapshot(checkpoint: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    snapshot = dict(checkpoint)
    operation_counters = snapshot.get("operation_counters")
    if isinstance(operation_counters, list):
        snapshot["operation_counters"] = [
            dict(counter) if isinstance(counter, Mapping) else counter
            for counter in operation_counters
        ]
    return snapshot


def _operation_counters_snapshot(operation_counters):
    snapshot = []
    for counter in operation_counters:
        snapshot.append(dict(counter) if isinstance(counter, Mapping) else int(counter))
    return snapshot


def _update_checkpoint(
    checkpoint: dict[str, Any] | None,
    progress: PipelineProgress,
    operation_counters=None,
) -> None:
    if checkpoint is None:
        return
    checkpoint.update({
        "processed": int(progress.processed),
        "emitted": int(progress.emitted),
        "skipped": int(progress.skipped),
        "topic": progress.topic,
        "message_id": progress.message_id,
        "timestamp": progress.timestamp,
        "done": bool(progress.done),
        "cancelled": bool(progress.cancelled),
    })
    if operation_counters is not None:
        checkpoint["operation_counters"] = _operation_counters_snapshot(operation_counters)


def _notify_progress(
    progress_callback: Callable[[PipelineProgress], Any] | None,
    progress: PipelineProgress,
    progress_interval: int,
    force: bool = False,
) -> None:
    if progress_callback is None:
        return
    if force or progress.done or progress.cancelled or progress.processed % progress_interval == 0:
        progress_callback(progress)


def _cancel_requested(cancel_token: Any | None) -> bool:
    if cancel_token is None:
        return False
    if isinstance(cancel_token, CancellationToken):
        return cancel_token.cancelled

    cancelled = getattr(cancel_token, "cancelled", None)
    if callable(cancelled):
        return bool(cancelled())
    if cancelled is not None:
        return bool(cancelled)

    is_cancelled = getattr(cancel_token, "is_cancelled", None)
    if callable(is_cancelled):
        return bool(is_cancelled())

    if callable(cancel_token):
        return bool(cancel_token())
    return bool(cancel_token)


def _raise_if_cancelled(
    cancel_token: Any | None,
    checkpoint: dict[str, Any] | None,
    progress: PipelineProgress,
    operation_counters=None,
) -> None:
    if not _cancel_requested(cancel_token):
        return
    cancelled_progress = PipelineProgress(
        processed=progress.processed,
        emitted=progress.emitted,
        skipped=progress.skipped,
        topic=progress.topic,
        message_id=progress.message_id,
        timestamp=progress.timestamp,
        done=False,
        cancelled=True,
        checkpoint=_checkpoint_snapshot(checkpoint),
    )
    _update_checkpoint(checkpoint, cancelled_progress, operation_counters=operation_counters)
    raise PipelineCancelled("pipeline execution cancelled")


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


def _decode_text(value: Any) -> str | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    return None


def _normalize_topic_selection(topics: tuple[str | Iterable[str], ...]) -> tuple[str, ...]:
    if len(topics) == 1 and not isinstance(topics[0], (str, bytes)):
        topics = tuple(topics[0])
    if not topics:
        raise ValueError("at least one topic must be selected")
    return tuple(str(topic) for topic in topics)


def _normalize_text_selection(values: tuple[str | Iterable[str], ...], name: str) -> frozenset[str]:
    if len(values) == 1 and not isinstance(values[0], (str, bytes)):
        values = tuple(values[0])
    normalized = frozenset(decoded for value in values if (decoded := _decode_text(value)) is not None)
    if not normalized:
        raise ValueError(f"at least one {name} must be selected")
    return normalized


def _field_value(value: Any, names: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]

    for name in names:
        if hasattr(value, name):
            candidate = getattr(value, name)
            if _decode_text(candidate) is not None or not callable(candidate):
                return candidate

    array = np.asarray(value)
    if array.dtype.fields:
        for name in names:
            if name in array.dtype.fields:
                field = array[name]
                return field.item() if isinstance(field, np.ndarray) and field.ndim == 0 else field

    if array.dtype == object and array.ndim == 0:
        item = array.item()
        if item is not value:
            return _field_value(item, names)

    return None


def _row_frame_id(data: Any, message_id: Any = None) -> str | None:
    frame_id = _decode_text(_field_value(data, ("frame_id", "frame", "frameid")))
    if frame_id is not None:
        return frame_id
    return _decode_text(_field_value(message_id, ("frame_id", "frame", "frameid")))


def _geographic_value_in_bounds(
    value: Any,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    columns: tuple[int, int],
) -> bool:
    coordinates = _coordinate_array(
        value,
        field_groups=(("lat", "latitude"), ("lon", "longitude")),
        columns=columns,
    )
    return _coordinates_in_bounds(
        coordinates,
        np.array([min_lat, min_lon], dtype=np.float64),
        np.array([max_lat, max_lon], dtype=np.float64),
    )


def _spatial_value_in_bounds(
    value: Any,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    columns: tuple[int, ...],
) -> bool:
    field_groups = tuple((name,) for name in ("x", "y", "z", "w")[: min_bound.size])
    coordinates = _coordinate_array(value, field_groups=field_groups, columns=columns)
    return _coordinates_in_bounds(coordinates, min_bound, max_bound)


def _normalize_bounds(min_bound, max_bound, columns: tuple[int, ...] | None = None):
    min_array = np.asarray(min_bound, dtype=np.float64)
    max_array = np.asarray(max_bound, dtype=np.float64)
    if min_array.ndim != 1 or max_array.ndim != 1 or min_array.shape != max_array.shape:
        raise ValueError("min_bound and max_bound must be one-dimensional arrays with the same shape")
    if np.any(min_array > max_array):
        raise ValueError("min_bound must be less than or equal to max_bound")
    if columns is None:
        columns = tuple(range(min_array.size))
    if len(columns) != min_array.size:
        raise ValueError("columns length must match bound dimensionality")
    if any(column < 0 for column in columns):
        raise ValueError("columns must be non-negative")
    return min_array, max_array, tuple(int(column) for column in columns)


def _coordinate_array(
    value: Any,
    field_groups: tuple[tuple[str, ...], ...],
    columns: tuple[int, ...],
) -> np.ndarray | None:
    fields = [_field_value(value, names) for names in field_groups]
    if all(field is not None for field in fields):
        try:
            return np.stack([np.asarray(field, dtype=np.float64) for field in fields], axis=-1)
        except (TypeError, ValueError):
            return None

    array = np.asarray(value)
    if array.dtype == object and array.ndim == 0:
        item = array.item()
        if item is not value:
            return _coordinate_array(item, field_groups=field_groups, columns=columns)

    if array.size == 0 or array.ndim == 0:
        return None
    if max(columns) >= array.shape[-1]:
        return None

    try:
        return np.take(array.astype(np.float64, copy=False), columns, axis=-1)
    except (TypeError, ValueError):
        return None


def _coordinates_in_bounds(coordinates: np.ndarray | None, min_bound: np.ndarray, max_bound: np.ndarray) -> bool:
    if coordinates is None:
        return False
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.size == 0 or coords.shape[-1] != min_bound.size:
        return False
    mask = np.logical_and(coords >= min_bound, coords <= max_bound).all(axis=-1)
    return bool(np.any(mask))


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


def _validated_max_workers(max_workers: int | None) -> int:
    if max_workers is None:
        return 1
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    return max_workers


def _topic_view_nbytes(view: TopicView, include_data: bool = True) -> int:
    ids_nbytes = 0 if view.ids is None else view.ids.nbytes
    data_nbytes = view.data.nbytes if include_data else 0
    return int(data_nbytes + view.timestamps.nbytes + ids_nbytes)


def _topic_result_nbytes(topic: Mapping[str, Any]) -> int:
    total = np.asarray(topic["ts"]).nbytes + np.asarray(topic["data"]).nbytes
    ids = topic.get("id", topic.get("name"))
    if ids is not None:
        total += np.asarray(ids).nbytes
    return int(total)


def _check_collect_limits(
    rows: int,
    nbytes: int,
    max_rows: int | None,
    max_bytes: int | None,
    allow_large: bool,
) -> None:
    if allow_large:
        return
    if max_rows is not None and rows > max_rows:
        raise MemoryError(
            f"collect() would materialize {rows} rows, which exceeds max_rows={max_rows}; "
            "tighten the query, iterate chunks, or pass a larger max_rows"
        )
    if max_bytes is not None and nbytes > max_bytes:
        raise MemoryError(
            f"collect() would materialize about {nbytes} bytes, which exceeds max_bytes={max_bytes}; "
            "tighten the query, iterate chunks, pass out=, or set allow_large=True"
        )


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


def align_topic(
    reference_topic: dict | np.ndarray | TopicView | None,
    target_topic: dict | np.ndarray | TopicView,
    mode: str = "nearest",
    tolerance: float | None = None,
    rate_hz: float | None = None,
    period: float | None = None,
    start: float | None = None,
    end: float | None = None,
    interpolation: str = "linear",
    seconds: float | None = None,
    size: int | None = None,
    lookback: float | None = None,
    lookahead: float = 0.0,
    copy: bool = True,
) -> dict:
    """Align or resample topic data using a named timestamp mode."""

    normalized_mode = mode.lower().replace("-", "_")
    if normalized_mode == "exact":
        if reference_topic is None:
            raise ValueError("reference_topic is required for exact alignment")
        return align_exact(reference_topic, target_topic)
    if normalized_mode in {"nearest", "nearest_neighbor"}:
        if reference_topic is None:
            raise ValueError("reference_topic is required for nearest alignment")
        return align_nearest(reference_topic, target_topic, tolerance=tolerance)
    if normalized_mode in {"bounded", "bounded_tolerance", "tolerance"}:
        if reference_topic is None:
            raise ValueError("reference_topic is required for bounded alignment")
        return align_bounded(reference_topic, target_topic, tolerance=tolerance)
    if normalized_mode in {"fixed_rate", "resample", "fixed_rate_resampling"}:
        return resample_topic(
            target_topic,
            rate_hz=rate_hz,
            period=period,
            start=start,
            end=end,
            method=interpolation,
            tolerance=tolerance,
        )
    if normalized_mode in {"rolling_window", "window", "rolling_window_join"}:
        if reference_topic is None:
            raise ValueError("reference_topic is required for rolling window joins")
        return rolling_window_join(
            reference_topic,
            target_topic,
            seconds=seconds,
            size=size,
            lookback=lookback,
            lookahead=lookahead,
            copy=copy,
        )
    raise ValueError(f"unsupported alignment mode: {mode}")


def align_exact(reference_topic: dict | np.ndarray | TopicView, target_topic: dict | np.ndarray | TopicView) -> dict:
    """Align target messages to exactly matching reference timestamps."""

    reference = topic_view(reference_topic)
    target = topic_view(target_topic)
    indices = _exact_alignment_indices(reference.timestamps, target.timestamps)
    return _aligned_topic_result(reference, target, indices, mode="exact")


def align_nearest(
    reference_topic: dict | np.ndarray | TopicView,
    target_topic: dict | np.ndarray | TopicView,
    tolerance: float | None = None,
) -> dict:
    """Align each reference timestamp to the nearest target message."""

    reference = topic_view(reference_topic)
    target = topic_view(target_topic)
    indices = _nearest_alignment_indices(reference.timestamps, target.timestamps, tolerance=tolerance)
    return _aligned_topic_result(reference, target, indices, mode="nearest")


def align_bounded(
    reference_topic: dict | np.ndarray | TopicView,
    target_topic: dict | np.ndarray | TopicView,
    tolerance: float | None,
) -> dict:
    """Align to the nearest target message, requiring a maximum time delta."""

    if tolerance is None:
        raise ValueError("tolerance is required for bounded alignment")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    reference = topic_view(reference_topic)
    target = topic_view(target_topic)
    indices = _nearest_alignment_indices(reference.timestamps, target.timestamps, tolerance=tolerance)
    return _aligned_topic_result(reference, target, indices, mode="bounded_tolerance")


def resample_topic(
    topic_data: dict | np.ndarray | TopicView,
    rate_hz: float | None = None,
    period: float | None = None,
    start: float | None = None,
    end: float | None = None,
    method: str = "linear",
    tolerance: float | None = None,
) -> dict:
    """Resample a topic onto a fixed-rate timestamp grid."""

    view = topic_view(topic_data)
    sample_ts = _fixed_rate_timestamps(view.timestamps, rate_hz=rate_hz, period=period, start=start, end=end)
    normalized_method = method.lower().replace("-", "_")

    if normalized_method in {"nearest", "nearest_neighbor"}:
        indices = _nearest_alignment_indices(sample_ts, view.timestamps, tolerance=tolerance)
        result = _aligned_topic_arrays(sample_ts, view, indices)
        result["mode"] = "fixed_rate_nearest"
        result["rate_hz"] = None if period is not None else rate_hz
        result["period"] = _resolve_period(rate_hz=rate_hz, period=period)
        return result

    if normalized_method != "linear":
        raise ValueError("method must be 'linear' or 'nearest'")

    if sample_ts.size == 0:
        data = np.empty((0,) + view.data.shape[1:], dtype=view.data.dtype)
        valid = np.zeros(0, dtype=bool)
    elif view.timestamps.size == 0:
        data = np.full((sample_ts.size,) + view.data.shape[1:], np.nan, dtype=np.float64)
        valid = np.zeros(sample_ts.size, dtype=bool)
    else:
        data = _interpolate_topic_data(view.timestamps, view.data, sample_ts)
        valid = (sample_ts >= view.timestamps[0]) & (sample_ts <= view.timestamps[-1])

    return {
        "mode": "fixed_rate",
        "ts": sample_ts,
        "data": data,
        "valid": valid,
        "rate_hz": None if period is not None else rate_hz,
        "period": _resolve_period(rate_hz=rate_hz, period=period),
        "topic": view.metadata.topic,
        "metadata": view.metadata,
    }


def rolling_window_join(
    reference_topic: dict | np.ndarray | TopicView,
    target_topic: dict | np.ndarray | TopicView,
    seconds: float | None = None,
    size: int | None = None,
    lookback: float | None = None,
    lookahead: float = 0.0,
    copy: bool = True,
) -> dict:
    """Join each reference timestamp with a trailing target-topic window."""

    if seconds is None and lookback is None and size is None:
        raise ValueError("seconds, lookback, or size must be provided")
    if seconds is not None and seconds < 0:
        raise ValueError("seconds must be non-negative")
    if lookback is not None and lookback < 0:
        raise ValueError("lookback must be non-negative")
    if lookahead < 0:
        raise ValueError("lookahead must be non-negative")
    if size is not None and size < 1:
        raise ValueError("size must be at least 1")

    reference = topic_view(reference_topic)
    target = topic_view(target_topic)
    window_lookback = seconds if lookback is None else lookback
    counts = []
    windows = []

    for timestamp in reference.timestamps:
        if window_lookback is None:
            left = 0
        else:
            left = int(np.searchsorted(target.timestamps, timestamp - window_lookback, side="left"))
        right = int(np.searchsorted(target.timestamps, timestamp + lookahead, side="right"))
        if size is not None:
            left = max(left, right - size)

        ids = None if target.ids is None else target.ids[left:right]
        ts = target.timestamps[left:right]
        data = target.data[left:right]
        window = TopicView(ids, ts, data, metadata=target.metadata, copy=copy)
        windows.append(window)
        counts.append(len(window))

    result = {
        "mode": "rolling_window",
        "reference_ts": reference.timestamps.copy(),
        "windows": windows,
        "counts": np.asarray(counts, dtype=np.int64),
        "valid": np.asarray(counts, dtype=np.int64) > 0,
    }
    if reference.ids is not None:
        result["reference_id"] = reference.ids.copy()
    return result


def _exact_alignment_indices(reference_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if target_ts.size == 0:
        return np.full(reference_ts.shape, -1, dtype=np.int64)

    positions = np.searchsorted(target_ts, reference_ts, side="left")
    valid = (positions < target_ts.size) & (target_ts[np.minimum(positions, target_ts.size - 1)] == reference_ts)
    return np.where(valid, positions, -1).astype(np.int64)


def _nearest_alignment_indices(
    reference_ts: np.ndarray,
    target_ts: np.ndarray,
    tolerance: float | None = None,
) -> np.ndarray:
    if tolerance is not None and tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return np.array([
        -1 if (idx := nearest_time_index(target_ts, float(timestamp), tolerance)) is None else idx
        for timestamp in reference_ts
    ], dtype=np.int64)


def _aligned_topic_result(reference: TopicView, target: TopicView, indices: np.ndarray, mode: str) -> dict:
    result = _aligned_topic_arrays(reference.timestamps, target, indices)
    result["mode"] = mode
    result["reference_ts"] = reference.timestamps.copy()
    if reference.ids is not None:
        result["reference_id"] = reference.ids.copy()
    return result


def _aligned_topic_arrays(reference_ts: np.ndarray, target: TopicView, indices: np.ndarray) -> dict:
    valid = indices >= 0
    safe_indices = np.where(valid, indices, 0)
    aligned_data = _empty_aligned_data(target.data, reference_ts.shape[0])
    if valid.any():
        aligned_data[valid] = target.data[indices[valid]]
    target_ts = np.full(reference_ts.shape, np.nan, dtype=np.float64)
    if target.timestamps.size:
        target_ts = np.where(valid, target.timestamps[safe_indices], np.nan)

    aligned = {
        "ts": reference_ts.copy(),
        "target_ts": target_ts,
        "target_index": indices,
        "valid": valid,
        "data": aligned_data,
        "metadata": target.metadata,
    }
    if target.metadata.topic is not None:
        aligned["topic"] = target.metadata.topic
    if target.ids is not None:
        aligned["id"] = np.asarray([target.ids[i] if ok else None for i, ok in zip(safe_indices, valid)], dtype=object)
        aligned["name"] = aligned["id"]
    return aligned


def _empty_aligned_data(data: np.ndarray, count: int) -> np.ndarray:
    shape = (count,) + data.shape[1:]
    if np.issubdtype(data.dtype, np.number):
        return np.full(shape, np.nan, dtype=np.result_type(data.dtype, np.float64))

    aligned = np.empty(shape, dtype=object)
    aligned[...] = None
    return aligned


def _resolve_period(rate_hz: float | None = None, period: float | None = None) -> float:
    if period is not None:
        period = float(period)
        if period <= 0:
            raise ValueError("period must be positive")
        return period
    if rate_hz is None:
        raise ValueError("rate_hz or period must be provided")
    rate_hz = float(rate_hz)
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    return 1.0 / rate_hz


def _fixed_rate_timestamps(
    source_ts: np.ndarray,
    rate_hz: float | None = None,
    period: float | None = None,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    sample_period = _resolve_period(rate_hz=rate_hz, period=period)
    if source_ts.size == 0 and (start is None or end is None):
        return np.array([], dtype=np.float64)

    sample_start = float(source_ts[0] if start is None else start)
    sample_end = float(source_ts[-1] if end is None else end)
    if sample_start > sample_end:
        raise ValueError("start must be less than or equal to end")

    count = int(np.floor((sample_end - sample_start) / sample_period + 1.0e-12)) + 1
    return sample_start + np.arange(count, dtype=np.float64) * sample_period


def _interpolate_topic_data(timestamps: np.ndarray, data: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    if target_timestamps.size == 0:
        return np.empty((0,) + data.shape[1:], dtype=np.result_type(data.dtype, np.float64))
    if timestamps.size == 0:
        return np.full((target_timestamps.size,) + data.shape[1:], np.nan, dtype=np.float64)
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("linear fixed-rate resampling requires numeric topic data")

    flat = np.asarray(data, dtype=np.float64).reshape((data.shape[0], -1))
    interpolated = np.column_stack([
        np.interp(target_timestamps, timestamps, flat[:, dim])
        for dim in range(flat.shape[1])
    ])
    return interpolated.reshape((target_timestamps.size,) + data.shape[1:])
