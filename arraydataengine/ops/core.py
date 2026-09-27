from __future__ import annotations

import functools
import inspect
import types
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import copy as _shallow_copy
from dataclasses import dataclass, replace
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
    """Raised when a pipeline cancellation token is set.

    `partial` holds the rows gathered before cancellation when the raising
    call materializes results (`TopicPipeline.collect()` sets it to the same
    dict shape `collect()` returns); it is None otherwise.
    """

    partial: dict | None = None


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
        frame_ids: np.ndarray | None = None,
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

        self.frame_ids = _normalize_ids(frame_ids, ts.size)
        if copy and self.frame_ids is not None:
            self.frame_ids = self.frame_ids.copy()
        self.ids = normalized_ids
        self.timestamps = ts
        self.data = values
        self._metadata = TopicMetadata.from_arrays(
            self.timestamps,
            self.data,
            self.ids,
            metadata=metadata,
            topic=topic,
            source_uri=source_uri,
            frame_id=frame_id,
        )
        # The topic-level frame id implied by per-row frame ids is resolved
        # lazily: scanning every row on construction made each _select() (and
        # so every window/chunk) O(n) in Python.
        self._frame_metadata_pending = self.frame_ids is not None

    @property
    def metadata(self) -> TopicMetadata:
        if self._frame_metadata_pending:
            frames = {_decode_text(frame) for frame in self.frame_ids}
            self._metadata = replace(
                self._metadata, frame_id=next(iter(frames)) if len(frames) == 1 else None
            )
            self._frame_metadata_pending = False
        return self._metadata

    @metadata.setter
    def metadata(self, value: TopicMetadata) -> None:
        self._metadata = value
        self._frame_metadata_pending = False

    def _frame_id_at(self, index: int) -> str | None:
        if self.frame_ids is not None:
            return _decode_text(self.frame_ids[index])
        return _decode_text(self.metadata.frame_id) or _row_frame_id(
            self.data[index], None if self.ids is None else self.ids[index]
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
        if self.frame_ids is not None:
            result["frame_ids"] = self.frame_ids.copy() if copy else self.frame_ids
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
        """Apply `fn` to every row and return a new view of the results.

        `fn` receives `(data)`, `(data, ts)` or `(data, ts, id)` depending on
        how many positional parameters it requires (optional parameters are
        left at their defaults). Rows whose results differ in shape are
        returned as a 1-D object array.
        """

        chunk_size = _validated_chunk_size(chunk_size) if chunk_size is not None else None
        ids = None if self.ids is None else self.ids.copy()
        ts = self.timestamps.copy()
        call = _metadata_caller(fn)

        if out is not None:
            mapped = np.asarray(out)
            if mapped.shape[:1] != self.data.shape[:1]:
                raise ValueError("out must have the same leading dimension as topic data")
            for index, timestamp, value, message_id in self._iter_rows(chunk_size):
                mapped[index] = call(
                    _copy_value(value) if copy else value,
                    float(timestamp),
                    message_id,
                )
            return TopicView(ids, ts, mapped, metadata=self.metadata, frame_ids=self.frame_ids)

        mapped_values = [
            call(_copy_value(value) if copy else value, float(timestamp), message_id)
            for _, timestamp, value, message_id in self._iter_rows(chunk_size)
        ]
        return TopicView(ids, ts, _stack_values(mapped_values), metadata=self.metadata, frame_ids=self.frame_ids)

    def filter(self, predicate: Callable, copy: bool = True, chunk_size: int | None = None) -> "TopicView":
        """Keep rows where `predicate` is truthy (called like `map`'s `fn`)."""

        chunk_size = _validated_chunk_size(chunk_size) if chunk_size is not None else None
        call = _metadata_caller(predicate)
        mask = np.zeros(len(self), dtype=bool)
        for index, timestamp, value, message_id in self._iter_rows(chunk_size):
            mask[index] = bool(
                call(
                    _copy_value(value) if copy else value,
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
        """Fold rows with `fn(acc, data)`, `fn(acc, data, ts)` or
        `fn(acc, data, ts, id)`, chosen by its required positional
        parameters."""

        chunk_size = _validated_chunk_size(chunk_size) if chunk_size is not None else None
        call = _reduce_caller(fn)
        iterator = self._iter_rows(chunk_size)
        if initial is None:
            try:
                _, _, value, _ = next(iterator)
            except StopIteration as exc:
                raise ValueError("cannot reduce an empty topic without an initial value") from exc
            acc = _copy_value(value) if copy else value
        else:
            acc = initial

        for _, timestamp, value, message_id in iterator:
            acc = call(acc, _copy_value(value) if copy else value, float(timestamp), message_id)
        return acc

    def window(
        self,
        size: int | None = None,
        seconds: float | None = None,
        copy: bool = True,
    ) -> Iterable["TopicView"]:
        """Yield the trailing window ending at each row, in row order.

        `seconds` drops leading rows older than `ts[row] - seconds`, exactly
        like `TopicPipeline.window()`. Windows are runs of consecutive rows,
        so with non-monotonic timestamps an old row is only dropped once
        every row before it has been dropped; sort the topic by time first
        when strict time windows are needed.
        """

        if size is None and seconds is None:
            raise ValueError("size or seconds must be provided")
        if size is not None and size < 1:
            raise ValueError("size must be at least 1")
        if seconds is not None and seconds < 0:
            raise ValueError("seconds must be non-negative")

        monotonic = seconds is None or _is_non_decreasing(self.timestamps)
        start_index = 0
        for end_index in range(len(self)):
            if seconds is not None:
                cutoff = self.timestamps[end_index] - seconds
                if monotonic:
                    start_index = max(
                        start_index,
                        int(np.searchsorted(self.timestamps, cutoff, side="left")),
                    )
                else:
                    # searchsorted is meaningless on unsorted timestamps.
                    while start_index < end_index and self.timestamps[start_index] < cutoff:
                        start_index += 1
            if size is not None:
                start_index = max(start_index, end_index - size + 1)
            yield self._select(slice(start_index, end_index + 1), copy=copy)

    def _select(self, selection: slice | np.ndarray, copy: bool) -> "TopicView":
        ids = None if self.ids is None else self.ids[selection]
        ts = self.timestamps[selection]
        data = self.data[selection]
        # Mask/index selections already produce new arrays.
        copy = copy and isinstance(selection, slice)
        return TopicView(ids, ts, data, metadata=self.metadata, copy=copy,
                         frame_ids=None if self.frame_ids is None else self.frame_ids[selection])

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
    rows: tuple[tuple[Any, float, np.ndarray, str | None], ...]
    processed: int
    emitted: int
    skipped: int
    message_id: Any | None = None
    timestamp: float | None = None
    # Zero-based position of each emitted row within its source chunk, so
    # checkpoints can advance per delivered row.
    offsets: tuple[int, ...] = ()


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
        base = _coerce_metadata(metadata)
        if base is None:
            self.metadata = TopicMetadata(topic=topic, source_uri=source_uri, frame_id=frame_id)
        else:
            # Merge explicit kwargs over the base metadata like TopicView does
            # instead of silently discarding them.
            self.metadata = replace(
                base,
                topic=base.topic if topic is None else topic,
                source_uri=base.source_uri if source_uri is None else source_uri,
                frame_id=base.frame_id if frame_id is None else frame_id,
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
        """Yield processed rows as dicts.

        A `checkpoint` always covers exactly the rows yielded so far, so
        stopping early (break, exception, cancellation) and resuming with the
        same checkpoint neither repeats nor skips rows.
        """

        if _validated_max_workers(max_workers) > 1:
            rows = self._iter_processed_rows_parallel(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
                max_workers=max_workers,
            )
        else:
            rows = self._iter_processed_rows(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
            )

        for message_id, timestamp, value, frame_id in rows:
            yield {
                "id": message_id,
                "name": message_id,
                "ts": timestamp,
                "data": value,
                "frame_id": frame_id,
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
        """Yield processed rows as `TopicView` chunks of `chunk_size` rows.

        Chunks never alias the source arrays. A `checkpoint` covers exactly
        the rows of the chunks yielded so far: rows still being gathered for
        the next chunk are rolled back out of it when an exception (or
        closing the iterator) interrupts iteration, and are flushed as a
        final short chunk before `PipelineCancelled` propagates, so resuming
        neither repeats nor skips rows.
        """

        yield from self._iter_output_chunks(
            chunk_size=chunk_size,
            copy=copy,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
            max_workers=max_workers,
            own_data=True,
        )

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
        """Fold processed rows with `fn(acc, data)`, `fn(acc, data, ts)` or
        `fn(acc, data, ts, id)`, chosen by its required positional
        parameters. Cannot resume from a checkpoint."""

        if _checkpoint_processed(checkpoint) > 0:
            raise ValueError(
                "reduce cannot resume from a checkpoint: the accumulator state "
                "is not persisted, so the result would be silently wrong. "
                "Restart with a fresh checkpoint or use collect()/iter_rows()."
            )
        call = _reduce_caller(fn)
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
                _, _, value, _ = next(iterator)
            except StopIteration as exc:
                raise ValueError("cannot reduce an empty topic without an initial value") from exc
            acc = value
        else:
            acc = initial

        # Rows are already copied by the iterator when copy=True.
        for message_id, timestamp, value, _ in iterator:
            acc = call(acc, value, float(timestamp), message_id)
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
        """Materialize the pipeline as a topic dict.

        The result never aliases the source arrays. Rows whose mapped data
        differ in shape are returned as a 1-D object array.

        Checkpoints: when `cancel_token` stops the run, the raised
        `PipelineCancelled` carries the rows collected so far in `.partial`
        and the checkpoint points just past them, so `partial` followed by a
        `collect()` resumed from the checkpoint equals an uninterrupted run.
        When any other exception escapes (including the `max_rows` /
        `max_bytes` MemoryError), no rows were returned, so the checkpoint is
        rolled back to its state when `collect()` started.
        """

        return self._collect(
            chunk_size=chunk_size,
            copy=copy,
            out=out,
            max_rows=max_rows,
            max_bytes=max_bytes,
            allow_large=allow_large,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
            max_workers=max_workers,
        )

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

    def _collect(
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
        rows_before: int = 0,
        bytes_before: int = 0,
    ) -> dict:
        # `rows_before` / `bytes_before` let DatasetQuery enforce one
        # collect budget across topics.
        chunk_size = _validated_chunk_size(chunk_size)
        ids_parts = []
        frame_parts = []
        chunk_lengths = []
        ts_parts = []
        data_parts = []
        output = None if out is None else np.asarray(out)
        offset = 0
        collected_bytes = int(bytes_before)
        start_checkpoint = _checkpoint_snapshot(checkpoint)

        def assemble() -> dict:
            # Concatenation always allocates, so the result never aliases the
            # source even though chunk arrays are gathered without copying.
            ids = _concat_chunk_ids(ids_parts, chunk_lengths)
            timestamps = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.float64)
            if output is None:
                if data_parts:
                    data = _concat_data_parts(data_parts)
                else:
                    template = self._empty_data_template()
                    data = np.array([]) if template is None else template
            else:
                data = output if offset == output.shape[0] else output[:offset]
            return TopicView(ids, timestamps, data, metadata=self.metadata,
                             frame_ids=_concat_chunk_ids(frame_parts, chunk_lengths)).as_dict(copy=False)

        try:
            # `copy` keeps callables off the source and copies objects held
            # by object arrays; chunks may alias the source otherwise.
            for chunk in self._iter_output_chunks(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
                max_workers=max_workers,
                own_data=False,
            ):
                offset += len(chunk)
                collected_bytes += _topic_view_nbytes(chunk, include_data=output is None)
                _check_collect_limits(rows_before + offset, collected_bytes, max_rows, max_bytes, allow_large)

                frame_parts.append(chunk.frame_ids)
                chunk_lengths.append(len(chunk))
                ids_parts.append(chunk.ids)
                ts_parts.append(chunk.timestamps)
                if output is None:
                    data_parts.append(chunk.data)
                else:
                    if offset > output.shape[0]:
                        raise ValueError("out is too small for collected pipeline output")
                    output[offset - len(chunk):offset] = chunk.data
        except PipelineCancelled as exc:
            # The checkpoint covers every row gathered so far; hand them to
            # the caller instead of discarding them.
            exc.partial = assemble()
            raise
        except BaseException:
            _restore_checkpoint(checkpoint, start_checkpoint)
            raise
        return assemble()

    def _iter_output_chunks(
        self,
        chunk_size: int,
        copy: bool,
        progress_callback: Callable[[PipelineProgress], Any] | None,
        cancel_token: Any | None,
        checkpoint: dict[str, Any] | None,
        progress_interval: int,
        max_workers: int | None,
        own_data: bool,
    ) -> Iterable[TopicView]:
        chunk_size = _validated_chunk_size(chunk_size)
        max_workers = _validated_max_workers(max_workers)
        _, operations = self._split_pushdown_operations()
        # Without row operations whole chunks can pass through. Progress
        # callbacks keep the row-by-row path so they still see every
        # `progress_interval` boundary exactly.
        if not operations and progress_callback is None:
            yield from self._iter_passthrough_chunks(
                chunk_size=chunk_size,
                copy=copy,
                own_data=own_data,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
            )
            return

        # Stacking rows into a chunk allocates new arrays, so rows need no
        # output copy here; `copy` still keeps callables off the source.
        if max_workers == 1:
            processed_rows = self._iter_processed_rows(
                chunk_size=chunk_size,
                copy=copy,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
                copy_output=False,
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
                copy_output=False,
            )

        ids: list[Any] = []
        timestamps: list[float] = []
        values: list[Any] = []
        frame_ids: list[str | None] = []
        # Checkpoint state matching the rows actually handed to the caller.
        delivered = _checkpoint_snapshot(checkpoint)
        try:
            for message_id, timestamp, value, frame_id in processed_rows:
                frame_ids.append(frame_id)
                ids.append(message_id)
                timestamps.append(timestamp)
                values.append(value)
                if len(values) == chunk_size:
                    chunk = self._make_chunk(ids, timestamps, values, copy=copy, frame_ids=frame_ids)
                    ids, timestamps, values, frame_ids = [], [], [], []
                    delivered = _checkpoint_snapshot(checkpoint)
                    yield chunk
            if values:
                chunk = self._make_chunk(ids, timestamps, values, copy=copy, frame_ids=frame_ids)
                ids, timestamps, values, frame_ids = [], [], [], []
                delivered = _checkpoint_snapshot(checkpoint)
                yield chunk
        except PipelineCancelled:
            # Rows buffered here are already recorded in the checkpoint;
            # flush them so cancel + resume does not silently lose them.
            if values:
                yield self._make_chunk(ids, timestamps, values, copy=copy, frame_ids=frame_ids)
            raise
        except BaseException:
            # Any other interruption: rows buffered here never reached the
            # caller, so take them back out of the checkpoint.
            processed_rows.close()
            _restore_checkpoint(checkpoint, delivered)
            raise

    def _iter_passthrough_chunks(
        self,
        chunk_size: int,
        copy: bool,
        own_data: bool,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ) -> Iterable[TopicView]:
        """Chunk fast path for pipelines without row-level operations.

        Source chunks (already narrowed by pushed-down selections) are
        re-chunked to `chunk_size` and passed through whole instead of being
        unpacked and restacked row by row. With `own_data`, yielded chunks
        never alias the source; otherwise they may (collect() concatenates).
        """

        progress_interval = _validated_progress_interval(progress_interval)
        resume_processed = _checkpoint_processed(checkpoint)
        processed = resume_processed
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        operation_counters: list = []
        topic = self.metadata.topic
        fallback_frame_id = _decode_text(self.metadata.frame_id)
        source_seen = 0
        pending: list[tuple] = []
        pending_rows = 0
        last_id = None
        last_timestamp = None

        def deliver(parts: list[tuple]) -> TopicView:
            nonlocal processed, emitted, last_id, last_timestamp
            if cancel_token is not None:
                _raise_if_cancelled(
                    cancel_token,
                    checkpoint,
                    PipelineProgress(processed=processed, emitted=emitted, skipped=skipped, topic=topic),
                    operation_counters=operation_counters,
                )
            chunk = self._assemble_passthrough_chunk(parts, copy=copy, own_data=own_data)
            previous = processed
            processed += len(chunk)
            emitted += len(chunk)
            last_id = None if chunk.ids is None else chunk.ids[-1]
            last_timestamp = float(chunk.timestamps[-1])
            if checkpoint is not None or progress_callback is not None:
                progress = _record_progress(
                    checkpoint,
                    PipelineProgress(
                        processed=processed,
                        emitted=emitted,
                        skipped=skipped,
                        topic=topic,
                        message_id=last_id,
                        timestamp=last_timestamp,
                    ),
                    operation_counters=operation_counters,
                )
                _notify_progress(progress_callback, progress, progress_interval, previous=previous)
            return chunk

        for chunk in self._source_chunks(chunk_size=chunk_size, copy=False):
            length = len(chunk)
            if length == 0:
                continue
            if source_seen + length <= resume_processed:
                # Whole chunk already covered by the checkpoint.
                source_seen += length
                continue
            if source_seen < resume_processed:
                chunk = chunk._select(slice(resume_processed - source_seen, None), copy=False)
            source_seen += length
            pending.append(_passthrough_parts(chunk, fallback_frame_id))
            pending_rows += len(chunk)
            while pending_rows >= chunk_size:
                parts, pending = _split_parts(pending, chunk_size)
                pending_rows -= chunk_size
                yield deliver(parts)

        if pending_rows:
            yield deliver(pending)

        done_progress = _record_progress(
            checkpoint,
            PipelineProgress(
                processed=max(processed, resume_processed),
                emitted=emitted,
                skipped=skipped,
                topic=topic,
                message_id=last_id,
                timestamp=last_timestamp,
                done=True,
            ),
            operation_counters=operation_counters,
        )
        _notify_progress(progress_callback, done_progress, progress_interval, force=True)

    def _assemble_passthrough_chunk(self, parts: list[tuple], copy: bool, own_data: bool) -> TopicView:
        lengths = [part[1].shape[0] for part in parts]
        if len(parts) == 1:
            ids, timestamps, data, frames = parts[0]
            if own_data:
                timestamps = timestamps.copy()
                data = data.copy()
        else:
            ids = _concat_chunk_ids([part[0] for part in parts], lengths)
            timestamps = np.concatenate([part[1] for part in parts])
            data = _concat_data_parts([part[2] for part in parts])
            frames = _concat_chunk_ids([part[3] for part in parts], lengths)

        # Match the row-by-row path: object ids, None when no row has one.
        if ids is not None:
            ids = ids.astype(object)
            if all(value is None for value in ids):
                ids = None
        if copy and data.dtype == object:
            data = _copy_object_elements(data)
        if frames is not None:
            has_missing = any(frame is None for frame in frames)
            if has_missing and self.metadata.frame_id is None and all(frame is None for frame in frames):
                frames = None
            elif not has_missing:
                # The row path stacks frame lists with np.asarray -> unicode.
                frames = frames.astype(str)
        return TopicView(ids, timestamps, data, metadata=self.metadata, frame_ids=frames)

    def _iter_processed_rows(
        self,
        chunk_size: int,
        copy: bool,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
        copy_output: bool | None = None,
    ):
        """Yield `(id, ts, data, frame_id)` for every row that survives the
        row-level operations.

        With `copy`, callables never see source memory. With `copy_output`
        (defaults to `copy`), yielded data never aliases the source either.
        """

        chunk_size = _validated_chunk_size(chunk_size)
        copy_output = copy if copy_output is None else copy_output
        _, operations = self._split_pushdown_operations()
        callers = _operation_callers(operations)
        index_counters = _checkpoint_operation_counters(checkpoint, len(operations), kind="topic")
        resume_processed = _checkpoint_processed(checkpoint)
        processed = 0
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        progress_interval = _validated_progress_interval(progress_interval)
        track_progress = checkpoint is not None or progress_callback is not None
        topic = self.metadata.topic
        fallback_frame_id = _decode_text(self.metadata.frame_id)
        last_id = None
        last_timestamp = None

        for chunk in self._source_chunks(chunk_size=chunk_size, copy=False):
            length = len(chunk)
            if processed + length <= resume_processed:
                # Skip whole chunks already covered by the checkpoint.
                processed += length
                continue
            first_row = max(0, resume_processed - processed)
            processed += first_row
            chunk_ids = chunk.ids
            chunk_timestamps = chunk.timestamps
            chunk_data = chunk.data

            for row_index in range(first_row, length):
                processed += 1
                if cancel_token is not None:
                    _raise_if_cancelled(
                        cancel_token,
                        checkpoint,
                        PipelineProgress(
                            processed=processed - 1,
                            emitted=emitted,
                            skipped=skipped,
                            topic=topic,
                        ),
                        operation_counters=index_counters,
                    )
                current_frame_id = chunk._frame_id_at(row_index)
                if chunk.frame_ids is None and current_frame_id is None:
                    current_frame_id = fallback_frame_id
                current_id = None if chunk_ids is None else chunk_ids[row_index]
                current_timestamp = float(chunk_timestamps[row_index])

                keep, current_value, aliases_source = _apply_row_operations(
                    operations,
                    callers,
                    chunk_data[row_index],
                    current_timestamp,
                    current_id,
                    current_frame_id,
                    index_counters=index_counters,
                    protect_source=copy,
                )
                if keep:
                    emitted += 1
                    if copy_output and aliases_source:
                        current_value = _copy_value(current_value)
                else:
                    skipped += 1

                last_id = current_id
                last_timestamp = current_timestamp
                if track_progress:
                    progress = _record_progress(
                        checkpoint,
                        PipelineProgress(
                            processed=processed,
                            emitted=emitted,
                            skipped=skipped,
                            topic=topic,
                            message_id=current_id,
                            timestamp=current_timestamp,
                        ),
                        operation_counters=index_counters,
                    )
                    _notify_progress(progress_callback, progress, progress_interval)

                if keep:
                    yield current_id, current_timestamp, current_value, current_frame_id

        done_progress = _record_progress(
            checkpoint,
            PipelineProgress(
                processed=max(processed, resume_processed),
                emitted=emitted,
                skipped=skipped,
                topic=topic,
                message_id=last_id,
                timestamp=last_timestamp,
                done=True,
            ),
            operation_counters=index_counters,
        )
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
        copy_output: bool | None = None,
    ):
        chunk_size = _validated_chunk_size(chunk_size)
        max_workers = _validated_max_workers(max_workers)
        copy_output = copy if copy_output is None else copy_output
        _, operations = self._split_pushdown_operations()
        _validate_parallel_operations(operations)
        callers = _operation_callers(operations)

        resume_processed = _checkpoint_processed(checkpoint)
        processed = resume_processed
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        progress_interval = _validated_progress_interval(progress_interval)
        operation_counters = _checkpoint_operation_counters(checkpoint, len(operations), kind="topic")
        topic = self.metadata.topic
        source_seen = 0
        next_sequence = 0
        next_yield = 0
        pending = {}
        last_id = None
        last_timestamp = None

        def check_cancelled():
            if cancel_token is not None:
                _raise_if_cancelled(
                    cancel_token,
                    checkpoint,
                    PipelineProgress(processed=processed, emitted=emitted, skipped=skipped, topic=topic),
                    operation_counters=operation_counters,
                )

        def submit_ready(executor, source_iter):
            nonlocal next_sequence, source_seen
            while len(pending) < max_workers * 2:
                check_cancelled()
                try:
                    chunk = next(source_iter)
                except StopIteration:
                    return

                chunk_length = len(chunk)
                if source_seen + chunk_length <= resume_processed:
                    source_seen += chunk_length
                    continue
                if source_seen < resume_processed:
                    chunk = chunk._select(slice(resume_processed - source_seen, None), copy=False)
                    source_seen = resume_processed

                source_seen += len(chunk)
                pending[next_sequence] = executor.submit(
                    _process_pipeline_chunk,
                    chunk,
                    operations,
                    self.metadata,
                    copy,
                    callers,
                    copy_output,
                )
                next_sequence += 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            source_iter = iter(self._source_chunks(chunk_size=chunk_size, copy=False))
            submit_ready(executor, source_iter)
            while pending:
                future = pending.pop(next_yield)
                result = future.result()
                next_yield += 1

                # Check cancellation before delivering this chunk's rows.
                check_cancelled()

                chunk_start = processed
                emitted_start = emitted
                skipped_start = skipped
                for kept, (row, offset) in enumerate(zip(result.rows, result.offsets)):
                    if checkpoint is not None:
                        # Advance per delivered row: a consumer that stops
                        # mid-chunk must resume right after its last row.
                        _update_checkpoint(
                            checkpoint,
                            PipelineProgress(
                                processed=chunk_start + offset + 1,
                                emitted=emitted_start + kept + 1,
                                skipped=skipped_start + offset - kept,
                                topic=topic,
                                message_id=row[0],
                                timestamp=row[1],
                            ),
                            operation_counters=operation_counters,
                        )
                    yield row

                processed += result.processed
                emitted += result.emitted
                skipped += result.skipped
                if result.processed:
                    last_id = result.message_id
                    last_timestamp = result.timestamp
                if checkpoint is not None or progress_callback is not None:
                    progress = _record_progress(
                        checkpoint,
                        PipelineProgress(
                            processed=processed,
                            emitted=emitted,
                            skipped=skipped,
                            topic=topic,
                            message_id=result.message_id,
                            timestamp=result.timestamp,
                        ),
                        operation_counters=operation_counters,
                    )
                    _notify_progress(progress_callback, progress, progress_interval, previous=chunk_start)

                submit_ready(executor, source_iter)

        done_progress = _record_progress(
            checkpoint,
            PipelineProgress(
                processed=processed,
                emitted=emitted,
                skipped=skipped,
                topic=topic,
                message_id=last_id,
                timestamp=last_timestamp,
                done=True,
            ),
            operation_counters=operation_counters,
        )
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

    def _empty_data_template(self) -> np.ndarray | None:
        """Schema-correct empty data array for no-match collects, or None when
        the output schema is unknowable (a map operation may change it)."""

        _, remaining = self._split_pushdown_operations()
        if any(operation.kind == "map" for operation in remaining):
            return None
        try:
            for chunk in self._chunk_source(1, False, ()):
                return np.asarray(chunk.data)[:0].copy()
        except Exception:
            return None
        return None

    def _make_chunk(self, ids: list[Any], timestamps: list[float], values: list[np.ndarray], copy: bool, frame_ids=None) -> TopicView:
        # Preserve "no ids" instead of fabricating a column of Nones so the
        # lazy path returns the same schema as the eager path.
        ids_array = None if all(i is None for i in ids) else _object_vector(ids)
        ts_array = np.asarray(timestamps, dtype=np.float64)
        # Stacking allocates, so numeric chunks never alias their rows; object
        # chunks hold references, which copy=True replaces with copies.
        data_array = _stack_values(values)
        if copy and data_array.dtype == object:
            data_array = _copy_object_elements(data_array)
        return TopicView(ids_array, ts_array, data_array, metadata=self.metadata,
                         frame_ids=frame_ids if frame_ids and (self.metadata.frame_id is not None or
                                                             any(f is not None for f in frame_ids)) else None)


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
        if _checkpoint_processed(checkpoint) > 0:
            raise ValueError(
                "window pipelines cannot resume from a checkpoint: the rolling "
                "window state is not persisted, so resumed windows would "
                "silently differ from an uninterrupted run. Restart with a "
                "fresh checkpoint."
            )
        ids = deque()
        timestamps = deque()
        values = deque()
        frame_ids = deque()

        for message_id, timestamp, value, frame_id in self.pipeline._iter_processed_rows(
            chunk_size=chunk_size,
            copy=self.copy,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            checkpoint=checkpoint,
            progress_interval=progress_interval,
        ):
            ids.append(message_id)
            frame_ids.append(frame_id)
            timestamps.append(timestamp)
            # Rows are already copied by the iterator when self.copy is set.
            values.append(value)

            if self.seconds is not None:
                cutoff = timestamp - self.seconds
                while timestamps and timestamps[0] < cutoff:
                    ids.popleft()
                    timestamps.popleft()
                    values.popleft()
                    frame_ids.popleft()

            if self.size is not None:
                while len(values) > self.size:
                    ids.popleft()
                    timestamps.popleft()
                    values.popleft()
                    frame_ids.popleft()

            window_ids = None if all(i is None for i in ids) else _object_vector(ids)
            # Stacking allocates new arrays, so windows never share memory
            # with each other or with the source.
            yield TopicView(
                window_ids,
                np.asarray(timestamps, dtype=np.float64),
                _stack_values(values),
                metadata=self.pipeline.metadata,
                frame_ids=_object_vector(frame_ids),
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
                # Selection semantics (pinned by tests): topics whose known
                # frame does not match are dropped from the dataset entirely.
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
            # Same result shape as TopicPipeline.collect() (including
            # per-row frame_ids); the limits apply to the running totals.
            result = pipeline._collect(
                chunk_size=chunk_size,
                copy=copy,
                max_rows=max_rows,
                max_bytes=max_bytes,
                allow_large=allow_large,
                max_workers=max_workers,
                rows_before=total_rows,
                bytes_before=total_bytes,
            )
            total_rows += int(np.asarray(result["ts"]).shape[0])
            total_bytes += _topic_result_nbytes(result)
            collected[topic] = result

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
        """Transform each streamed message.

        Unlike topic pipelines, a callable taking one positional parameter
        receives the whole message dict (``topic``, ``timestamp``, ``name``,
        ``data``, ...), not just its data. Callables that require two or
        more positional parameters are called as ``fn(data, ts)`` or
        ``fn(data, ts, name)``. `fn` returns either a replacement message
        mapping or replacement data. With `copy`, `fn` gets a private copy
        of the message and its data.
        """

        return self._with_operation("source_map", fn, copy=copy)

    def filter(self, predicate: Callable, copy: bool = True) -> "SourcePipeline":
        """Keep messages for which `predicate` is truthy.

        As with `map()`, a one-parameter `predicate` receives the whole
        message dict; callables requiring two or more positional parameters
        get ``(data, ts)`` or ``(data, ts, name)``.
        """

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
        """Yield processed source messages.

        With `copy`, message data is copied once up front, so neither the
        callables nor the consumer can alias the source's arrays. A
        `checkpoint` covers exactly the messages processed so far (it also
        records per-topic emitted counts under ``"topic_emitted"``).
        """

        index_counters = _checkpoint_operation_counters(checkpoint, len(self._operations), kind="source")
        resume_processed = _checkpoint_processed(checkpoint)
        processed = 0
        emitted = _checkpoint_emitted(checkpoint)
        skipped = _checkpoint_skipped(checkpoint)
        topic_emitted = _checkpoint_topic_emitted(checkpoint)
        progress_interval = _validated_progress_interval(progress_interval)
        track_progress = checkpoint is not None or progress_callback is not None
        callers = [
            _source_caller(operation.args[0]) if operation.kind in {"source_map", "source_filter"} else None
            for operation in self._operations
        ]
        last_topic = None
        last_id = None
        last_timestamp = None
        for raw_message in _source_messages(self.data_source):
            processed += 1
            if processed <= resume_processed:
                continue
            if cancel_token is not None:
                _raise_if_cancelled(
                    cancel_token,
                    checkpoint,
                    PipelineProgress(
                        processed=processed - 1,
                        emitted=emitted,
                        skipped=skipped,
                    ),
                    operation_counters=index_counters,
                )
            # The one copy: after it, nothing below aliases the source.
            message = _normalize_source_message(raw_message, copy=copy)
            keep = True

            for operation_index, operation in enumerate(self._operations):
                if operation.kind == "source_topics":
                    keep = message["topic"] in operation.args[0]
                elif operation.kind == "source_map":
                    mapped = callers[operation_index](
                        _copy_source_message(message) if operation.kwargs.get("copy", True) else message,
                    )
                    message = _mapped_source_message(message, mapped, copy=False)
                elif operation.kind == "source_filter":
                    keep = bool(
                        callers[operation_index](
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
                if topic_emitted is not None:
                    topic_emitted[message["topic"]] = topic_emitted.get(message["topic"], 0) + 1
            else:
                skipped += 1

            last_topic = message["topic"]
            last_id = message.get("name")
            last_timestamp = float(message["timestamp"])
            if track_progress:
                progress = _record_progress(
                    checkpoint,
                    PipelineProgress(
                        processed=processed,
                        emitted=emitted,
                        skipped=skipped,
                        topic=last_topic,
                        message_id=last_id,
                        timestamp=last_timestamp,
                    ),
                    operation_counters=index_counters,
                    topic_emitted=topic_emitted,
                )
                _notify_progress(progress_callback, progress, progress_interval)

            if keep:
                yield message

        done_progress = _record_progress(
            checkpoint,
            PipelineProgress(
                processed=max(processed, resume_processed),
                emitted=emitted,
                skipped=skipped,
                topic=last_topic,
                message_id=last_id,
                timestamp=last_timestamp,
                done=True,
            ),
            operation_counters=index_counters,
            topic_emitted=topic_emitted,
        )
        _notify_progress(progress_callback, done_progress, progress_interval, force=True)

    def nearest_topic_pairs(
        self,
        reference_topic: str,
        target_topic: str,
        tolerance: float | None = None,
        copy: bool = True,
    ) -> Iterable[tuple[dict, dict]]:
        """Yield reference messages paired with the nearest target message seen in the stream.

        This streaming join is intended for time-ordered sources. It buffers target
        messages and pairs each reference message with the nearest target timestamp
        available once a newer target has been observed.
        """

        if reference_topic not in self.topics:
            raise ValueError(f"reference topic not found: {reference_topic!r}")
        if target_topic not in self.topics:
            raise ValueError(f"target topic not found: {target_topic!r}")

        targets: deque[dict] = deque()
        pending_references: deque[dict] = deque()

        def emit_ready(current_time: float):
            while pending_references and targets and targets[-1]["timestamp"] >= pending_references[0]["timestamp"]:
                reference = pending_references.popleft()
                target = _nearest_buffered_message(targets, float(reference["timestamp"]), tolerance)
                if target is None:
                    continue
                yield (
                    _copy_source_message(reference) if copy else reference,
                    _copy_source_message(target) if copy else target,
                )
                while len(targets) > 2 and targets[1]["timestamp"] <= current_time:
                    targets.popleft()

        selected = self.select_topics(reference_topic, target_topic)
        for message in selected.iter_messages(copy=copy):
            if message["topic"] == target_topic:
                targets.append(message)
                yield from emit_ready(float(message["timestamp"]))
            elif message["topic"] == reference_topic:
                pending_references.append(message)
                yield from emit_ready(float(message["timestamp"]))

        for reference in pending_references:
            target = _nearest_buffered_message(targets, float(reference["timestamp"]), tolerance)
            if target is not None:
                yield (
                    _copy_source_message(reference) if copy else reference,
                    _copy_source_message(target) if copy else target,
                )

    def to_buffer(
        self,
        buffer_depth: int = 1,
        data_uri: str = "/tmp/tiledb/my_group/",
        use_db: bool = False,
        axis: str | None = None,
        buffer=None,
        backend: str | None = None,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
    ):
        """Stream processed messages into a `DataBuffer`.

        `backend` follows DataBuffer semantics: None picks memory for
        `use_db=False` and arrow (or an existing TileDB store) for
        `use_db=True`.

        Resuming a persistent store: the store's per-topic row counts are
        the source of truth. A checkpoint is used to skip already-processed
        source messages only when every topic it counts as emitted is fully
        stored; rows the store holds beyond the checkpoint are not appended
        twice. When the checkpoint is ahead of the store (e.g. the process
        died before staged rows were flushed), it is reset and the source is
        replayed from the start, skipping the rows already stored.
        """

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
            backend=backend,
            preload=0,
        )
        if result.use_db and _checkpoint_processed(checkpoint) > 0:
            # The checkpoint may run ahead of (crash before flush) or behind
            # (older saved copy) what the store actually holds; reconcile it
            # with the stored per-topic counts before trusting it.
            skip_counts = self._reconcile_checkpoint_with_store(result, checkpoint)
            return self._append_to_buffer(
                result,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
                skip_counts=skip_counts,
            )
        try:
            result.load_data_db(selected_axis)
        except BaseException:
            # Persist what was appended so the store agrees with the
            # checkpoint, which already covers those rows.
            _close_after_failure(result)
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
            backend="tiledb",
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

    def _reconcile_checkpoint_with_store(self, buffer, checkpoint: dict[str, Any]) -> dict[str, int]:
        """Make a resume checkpoint consistent with a persistent store.

        Returns how many upcoming emitted messages per topic are already
        stored and must be skipped. Resets `checkpoint` in place (so the
        source is replayed from the start) when it claims rows the store
        does not hold.
        """

        stored = {str(topic): int(count) for topic, count in dict(buffer.counters).items()}
        topic_emitted = _checkpoint_topic_emitted(checkpoint)
        if topic_emitted is not None:
            if all(stored.get(topic, 0) >= count for topic, count in topic_emitted.items()):
                return {
                    topic: count - topic_emitted.get(topic, 0)
                    for topic, count in stored.items()
                    if count > topic_emitted.get(topic, 0)
                }
        elif sum(stored.get(topic, 0) for topic in self.topics) == _checkpoint_emitted(checkpoint):
            # Legacy checkpoint without per-topic counts: trust it only when
            # its emitted total matches the store exactly.
            return {}

        checkpoint.clear()
        return {topic: count for topic, count in stored.items() if count > 0}

    def _append_to_buffer(
        self,
        buffer,
        progress_callback: Callable[[PipelineProgress], Any] | None = None,
        cancel_token: Any | None = None,
        checkpoint: dict[str, Any] | None = None,
        progress_interval: int = 1,
        skip_counts: Mapping[str, int] | None = None,
    ):
        # Emitted messages per topic that the store already holds.
        remaining_skips = {topic: int(count) for topic, count in (skip_counts or {}).items() if count > 0}
        try:
            for message in self.iter_messages(
                copy=True,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                checkpoint=checkpoint,
                progress_interval=progress_interval,
            ):
                topic = message["topic"]
                if remaining_skips.get(topic, 0) > 0:
                    remaining_skips[topic] -= 1
                    continue
                if topic not in buffer.topics:
                    buffer.topics.append(topic)
                if hasattr(buffer.buffer_impl, "topics") and topic not in buffer.buffer_impl.topics:
                    buffer.buffer_impl.topics.append(topic)
                if getattr(buffer, "backend", None) == "tiledb":
                    self._prepare_tiledb_append(buffer, message)
                buffer.append_buffer(message)
        except BaseException:
            # Flush appended rows so the store matches the checkpoint, which
            # already covers them (cancellation, callable errors, Ctrl-C).
            _close_after_failure(buffer)
            raise

        if getattr(buffer, "use_db", False):
            if hasattr(buffer, "_source_exhausted"):
                buffer._source_exhausted = True
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
            frame_ids=topic_data.frame_ids,
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

    row_frames = None
    if isinstance(topic_data, dict):
        row_frames = topic_data.get("frame_ids")
    elif isinstance(topic_data, np.ndarray) and topic_data.dtype.names and "frame_id" in topic_data.dtype.names:
        row_frames = topic_data["frame_id"]
    ids, ts, data = topic_parts(topic_data)
    return TopicView(
        ids,
        ts,
        data,
        frame_ids=row_frames,
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
            if selected.frame_ids is not None:
                selected = selected._select(np.array([
                    selected._frame_id_at(i) in targets for i in range(len(selected))
                ], dtype=bool), copy=False)
            elif metadata_frame_id is not None:
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
    callers: list | None = None,
    copy_output: bool | None = None,
) -> _ProcessedChunk:
    callers = _operation_callers(operations) if callers is None else callers
    copy_output = copy if copy_output is None else copy_output
    fallback_frame_id = _decode_text(metadata.frame_id)
    rows = []
    offsets = []
    processed = 0
    emitted = 0
    skipped = 0
    last_id = None
    last_timestamp = None
    for row_index in range(len(chunk)):
        processed += 1
        frame_id = chunk._frame_id_at(row_index)
        if chunk.frame_ids is None and frame_id is None:
            frame_id = fallback_frame_id
        current_id = None if chunk.ids is None else chunk.ids[row_index]
        current_timestamp = float(chunk.timestamps[row_index])
        last_id = current_id
        last_timestamp = current_timestamp

        keep, value, aliases_source = _apply_row_operations(
            operations,
            callers,
            chunk.data[row_index],
            current_timestamp,
            current_id,
            frame_id,
            index_counters=None,
            protect_source=copy,
        )
        if keep:
            emitted += 1
            if copy_output and aliases_source:
                value = _copy_value(value)
            rows.append((current_id, current_timestamp, value, frame_id))
            offsets.append(row_index)
        else:
            skipped += 1

    return _ProcessedChunk(
        rows=tuple(rows),
        processed=processed,
        emitted=emitted,
        skipped=skipped,
        message_id=last_id,
        timestamp=last_timestamp,
        offsets=tuple(offsets),
    )


def _apply_row_operations(
    operations: tuple[_PipelineOperation, ...],
    callers: list,
    value: Any,
    timestamp: float,
    message_id: Any,
    frame_id: str | None,
    index_counters: list[int] | None = None,
    protect_source: bool = True,
) -> tuple[bool, Any, bool]:
    """Run the row-level pipeline operations on one source row.

    Returns ``(keep, value, aliases_source)``. Callables whose operation has
    ``copy=True`` always receive a private copy. With `protect_source`, a
    ``copy=False`` callable gets a (single, reused) private copy instead of
    the source row. `aliases_source` tells callers whether `value` may still
    share memory with the source row. `index_counters` is None on the
    parallel path, where non-leading index ranges are unsupported.
    """

    aliases_source = True
    for op_index, operation in enumerate(operations):
        kind = operation.kind
        if kind == "map" or kind == "filter":
            op_copy = operation.kwargs.get("copy", True)
            if op_copy:
                argument = _copy_value(value)
            else:
                if protect_source and aliases_source:
                    value = _copy_value(value)
                    aliases_source = False
                argument = value
            result = callers[op_index](argument, timestamp, message_id)
            if kind == "map":
                value = result
                # A callable given a private copy cannot return source memory.
                aliases_source = aliases_source and not op_copy
                continue
            keep = bool(result)
        elif kind == "time_range":
            start, end = operation.args
            if operation.kwargs.get("inclusive", True):
                keep = timestamp >= start and timestamp <= end
            else:
                keep = timestamp > start and timestamp < end
        elif kind == "index_range":
            if index_counters is None:
                raise ValueError("parallel topic execution does not support non-leading index_range operations")
            current_index = index_counters[op_index]
            index_counters[op_index] += 1
            keep = _slice_contains(current_index, *operation.args)
        elif kind == "frame_id":
            targets = operation.args[0]
            if frame_id is not None:
                keep = frame_id in targets
            else:
                keep = _row_frame_id(value, message_id) in targets
        elif kind == "spatial_bounds":
            min_bound, max_bound = operation.args
            keep = _spatial_value_in_bounds(
                value,
                min_bound=min_bound,
                max_bound=max_bound,
                columns=operation.kwargs["columns"],
            )
        else:
            raise ValueError(f"unsupported pipeline operation: {kind}")

        if not keep:
            return False, value, aliases_source
    return True, value, aliases_source


def _operation_callers(operations: Iterable[_PipelineOperation]) -> list:
    """Resolve each map/filter callable's argument dispatch once per run."""

    return [
        _metadata_caller(operation.args[0]) if operation.kind in {"map", "filter"} else None
        for operation in operations
    ]


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



def _nearest_buffered_message(messages: deque[dict], timestamp: float, tolerance: float | None = None) -> dict | None:
    if not messages:
        return None
    best = min(messages, key=lambda message: abs(float(message["timestamp"]) - timestamp))
    if tolerance is not None and abs(float(best["timestamp"]) - timestamp) > tolerance:
        return None
    return best

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


def _source_caller(fn: Callable) -> Callable[[Mapping[str, Any]], Any]:
    """Dispatch for source-pipeline callables, resolved once per run.

    Callables requiring two or more positional parameters get
    ``(data, ts[, name])``; everything else gets the message dict first,
    falling back to data-style dispatch when that call cannot bind.
    """

    required = _required_positional_count(fn)
    if required is not None and required >= 2:
        data_call = _metadata_caller(fn)
        return lambda message: data_call(message["data"], float(message["timestamp"]), message.get("name"))

    data_call = None

    def call(message: Mapping[str, Any]):
        nonlocal data_call
        try:
            return fn(message)
        except TypeError as exc:
            if _type_error_from_inside(exc, fn):
                raise
        if data_call is None:
            data_call = _metadata_caller(fn)
        return data_call(message["data"], float(message["timestamp"]), message.get("name"))

    return call


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


def _checkpoint_topic_emitted(checkpoint: Mapping[str, Any] | None) -> dict[str, int] | None:
    """Per-topic emitted counts of a source checkpoint, or None when unknown
    (no checkpoint, or a resumed checkpoint written before they existed)."""

    if checkpoint is None:
        return None
    saved = checkpoint.get("topic_emitted")
    if isinstance(saved, Mapping):
        return {str(topic): int(count) for topic, count in saved.items()}
    if _checkpoint_processed(checkpoint) > 0:
        return None
    return {}


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
    topic_emitted: Mapping[str, int] | None = None,
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
    if topic_emitted is not None:
        checkpoint["topic_emitted"] = dict(topic_emitted)


def _record_progress(
    checkpoint: dict[str, Any] | None,
    progress: PipelineProgress,
    operation_counters=None,
    topic_emitted: Mapping[str, int] | None = None,
) -> PipelineProgress:
    """Write `progress` into `checkpoint`, then return it carrying the
    updated checkpoint snapshot (so ``progress.checkpoint`` matches
    ``progress`` itself, including ``done``)."""

    if checkpoint is None:
        return progress
    _update_checkpoint(checkpoint, progress, operation_counters=operation_counters, topic_emitted=topic_emitted)
    return replace(progress, checkpoint=_checkpoint_snapshot(checkpoint))


def _restore_checkpoint(checkpoint: dict[str, Any] | None, snapshot: Mapping[str, Any] | None) -> None:
    """Roll a live checkpoint back to an earlier snapshot, in place."""

    if checkpoint is None or snapshot is None:
        return
    checkpoint.clear()
    checkpoint.update(_checkpoint_snapshot(snapshot))


def _close_after_failure(buffer) -> None:
    """Flush a persistent buffer while an exception propagates, without
    letting a secondary close error replace the original exception."""

    if not getattr(buffer, "use_db", False):
        return
    try:
        buffer.close(closed=False)
    except Exception:
        pass


def _notify_progress(
    progress_callback: Callable[[PipelineProgress], Any] | None,
    progress: PipelineProgress,
    progress_interval: int,
    force: bool = False,
    previous: int | None = None,
) -> None:
    """Call `progress_callback` whenever `processed` crosses a multiple of
    `progress_interval` since `previous` (default: the row before), so
    chunked execution that advances many rows at once still reports."""

    if progress_callback is None:
        return
    if force or progress.done or progress.cancelled:
        progress_callback(progress)
        return
    previous = progress.processed - 1 if previous is None else previous
    if progress.processed // progress_interval > previous // progress_interval:
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
    frames_nbytes = 0 if view.frame_ids is None else view.frame_ids.nbytes
    return int(data_nbytes + view.timestamps.nbytes + ids_nbytes + frames_nbytes)


def _topic_result_nbytes(topic: Mapping[str, Any]) -> int:
    total = np.asarray(topic["ts"]).nbytes + np.asarray(topic["data"]).nbytes
    ids = topic.get("id", topic.get("name"))
    if ids is not None:
        total += np.asarray(ids).nbytes
    if topic.get("frame_ids") is not None:
        total += np.asarray(topic["frame_ids"]).nbytes
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


def _callable_code(fn: Callable):
    """The code object of fn's Python body, unwrapping partials and wrappers."""

    seen = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        if isinstance(fn, functools.partial):
            fn = fn.func
            continue
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            fn = wrapped
            continue
        break
    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    if code is None:
        call = getattr(fn, "__call__", None)
        code = getattr(getattr(call, "__func__", call), "__code__", None)
    return code


def _type_error_from_inside(exc: TypeError, fn: Callable) -> bool:
    """True when the TypeError was raised inside fn's own body (its frame ran).

    Retrying with fewer arguments is only safe for argument-binding errors;
    re-invoking a function that already executed masks the user's real error
    and can duplicate side effects."""

    code = _callable_code(fn)
    if code is None:
        return False
    traceback = exc.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code is code:
            return True
        traceback = traceback.tb_next
    return False


def _concat_chunk_ids(ids_parts: list, chunk_lengths: list[int]):
    """Concatenate per-chunk id arrays, backfilling id-less chunks with None
    entries when any sibling chunk carries ids; None when no chunk does."""

    if not ids_parts or all(part is None for part in ids_parts):
        return None
    return np.concatenate([
        part if part is not None else np.full(length, None, dtype=object)
        for part, length in zip(ids_parts, chunk_lengths)
    ])


_BUILTIN_CALLABLE_TYPES = (
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.ClassMethodDescriptorType,
    type,
)
_VARIADIC = 1 << 30


def _required_positional_count(fn: Callable) -> int | None:
    """How many positional arguments `fn` requires.

    Parameters with defaults do not count (so ``def scale(d, factor=2.0)``
    requires one); ``*args`` counts as unbounded. NumPy ufuncs and builtins
    without an introspectable signature (``max``, ``np.asarray``) count as
    data-only (0). None means the signature is unknown.
    """

    if isinstance(fn, np.ufunc):
        return 0
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return 0 if isinstance(fn, _BUILTIN_CALLABLE_TYPES) else None

    required = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return _VARIADIC
        if (
            parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ):
            required += 1
    return required


def _metadata_caller(fn: Callable) -> Callable[[Any, float, Any], Any]:
    """Return ``call(data, ts, message_id)`` passing `fn` only the leading
    arguments it requires: ``fn(data)``, ``fn(data, ts)`` or
    ``fn(data, ts, id)``.

    Probing by calling ``fn(data, ts, id)`` first would feed metadata into
    optional parameters (``np.linalg.norm(data, ord=ts)``), so the signature
    decides; probing remains only for callables without one.
    """

    required = _required_positional_count(fn)
    if required is None:
        return functools.partial(_probe_call_with_metadata, fn)
    if required <= 1:
        return lambda data, ts, message_id: fn(data)
    if required == 2:
        return lambda data, ts, message_id: fn(data, ts)
    return fn


def _reduce_caller(fn: Callable) -> Callable[[Any, Any, float, Any], Any]:
    """Like `_metadata_caller` for reducers: ``fn(acc, data)``,
    ``fn(acc, data, ts)`` or ``fn(acc, data, ts, id)``."""

    required = _required_positional_count(fn)
    if required is None:
        return functools.partial(_probe_reduce_call, fn)
    if required <= 2:
        return lambda acc, data, ts, message_id: fn(acc, data)
    if required == 3:
        return lambda acc, data, ts, message_id: fn(acc, data, ts)
    return fn


def _probe_call_with_metadata(fn: Callable, data: Any, ts: float, message_id: Any) -> Any:
    try:
        return fn(data, ts, message_id)
    except TypeError as exc:
        if _type_error_from_inside(exc, fn):
            raise
    try:
        return fn(data, ts)
    except TypeError as exc:
        if _type_error_from_inside(exc, fn):
            raise
    return fn(data)


def _probe_reduce_call(fn: Callable, acc: Any, data: Any, ts: float, message_id: Any) -> Any:
    try:
        return fn(acc, data, ts, message_id)
    except TypeError as exc:
        if _type_error_from_inside(exc, fn):
            raise
    return fn(acc, data)


def _copy_value(value: Any) -> Any:
    """Copy one row value: arrays and NumPy scalars via ``.copy()``, other
    Python objects (floats, strings, dicts from a map...) shallowly."""

    if isinstance(value, (np.ndarray, np.generic)):
        return value.copy()
    return _shallow_copy(value)


def _object_vector(items: list) -> np.ndarray:
    """1-D object array holding `items` as-is (np.asarray would broadcast
    nested sequences into extra dimensions)."""

    result = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        result[index] = item
    return result


def _stack_values(values: list) -> np.ndarray:
    """Stack row values; rows of differing shape become a 1-D object array."""

    try:
        return np.asarray(values)
    except ValueError:
        return _object_vector(values)


def _concat_data_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate chunk data; chunks whose row shapes differ fall back to a
    1-D object array of rows."""

    try:
        return np.concatenate(parts, axis=0)
    except ValueError:
        rows = []
        for part in parts:
            if part.dtype == object and part.ndim == 1:
                rows.extend(part.tolist())
            else:
                rows.extend(part[index] for index in range(part.shape[0]))
        return _object_vector(rows)


def _copy_object_elements(data: np.ndarray) -> np.ndarray:
    """Copy an object array together with the objects it references."""

    result = np.empty(data.shape, dtype=object)
    for index in np.ndindex(data.shape):
        result[index] = _copy_value(data[index])
    return result


def _is_non_decreasing(values: np.ndarray) -> bool:
    values = np.asarray(values)
    return bool(values.size < 2 or np.all(values[1:] >= values[:-1]))


def _decoded_frame_array(frame_ids: np.ndarray) -> np.ndarray:
    """Per-row frame ids as an object array of str/None."""

    frame_ids = np.asarray(frame_ids)
    if frame_ids.dtype.kind == "U":
        return frame_ids.astype(object)
    if frame_ids.dtype.kind == "S":
        return np.char.decode(frame_ids, "utf-8", errors="replace").astype(object)
    return _object_vector([_decode_text(frame) for frame in frame_ids])


def _rows_may_carry_frame_ids(chunk: TopicView) -> bool:
    """Whether `_row_frame_id` can find a frame inside any row's data or id."""

    if chunk.data.dtype.fields is not None or chunk.data.dtype == object:
        return True
    ids = chunk.ids
    if ids is None:
        return False
    if ids.dtype.fields is not None:
        return True
    if ids.dtype != object:
        return False
    return not all(
        item is None or (isinstance(item, (str, bytes, int, float, np.number, np.bool_)) and not isinstance(item, np.void))
        for item in ids
    )


def _chunk_frame_ids(chunk: TopicView, fallback: str | None) -> np.ndarray | None:
    """Vectorized equivalent of the per-row frame id resolution in the row
    path (chunk frame ids, then the chunk's frame, then fields inside each
    row, then the pipeline's frame). None when every row has no frame."""

    count = len(chunk)
    if chunk.frame_ids is not None:
        return _decoded_frame_array(chunk.frame_ids)
    own = _decode_text(chunk.metadata.frame_id)
    if own:
        return np.full(count, own, dtype=object)
    if _rows_may_carry_frame_ids(chunk):
        frames = np.empty(count, dtype=object)
        for index in range(count):
            frame = _row_frame_id(chunk.data[index], None if chunk.ids is None else chunk.ids[index])
            frames[index] = fallback if frame is None else frame
        return None if all(frame is None for frame in frames) else frames
    if fallback is not None:
        return np.full(count, fallback, dtype=object)
    return None


def _passthrough_parts(chunk: TopicView, fallback_frame_id: str | None) -> tuple:
    return (chunk.ids, chunk.timestamps, chunk.data, _chunk_frame_ids(chunk, fallback_frame_id))


def _split_parts(parts: list[tuple], count: int) -> tuple[list[tuple], list[tuple]]:
    """Split pending `(ids, ts, data, frames)` parts after `count` rows."""

    taken: list[tuple] = []
    rest: list[tuple] = []
    needed = count
    for part in parts:
        length = part[1].shape[0]
        if needed == 0:
            rest.append(part)
        elif length <= needed:
            taken.append(part)
            needed -= length
        else:
            taken.append(tuple(None if item is None else item[:needed] for item in part))
            rest.append(tuple(None if item is None else item[needed:] for item in part))
            needed = 0
    return taken, rest


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
    """Index of the timestamp nearest `query_time` (ties prefer the later
    sample), or None when empty or farther than `tolerance`. `timestamps`
    need not be sorted."""

    ts = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))
    index = int(_nearest_indices(np.array([query_time], dtype=np.float64), ts, tolerance)[0])
    return None if index < 0 else index


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
    """Align target messages to exactly matching reference timestamps.

    Results follow the reference order; the target need not be
    time-ordered and ``target_index`` refers to its original rows."""

    reference = topic_view(reference_topic)
    target = topic_view(target_topic)
    indices = _exact_alignment_indices(reference.timestamps, target.timestamps)
    return _aligned_topic_result(reference, target, indices, mode="exact")


def align_nearest(
    reference_topic: dict | np.ndarray | TopicView,
    target_topic: dict | np.ndarray | TopicView,
    tolerance: float | None = None,
) -> dict:
    """Align each reference timestamp to the nearest target message.

    Results follow the reference order; the target need not be
    time-ordered and ``target_index`` refers to its original rows."""

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
    """Resample a topic onto a fixed-rate timestamp grid.

    Source rows need not be time-ordered (they are stably sorted first;
    rows with non-finite timestamps are ignored), and ``target_index``
    always refers to the caller's original rows. `tolerance` bounds the
    distance to the nearest source sample for ``method="nearest"``; for
    ``method="linear"`` it is the largest source gap that may be
    interpolated across. Samples outside the source time range, or inside a
    larger gap, are marked invalid and their data is NaN.
    """

    if tolerance is not None and tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    original = topic_view(topic_data)
    view, positions = _time_ordered_view(original, finite_only=True)
    sample_ts = _fixed_rate_timestamps(view.timestamps, rate_hz=rate_hz, period=period, start=start, end=end)
    normalized_method = method.lower().replace("-", "_")

    if normalized_method in {"nearest", "nearest_neighbor"}:
        indices = _nearest_alignment_indices(sample_ts, view.timestamps, tolerance=tolerance)
        result = _aligned_topic_arrays(sample_ts, view, indices)
        if positions is not None:
            result["target_index"] = np.where(indices >= 0, positions[np.maximum(indices, 0)], -1).astype(np.int64)
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
        valid = _linear_sample_validity(view.timestamps, sample_ts, tolerance)
        if not valid.all():
            data[~valid] = np.nan

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
    """Join each reference timestamp with a trailing target-topic window.

    An unsorted target is stably sorted by time first, so each window
    holds its rows in time order."""

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
    # Windows are time slices of the target, so it must be time-ordered.
    target, _ = _time_ordered_view(topic_view(target_topic))
    window_lookback = seconds if lookback is None else lookback
    counts = []
    windows = []

    if window_lookback is None:
        lefts = np.zeros(reference.timestamps.shape, dtype=np.int64)
    else:
        lefts = np.searchsorted(target.timestamps, reference.timestamps - window_lookback, side="left")
    rights = np.searchsorted(target.timestamps, reference.timestamps + lookahead, side="right")
    if size is not None:
        lefts = np.maximum(lefts, rights - size)
    lefts = np.minimum(lefts, rights)

    for left, right in zip(lefts.tolist(), rights.tolist()):
        ids = None if target.ids is None else target.ids[left:right]
        ts = target.timestamps[left:right]
        data = target.data[left:right]
        frames = None if target.frame_ids is None else target.frame_ids[left:right]
        window = TopicView(ids, ts, data, metadata=target.metadata, copy=copy, frame_ids=frames)
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


def _time_order(timestamps: np.ndarray) -> np.ndarray | None:
    """Stable time-sorting permutation, or None when already non-decreasing.
    Non-finite timestamps sort last (NaN compares false, so arrays holding
    one are never considered sorted)."""

    if _is_non_decreasing(timestamps):
        return None
    return np.argsort(timestamps, kind="stable")


def _time_ordered_view(view: TopicView, finite_only: bool = False) -> tuple[TopicView, np.ndarray | None]:
    """`view` sorted by time (optionally dropping non-finite timestamps),
    plus the original row index of each returned row (None when unchanged)."""

    positions = None
    if finite_only:
        finite = np.isfinite(view.timestamps)
        if not finite.all():
            positions = np.flatnonzero(finite)
    ordered_ts = view.timestamps if positions is None else view.timestamps[positions]
    order = _time_order(ordered_ts)
    if order is not None:
        positions = order if positions is None else positions[order]
    if positions is None:
        return view, None
    return view._select(positions, copy=False), positions


def _exact_alignment_indices(reference_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    reference_ts = np.asarray(reference_ts, dtype=np.float64)
    target_ts = np.asarray(target_ts, dtype=np.float64)
    if target_ts.size == 0:
        return np.full(reference_ts.shape, -1, dtype=np.int64)

    # searchsorted needs sorted input; map sorted positions back to the
    # caller's row order (stable sort keeps the first of equal stamps).
    order = _time_order(target_ts)
    sorted_ts = target_ts if order is None else target_ts[order]
    positions = np.searchsorted(sorted_ts, reference_ts, side="left")
    clipped = np.minimum(positions, sorted_ts.size - 1)
    valid = (positions < sorted_ts.size) & (sorted_ts[clipped] == reference_ts)
    matches = clipped if order is None else order[clipped]
    return np.where(valid, matches, -1).astype(np.int64)


def _nearest_alignment_indices(
    reference_ts: np.ndarray,
    target_ts: np.ndarray,
    tolerance: float | None = None,
) -> np.ndarray:
    if tolerance is not None and tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return _nearest_indices(reference_ts, target_ts, tolerance)


def _nearest_indices(reference_ts, target_ts, tolerance: float | None = None) -> np.ndarray:
    """Vectorized nearest-target lookup for every reference timestamp.

    Targets need not be sorted; returned indices refer to the caller's
    target order. Ties prefer the later target; NaN references, NaN targets
    and matches farther than `tolerance` yield -1.
    """

    reference_ts = np.asarray(reference_ts, dtype=np.float64)
    target_ts = np.asarray(target_ts, dtype=np.float64)
    result = np.full(reference_ts.shape, -1, dtype=np.int64)
    order = _time_order(target_ts)
    sorted_ts = target_ts if order is None else target_ts[order]
    # NaNs sort last; only the leading non-NaN run is searchable.
    count = int(sorted_ts.size - np.count_nonzero(np.isnan(sorted_ts)))
    if count == 0 or reference_ts.size == 0:
        return result
    searchable = sorted_ts[:count]

    insert = np.searchsorted(searchable, reference_ts, side="left")
    right = np.minimum(insert, count - 1)
    left = np.maximum(insert - 1, 0)
    with np.errstate(invalid="ignore"):
        right_distance = np.abs(searchable[right] - reference_ts)
        left_distance = np.abs(searchable[left] - reference_ts)
        use_right = (insert < count) & ((insert == 0) | (right_distance <= left_distance))
        best = np.where(use_right, right, left)
        distance = np.where(use_right, right_distance, left_distance)
        found = ~np.isnan(distance)
        if tolerance is not None:
            found &= distance <= tolerance
    matches = best if order is None else order[best]
    result[found] = matches[found]
    return result


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
    source_ts = np.asarray(source_ts, dtype=np.float64)
    source_ts = source_ts[np.isfinite(source_ts)]
    if source_ts.size == 0 and (start is None or end is None):
        return np.array([], dtype=np.float64)

    sample_start = float(source_ts.min() if start is None else start)
    sample_end = float(source_ts.max() if end is None else end)
    if sample_start > sample_end:
        raise ValueError("start must be less than or equal to end")

    # Endpoints are only known to a few ulps (epoch-scale stamps have ulps
    # of ~2e-7 s), so an absolute epsilon drops the last grid point there.
    slack = 4.0 * (np.spacing(abs(sample_start)) + np.spacing(abs(sample_end))) + sample_period * 1.0e-9
    count = int(np.floor((sample_end - sample_start + slack) / sample_period)) + 1
    grid = sample_start + np.arange(count, dtype=np.float64) * sample_period
    # The last point may overshoot `end` by float noise; keep it in range.
    return np.minimum(grid, sample_end)


def _linear_sample_validity(source_ts: np.ndarray, sample_ts: np.ndarray, tolerance: float | None) -> np.ndarray:
    """Samples inside the (sorted, finite) source range whose bracketing
    source gap is at most `tolerance` (exact hits are always valid)."""

    valid = (sample_ts >= source_ts[0]) & (sample_ts <= source_ts[-1])
    if tolerance is None or source_ts.size < 2:
        return valid
    right = np.clip(np.searchsorted(source_ts, sample_ts, side="left"), 1, source_ts.size - 1)
    exact = (source_ts[right] == sample_ts) | (source_ts[right - 1] == sample_ts)
    gap = source_ts[right] - source_ts[right - 1]
    return valid & (exact | (gap <= tolerance))


def _interpolate_topic_data(timestamps: np.ndarray, data: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    if target_timestamps.size == 0:
        return np.empty((0,) + data.shape[1:], dtype=np.result_type(data.dtype, np.float64))
    if timestamps.size == 0:
        return np.full((target_timestamps.size,) + data.shape[1:], np.nan, dtype=np.float64)
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("linear fixed-rate resampling requires numeric topic data")

    flat = np.asarray(data, dtype=np.float64).reshape((data.shape[0], -1))
    if flat.shape[1] == 0:
        return np.empty((target_timestamps.size,) + data.shape[1:], dtype=np.float64)
    interpolated = np.column_stack([
        np.interp(target_timestamps, timestamps, flat[:, dim])
        for dim in range(flat.shape[1])
    ])
    return interpolated.reshape((target_timestamps.size,) + data.shape[1:])
