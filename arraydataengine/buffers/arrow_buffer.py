"""Apache Arrow / Parquet persistent buffer backend.

Each topic is a directory of Parquet fragment files plus a ``manifest.json``:

    <group_uri>/<topic>/part-00000.parquet
    <group_uri>/<topic>/part-00001.parquet
    <group_uri>/<topic>/manifest.json

The manifest lists committed fragments. Temporary files and fragments absent
from that list are ignored after an interrupted write; replaying the source
replaces uncommitted fragments. Legacy stores without a fragment list reconcile
their message counts against Parquet metadata before resuming.

Appends stage in memory and flush a fragment when the staged payload reaches
``flush_bytes`` (or on close), so ingest memory stays bounded regardless of
dataset size. Reads stream through ``pyarrow.dataset`` with bounded readahead,
so scans of larger-than-memory topics stay bounded too. Row-group size is
derived from the first message's byte size (targeting ``row_group_bytes``),
which keeps single-message random access cheap for large payloads.

Tuning knobs (all exposed through ``DataBuffer(backend_options=...)``):

- ``flush_bytes`` (default 32 MB): staged bytes per topic before a fragment
  is written. Larger values mean fewer fragments and faster scans at the cost
  of ingest memory.
- ``row_group_bytes`` (default 16 MB): target Parquet row-group size. Smaller
  groups make random access finer-grained; larger groups scan faster.
- ``compression`` (default ``"zstd"``): Parquet codec (``"zstd"``,
  ``"snappy"``, ``"lz4"``, ``"none"``...).
- ``batch_readahead`` / ``fragment_readahead`` (default 1 each): scanner
  prefetch depth. Raise for throughput at the cost of scan memory.
- ``use_threads`` (default False): parallel scanning. Off by default so
  streamed batches preserve append order and memory stays bounded.

Reads are served from committed fragments plus rows still staged in memory,
so reading never forces a flush. Single-row and index-range reads open only the
fragments and row groups that hold the requested rows. A manifest is rewritten
only when a flush or a closed-flag change alters it, and a handle refuses to
write over a manifest that another writer changed after this handle read it.

Payloads have a fixed per-topic shape and dtype. Scalars, bool, datetime64,
complex, and fixed-width byte/str payloads are supported (non-numeric dtypes
are stored as raw bytes and viewed back on read); object and structured dtypes
and zero-length payloads are rejected on append.

Unlike the TileDB backend, fragments are immutable: ``__setitem__`` is not
supported. Use ``backend="tiledb"`` when in-place cell updates are needed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .common import (
    apply_index_range,
    check_message_schema,
    check_persistable_data,
    decode_frame_id,
    encode_frame_id,
    encode_name,
    index_ranges_exhausted,
    native_message_data,
    raise_collected,
    resolve_topic_path,
    spatial_bounds_for_data,
    spatial_overlap_mask,
)

_logger = logging.getLogger(__name__)

DEFAULT_FLUSH_BYTES = 32 * 1024 * 1024
DEFAULT_ROW_GROUP_BYTES = 16 * 1024 * 1024
MANIFEST_NAME = "manifest.json"
# Arrow tensors hold integers and floats; other fixed-width dtypes are stored
# as raw bytes with a trailing itemsize axis and viewed back on read.
_TENSOR_KINDS = "iuf"


def _fsync_directory(path: Path) -> None:
    """Persist a rename inside `path` (POSIX); skipped where directories cannot be opened."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def _atomic_output(path: Path):
    """Publish a complete file using a same-directory rename."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _native_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    return dtype if dtype.byteorder in ("=", "|") else dtype.newbyteorder("=")


def _stores_raw_bytes(dtype: np.dtype) -> bool:
    return dtype.kind not in _TENSOR_KINDS


def _encode_tensor(data: np.ndarray, shape: tuple, dtype: np.dtype):
    import pyarrow as pa

    rows = data.shape[0]
    if _stores_raw_bytes(dtype):
        data = np.ascontiguousarray(data).view(np.uint8).reshape((rows, *shape, dtype.itemsize))
    elif not shape:
        data = data.reshape((rows, 1))
    return pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(data))


def _decode_tensor(column, schema) -> np.ndarray:
    """NumPy rows of a tensor column, restored to the topic's shape and dtype."""
    import pyarrow as pa

    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    stored_shape = None
    if isinstance(array, pa.ExtensionArray):
        stored_shape = tuple(array.type.shape)
        array = array.storage
    rows = len(array)
    values = array.flatten().to_numpy(zero_copy_only=False)
    if schema is None:
        return values.reshape((rows, *(stored_shape or (array.type.list_size,))))
    shape, dtype = schema
    if _stores_raw_bytes(dtype):
        values = np.ascontiguousarray(values).view(dtype)
    elif values.dtype != dtype:
        values = values.astype(dtype)
    return values.reshape((rows, *shape))


def _new_stage() -> dict:
    return {
        "ts": [], "name": [], "frame_id": [], "spatial_valid": [], "spatial_min": [],
        "spatial_max": [], "data": [], "bytes": 0, "table": None,
    }


class ArrowBuffer:
    def __init__(
        self,
        data_source,
        init_source,
        group_uri,
        axis: str = "",
        topics=None,
        flush_bytes: int = DEFAULT_FLUSH_BYTES,
        row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
        compression: str = "zstd",
        batch_readahead: int = 1,
        fragment_readahead: int = 1,
        use_threads: bool = False,
    ):
        import pyarrow  # noqa: F401  (fail fast with a clear error)

        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = str(group_uri)
        self._axis = axis
        self.topics = [] if topics is None else list(topics)

        self.flush_bytes = int(flush_bytes)
        self.row_group_bytes = int(row_group_bytes)
        self.compression = None if str(compression).lower() == "none" else compression
        self.batch_readahead = int(batch_readahead)
        self.fragment_readahead = int(fragment_readahead)
        self.use_threads = bool(use_threads)

        self.counters: dict[str, int] = {}
        self.frame_ids: dict[str, str | None] = {}
        self.names: dict[str, bytes] = {}
        self.closed_topics: dict[str, bool] = {}
        self.timestamps: dict = {}
        self._resume_seen: dict[str, int] = {}
        self._persisted: dict[str, int] = {}
        self._fragments: dict[str, list[Path]] = {}
        self._fragment_rows: dict[str, list[int]] = {}
        self._staged: dict[str, dict] = {}
        self._schemas: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self._topic_paths: dict[str, str] = {}
        # Manifest content this handle last read from or wrote to disk.
        self._manifests: dict[str, dict] = {}
        self._source_counts: dict[str, int | None] = {}

        self.read_only = data_source is None or init_source is None
        try:
            self._hydrate_existing_topics()
        except Exception:
            # A failed open must not rewrite partially hydrated metadata on GC.
            self.read_only = True
            raise

    # -- properties ----------------------------------------------------------

    @property
    def msg_len(self) -> dict:
        """Messages appended per topic (persisted + staged)."""
        return dict(self.counters)

    # -- topic layout --------------------------------------------------------

    def _topic_dir(self, topic: str) -> Path:
        return Path(resolve_topic_path(self.group_uri, topic, self._topic_paths))

    def _manifest_path(self, topic: str) -> Path:
        return self._topic_dir(topic) / MANIFEST_NAME

    def _register_topic(self, topic: str) -> None:
        self.counters.setdefault(topic, 0)
        self._persisted.setdefault(topic, 0)
        self._fragments.setdefault(topic, [])
        self._fragment_rows.setdefault(topic, [])
        if topic not in self.topics:
            self.topics.append(topic)

    def _schema(self, topic: str):
        return self._schemas.get(topic)

    # -- hydration / resume --------------------------------------------------

    def _hydrate_existing_topics(self) -> None:
        root = Path(self.group_uri)
        if not root.is_dir():
            return
        for entry in sorted(root.iterdir()):
            manifest_path = entry / MANIFEST_NAME
            if not entry.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unreadable Arrow manifest {manifest_path}") from exc
            topic = manifest.get("topic", entry.name)
            self._topic_paths[topic] = str(entry)
            committed = manifest.get("fragments")
            if committed is None:
                # Legacy manifests predate explicit commit lists. Reconcile
                # their counters with complete fragments before replaying.
                fragments = sorted(entry.glob("part-*.parquet"))
            else:
                if not isinstance(committed, list) or any(
                    not isinstance(name, str) or Path(name).name != name
                    or not name.startswith("part-") or not name.endswith(".parquet")
                    for name in committed
                ) or len(set(committed)) != len(committed):
                    raise ValueError(f"Invalid fragment list in {manifest_path}")
                fragments = [entry / name for name in committed]
            import pyarrow.parquet as pq

            rows = [pq.read_metadata(path).num_rows for path in fragments]
            count = sum(rows)
            if committed is not None and count != int(manifest.get("count", 0)):
                raise ValueError(f"Fragment count disagrees with {manifest_path}")
            self.counters[topic] = count
            self._persisted[topic] = count
            self._fragments[topic] = fragments
            self._fragment_rows[topic] = rows
            self._manifests[topic] = manifest
            self.closed_topics[topic] = (
                bool(manifest.get("closed", False)) and count == int(manifest.get("count", 0))
            )
            self.frame_ids[topic] = manifest.get("frame_id")
            name = manifest.get("name")
            if name is not None:
                self.names[topic] = name.encode() if isinstance(name, str) else bytes(name)
            shape = manifest.get("shape")
            dtype = manifest.get("dtype")
            # An empty topic's schema is not binding (older versions wrote a
            # placeholder); the first append defines it.
            if count and shape is not None and dtype is not None:
                self._schemas[topic] = (tuple(int(v) for v in shape), _native_dtype(dtype))
            if topic not in self.topics:
                self.topics.append(topic)

    def _should_skip_replayed_message(self, topic: str) -> bool:
        existing_count = self.counters.get(topic, 0)
        seen = self._resume_seen.get(topic, 0)
        self._resume_seen[topic] = seen + 1
        return seen < existing_count

    def _source_count(self, topic: str) -> int | None:
        """The source's total message count for `topic`, when it reports one."""
        if topic not in self._source_counts:
            get_count = getattr(self.init_source, "get_count", None)
            count = None
            if callable(get_count):
                try:
                    count = int(get_count(topic))
                except Exception:
                    count = None
            self._source_counts[topic] = count
        return self._source_counts[topic]

    def _topic_complete(self, topic: str) -> bool | None:
        """Whether `topic` holds every source message (None when unknown)."""
        expected = self._source_count(topic)
        if expected is None:
            return None
        return self.counters.get(topic, 0) >= expected

    def reset(self) -> None:
        self.close()
        self.counters = {}
        self.frame_ids = {}
        self.names = {}
        self.closed_topics = {}
        self.timestamps = {}
        self._resume_seen = {}
        self._persisted = {}
        self._fragments = {}
        self._fragment_rows = {}
        self._staged = {}
        self._schemas = {}
        self._topic_paths = {}
        self._manifests = {}
        self._source_counts = {}
        self._hydrate_existing_topics()

    # -- write path ----------------------------------------------------------

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            topic = msg["topic"]

            if topic not in self.counters:
                self._register_topic(topic)

            if self._should_skip_replayed_message(topic):
                if topic == self._axis:
                    break
                continue

            if self.closed_topics.get(topic, False):
                if self._topic_complete(topic) is False:
                    # Older versions marked partially ingested topics closed
                    # when a `with` block exited; keep resuming those.
                    self.closed_topics[topic] = False
                else:
                    if topic == self._axis:
                        break
                    continue

            self.append_buffer(msg)

            if topic == self._axis:
                break

    def append_buffer(self, msg: dict) -> None:
        topic = msg["topic"]
        data = native_message_data(msg["data"])

        # Validate everything before touching any state so a rejected message
        # cannot poison the stage (and every later flush of this topic).
        schema = self._schemas.get(topic)
        if schema is None:
            check_persistable_data(topic, data)
        else:
            check_message_schema(topic, data, *schema)
        timestamp = float(msg["timestamp"])
        name = encode_name(msg.get("name", topic))
        frame_id = encode_frame_id(msg.get("frame_id"))
        valid, mins, maxs = spatial_bounds_for_data(data)

        # An explicit append is a write intent (mirrors the TileDB backend).
        self.read_only = False
        self._register_topic(topic)
        if schema is None:
            self._schemas[topic] = (tuple(data.shape), data.dtype)

        self.names[topic] = name
        self._record_frame_id(msg)

        staged = self._staged.get(topic)
        if staged is None:
            staged = self._staged[topic] = _new_stage()
        staged["ts"].append(timestamp)
        staged["name"].append(name)
        staged["frame_id"].append(frame_id)
        staged["spatial_valid"].append(valid)
        staged["spatial_min"].append(mins)
        staged["spatial_max"].append(maxs)
        # Own staged payloads: sources may immediately reuse their arrays.
        staged["data"].append(np.array(data, copy=True))
        staged["bytes"] += data.nbytes
        self.counters[topic] += 1

        if staged["bytes"] >= self.flush_bytes:
            self._flush_topic(topic)

    def _record_frame_id(self, msg: dict) -> None:
        topic = msg["topic"]
        frame_id = decode_frame_id(msg.get("frame_id")) or None
        if self.counters.get(topic, 0) == 0 or topic not in self.frame_ids:
            self.frame_ids[topic] = frame_id
        elif self.frame_ids[topic] != frame_id:
            self.frame_ids[topic] = None

    def _build_table(self, topic: str, staged: dict, start: int, stop: int):
        import pyarrow as pa

        shape, dtype = self._schemas[topic]
        data = np.stack(staged["data"][start:stop])
        mins = np.asarray(staged["spatial_min"][start:stop], dtype=np.float64).reshape((-1, 3))
        maxs = np.asarray(staged["spatial_max"][start:stop], dtype=np.float64).reshape((-1, 3))
        columns: dict = {
            "ts": pa.array(staged["ts"][start:stop], type=pa.float64()),
            "name": pa.array(staged["name"][start:stop], type=pa.binary()),
            "frame_id": pa.array(staged["frame_id"][start:stop], type=pa.binary()),
            "spatial_valid": pa.array(staged["spatial_valid"][start:stop], type=pa.bool_()),
        }
        for dim in range(3):
            columns[f"spatial_min_{dim}"] = pa.array(mins[:, dim], type=pa.float64())
            columns[f"spatial_max_{dim}"] = pa.array(maxs[:, dim], type=pa.float64())
        columns["data"] = _encode_tensor(data, shape, dtype)
        return pa.table(columns)

    def _staged_table(self, topic: str):
        """Arrow table of the rows staged for `topic`, built incrementally."""
        staged = self._staged.get(topic)
        rows = len(staged["ts"]) if staged else 0
        if rows == 0:
            return None
        table = staged["table"]
        built = 0 if table is None else table.num_rows
        if built < rows:
            import pyarrow as pa

            part = self._build_table(topic, staged, built, rows)
            table = part if table is None else pa.concat_tables([table, part])
            staged["table"] = table
            # The table now owns these payloads; drop the NumPy copies.
            staged["data"][built:rows] = [None] * (rows - built)
        return table

    def _flush_topic(self, topic: str) -> None:
        staged = self._staged.get(topic)
        if not staged or not staged["ts"]:
            return

        import pyarrow.parquet as pq

        # Never write over fragments another writer committed after we read
        # the manifest: our next fragment name could collide with theirs.
        self._verify_manifest(topic)
        table = self._staged_table(topic)
        rows = table.num_rows
        shape, dtype = self._schemas[topic]
        message_bytes = max(1, int(np.prod(shape, dtype=np.int64)) * dtype.itemsize)
        rows_per_group = max(1, self.row_group_bytes // message_bytes)

        topic_dir = self._topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        fragment_path = topic_dir / f"part-{len(self._fragments[topic]):05d}.parquet"
        # Establish a commit list before writing, including when upgrading a
        # legacy store. An interrupted first write retains the topic's path.
        self._write_manifest(topic)
        with _atomic_output(fragment_path) as temporary:
            pq.write_table(
                table,
                temporary,
                row_group_size=int(rows_per_group),
                compression=self.compression or "none",
            )
        self._fragments[topic].append(fragment_path)
        self._fragment_rows[topic].append(rows)
        self._persisted[topic] += rows
        try:
            self._write_manifest(topic)
        except Exception:
            self._fragments[topic].pop()
            self._fragment_rows[topic].pop()
            self._persisted[topic] -= rows
            # Keep staged rows for retry; the uncommitted file is ignored on
            # reopen and atomically replaced by the next successful flush.
            raise
        self._staged[topic] = _new_stage()

    def _manifest_content(self, topic: str) -> dict:
        schema = self._schemas.get(topic)
        return {
            "topic": topic,
            "count": self._persisted.get(topic, 0),
            "fragments": [path.name for path in self._fragments.get(topic, [])],
            "closed": bool(self.closed_topics.get(topic, False)),
            "name": self.names.get(topic, b"").decode(errors="replace"),
            "frame_id": self.frame_ids.get(topic),
            "shape": None if schema is None else list(schema[0]),
            "dtype": None if schema is None else str(schema[1]),
        }

    def _verify_manifest(self, topic: str) -> None:
        path = self._manifest_path(topic)
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            on_disk = None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable Arrow manifest {path}") from exc
        if on_disk != self._manifests.get(topic):
            raise RuntimeError(
                f"Arrow manifest {path} was changed by another writer since this handle read it; "
                f"reopen the store before writing topic {topic!r}"
            )

    def _write_manifest(self, topic: str, closed: bool | None = None) -> None:
        if closed is not None:
            self.closed_topics[topic] = bool(closed)
        manifest = self._manifest_content(topic)
        if manifest == self._manifests.get(topic):
            return
        self._verify_manifest(topic)
        topic_dir = self._topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        with _atomic_output(self._manifest_path(topic)) as temporary:
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
        self._manifests[topic] = manifest

    def close_topic(self, topic: str, closed: bool | None = None) -> None:
        if self.read_only:
            return
        if topic not in self.counters:
            return
        self._flush_topic(topic)
        self._write_manifest(topic, closed=closed)

    def _close_topics(self, closed_by_topic: dict) -> None:
        # Attempt every topic so one failure cannot strand other topics'
        # staged rows, then report all failures together.
        errors = []
        for topic, closed in closed_by_topic.items():
            try:
                self.close_topic(topic, closed)
            except Exception as exc:
                exc.add_note(f"while closing Arrow topic {topic!r}")
                errors.append(exc)
        raise_collected(errors, "Arrow")

    def close(self, closed: bool | None = None) -> None:
        self._close_topics({topic: closed for topic in list(self.counters)})

    def close_completed(self) -> None:
        """Close, marking closed only topics that hold every source message."""
        self._close_topics({
            topic: True if self._topic_complete(topic) else None for topic in list(self.counters)
        })

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- read path -----------------------------------------------------------

    def _dataset(self, topic: str):
        import pyarrow.dataset as ds

        fragments = self._fragments.get(topic, [])
        if not fragments:
            return None
        return ds.dataset([str(path) for path in fragments], format="parquet")

    def _scanner(self, dataset, **kwargs):
        kwargs.setdefault("batch_readahead", self.batch_readahead)
        kwargs.setdefault("fragment_readahead", self.fragment_readahead)
        kwargs.setdefault("use_threads", self.use_threads)
        return dataset.scanner(**kwargs)

    def _iter_batches(self, topic: str, batch_size: int | None = None, scan_filter=None):
        """Committed fragments then staged rows, as record batches in append order."""
        import pyarrow.dataset as ds

        # Snapshot both sources now: a flush during iteration moves rows.
        dataset = self._dataset(topic)
        staged = self._staged_table(topic)
        options = {"filter": scan_filter}
        if batch_size is not None:
            options["batch_size"] = int(batch_size)

        def batches():
            if dataset is not None:
                yield from self._scanner(dataset, **options).to_batches()
            if staged is not None:
                yield from ds.dataset(staged).scanner(use_threads=False, **options).to_batches()

        return batches()

    def _take_fragment(self, path: Path, local: np.ndarray, columns=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        try:
            metadata = parquet.metadata
            group_rows = np.array(
                [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)], dtype=np.int64
            )
            group_starts = np.concatenate(([0], np.cumsum(group_rows)))
            group_ids = np.searchsorted(group_starts, local, side="right") - 1
            groups = np.unique(group_ids)
            table = parquet.read_row_groups(
                [int(group) for group in groups], columns=columns, use_threads=self.use_threads
            )
        finally:
            parquet.close()
        selected_starts = np.concatenate(([0], np.cumsum(group_rows[groups])))
        positions = local - group_starts[group_ids] + selected_starts[np.searchsorted(groups, group_ids)]
        if int(positions[-1]) - int(positions[0]) + 1 == positions.size:
            return table.slice(int(positions[0]), int(positions.size))
        return table.take(pa.array(positions))

    def _take(self, topic: str, indices, columns=None):
        """Rows at `indices` (each in ``[0, count)``) reading only the fragments,
        row groups, and staged rows that hold them."""
        import pyarrow as pa

        indices = np.asarray(indices, dtype=np.int64)
        order = None
        if indices.size > 1 and np.any(np.diff(indices) < 0):
            order = np.argsort(indices, kind="stable")
            indices = indices[order]

        persisted = self._persisted.get(topic, 0)
        split = int(np.searchsorted(indices, persisted))
        tables = []
        if split:
            starts = np.concatenate(([0], np.cumsum(self._fragment_rows[topic], dtype=np.int64)))
            fragment_ids = np.searchsorted(starts, indices[:split], side="right") - 1
            for fragment in np.unique(fragment_ids):
                local = indices[:split][fragment_ids == fragment] - starts[fragment]
                tables.append(self._take_fragment(self._fragments[topic][int(fragment)], local, columns))
        if split < indices.size:
            staged = self._staged_table(topic).take(pa.array(indices[split:] - persisted))
            tables.append(staged.select(columns) if columns else staged)

        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
        if order is not None:
            table = table.take(pa.array(np.argsort(order)))
        return table

    def _empty_topic_result(self, topic: str) -> dict:
        shape, dtype = self._schemas.get(topic, ((), np.dtype(np.float64)))
        result = {
            "id": np.array([], dtype=object),
            "name": np.array([], dtype=object),
            "ts": np.array([], dtype=np.float64),
            "data": np.empty((0, *shape), dtype=dtype),
            "frame_ids": np.array([], dtype=object),
            "topic": topic,
            "source_uri": str(self._topic_dir(topic)),
        }
        if self.frame_ids.get(topic) is not None:
            result["frame_id"] = self.frame_ids[topic]
        return result

    def _table_to_result(self, topic: str, table, copy: bool = True) -> dict:
        names = np.array(table["name"].to_pylist(), dtype=object)
        data = _decode_tensor(table["data"], self._schema(topic))
        if copy or not data.flags.writeable:
            data = data.copy()
        if "frame_id" in table.column_names:
            frames = [decode_frame_id(frame) or None for frame in table["frame_id"].to_pylist()]
        else:
            frames = [None] * table.num_rows
        result = {
            "id": names,
            "name": names.copy() if copy else names,
            "ts": np.array(table["ts"], dtype=np.float64),
            "data": data,
            "frame_ids": np.array(frames, dtype=object),
            "topic": topic,
            "source_uri": str(self._topic_dir(topic)),
        }
        if self.frame_ids.get(topic) is not None:
            result["frame_id"] = self.frame_ids[topic]
        return result

    def get_index_range(
        self,
        axis: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        copy: bool = True,
    ) -> dict:
        if axis not in self.counters:
            return self._empty_topic_result(axis)

        range_start, range_stop, range_step = slice(start, stop, step).indices(self.counters[axis])
        if range_step < 1:
            raise ValueError("step must be positive")
        indices = np.arange(range_start, range_stop, range_step, dtype=np.int64)
        if indices.size == 0:
            return self._empty_topic_result(axis)
        return self._table_to_result(axis, self._take(axis, indices), copy=copy)

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        import pyarrow as pa
        import pyarrow.dataset as ds

        if axis not in self.counters:
            return self._empty_topic_result(axis)
        scan_filter = (ds.field("ts") >= float(start)) & (ds.field("ts") <= float(end))
        tables = [
            pa.Table.from_batches([batch])
            for batch in self._iter_batches(axis, scan_filter=scan_filter)
            if batch.num_rows
        ]
        if not tables:
            return self._empty_topic_result(axis)
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
        return self._table_to_result(axis, table)

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if not self.counters.get(axis, 0):
            return self._empty_topic_result(axis)
        staged = self._staged.get(axis)
        if staged and staged["ts"]:
            end = float(staged["ts"][-1])
        else:
            last = self._take(axis, [self._persisted[axis] - 1], columns=["ts"])
            end = float(last["ts"][0].as_py())
        return self.get_time_range(axis, end - float(seconds), end)

    def get_buffer(self, copy: bool = True) -> dict:
        return {
            topic: self.get_index_range(topic, 0, self.counters.get(topic, 0), copy=copy)
            for topic in self.counters
        }

    def get_topic(self, topic: str, copy: bool = True) -> dict:
        return self.get_index_range(topic, 0, self.counters.get(topic, 0), copy=copy)

    def _operation_mask(self, batch, operations, counters) -> np.ndarray:
        rows = batch.num_rows
        keep = np.ones(rows, dtype=bool)
        ts = None
        for operation_index, operation in enumerate(operations):
            if not keep.any():
                break
            if operation.kind == "time_range":
                op_start, op_end = operation.args
                if ts is None:
                    ts = np.asarray(batch["ts"], dtype=np.float64)
                if operation.kwargs.get("inclusive", True):
                    keep &= (ts >= op_start) & (ts <= op_end)
                else:
                    keep &= (ts > op_start) & (ts < op_end)
            elif operation.kind == "index_range":
                keep, counters[operation_index] = apply_index_range(
                    keep, counters[operation_index], *operation.args
                )
            elif operation.kind == "frame_id":
                targets = operation.args[0]
                if "frame_id" in batch.column_names:
                    frames = batch["frame_id"].to_pylist()
                else:
                    frames = [None] * rows
                keep &= np.fromiter(
                    ((decode_frame_id(frame) or None) in targets for frame in frames), dtype=bool, count=rows
                )
            elif operation.kind == "spatial_bounds":
                min_bound, max_bound = operation.args
                columns = operation.kwargs["columns"]
                # Bounds prune conservatively; the exact filter runs afterwards.
                if all(column < 3 for column in columns) and "spatial_valid" in batch.column_names:
                    keep &= spatial_overlap_mask(
                        np.asarray(batch["spatial_valid"], dtype=bool),
                        [batch[f"spatial_min_{column}"] for column in columns],
                        [batch[f"spatial_max_{column}"] for column in columns],
                        min_bound,
                        max_bound,
                    )
            else:
                raise ValueError(f"unsupported pushdown operation: {operation.kind}")
        return keep

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False, operations=()):
        import pyarrow as pa
        import pyarrow.dataset as ds

        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not self.counters.get(axis, 0):
            return

        operations = tuple(operations or ())
        if operations and operations[0].kind == "index_range":
            # Every row reaches a leading index range, so it maps directly to
            # row positions: read only those rows instead of scanning.
            indices = np.arange(*slice(*operations[0].args).indices(self.counters[axis]), dtype=np.int64)
            operations = operations[1:]
            batches = (
                self._take(axis, indices[offset:offset + chunk_size])
                for offset in range(0, indices.size, chunk_size)
            )
        else:
            scan_filter = None
            if operations and operations[0].kind == "time_range":
                start, end = operations[0].args
                if operations[0].kwargs.get("inclusive", True):
                    scan_filter = (ds.field("ts") >= float(start)) & (ds.field("ts") <= float(end))
                else:
                    scan_filter = (ds.field("ts") > float(start)) & (ds.field("ts") < float(end))
                operations = operations[1:]
            batches = self._iter_batches(axis, chunk_size, scan_filter)

        counters = [0] * len(operations)
        for batch in batches:
            if batch.num_rows == 0:
                continue
            keep = self._operation_mask(batch, operations, counters)
            if keep.any():
                selected = batch if keep.all() else batch.filter(pa.array(keep))
                yield self._table_to_result(axis, selected, copy=copy)
            if index_ranges_exhausted(operations, counters):
                break

    # -- subscripts ----------------------------------------------------------

    def __getitem__(self, subscript):
        topic = self._axis
        if topic not in self.counters:
            return None
        count = self.counters[topic]

        if isinstance(subscript, slice):
            indices = np.arange(*subscript.indices(count), dtype=np.int64)
        elif isinstance(subscript, (int, np.integer)):
            index = int(subscript)
            if index < 0:
                index += count
            if not 0 <= index < count:
                raise IndexError(f"index {int(subscript)} is out of range for topic {topic!r} with {count} messages")
            indices = np.array([index], dtype=np.int64)
        else:
            raise TypeError(f"unsupported subscript type: {type(subscript).__name__}")

        if indices.size == 0:
            shape, dtype = self._schemas.get(topic, ((), np.dtype(np.float64)))
            return np.squeeze(np.empty((0, *shape), dtype=dtype))
        data = _decode_tensor(self._take(topic, indices, columns=["data"])["data"], self._schema(topic))
        return np.squeeze(data.copy())

    def __setitem__(self, subscript, newval):
        raise NotImplementedError(
            "the arrow backend stores immutable Parquet fragments and does not "
            "support in-place writes; use DataBuffer(backend='tiledb') for that"
        )
