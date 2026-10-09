"""Apache Arrow / Parquet persistent buffer backend.

Each topic is a directory of Parquet fragment files plus a ``manifest.json``:

    <group_uri>/topic-<escaped topic>.data/part-00000.parquet
    <group_uri>/topic-<escaped topic>.data/part-00001.parquet
    <group_uri>/topic-<escaped topic>.data/manifest.json

The manifest lists committed fragments. Temporary files and fragments absent
from that list are ignored after an interrupted write; replaying the source
replaces uncommitted fragments, and a writable open removes temporary files
older than a few minutes. Legacy stores without a fragment list (0.3.1 used
``<group_uri>/<topic>/``) reconcile their message counts against the longest
readable run of Parquet fragments before resuming. Only directories named like
topic directories, or holding a manifest with the Arrow keys, are treated as
topics; other directories are never adopted or rewritten.

Appends stage in memory and flush a fragment when the staged rows reach
``flush_bytes`` (payload plus an estimated ``row_overhead_bytes`` of
bookkeeping per row) or on close, so ingest memory stays bounded regardless of
dataset size. Reads stream through ``pyarrow.dataset`` with bounded readahead,
so scans of larger-than-memory topics stay bounded too. Row-group size is
derived from the first message's byte size (targeting ``row_group_bytes``),
which keeps single-message random access cheap for large payloads.

Tuning knobs (all exposed through ``DataBuffer(backend_options=...)``):

- ``flush_bytes`` (default 32 MB): staged bytes per topic before a fragment
  is written. Larger values mean fewer fragments and faster scans at the cost
  of ingest memory.
- ``row_overhead_bytes`` (default 256): staging bookkeeping counted per row.
- ``row_group_bytes`` (default 16 MB): target Parquet row-group size. Smaller
  groups make random access finer-grained; larger groups scan faster.
- ``compression`` (default ``"zstd"``): Parquet codec (``"zstd"``,
  ``"snappy"``, ``"lz4"``, ``"none"``...).
- ``batch_readahead`` / ``fragment_readahead`` (default 1 each): scanner
  prefetch depth. Raise for throughput at the cost of scan memory.
- ``use_threads`` (default False): parallel scanning. Off by default so
  streamed batches preserve append order and memory stays bounded.

Reads are served from committed fragments plus rows still staged in memory,
so reading never forces a flush. Single-row and sparse index-range reads open
only the fragments and row groups that hold the requested rows (the last
decoded row groups are reused, so ``buf[i]`` loops stay cheap); dense ranges
are scanned. A manifest is rewritten only when a flush or a closed-flag change
alters it; a handle refuses to write over a manifest that another writer
changed after this handle read it, and refuses to create a topic another
writer created meanwhile.

Payloads have a fixed per-topic shape and dtype; safe widening casts (e.g.
``float32`` into a ``float64`` topic) are applied on append. Scalars, bool,
datetime64, complex, and fixed-width byte/str payloads are supported
(non-numeric dtypes are stored as raw bytes and viewed back on read); object
and structured dtypes and zero-length payloads are rejected on append.
Variable-size topics (messages with ``point_count``, or listed in
``ragged_topics``) are stored zero-padded to their capacity with an ``int64``
``length`` column; reads return a ``lengths`` array for them.

Unlike the TileDB backend, fragments are immutable: ``__setitem__`` is not
supported. Use ``backend="tiledb"`` when in-place cell updates are needed.
"""

from __future__ import annotations

import array
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .common import (
    apply_index_range,
    check_persistable_data,
    conform_message,
    decode_frame_id,
    encode_frame_id,
    encode_name,
    index_ranges_exhausted,
    is_arrow_topic_dir_name,
    is_tiledb_store,
    message_is_ragged,
    native_message_data,
    new_ragged_schema,
    normalize_ragged_topics,
    pad_rows,
    parse_arrow_manifest,
    ragged_capacity,
    raise_collected,
    resolve_topic_path,
    source_topic_count,
    spatial_bounds_for_data,
    spatial_overlap_mask,
)

_logger = logging.getLogger(__name__)

DEFAULT_FLUSH_BYTES = 32 * 1024 * 1024
DEFAULT_ROW_GROUP_BYTES = 16 * 1024 * 1024
# Python bookkeeping per staged row (columns, name/frame slots, the payload
# array header), counted toward flush_bytes so many small messages cannot
# stage far more memory than the payload suggests.
DEFAULT_ROW_OVERHEAD_BYTES = 256
MANIFEST_NAME = "manifest.json"
# Arrow tensors hold integers and floats; other fixed-width dtypes are stored
# as raw bytes with a trailing itemsize axis and viewed back on read.
_TENSOR_KINDS = "iuf"
# Temporary files a crashed writer left behind are removed by writable opens
# once they have not been modified for this long (an active write keeps
# touching its file).
_STALE_TEMP_SECONDS = 300
# Decoded row groups kept for repeated point reads (``buf[i]`` loops, chunked
# index ranges).
_ROW_GROUP_CACHE_SIZE = 2
# A leading index range selecting at least this share of a topic is scanned
# instead of read row group by row group.
_SCAN_FRACTION = 0.5

_UMASK: int | None = None


def _process_umask() -> int:
    """The process umask (read once; files are published with its permissions)."""
    global _UMASK
    if _UMASK is None:
        umask = None
        try:
            with open("/proc/self/status", encoding="ascii") as status:
                for line in status:
                    if line.startswith("Umask:"):
                        umask = int(line.split()[1], 8)
                        break
        except (OSError, ValueError):
            umask = None
        if umask is None:
            umask = os.umask(0o022)
            os.umask(umask)
        _UMASK = umask
    return _UMASK


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
        # mkstemp creates 0600 files; publish with the usual umask permissions.
        os.chmod(temporary, 0o666 & ~_process_umask())
        # Windows refuses fsync on read-only handles.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_stale_temporaries(directory: Path) -> None:
    cutoff = time.time() - _STALE_TEMP_SECONDS
    for path in directory.glob(".*.tmp"):
        if not (path.name.startswith(f".{MANIFEST_NAME}.") or path.name.startswith(".part-")):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _native_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    return dtype if dtype.byteorder in ("=", "|") else dtype.newbyteorder("=")


def _stores_raw_bytes(dtype: np.dtype) -> bool:
    return dtype.kind not in _TENSOR_KINDS


def _detect_raw_encoding(arrow_type, shape: tuple, dtype: np.dtype) -> bool:
    """Whether a stored ``data`` column holds raw bytes (else a typed tensor).

    Stores written before raw-byte encoding kept bool payloads as bool tensors;
    appends must keep that encoding or later reads cannot merge fragments.
    """
    import pyarrow as pa

    if not isinstance(arrow_type, pa.BaseExtensionType):
        return _stores_raw_bytes(dtype)
    stored_shape = tuple(getattr(arrow_type, "shape", ()) or ())
    value_type = getattr(arrow_type.storage_type, "value_type", None)
    return (
        _stores_raw_bytes(dtype)
        and stored_shape == (*shape, dtype.itemsize)
        and value_type == pa.uint8()
    )


def _encode_tensor(data: np.ndarray, shape: tuple, dtype: np.dtype, raw: bool | None = None):
    import pyarrow as pa

    rows = data.shape[0]
    if raw is None:
        raw = _stores_raw_bytes(dtype)
    if raw:
        data = np.ascontiguousarray(data).view(np.uint8).reshape((rows, *shape, dtype.itemsize))
    elif not shape:
        data = data.reshape((rows, 1))
    return pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(data))


def _decode_tensor(column, schema) -> np.ndarray:
    """NumPy rows of a tensor column, restored to the topic's shape and dtype."""
    import pyarrow as pa

    array_ = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    stored_shape = None
    if isinstance(array_, pa.ExtensionArray):
        stored_shape = tuple(array_.type.shape)
        array_ = array_.storage
    rows = len(array_)
    values = array_.flatten().to_numpy(zero_copy_only=False)
    if schema is None:
        return values.reshape((rows, *(stored_shape or (array_.type.list_size,))))
    shape, dtype = schema
    raw = (
        _stores_raw_bytes(dtype)
        and values.dtype == np.uint8
        and stored_shape == (*shape, dtype.itemsize)
    )
    if raw:
        values = np.ascontiguousarray(values).view(dtype)
    elif values.dtype != dtype:
        values = values.astype(dtype)
    return values.reshape((rows, *shape))


def _new_stage() -> dict:
    return {
        "ts": array.array("d"), "name": [], "frame_id": [], "spatial_valid": array.array("b"),
        "spatial_min": array.array("d"), "spatial_max": array.array("d"), "data": [], "length": [],
        "bytes": 0, "table": None,
    }


def _stage_rows(stage: dict) -> int:
    return len(stage["ts"])


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
        row_overhead_bytes: int = DEFAULT_ROW_OVERHEAD_BYTES,
        ragged_topics=None,
        max_points: int | None = None,
    ):
        import pyarrow  # noqa: F401  (fail fast with a clear error)

        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = str(group_uri)
        self._axis = axis
        self.topics = [] if topics is None else list(topics)

        self.flush_bytes = int(flush_bytes)
        self.row_overhead_bytes = int(row_overhead_bytes)
        self.row_group_bytes = int(row_group_bytes)
        self.compression = None if str(compression).lower() == "none" else compression
        self.batch_readahead = int(batch_readahead)
        self.fragment_readahead = int(fragment_readahead)
        self.use_threads = bool(use_threads)
        self.ragged_topics = normalize_ragged_topics(ragged_topics)
        self.max_points = max_points

        self._init_state()
        self.read_only = data_source is None or init_source is None
        try:
            if not self.read_only:
                self._check_writable_location()
            self._hydrate_existing_topics()
        except Exception:
            # A failed open must not rewrite partially hydrated metadata on GC.
            self.read_only = True
            raise

    def _init_state(self) -> None:
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
        self._ragged: dict[str, bool] = {}
        self._raw: dict[str, bool] = {}
        self._topic_paths: dict[str, str] = {}
        # Manifest content this handle last read from or wrote to disk.
        self._manifests: dict[str, dict] = {}
        self._source_counts: dict = {}
        self._group_rows: dict[str, np.ndarray] = {}
        self._row_group_cache: OrderedDict = OrderedDict()
        self._interned: dict[bytes, bytes] = {}

    # -- properties ----------------------------------------------------------

    @property
    def msg_len(self) -> dict:
        """Messages appended per topic (persisted + staged)."""
        return dict(self.counters)

    def get_size(self, topic: str) -> int:
        """Messages stored (persisted + staged) for `topic`."""
        return int(self.counters.get(topic, 0))

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

    def _check_writable_location(self) -> None:
        if is_tiledb_store(self.group_uri):
            raise ValueError(
                f"{self.group_uri} holds a TileDB store; open it with backend='tiledb' "
                "instead of writing Arrow topics into it"
            )

    # -- hydration / resume --------------------------------------------------

    def _read_manifest(self, entry: Path) -> dict | None:
        """The Arrow manifest of directory `entry`, or None when it is not a topic."""
        manifest_path = entry / MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        topic_named = is_arrow_topic_dir_name(entry.name)
        try:
            manifest = parse_arrow_manifest(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            if topic_named:
                raise ValueError(f"Unreadable Arrow manifest {manifest_path}") from exc
            return None
        if manifest is None and topic_named:
            raise ValueError(f"Unreadable Arrow manifest {manifest_path}")
        # Directories not named like topics (0.3.1 used the raw topic name)
        # count only when their manifest has the Arrow keys: a web project's
        # public/manifest.json is not a topic and is never rewritten.
        return manifest

    @staticmethod
    def _legacy_fragments(entry: Path) -> tuple[list[Path], list]:
        """Longest readable run part-00000, part-00001, ... of a legacy topic."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        paths = sorted(entry.glob("part-*.parquet"))
        fragments, metadata = [], []
        for index, path in enumerate(paths):
            if path.name != f"part-{index:05d}.parquet":
                break
            try:
                metadata.append(pq.read_metadata(path))
            except (OSError, pa.ArrowException):
                break
            fragments.append(path)
        ignored = [path.name for path in paths[len(fragments):]]
        if ignored:
            _logger.warning(
                "Ignoring unreadable or out-of-sequence Arrow fragments in %s (an interrupted "
                "write by an older version): %s", entry, ", ".join(ignored),
            )
        return fragments, metadata

    def _load_topic(self, entry: Path, manifest: dict) -> dict:
        import pyarrow.parquet as pq

        manifest_path = entry / MANIFEST_NAME
        committed = manifest.get("fragments")
        if committed is None:
            # Legacy manifests predate explicit commit lists. Reconcile
            # their counters with complete fragments before replaying.
            fragments, metadata = self._legacy_fragments(entry)
        else:
            if any(
                not isinstance(name, str) or Path(name).name != name
                or not name.startswith("part-") or not name.endswith(".parquet")
                for name in committed
            ) or len(set(committed)) != len(committed):
                raise ValueError(f"Invalid fragment list in {manifest_path}")
            fragments = [entry / name for name in committed]
            metadata = [pq.read_metadata(path) for path in fragments]
        rows = [meta.num_rows for meta in metadata]
        count = sum(rows)
        if committed is not None and count != int(manifest.get("count", 0)):
            raise ValueError(f"Fragment count disagrees with {manifest_path}")
        data_type = None
        if metadata:
            schema = metadata[0].schema.to_arrow_schema()
            if "data" in schema.names:
                data_type = schema.field("data").type
        return {
            "entry": entry, "manifest": manifest, "fragments": fragments, "rows": rows,
            "count": count, "data_type": data_type,
        }

    def _hydrate_existing_topics(self) -> None:
        root = Path(self.group_uri)
        if not root.is_dir():
            return
        found: dict[str, dict] = {}
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if not self.read_only and is_arrow_topic_dir_name(entry.name):
                _remove_stale_temporaries(entry)
            manifest = self._read_manifest(entry)
            if manifest is None:
                continue
            loaded = self._load_topic(entry, manifest)
            if not self.read_only and not is_arrow_topic_dir_name(entry.name):
                _remove_stale_temporaries(entry)
            topic = manifest["topic"]
            previous = found.get(topic)
            if previous is not None:
                # Two handles once could create the same topic in separate
                # directories; keep the fuller one rather than listing order.
                kept, dropped = (loaded, previous) if loaded["count"] > previous["count"] else (previous, loaded)
                _logger.warning(
                    "Topic %r has several Arrow directories; using %s and ignoring %s",
                    topic, kept["entry"], dropped["entry"],
                )
                found[topic] = kept
            else:
                found[topic] = loaded
        for topic, loaded in found.items():
            self._adopt_topic(topic, loaded)

    def _adopt_topic(self, topic: str, loaded: dict) -> None:
        manifest = loaded["manifest"]
        count = loaded["count"]
        self._topic_paths[topic] = str(loaded["entry"])
        self.counters[topic] = count
        self._persisted[topic] = count
        self._fragments[topic] = loaded["fragments"]
        self._fragment_rows[topic] = loaded["rows"]
        self._manifests[topic] = manifest
        self.closed_topics[topic] = bool(manifest.get("closed", False)) and count == int(manifest.get("count", 0))
        self.frame_ids[topic] = manifest.get("frame_id")
        name = manifest.get("name")
        if name is not None:
            self.names[topic] = name.encode() if isinstance(name, str) else bytes(name)
        shape = manifest.get("shape")
        dtype = manifest.get("dtype")
        # An empty topic's schema is not binding (older versions wrote a
        # placeholder); the first append defines it.
        if count and shape is not None and dtype is not None:
            shape = tuple(int(v) for v in shape)
            dtype = _native_dtype(dtype)
            self._schemas[topic] = (shape, dtype)
            # Stores written before lengths existed are fixed-shape.
            self._ragged[topic] = bool(manifest.get("ragged", False))
            if loaded["data_type"] is not None:
                self._raw[topic] = _detect_raw_encoding(loaded["data_type"], shape, dtype)
        if topic not in self.topics:
            self.topics.append(topic)

    def _should_skip_replayed_message(self, topic: str) -> bool:
        existing_count = self.counters.get(topic, 0)
        seen = self._resume_seen.get(topic, 0)
        self._resume_seen[topic] = seen + 1
        return seen < existing_count

    def _source_count(self, topic: str) -> int | None:
        """The source's total message count for `topic`, when it reports one."""
        return source_topic_count(self.init_source, topic, self._source_counts)

    def _topic_complete(self, topic: str) -> bool | None:
        """Whether `topic` holds every source message (None when unknown)."""
        expected = self._source_count(topic)
        if expected is None:
            return None
        return self.counters.get(topic, 0) >= expected

    def reset(self) -> None:
        self.close()
        self._init_state()
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

    def _new_topic_schema(self, topic: str, data: np.ndarray, msg: dict):
        if message_is_ragged(msg, self.ragged_topics):
            capacity = ragged_capacity(topic, self.ragged_topics, self.max_points)
            shape, dtype = new_ragged_schema(topic, data, msg, capacity)
            check_persistable_data(topic, data, shape)
            return shape, dtype, True
        check_persistable_data(topic, data)
        return tuple(data.shape), data.dtype, False

    def append_buffer(self, msg: dict) -> None:
        topic = msg["topic"]
        data = native_message_data(msg["data"])

        # Validate everything before touching any state so a rejected message
        # cannot poison the stage (and every later flush of this topic).
        schema = self._schemas.get(topic)
        if schema is None:
            shape, dtype, ragged = self._new_topic_schema(topic, data, msg)
        else:
            (shape, dtype), ragged = schema, self._ragged.get(topic, False)
        rows, length = conform_message(topic, data, msg, shape, dtype, ragged)
        timestamp = float(msg["timestamp"])
        name = encode_name(msg.get("name", topic))
        frame_id = encode_frame_id(msg.get("frame_id"))
        valid, mins, maxs = spatial_bounds_for_data(rows)

        # An explicit append is a write intent (mirrors the TileDB backend).
        if self.read_only:
            self._check_writable_location()
            self.read_only = False
        self._register_topic(topic)
        if schema is None:
            self._schemas[topic] = (tuple(shape), np.dtype(dtype))
            self._ragged[topic] = ragged

        self.names[topic] = name
        self._record_frame_id(msg)

        staged = self._staged.get(topic)
        if staged is None:
            staged = self._staged[topic] = _new_stage()
        staged["ts"].append(timestamp)
        staged["name"].append(self._interned.setdefault(name, name))
        staged["frame_id"].append(self._interned.setdefault(frame_id, frame_id))
        staged["spatial_valid"].append(1 if valid else 0)
        staged["spatial_min"].extend(mins.tolist())
        staged["spatial_max"].extend(maxs.tolist())
        # Own staged payloads: sources may immediately reuse their arrays.
        staged["data"].append(np.array(rows, copy=True))
        if ragged:
            staged["length"].append(length)
        payload_bytes = rows.nbytes if length is None else int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        staged["bytes"] += payload_bytes + self.row_overhead_bytes
        self.counters[topic] += 1
        if len(self._interned) > 4096:
            self._interned.clear()

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
        ragged = self._ragged.get(topic, False)
        if ragged:
            data = pad_rows(staged["data"][start:stop], staged["length"][start:stop], shape, dtype)
        else:
            data = np.stack(staged["data"][start:stop])
        mins = np.array(staged["spatial_min"][3 * start:3 * stop], dtype=np.float64).reshape((-1, 3))
        maxs = np.array(staged["spatial_max"][3 * start:3 * stop], dtype=np.float64).reshape((-1, 3))
        columns: dict = {
            "ts": pa.array(np.array(staged["ts"][start:stop], dtype=np.float64), type=pa.float64()),
            "name": pa.array(staged["name"][start:stop], type=pa.binary()),
            "frame_id": pa.array(staged["frame_id"][start:stop], type=pa.binary()),
            "spatial_valid": pa.array(np.array(staged["spatial_valid"][start:stop], dtype=bool), type=pa.bool_()),
        }
        for dim in range(3):
            columns[f"spatial_min_{dim}"] = pa.array(mins[:, dim], type=pa.float64())
            columns[f"spatial_max_{dim}"] = pa.array(maxs[:, dim], type=pa.float64())
        if ragged:
            columns["length"] = pa.array(np.asarray(staged["length"][start:stop], dtype=np.int64), type=pa.int64())
        columns["data"] = _encode_tensor(data, shape, dtype, self._raw.get(topic))
        return pa.table(columns)

    def _staged_table(self, topic: str):
        """Arrow table of the rows staged for `topic`, built incrementally."""
        staged = self._staged.get(topic)
        rows = _stage_rows(staged) if staged else 0
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
        if not staged or not _stage_rows(staged):
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

        fragment_path = self._topic_dir(topic) / f"part-{len(self._fragments[topic]):05d}.parquet"
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
        content = {
            "topic": topic,
            "count": self._persisted.get(topic, 0),
            "fragments": [path.name for path in self._fragments.get(topic, [])],
            "closed": bool(self.closed_topics.get(topic, False)),
            "name": self.names.get(topic, b"").decode(errors="replace"),
            "frame_id": self.frame_ids.get(topic),
            "shape": None if schema is None else list(schema[0]),
            "dtype": None if schema is None else str(schema[1]),
        }
        if self._ragged.get(topic):
            # Rows are zero-padded to shape[0]; the `length` column holds each
            # message's real row count. Absent for fixed-shape topics.
            content["ragged"] = True
        return content

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

    def _check_topic_unclaimed(self, topic: str) -> None:
        """Refuse to create `topic` when another writer created it meanwhile."""
        self._check_writable_location()
        root = Path(self.group_uri)
        if not root.is_dir():
            return
        # Directories this handle hydrated or created belong to known topics.
        known = {Path(path) for path in self._topic_paths.values()}
        for entry in root.iterdir():
            if not entry.is_dir() or entry in known:
                continue
            try:
                manifest = self._read_manifest(entry)
            except ValueError:
                continue
            if manifest is not None and manifest.get("topic") == topic:
                raise RuntimeError(
                    f"Arrow topic {topic!r} was created at {entry} by another writer after this handle "
                    "opened the store; reopen the store before writing it"
                )

    def _write_manifest(self, topic: str, closed: bool | None = None) -> None:
        if closed is not None:
            self.closed_topics[topic] = bool(closed)
        manifest = self._manifest_content(topic)
        if manifest == self._manifests.get(topic):
            return
        creating = self._manifests.get(topic) is None
        if creating:
            self._check_topic_unclaimed(topic)
        self._verify_manifest(topic)
        topic_dir = self._topic_dir(topic)
        if not topic_dir.is_dir():
            topic_dir.mkdir(parents=True, exist_ok=True)
            # Persist the new topic directory's entry in the group directory.
            _fsync_directory(topic_dir.parent)
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
        self._row_group_cache.clear()
        raise_collected(errors, "Arrow")

    def close(self, closed: bool | None = None) -> None:
        self._close_topics({topic: closed for topic in list(self.counters)})

    def close_completed(self, source_exhausted: bool = False) -> None:
        """Close, marking closed only topics that hold every source message.

        With `source_exhausted`, topics the source delivered in this session
        are complete too; other topics keep their closed flag.
        """
        self._close_topics({
            topic: True if self._topic_complete(topic) or (source_exhausted and topic in self._resume_seen) else None
            for topic in list(self.counters)
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

    def _fragment_group_rows(self, path: Path, parquet=None) -> np.ndarray:
        key = str(path)
        rows = self._group_rows.get(key)
        if rows is None:
            import pyarrow.parquet as pq

            metadata = parquet.metadata if parquet is not None else pq.read_metadata(path)
            rows = np.array([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)], dtype=np.int64)
            self._group_rows[key] = rows
        return rows

    def _read_row_group(self, path: Path, group: int, columns, opened: dict):
        """One decoded row group of a committed (immutable) fragment, cached."""
        key = (str(path), int(group), None if columns is None else tuple(columns))
        table = self._row_group_cache.get(key)
        if table is not None:
            self._row_group_cache.move_to_end(key)
            return table
        parquet = opened.get(key[0])
        if parquet is None:
            import pyarrow.parquet as pq

            parquet = opened[key[0]] = pq.ParquetFile(path)
        table = parquet.read_row_groups([int(group)], columns=columns, use_threads=self.use_threads)
        self._row_group_cache[key] = table
        while len(self._row_group_cache) > _ROW_GROUP_CACHE_SIZE:
            self._row_group_cache.popitem(last=False)
        return table

    def _take_fragment(self, path: Path, local: np.ndarray, columns=None):
        import pyarrow as pa

        opened: dict = {}
        try:
            group_rows = self._fragment_group_rows(path)
            group_starts = np.concatenate(([0], np.cumsum(group_rows)))
            group_ids = np.searchsorted(group_starts, local, side="right") - 1
            groups = np.unique(group_ids)
            tables = [self._read_row_group(path, group, columns, opened) for group in groups]
        finally:
            for parquet in opened.values():
                parquet.close()
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
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
            "source_uri": self._source_uri(topic),
        }
        if self._ragged.get(topic):
            result["lengths"] = np.array([], dtype=np.int64)
        if self.frame_ids.get(topic) is not None:
            result["frame_id"] = self.frame_ids[topic]
        return result

    def _source_uri(self, topic: str) -> str:
        # Do not reserve a path for a topic this store does not hold.
        if topic in self._topic_paths:
            return self._topic_paths[topic]
        return resolve_topic_path(self.group_uri, topic, dict(self._topic_paths))

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
            "source_uri": self._source_uri(topic),
        }
        if self._ragged.get(topic):
            if "length" in table.column_names:
                result["lengths"] = np.asarray(table["length"], dtype=np.int64)
            else:
                result["lengths"] = np.full(table.num_rows, data.shape[1] if data.ndim > 1 else 0, dtype=np.int64)
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
        if staged and _stage_rows(staged):
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
        count = self.counters.get(axis, 0)
        if not count:
            return

        operations = tuple(operations or ())
        indices = None
        if operations and operations[0].kind == "index_range":
            # Every row reaches a leading index range, so it maps directly to
            # row positions. Sparse selections read only their row groups;
            # dense ones scan (and stop once the range is exhausted).
            indices = np.arange(*slice(*operations[0].args).indices(count), dtype=np.int64)
            if indices.size >= _SCAN_FRACTION * count:
                indices = None
            else:
                operations = operations[1:]
        if indices is not None:
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
        ragged = self._ragged.get(topic, False) and not isinstance(subscript, slice)
        table = self._take(topic, indices, columns=["data", "length"] if ragged else ["data"])
        data = _decode_tensor(table["data"], self._schema(topic))
        if ragged:
            # One message of a variable-size topic: its real rows only.
            return data[0, :int(table["length"][0].as_py())].copy()
        return np.squeeze(data.copy())

    def __setitem__(self, subscript, newval):
        raise NotImplementedError(
            "the arrow backend stores immutable Parquet fragments and does not "
            "support in-place writes; use DataBuffer(backend='tiledb') for that"
        )
