"""TileDB persistent buffer backend.

Each topic is a pair of dense TileDB arrays in the group at ``group_uri``:

- ``<topic>`` stores payloads in the ``features`` attribute, one message per
  cell of the first dimension (tile extent 1 keeps random access cheap).
- ``<topic>__timestamps`` is the row index: timestamp, message name, frame id,
  and per-message spatial bounds used for query pushdown.

Appends stage rows in memory and write them as one fragment per array when
the staged payload reaches ``flush_bytes`` or when the topic is closed, so the
fragment count follows the data size instead of the message count. Reads merge
persisted cells with staged rows and never force a write; metadata is written
only when it changes. New arrays compress every attribute with Zstd. Closing a
handle that wrote consolidates and vacuums the arrays it touched (data arrays
that received more than ``consolidate_max_bytes`` in this session consolidate
fragment metadata only, to avoid rewriting large payloads).

A topic's capacity is the source's ``get_count(topic)`` when available; topics
without a known count (direct appends, callable sources) get an effectively
unbounded domain. Payload shape and dtype are fixed per topic.
"""

from __future__ import annotations
import logging
import os
import re
import shutil
from contextlib import suppress
from urllib.parse import unquote
import numpy as np
import tiledb

from .common import (
    SPATIAL_INDEX_DIMS,
    apply_index_range,
    check_message_schema,
    check_persistable_data,
    decode_frame_id as _decode_frame_id,
    encode_frame_id as _encode_frame_id,
    encode_name as _encode_name,
    index_ranges_exhausted,
    native_message_data,
    raise_collected,
    resolve_topic_path,
    spatial_bounds_for_data as _spatial_bounds_for_data,
    spatial_overlap_mask,
)

_logger = logging.getLogger(__name__)

DEFAULT_FLUSH_BYTES = 32 * 1024 * 1024
CONSOLIDATE_MAX_BYTES = 256 * 1024 * 1024
# TileDB allocates this much per attribute while consolidating fragments; its
# 50 MB default costs ~1.5 GB of RSS on the many-attribute timestamp index.
_CONSOLIDATION_BUFFER_BYTES = 4 * 1024 * 1024
# Rows of a topic without a known message count: a multiple of the index tile
# extent that still fits the int32 dimensions used by existing stores.
UNBOUNDED_CAPACITY = 2**31 - 1024
_INDEX_TILE = 1024
_SCAN_ROWS = 65536
_TOPIC_DIR = re.compile(r"topic-(.*)\.data(?:\.\d+)?")


def _new_stage() -> dict:
    return {
        "ts": [], "name": [], "frame_id": [], "spatial_valid": [], "spatial_min": [],
        "spatial_max": [], "data": [], "bytes": 0,
    }


def _fallback_topic(entry: str) -> str:
    """Topic for an array without topic metadata (created, never flushed)."""
    match = _TOPIC_DIR.fullmatch(entry)
    return unquote(match.group(1)) if match else entry


def _filters():
    return tiledb.FilterList([tiledb.ZstdFilter()])


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of ascending unique indices."""
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [indices.size]))
    return [(int(indices[a]), int(indices[b - 1]) + 1) for a, b in zip(starts, stops)]


def _staged_column(stage: dict, attr: str, positions) -> np.ndarray:
    if attr == "features":
        return np.stack([stage["data"][i] for i in positions])
    if attr == "timestamp":
        return np.asarray([stage["ts"][i] for i in positions], dtype=np.float64)
    if attr in ("name", "frame_id"):
        return np.array([stage[attr][i] for i in positions], dtype="S256")
    if attr == "spatial_valid":
        return np.asarray([stage["spatial_valid"][i] for i in positions], dtype=np.uint8)
    for kind in ("min", "max"):
        prefix = f"spatial_{kind}_"
        if attr.startswith(prefix):
            dim = int(attr[len(prefix):])
            return np.asarray([stage[f"spatial_{kind}"][i][dim] for i in positions], dtype=np.float64)
    raise KeyError(attr)


class _TopicReader:
    """Row reads over a topic's persisted TileDB cells and its staged rows.

    The row count and stage are snapshotted: a flush replaces the stage dict,
    and later appends only add rows past ``count``.
    """

    def __init__(self, data_uri: str, index_uri: str, persisted: int, stage: dict, schema=None):
        self.data_uri = data_uri
        self.index_uri = index_uri
        self.persisted = persisted
        self.stage = stage
        self.count = persisted + len(stage["ts"])
        self.shape, self.dtype = schema if schema is not None else ((), None)
        self._arrays = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        for array in self._arrays.values():
            with suppress(Exception):
                array.close()
        self._arrays = {}

    def _array(self, attr: str):
        uri = self.data_uri if attr == "features" else self.index_uri
        if uri not in self._arrays:
            self._arrays[uri] = tiledb.open(uri, "r")
        return self._arrays[uri]

    def has_attr(self, attr: str) -> bool:
        if attr == "features":
            return True
        return attr in self._array(attr).schema.attr_names

    def _restore_dtype(self, attr: str, values: np.ndarray) -> np.ndarray:
        # TileDB returns fixed-width str cells as objects; restore the schema dtype.
        if attr == "features" and self.dtype is not None and values.dtype != self.dtype:
            return values.astype(self.dtype)
        return values

    def read(self, attr: str, first: int, last: int) -> np.ndarray:
        """Values of rows ``[first, last)``."""
        parts = []
        if first < self.persisted:
            stop = min(last, self.persisted)
            values = self._array(attr).query(attrs=[attr])[first:stop][attr]
            parts.append(self._restore_dtype(attr, np.asarray(values)))
        if last > self.persisted:
            lower = max(first, self.persisted) - self.persisted
            parts.append(_staged_column(self.stage, attr, range(lower, last - self.persisted)))
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def read_points(self, attr: str, indices: np.ndarray) -> np.ndarray:
        """Values of the rows at ascending, unique `indices`."""
        split = int(np.searchsorted(indices, self.persisted))
        parts = []
        if split:
            persisted = indices[:split]
            first, last = int(persisted[0]), int(persisted[-1]) + 1
            if last - first == persisted.size:
                parts.append(self.read(attr, first, last))
            elif last - first <= 2 * persisted.size:
                parts.append(self.read(attr, first, last)[persisted - first])
            else:
                # Sparse selections (e.g. strided ranges) read only their runs.
                ranges = [slice(start, stop - 1) for start, stop in _runs(persisted)]
                values = self._array(attr).query(attrs=[attr]).multi_index[ranges][attr]
                parts.append(self._restore_dtype(attr, np.asarray(values)))
        if split < indices.size:
            parts.append(_staged_column(self.stage, attr, indices[split:] - self.persisted))
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


class TileDBBuffer:
    def __init__(
        self,
        data_source,
        init_source,
        group_uri,
        axis="",
        topics=None,
        flush_bytes: int = DEFAULT_FLUSH_BYTES,
        consolidate_max_bytes: int = CONSOLIDATE_MAX_BYTES,
    ):
        self.data_source = data_source
        self.init_source = init_source
        self.group_uri = group_uri
        self._axis = axis
        self.topics = [] if topics is None else list(topics)
        self.flush_bytes = int(flush_bytes)
        self.consolidate_max_bytes = int(consolidate_max_bytes)
        self._init_state()
        # DataBuffer passes iter(()) (not None) for a source-less reopen, so
        # a missing init_source must also mark the buffer read-only.
        self.read_only = data_source is None or init_source is None
        try:
            self._hydrate_existing_topics()
        except Exception:
            # A failed open must not rewrite partially hydrated metadata on GC.
            self.read_only = True
            raise

    def _init_state(self) -> None:
        self.counters = {}
        self.msg_len = {}
        self.names = {}
        self.frame_ids = {}
        self.closed_topics = {}
        self.timestamps = {}
        self._resume_seen = {}
        self._topic_paths = {}
        self._persisted: dict[str, int] = {}
        self._staged: dict[str, dict] = {}
        self._schemas: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self._unbounded: set[str] = set()
        self._sorted: dict[str, bool | None] = {}
        self._last_ts: dict[str, float] = {}
        # Metadata this handle last read from or wrote to disk, per topic.
        self._meta_written: dict[str, dict] = {}
        # Fragments, bytes, and metadata writes since the last consolidation.
        self._session: dict[str, dict] = {}
        self._source_counts: dict[str, int | None] = {}

    # -- layout ----------------------------------------------------------------

    def _get_array_uri(self, topic: str) -> str:
        return resolve_topic_path(self.group_uri, topic, self._topic_paths)

    def _get_timestamp_array_uri(self, topic: str) -> str:
        return self._get_array_uri(topic) + "__timestamps"

    def _source_uri(self, topic: str) -> str:
        # Do not reserve a path for a topic this store does not hold.
        if topic in self._topic_paths:
            return self._topic_paths[topic]
        return resolve_topic_path(self.group_uri, topic, dict(self._topic_paths))

    def _ensure_group(self) -> None:
        if tiledb.object_type(str(self.group_uri)) != "group":
            os.makedirs(self.group_uri, exist_ok=True)
            tiledb.group_create(str(self.group_uri))

    def _add_group_member(self, uri: str, name: str) -> None:
        self._ensure_group()
        with tiledb.Group(str(self.group_uri), "w") as group:
            with suppress(Exception):
                group.add(uri, name)

    # -- hydration / resume ----------------------------------------------------

    def _hydrate_existing_topics(self) -> None:
        if not os.path.isdir(self.group_uri):
            return

        for entry in sorted(os.listdir(self.group_uri)):
            if entry.endswith("__timestamps"):
                continue
            uri = os.path.join(self.group_uri, entry)
            if not os.path.isdir(uri) or tiledb.object_type(uri) != "array":
                continue
            try:
                info = self._read_topic_info(uri, _fallback_topic(entry))
            except tiledb.TileDBError as exc:
                raise ValueError(f"Unreadable TileDB topic array {uri}") from exc
            if info is None:
                _logger.warning("Ignoring TileDB array %s without a 'features' attribute", uri)
                continue
            topic = info["topic"]
            if topic in self._topic_paths:
                # Older versions could create a second array for a topic whose
                # hydration failed. Keep the fuller one instead of relying on
                # directory listing order.
                kept, dropped = (self._topic_paths[topic], uri)
                if info["count"] > self.counters.get(topic, 0):
                    kept, dropped = dropped, kept
                _logger.warning("Topic %r has several TileDB arrays; using %s and ignoring %s", topic, kept, dropped)
                if kept == self._topic_paths[topic]:
                    continue
            self._adopt_topic(uri, info)

    def _read_topic_info(self, uri: str, fallback_topic: str) -> dict | None:
        with tiledb.open(uri, "r") as tiledb_array:
            schema = tiledb_array.schema
            if "features" not in schema.attr_names:
                return None
            meta = tiledb_array.meta
            topic = meta.get("topic", fallback_topic)
            count = int(meta.get("count", 0))
            closed = bool(meta.get("closed", False))
            name = meta.get("name", "")
            frame_id = _decode_frame_id(meta.get("frame_id"))
            domain = schema.domain
            first, last = domain.dim(0).domain
            capacity = int(last) - int(first) + 1
            shape = tuple(
                int(domain.dim(i).domain[1]) - int(domain.dim(i).domain[0]) + 1 for i in range(1, domain.ndim)
            )
            dtype = np.dtype(schema.attr("features").dtype)

        ts_sorted = None
        timestamp_uri = uri + "__timestamps"
        if os.path.exists(timestamp_uri):
            with tiledb.open(timestamp_uri, "r") as timestamp_array:
                count = int(timestamp_array.meta.get("count", count))
                closed = bool(timestamp_array.meta.get("closed", closed))
                if "ts_sorted" in timestamp_array.meta:
                    ts_sorted = bool(timestamp_array.meta["ts_sorted"])
        return {
            "topic": topic, "count": count, "closed": closed, "name": _encode_name(name),
            "frame_id": frame_id, "capacity": capacity, "shape": shape, "dtype": dtype,
            "ts_sorted": ts_sorted,
        }

    def _adopt_topic(self, uri: str, info: dict) -> None:
        topic = info["topic"]
        count = info["count"]
        self._topic_paths[topic] = uri
        self.counters[topic] = count
        self._persisted[topic] = count
        self._staged.pop(topic, None)
        if info["capacity"] >= UNBOUNDED_CAPACITY:
            self._unbounded.add(topic)
            self.msg_len[topic] = count
        else:
            self._unbounded.discard(topic)
            self.msg_len[topic] = info["capacity"]
        self.closed_topics[topic] = info["closed"]
        self.names[topic] = info["name"]
        if info["frame_id"] is not None or count:
            # Rows without a frame make the topic frame unknown (None).
            self.frame_ids[topic] = info["frame_id"]
        self._schemas[topic] = (info["shape"], info["dtype"])
        self._sorted[topic] = info["ts_sorted"]
        self._last_ts.pop(topic, None)
        self._meta_written[topic] = self._metadata_values(topic)
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

    def reset(self) -> None:
        self.close()
        self._init_state()
        self._hydrate_existing_topics()

    # -- write path ------------------------------------------------------------

    def _register_topic(self, msg: dict) -> None:
        topic = msg["topic"]
        self.counters.setdefault(topic, 0)
        if topic not in self.msg_len and topic not in self._unbounded:
            count = self._source_count(topic)
            if count is None:
                self._unbounded.add(topic)
            else:
                self.msg_len[topic] = max(count, 1)
        self._init_tdb(msg)

    def _init_tdb(self, msg: dict) -> None:
        topic = msg["topic"]
        data = native_message_data(msg["data"])
        check_persistable_data(topic, data)
        uri = self._get_array_uri(topic)
        timestamp_uri = self._get_timestamp_array_uri(topic)
        self.counters.setdefault(topic, 0)

        if os.path.exists(uri):
            if os.path.exists(timestamp_uri):
                info = self._read_topic_info(uri, topic)
                if info is None:
                    raise ValueError(f"TileDB array {uri} is not a topic array")
                self._adopt_topic(uri, info)
                return
            if tiledb.object_type(uri) == "array" and len(tiledb.array_fragments(uri)) == 0:
                # Creation was interrupted before the index existed; nothing was written.
                shutil.rmtree(uri)
            else:
                raise ValueError(
                    f"TileDB topic array {uri} has no timestamp index {timestamp_uri}; refusing to replace it"
                )

        if topic in self._unbounded or topic not in self.msg_len:
            self._unbounded.add(topic)
            rows = UNBOUNDED_CAPACITY
            self.msg_len[topic] = self.counters[topic]
        else:
            rows = max(int(self.msg_len[topic]), 1)
            self.msg_len[topic] = rows

        dims = [tiledb.Dim(name="images", domain=(0, rows - 1), tile=1, dtype=np.int32)]
        dims += [
            tiledb.Dim(name=f"dim_{index}", domain=(0, extent - 1), tile=extent, dtype=np.int32)
            for index, extent in enumerate(data.shape)
        ]
        try:
            schema = tiledb.ArraySchema(
                domain=tiledb.Domain(*dims),
                sparse=False,
                attrs=[tiledb.Attr(name="features", dtype=data.dtype, filters=_filters())],
            )
        except (TypeError, tiledb.TileDBError) as exc:
            raise ValueError(f"topic {topic} messages have dtype {data.dtype}, which TileDB cannot store") from exc

        timestamp_schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="message", domain=(0, rows - 1), tile=min(rows, _INDEX_TILE), dtype=np.int32)
            ),
            sparse=False,
            attrs=[
                tiledb.Attr(name="timestamp", dtype=np.float64, filters=_filters()),
                tiledb.Attr(name="name", dtype="S256", filters=_filters()),
                tiledb.Attr(name="frame_id", dtype="S256", filters=_filters()),
                tiledb.Attr(name="spatial_valid", dtype=np.uint8, filters=_filters()),
                *[
                    tiledb.Attr(name=f"spatial_min_{dim}", dtype=np.float64, filters=_filters())
                    for dim in range(SPATIAL_INDEX_DIMS)
                ],
                *[
                    tiledb.Attr(name=f"spatial_max_{dim}", dtype=np.float64, filters=_filters())
                    for dim in range(SPATIAL_INDEX_DIMS)
                ],
            ],
        )
        tiledb.Array.create(uri, schema)
        self._add_group_member(uri, os.path.basename(uri))
        tiledb.Array.create(timestamp_uri, timestamp_schema)
        self._add_group_member(timestamp_uri, os.path.basename(timestamp_uri))

        self._persisted[topic] = 0
        self._schemas[topic] = (tuple(data.shape), data.dtype)
        self._sorted[topic] = True
        if topic not in self.topics:
            self.topics.append(topic)

    def roll_buffer(self, axis: str) -> None:
        self._axis = axis
        while True:
            msg = next(self.data_source)
            topic = msg['topic']

            if topic not in self.counters:
                self._register_topic(msg)

            if self._should_skip_replayed_message(topic):
                if topic == self._axis:
                    break
                continue

            if self.closed_topics.get(topic, False) and self.counters[topic] >= self.msg_len.get(topic, 0):
                if topic == self._axis:
                    break
                continue

            self.append_buffer(msg)

            if topic == self._axis:
                break

    def _is_full(self, topic: str) -> bool:
        return topic not in self._unbounded and self.counters.get(topic, 0) >= self.msg_len.get(topic, 0)

    def append_buffer(self, msg: dict) -> None:
        topic = msg['topic']
        data = native_message_data(msg['data'])
        schema = self._schemas.get(topic)
        # Validate before touching any state: TileDB would otherwise cast
        # silently (e.g. 70000.0 into int16) or fail mid-write.
        if schema is None:
            check_persistable_data(topic, data)
        else:
            check_message_schema(topic, data, *schema)
        timestamp = float(msg['timestamp'])
        name = _encode_name(msg.get("name", topic))
        frame_id = _encode_frame_id(msg.get("frame_id"))
        if topic in self.counters and schema is not None and self._is_full(topic):
            raise ValueError(f"topic {topic} is already full at {self.counters[topic]} messages")

        # An explicit append is a write intent: leave read-only mode so the
        # updated counts and metadata are persisted on close. Otherwise data
        # appended to a reopened store would be silently invisible on reload.
        self.read_only = False
        if schema is None:
            self._register_topic({**msg, "data": data})
            check_message_schema(topic, data, *self._schemas[topic])
            if self._is_full(topic):
                raise ValueError(f"topic {topic} is already full at {self.counters[topic]} messages")

        valid, mins, maxs = _spatial_bounds_for_data(data)
        self.names[topic] = name
        self._record_frame_id(msg)
        self._update_sorted(topic, timestamp)

        stage = self._staged.get(topic)
        if stage is None:
            stage = self._staged[topic] = _new_stage()
        stage["ts"].append(timestamp)
        stage["name"].append(name)
        stage["frame_id"].append(frame_id)
        stage["spatial_valid"].append(valid)
        stage["spatial_min"].append(mins)
        stage["spatial_max"].append(maxs)
        # Own staged payloads: sources may immediately reuse their arrays.
        stage["data"].append(np.array(data, copy=True))
        stage["bytes"] += data.nbytes
        self.counters[topic] += 1
        if topic in self._unbounded:
            self.msg_len[topic] = self.counters[topic]

        if stage["bytes"] >= self.flush_bytes:
            self._flush_topic(topic)

    def _last_timestamp(self, topic: str) -> float | None:
        if topic not in self._last_ts:
            persisted = self._persisted.get(topic, 0)
            if not persisted:
                return None
            with tiledb.open(self._get_timestamp_array_uri(topic), "r") as index_array:
                self._last_ts[topic] = float(
                    index_array.query(attrs=["timestamp"])[persisted - 1:persisted]["timestamp"][0]
                )
        return self._last_ts[topic]

    def _update_sorted(self, topic: str, timestamp: float) -> None:
        if self._sorted.get(topic) is not False:
            last = self._last_timestamp(topic)
            if last is not None and timestamp < last:
                self._sorted[topic] = False
        self._last_ts[topic] = timestamp

    def _record_frame_id(self, msg: dict) -> None:
        topic = msg["topic"]
        frame_id = _decode_frame_id(msg.get("frame_id")) or None
        if self.counters.get(topic, 0) == 0 or topic not in self.frame_ids:
            self.frame_ids[topic] = frame_id
        elif self.frame_ids[topic] != frame_id:
            self.frame_ids[topic] = None

    def _index_values(self, attr_names, stage: dict) -> dict:
        rows = range(len(stage["ts"]))
        return {attr: _staged_column(stage, attr, rows) for attr in attr_names}

    def _record_session_write(self, topic: str, fragments: int = 0, nbytes: int = 0, meta: int = 0) -> None:
        session = self._session.setdefault(topic, {"fragments": 0, "bytes": 0, "meta": 0})
        session["fragments"] += fragments
        session["bytes"] += nbytes
        session["meta"] += meta

    def _flush_topic(self, topic: str) -> None:
        stage = self._staged.get(topic)
        if not stage or not stage["ts"]:
            return

        start = self._persisted.get(topic, 0)
        rows = len(stage["ts"])
        data = np.stack(stage["data"])
        key = (slice(start, start + rows),) + tuple(slice(None) for _ in data.shape[1:])
        with tiledb.open(self._get_array_uri(topic), "w") as tiledb_array:
            tiledb_array[key] = data
        with tiledb.open(self._get_timestamp_array_uri(topic), "w") as timestamp_array:
            timestamp_array[start:start + rows] = self._index_values(timestamp_array.schema.attr_names, stage)
        self._persisted[topic] = start + rows
        self._staged[topic] = _new_stage()
        self._record_session_write(topic, fragments=1, nbytes=data.nbytes)
        # Advance the committed count with each flush, so an interrupted
        # ingest resumes after the last flushed row.
        self._write_metadata(topic)

    def _metadata_values(self, topic: str) -> dict:
        return {
            "name": self.names.get(topic, b"").decode(errors="replace"),
            "count": self._persisted.get(topic, 0),
            "frame_id": self.frame_ids.get(topic),
            "closed": bool(self.closed_topics.get(topic, False)),
            "ts_sorted": self._sorted.get(topic),
        }

    def _write_metadata(self, topic: str) -> None:
        values = self._metadata_values(topic)
        if values == self._meta_written.get(topic):
            return
        with tiledb.open(self._get_array_uri(topic), "w") as tiledb_array:
            tiledb_array.meta["name"] = values["name"]
            tiledb_array.meta["topic"] = topic
            tiledb_array.meta["count"] = values["count"]
            if values["frame_id"] is not None:
                tiledb_array.meta["frame_id"] = values["frame_id"]
            else:
                with suppress(KeyError):
                    del tiledb_array.meta["frame_id"]
            tiledb_array.meta["closed"] = values["closed"]
        with tiledb.open(self._get_timestamp_array_uri(topic), "w") as timestamp_array:
            timestamp_array.meta["topic"] = topic
            timestamp_array.meta["count"] = values["count"]
            timestamp_array.meta["closed"] = values["closed"]
            if values["ts_sorted"] is not None:
                timestamp_array.meta["ts_sorted"] = int(values["ts_sorted"])
        self._meta_written[topic] = values
        self._record_session_write(topic, meta=1)

    def _consolidate(self, topic: str) -> None:
        session = self._session.pop(topic, None)
        if not session:
            return
        meta_config = tiledb.Config({"sm.consolidation.mode": "array_meta", "sm.vacuum.mode": "array_meta"})
        for uri, holds_payload in (
            (self._get_array_uri(topic), True),
            (self._get_timestamp_array_uri(topic), False),
        ):
            try:
                if session["fragments"] and len(tiledb.array_fragments(uri)) > 1:
                    mode = "fragments"
                    if holds_payload and session["bytes"] > self.consolidate_max_bytes:
                        mode = "fragment_meta"
                    config = tiledb.Config({
                        "sm.consolidation.mode": mode,
                        "sm.vacuum.mode": mode,
                        "sm.consolidation.buffer_size": str(_CONSOLIDATION_BUFFER_BYTES),
                    })
                    tiledb.consolidate(uri, config=config)
                    tiledb.vacuum(uri, config=config)
                if session["meta"]:
                    tiledb.consolidate(uri, config=meta_config)
                    tiledb.vacuum(uri, config=meta_config)
            except tiledb.TileDBError as exc:
                # Data is already committed; consolidation only compacts it.
                _logger.warning("Could not consolidate TileDB array %s: %s", uri, exc)

    def close_topic(self, topic: str, closed: bool | None = None) -> None:
        if self.read_only or topic not in self._schemas:
            return
        self._flush_topic(topic)
        if closed is not None:
            self.closed_topics[topic] = bool(closed)
        self._write_metadata(topic)
        self._consolidate(topic)

    def _close_topics(self, closed_by_topic: dict) -> None:
        # Attempt every topic so one failure cannot strand other topics'
        # staged rows, then report all failures together.
        errors = []
        for topic, closed in closed_by_topic.items():
            try:
                self.close_topic(topic, closed)
            except Exception as exc:
                exc.add_note(f"while closing TileDB topic {topic!r}")
                errors.append(exc)
        raise_collected(errors, "TileDB")

    def close(self, closed: bool | None = None):
        self._close_topics({topic: closed for topic in list(self.counters)})

    def close_completed(self) -> None:
        """Close, marking closed only topics that hold every source message."""
        self._close_topics({
            topic: True if topic not in self._unbounded and self._is_full(topic) else None
            for topic in list(self.counters)
        })

    def __del__(self):
        with suppress(Exception):
            self.close()

    # -- read path -------------------------------------------------------------

    def _reader(self, topic: str) -> _TopicReader:
        return _TopicReader(
            self._get_array_uri(topic),
            self._get_timestamp_array_uri(topic),
            self._persisted.get(topic, 0),
            self._staged.get(topic) or _new_stage(),
            schema=self._schemas.get(topic),
        )

    def _metadata_for_topic(self, topic: str) -> dict:
        frame_id = self.frame_ids.get(topic)
        return {} if frame_id is None else {"frame_id": frame_id}

    def _empty_result(self, topic: str) -> dict:
        shape, dtype = self._schemas.get(topic, ((), np.dtype(np.float64)))
        return {
            "id": np.array([], dtype=object),
            "name": np.array([], dtype=object),
            "ts": np.array([], dtype=np.float64),
            "data": np.empty((0, *shape), dtype=dtype),
            "frame_ids": np.array([], dtype=object),
            "topic": topic,
            "source_uri": self._source_uri(topic),
            **self._metadata_for_topic(topic),
        }

    def _read_timestamp_scalar(self, reader: _TopicReader, index: int) -> float:
        return float(reader.read("timestamp", index, index + 1)[0])

    def _timestamp_search(self, reader: _TopicReader, count: int, value: float, side: str = "left") -> int:
        left = 0
        right = count
        while left < right:
            mid = (left + right) // 2
            timestamp = self._read_timestamp_scalar(reader, mid)
            if timestamp < value or (side == "right" and timestamp <= value):
                left = mid + 1
            else:
                right = mid
        return left

    def _time_range_to_index_range(
        self,
        reader: _TopicReader,
        count: int,
        start: float,
        end: float,
        inclusive: bool = True,
    ) -> tuple[int, int]:
        """Row range of a time window; valid only for sorted timestamps."""
        if count == 0:
            return 0, 0

        lower_side = "left" if inclusive else "right"
        upper_side = "right" if inclusive else "left"
        if count <= _SCAN_ROWS:
            timestamps = reader.read("timestamp", 0, count)
            first = int(np.searchsorted(timestamps, start, side=lower_side))
            last = int(np.searchsorted(timestamps, end, side=upper_side))
        else:
            first = self._timestamp_search(reader, count, start, side=lower_side)
            last = self._timestamp_search(reader, count, end, side=upper_side)
        return first, max(first, last)

    def _timestamps_sorted(self, topic: str, reader: _TopicReader) -> bool:
        state = self._sorted.get(topic)
        if state is not None:
            return state
        # Stores written before sortedness was recorded: check once.
        previous = -np.inf
        state = True
        for lower in range(0, reader.count, _SCAN_ROWS):
            timestamps = reader.read("timestamp", lower, min(reader.count, lower + _SCAN_ROWS))
            if timestamps.size and (timestamps[0] < previous or np.any(np.diff(timestamps) < 0)):
                state = False
                break
            if timestamps.size:
                previous = timestamps[-1]
        self._sorted[topic] = state
        if topic in self._meta_written:
            # Learned by reading; not a reason to write metadata.
            self._meta_written[topic]["ts_sorted"] = state
        return state

    def _iter_range_blocks(self, start: int, stop: int, step: int = 1):
        for lower in range(start, stop, step * _SCAN_ROWS):
            yield np.arange(lower, min(stop, lower + step * _SCAN_ROWS), step, dtype=np.int64)

    def _iter_time_candidates(self, topic: str, reader: _TopicReader, start: float, end: float, inclusive: bool = True):
        count = reader.count
        if self._timestamps_sorted(topic, reader):
            first, last = self._time_range_to_index_range(reader, count, start, end, inclusive=inclusive)
            yield from self._iter_range_blocks(first, last)
            return
        # Out-of-order timestamps: filter every row instead of bisecting.
        for lower in range(0, count, _SCAN_ROWS):
            timestamps = reader.read("timestamp", lower, min(count, lower + _SCAN_ROWS))
            if inclusive:
                mask = (timestamps >= start) & (timestamps <= end)
            else:
                mask = (timestamps > start) & (timestamps < end)
            hits = np.flatnonzero(mask)
            if hits.size:
                yield hits.astype(np.int64) + lower

    def _normalize_index_range(
        self,
        count: int,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> tuple[int, int, int]:
        range_start, range_stop, range_step = slice(start, stop, step).indices(count)
        if range_step < 1:
            raise ValueError("step must be positive")
        return range_start, range_stop, range_step

    def _read_index_attr(self, reader: _TopicReader, attr: str, indices: np.ndarray, fallback: str | None = None) -> np.ndarray:
        if indices.size == 0:
            return np.array([])
        if not reader.has_attr(attr):
            if fallback is None:
                return np.array([])
            return np.full(indices.size, fallback, dtype=object)
        return reader.read_points(attr, indices)

    def _read_data_attr(self, reader: _TopicReader, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.empty((0, *reader.shape), dtype=reader.dtype or np.float64)
        return reader.read_points("features", indices)

    def _row_frame_ids(self, axis: str, reader: _TopicReader, indices: np.ndarray) -> np.ndarray:
        if reader.has_attr("frame_id"):
            frames = reader.read_points("frame_id", indices)
            return np.array([_decode_frame_id(frame) or None for frame in frames], dtype=object)
        # Stores that predate per-row frames only know the topic frame.
        return np.full(indices.size, self.frame_ids.get(axis), dtype=object)

    def _iter_selected_indices(self, axis: str, reader: _TopicReader, count: int, chunk_size: int, operations):
        operations = tuple(operations or ())

        if operations and operations[0].kind == "time_range":
            start, end = operations[0].args
            candidates = self._iter_time_candidates(
                axis, reader, start, end, inclusive=operations[0].kwargs.get("inclusive", True)
            )
            operations = operations[1:]
        elif operations and operations[0].kind == "index_range":
            # Every row reaches a leading index range: it maps to row positions.
            candidates = self._iter_range_blocks(*slice(*operations[0].args).indices(count))
            operations = operations[1:]
        else:
            candidates = self._iter_range_blocks(0, count)

        counters = [0] * len(operations)
        for block in candidates:
            for offset in range(0, block.size, chunk_size):
                selected = block[offset:offset + chunk_size]
                for operation_index, operation in enumerate(operations):
                    if selected.size == 0:
                        break
                    if operation.kind == "time_range":
                        start, end = operation.args
                        timestamps = self._read_index_attr(reader, "timestamp", selected)
                        if operation.kwargs.get("inclusive", True):
                            mask = (timestamps >= start) & (timestamps <= end)
                        else:
                            mask = (timestamps > start) & (timestamps < end)
                        selected = selected[mask]
                    elif operation.kind == "index_range":
                        keep, counters[operation_index] = apply_index_range(
                            np.ones(selected.size, dtype=bool), counters[operation_index], *operation.args
                        )
                        selected = selected[keep]
                    elif operation.kind == "frame_id":
                        selected = self._filter_frame_indices(axis, reader, selected, operation.args[0])
                    elif operation.kind == "spatial_bounds":
                        min_bound, max_bound = operation.args
                        selected = self._filter_spatial_indices(
                            reader,
                            selected,
                            min_bound=min_bound,
                            max_bound=max_bound,
                            columns=operation.kwargs["columns"],
                        )
                    else:
                        raise ValueError(f"unsupported pushdown operation: {operation.kind}")

                if selected.size:
                    yield selected
                if index_ranges_exhausted(operations, counters):
                    return

    def _filter_frame_indices(self, axis: str, reader: _TopicReader, indices: np.ndarray, targets) -> np.ndarray:
        frames = self._row_frame_ids(axis, reader, indices)
        return indices[np.fromiter((frame in targets for frame in frames), dtype=bool, count=indices.size)]

    def _filter_spatial_indices(
        self,
        reader: _TopicReader,
        indices: np.ndarray,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
        columns: tuple[int, ...],
    ) -> np.ndarray:
        required_attrs = {"spatial_valid"}
        for column in columns:
            if column >= SPATIAL_INDEX_DIMS:
                return indices
            required_attrs.add(f"spatial_min_{column}")
            required_attrs.add(f"spatial_max_{column}")
        if not all(reader.has_attr(attr) for attr in required_attrs):
            return indices

        # Bounds prune conservatively; the exact filter runs afterwards.
        keep = spatial_overlap_mask(
            reader.read_points("spatial_valid", indices),
            [reader.read_points(f"spatial_min_{column}", indices) for column in columns],
            [reader.read_points(f"spatial_max_{column}", indices) for column in columns],
            min_bound,
            max_bound,
        )
        return indices[keep]

    def _read_topic_indices(self, axis: str, reader: _TopicReader, indices: np.ndarray, copy: bool = False) -> dict:
        # Every array read here is freshly allocated, so `copy` needs no extra work.
        data = self._read_data_attr(reader, indices)
        timestamps = np.asarray(self._read_index_attr(reader, "timestamp", indices), dtype=np.float64)
        names = self._read_index_attr(reader, "name", indices, fallback=axis)
        result = {
            "id": names,
            "name": names.copy(),
            "ts": timestamps,
            "data": data,
            "frame_ids": self._row_frame_ids(axis, reader, indices),
            "topic": axis,
            "source_uri": reader.data_uri,
        }
        if self.frame_ids.get(axis) is not None:
            result["frame_id"] = self.frame_ids[axis]
        return result

    def get_buffer(self, copy: bool = True) -> dict:
        buffer = {}
        for topic, count in self.counters.items():
            buffer[topic] = self.get_index_range(topic, 0, count, copy=copy)
        return buffer

    def get_topic(self, topic: str, copy: bool = True) -> dict:
        return self.get_index_range(topic, 0, self.counters.get(topic, 0), copy=copy)

    def iter_topic_chunks(self, axis: str, chunk_size: int, copy: bool = False, operations=()):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not self.counters.get(axis, 0) or axis not in self._schemas:
            return

        with self._reader(axis) as reader:
            for indices in self._iter_selected_indices(axis, reader, reader.count, chunk_size, operations):
                yield self._read_topic_indices(axis, reader, indices, copy=copy)

    def get_index_range(
        self,
        axis: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        copy: bool = True,
    ) -> dict:
        if axis not in self.counters or axis not in self._schemas:
            return self._empty_result(axis)

        range_start, range_stop, range_step = self._normalize_index_range(self.counters[axis], start, stop, step)
        indices = np.arange(range_start, range_stop, range_step, dtype=np.int64)
        if indices.size == 0:
            return self._empty_result(axis)
        with self._reader(axis) as reader:
            return self._read_topic_indices(axis, reader, indices, copy=copy)

    def get_time_range(self, axis: str, start: float, end: float) -> dict:
        if not self.counters.get(axis, 0) or axis not in self._schemas:
            return self._empty_result(axis)

        with self._reader(axis) as reader:
            blocks = list(self._iter_time_candidates(axis, reader, start, end))
            if not blocks:
                return self._empty_result(axis)
            return self._read_topic_indices(axis, reader, np.concatenate(blocks))

    def get_last_seconds(self, axis: str, seconds: float) -> dict:
        if not self.counters.get(axis, 0) or axis not in self._schemas:
            return self._empty_result(axis)

        with self._reader(axis) as reader:
            end = self._read_timestamp_scalar(reader, reader.count - 1)
        return self.get_time_range(axis, end - seconds, end)

    # -- subscripts ------------------------------------------------------------

    def _subscript_indices(self, topic: str, subscript) -> np.ndarray:
        count = self.counters.get(topic, 0)
        if isinstance(subscript, slice):
            return np.arange(*subscript.indices(count), dtype=np.int64)
        if isinstance(subscript, (int, np.integer)):
            index = int(subscript)
            if index < 0:
                index += count
            if not 0 <= index < count:
                raise IndexError(f"index {int(subscript)} is out of range for topic {topic!r} with {count} messages")
            return np.array([index], dtype=np.int64)
        raise TypeError(f"unsupported subscript type: {type(subscript).__name__}")

    def __getitem__(self, subscript):
        topic = self._axis
        if topic not in self.counters or topic not in self._schemas:
            return None

        indices = self._subscript_indices(topic, subscript)
        if indices.size == 0:
            shape, dtype = self._schemas[topic]
            return np.squeeze(np.empty((0, *shape), dtype=dtype))
        order = np.argsort(indices, kind="stable")
        with self._reader(topic) as reader:
            data = self._read_data_attr(reader, indices[order])
        if not np.array_equal(order, np.arange(order.size)):
            data = data[np.argsort(order)]
        return np.squeeze(data)

    def __setitem__(self, subscript, newval) -> bool | None:
        topic = self._axis
        if topic not in self.counters or topic not in self._schemas:
            return None

        indices = self._subscript_indices(topic, subscript)
        if indices.size == 0:
            return True
        shape, dtype = self._schemas[topic]
        values = np.broadcast_to(np.asarray(newval, dtype=dtype), (indices.size, *shape))
        order = np.argsort(indices, kind="stable")
        indices, values = indices[order], values[order]

        self.read_only = False
        # Staged rows must be cells before they can be overwritten.
        self._flush_topic(topic)
        offset = 0
        uri = self._get_array_uri(topic)
        timestamp_uri = self._get_timestamp_array_uri(topic)
        for start, stop in _runs(indices):
            part = values[offset:offset + stop - start]
            offset += stop - start
            key = (slice(start, stop),) + tuple(slice(None) for _ in shape)
            with tiledb.open(uri, "w") as tiledb_array:
                tiledb_array[key] = np.ascontiguousarray(part)
            self._refresh_spatial_index(timestamp_uri, start, stop, part)
            self._record_session_write(topic, fragments=1, nbytes=part.nbytes)
        return True

    def _refresh_spatial_index(self, timestamp_uri: str, start: int, stop: int, values: np.ndarray) -> None:
        """Keep stored bounds in sync with overwritten payloads (pruning relies on them)."""
        with tiledb.open(timestamp_uri, "r") as timestamp_array:
            attr_names = timestamp_array.schema.attr_names
            if "spatial_valid" not in attr_names:
                return
            rows = dict(timestamp_array[start:stop])
        bounds = [_spatial_bounds_for_data(value) for value in values]
        rows["spatial_valid"] = np.array([valid for valid, _, _ in bounds], dtype=np.uint8)
        for dim in range(SPATIAL_INDEX_DIMS):
            for kind, position in (("min", 1), ("max", 2)):
                attr = f"spatial_{kind}_{dim}"
                if attr in attr_names:
                    rows[attr] = np.array([bound[position][dim] for bound in bounds], dtype=np.float64)
        with tiledb.open(timestamp_uri, "w") as timestamp_array:
            timestamp_array[start:stop] = rows
