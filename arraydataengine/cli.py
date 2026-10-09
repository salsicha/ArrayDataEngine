"""Command-line interface for ArrayDataEngine.

Installed as the `ade` console script:

    ade info PATH               # topics, counts, duration, optional stats
    ade topics PATH             # one topic per line
    ade export PATH -t TOPIC -o out.npz
    ade ingest PATH -o /path/to/store/
    ade viewer PATH -t TOPIC -o viewer.html
    ade demo -o demo_dir/       # synthetic data end-to-end showcase

Point clouds of any size are read by default; `--max-points N` skips clouds
with more than N points (and sets the per-message capacity of new point-cloud
topics in `ade ingest` stores, 30000 rows by default).

Common errors (missing or unreadable files, missing optional dependencies,
bad arguments) print a one-line message and exit with status 2; Ctrl-C exits
with status 130. Pass --debug (or set ADE_DEBUG=1) for the full traceback.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sqlite3
import sys
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

import numpy as np


class CLIError(Exception):
    """A user-facing error: printed as one line, exit status 2."""


# Optional dependency -> extra that installs it, for ModuleNotFoundError hints.
_EXTRA_FOR_MODULE = {
    "pyarrow": "arrow",
    "tiledb": "tiledb",
    "rosbags": "ros",
    "cv2": "image",
    "skimage": "image",
    "scipy": "image",
    "matplotlib": "visualization",
    "PIL": "visualization",
    "open3d": "visualization",
    "IPython": "visualization",
    "requests": "dem",
    "torch": "ml",
}

# Exceptions raised by these packages while opening or reading a recording
# are input problems (corrupt, truncated, or incomplete files), not bugs.
_READER_PACKAGES = frozenset({"rosbags", "apsw", "mcap", "lz4", "zstandard", "yaml", "ruamel"})

_RECORDING_SUFFIXES = (".bag", ".db3", ".mcap")
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"

# Per-store record of the recording an `ade ingest` store was written from.
SOURCE_FINGERPRINT = "ade_source.json"
_FINGERPRINT_FORMAT = 1
_SAMPLE_BYTES = 64 * 1024
_SAMPLED_FILES = 32

# Entries that do not make a directory "non-empty" for `ade ingest`.
_IGNORED_STORE_ENTRIES = frozenset({".gitkeep", ".keep", "lost+found", ".DS_Store"})
_TILEDB_MARKERS = frozenset({"__tiledb_group.tdb", "__group", "__meta"})
_ARROW_TOPIC_DIR = re.compile(r"topic-.+\.data(\.\d+)?")
_ARROW_MANIFEST = "manifest.json"
_ARROW_MANIFEST_KEYS = frozenset({"topic", "count", "shape", "dtype"})

_TRUE_WORDS = frozenset({"1", "true", "yes", "on", "y", "t"})


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {number}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be zero or a positive integer, got {number}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}") from None
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value}")
    return number


def _env_flag(value: str | None) -> bool:
    """Parse a boolean environment variable: 1/true/yes/on (any case) or a
    non-zero integer are true; everything else, including 0/false/no, is false."""

    if value is None:
        return False
    text = value.strip().lower()
    if text in _TRUE_WORDS:
        return True
    try:
        return int(text) != 0
    except ValueError:
        return False


def _accepts_keyword(func, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip()
    first = text.splitlines()[0].strip() if text else ""
    return first or type(exc).__name__


# --- opening sources ---------------------------------------------------------------


def _recording_files(path: str) -> list[Path]:
    """Files that back a source path: the file itself, the recording files of a
    directory, or the matches of a glob pattern."""

    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        return sorted(
            entry for entry in candidate.iterdir()
            if entry.is_file() and (entry.suffix.lower() in _RECORDING_SUFFIXES or entry.name == "metadata.yaml")
        )
    return [Path(match) for match in sorted(glob.glob(path)) if os.path.isfile(match)]


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _check_recording_files(path: str) -> None:
    """Explain unfetched Git LFS pointers and empty recordings up front: the
    readers otherwise fail with "file is not a database" or "File magic is
    invalid" (or ask for an extra the real file would not need)."""

    for candidate in _recording_files(path):
        if _is_lfs_pointer(candidate):
            raise CLIError(
                f"{candidate} is a Git LFS pointer, not the recording; run `git lfs pull` to fetch it"
            )
        if candidate.suffix.lower() in _RECORDING_SUFFIXES:
            try:
                empty = candidate.stat().st_size == 0
            except OSError:
                empty = False
            if empty:
                raise CLIError(f"{candidate} is empty (0 bytes); it is not a readable recording")


def _is_reader_error(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.Error):
        return True
    for cls in type(exc).__mro__:
        if cls.__module__.split(".")[0] in _READER_PACKAGES:
            return True
    return False


def _open_source(path: str, **options):
    """Open `path` with DataSources and read its topic list, turning reader
    failures (corrupt, truncated, or incomplete recordings) into CLIError.

    `options` (e.g. ``max_points``) are passed to DataSources.
    """

    from .source import DataSources

    # Check up front: a missing bag path otherwise falls back to scanning its
    # parent directory, which can silently open a different recording.
    if path != "DEM" and not os.path.exists(path) and not glob.glob(path):
        raise FileNotFoundError(f"no such file or directory: {path}")
    if path != "DEM":
        _check_recording_files(path)
    try:
        source = DataSources(path, **options)
        # Reading the topic list opens the recording, so unreadable inputs
        # fail here with a one-line message instead of mid-command.
        source.get_topics()
    except ValueError as exc:
        detail = _one_line(exc)
        if "not supported file type" in detail:
            detail = "unsupported file type; expected one of " + detail.rsplit(":", 1)[-1].strip()
        raise CLIError(f"cannot open {path}: {detail}") from exc
    except Exception as exc:
        if _is_reader_error(exc):
            raise CLIError(f"cannot read {path}: {_one_line(exc)}") from exc
        raise
    return source


def _source_for(args, topics=None):
    """Open the source named on the command line with its reader options
    (only the options that were given are passed on)."""

    options = {}
    max_points = getattr(args, "max_points", None)
    if max_points is not None:
        options["max_points"] = max_points
    if topics is not None:
        options["topics"] = list(topics)
    return _open_source(args.path, **options)


def _topic_source(args, source, topic: str):
    """A source that reads (decodes) only `topic`, for streaming one topic.

    A DataSources is reopened restricted to the topic when it supports that;
    any other source is used as is (`_iter_topic` filters its messages).
    The unrestricted `source` stays in use for metadata and error messages.
    """

    from .source import DataSources

    if isinstance(source, DataSources) and _accepts_keyword(DataSources.__init__, "topics"):
        return _source_for(args, topics=[topic])
    return source


def _require_topic(source, topic: str, max_points: int | None) -> None:
    """Fail fast, before streaming, when `topic` is missing or undecodable."""

    try:
        topics = list(source.get_topics())
    except Exception:
        return
    if topic not in topics:
        raise _no_messages_error(source, topic, max_points)


def _frame_text(value) -> str | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if value is None or value == "":
        return None
    return str(value)


def _topic_messages(source, topic: str):
    """The source's message stream, restricted to `topic` when the source can
    skip decoding other topics."""

    getter = source.get_message
    if _accepts_keyword(getter, "topics"):
        return getter(topics=[topic])
    return getter()


def _iter_topic(source, topic: str, limit: int | None = None, stride: int = 1):
    """Yield every `stride`-th message of `topic`, at most `limit` of them."""

    seen = 0
    kept = 0
    for message in _topic_messages(source, topic):
        if message["topic"] != topic:
            continue
        if seen % stride == 0:
            yield message
            kept += 1
            if limit is not None and kept >= limit:
                return
        seen += 1


# --- topic types and skip reasons -------------------------------------------------


def _topic_types(source) -> dict:
    """topic -> recorded message type (including undecodable topics), if known."""

    inner = getattr(source, "source", None)
    for owner in (source, inner):
        getter = getattr(owner, "get_topic_types", None)
        if callable(getter):
            try:
                types = getter()
            except Exception:
                types = None
            if isinstance(types, dict) and types:
                return {str(topic): str(msgtype) for topic, msgtype in types.items()}
    return {}


def _topic_msgtype(source, topic: str) -> str | None:
    """Best-effort message type lookup for error messages."""

    msgtype = _topic_types(source).get(topic)
    if msgtype:
        return msgtype

    inner = getattr(source, "source", None)
    # rosbag2 sqlite storage: read the topics table of the requested file(s).
    db_paths = getattr(inner, "_db3_paths", None)
    if callable(db_paths):
        try:
            for db_path in db_paths():
                uri = Path(db_path).resolve().as_uri() + "?mode=ro"
                with closing(sqlite3.connect(uri, uri=True)) as connection:
                    row = connection.execute(
                        "select type from topics where name = ? limit 1", (topic,)
                    ).fetchone()
                if row:
                    return str(row[0])
        except Exception:
            pass

    reader = getattr(inner, "reader", None)
    if callable(reader):
        try:
            with reader() as opened:
                for connection in opened.connections:
                    if connection.topic == topic:
                        return str(connection.msgtype)
        except Exception:
            pass
    return None


def _short_type(msgtype: str) -> str:
    return str(msgtype).rsplit("/", 1)[-1].rsplit("__", 1)[-1].lower()


def _undecodable_reason(source, msgtype: str) -> str:
    """Why messages of `msgtype` cannot be decoded, with the extra to install."""

    short = _short_type(msgtype)
    if short == "compressedimage" and not _module_available("cv2"):
        return f'{msgtype} needs OpenCV; install it with: pip install "arraydataengine[image]"'
    inner = getattr(source, "source", None)
    sensor_class = getattr(inner, "_sensor_class", None)
    has_converter = False
    if callable(sensor_class):
        try:
            has_converter = sensor_class(msgtype) is not None
        except Exception:
            has_converter = False
    if has_converter and not _module_available("rosbags"):
        return f'{msgtype} needs rosbags; install it with: pip install "arraydataengine[ros]"'
    return f"unsupported message type {msgtype}"


def _skip_reason(msgtype: str | None, max_points: int | None) -> str:
    if max_points is not None and (msgtype is None or _short_type(msgtype) == "pointcloud2"):
        return (
            f"point clouds with more than --max-points {max_points} points are skipped; "
            "raise --max-points or omit it"
        )
    return "they failed to decode; see the warnings above"


def _no_messages_error(source, topic: str, max_points: int | None = None) -> CLIError:
    try:
        topics = list(source.get_topics())
    except Exception:
        topics = []

    if topic in topics:
        try:
            count = int(source.get_count(topic))
        except Exception:
            count = 0
        if count > 0:
            reason = _skip_reason(_topic_msgtype(source, topic), max_points)
            return CLIError(f"all {count} messages of topic {topic!r} were skipped: {reason}")
        return CLIError(f"no messages found for topic {topic!r}")

    msgtype = _topic_msgtype(source, topic)
    if msgtype:
        return CLIError(f"topic {topic!r} cannot be decoded: {_undecodable_reason(source, msgtype)}")
    if topics:
        return CLIError(f"topic {topic!r} not found; available topics: {', '.join(topics)}")
    return CLIError(f"no messages found for topic {topic!r}")


# --- stacking ------------------------------------------------------------------------


def _row_counts(arrays: list[np.ndarray], point_counts) -> list[int] | None:
    """Real row count per message from `point_count` (point clouds; older
    sources zero-pad clouds past it), or None when no message carries one."""

    if point_counts is None or all(count is None for count in point_counts):
        return None
    counts = []
    for array, count in zip(arrays, point_counts):
        if array.ndim == 0:
            raise ValueError("messages have incompatible shapes; export them individually")
        rows = array.shape[0]
        if count is None:
            counts.append(rows)
            continue
        count = int(count)
        if count < 0 or count > rows:
            raise ValueError(f"point_count {count} does not fit a message of {rows} rows")
        counts.append(count)
    return counts


def _stack_ragged(values: list, point_counts=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Stack messages into one array, zero-padding a ragged leading dimension
    (variable-size point clouds) to the largest message.

    `point_counts` (one entry per message, None where unknown) marks point
    clouds: their first `point_count` rows are real, so each message is
    trimmed to it, the stack is padded to the largest count (never to a
    fixed source size), and lengths are always returned.

    Returns ``(stacked, lengths)``; `lengths` holds each message's real
    leading size for point clouds or when padding was applied, else None.
    Raises ValueError when the shapes cannot be reconciled.
    """

    arrays = [np.asarray(value) for value in values]
    counts = _row_counts(arrays, point_counts)
    if counts is not None:
        arrays = [array[:count] for array, count in zip(arrays, counts)]
    shapes = {arr.shape for arr in arrays}
    if len(shapes) == 1:
        stacked = np.stack(arrays) if arrays else np.asarray(arrays)
        lengths = None if counts is None else np.asarray(counts, dtype=np.int64)
        return stacked, lengths

    trailing = {shape[1:] for shape in shapes}
    if len(trailing) != 1 or any(len(shape) == 0 for shape in shapes):
        raise ValueError("messages have incompatible shapes; export them individually")

    rows = max(shape[0] for shape in shapes)
    dtype = np.result_type(*{arr.dtype for arr in arrays})
    stacked = np.zeros((len(arrays), rows) + arrays[0].shape[1:], dtype=dtype)
    lengths = np.empty(len(arrays), dtype=np.int64)
    for index, arr in enumerate(arrays):
        stacked[index, : arr.shape[0]] = arr
        lengths[index] = arr.shape[0]
    return stacked, lengths


def _stack_messages(values: list[np.ndarray]) -> np.ndarray:
    """Stack messages into one array, zero-padding a ragged leading dimension
    (variable-size point clouds) when the trailing dimensions agree."""

    try:
        return _stack_ragged(values)[0]
    except ValueError as exc:
        raise CLIError(f"topic {exc}") from exc


def _shape_text(data: np.ndarray, lengths: np.ndarray | None) -> str | None:
    """`(677..1619, 3)` for ragged topics; None to keep the regular shape."""

    if lengths is None or lengths.size == 0:
        return None
    low, high = int(lengths.min()), int(lengths.max())
    dims = [str(low) if low == high else f"{low}..{high}", *(str(dim) for dim in data.shape[2:])]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def _collect_topic(source, topic: str, limit: int | None = None, stride: int = 1,
                   max_points: int | None = None, stream=None) -> dict:
    """Stack the messages of `topic` read from `stream` (default: `source`);
    `source` answers metadata questions for error messages."""

    ids = []
    timestamps = []
    values = []
    point_counts = []
    frames = []
    stream = source if stream is None else stream
    for message in _iter_topic(stream, topic, limit=limit, stride=stride):
        ids.append(message.get("name"))
        timestamps.append(message["timestamp"])
        values.append(message["data"])
        point_counts.append(message.get("point_count"))
        frames.append(_frame_text(message.get("frame_id")))
    if not values:
        raise _no_messages_error(source, topic, max_points)

    try:
        data, lengths = _stack_ragged(values, point_counts)
    except ValueError as exc:
        raise CLIError(f"topic {topic!r}: {exc}") from exc
    result = {
        "id": np.asarray(ids, dtype=object),
        "name": np.asarray(ids, dtype=object),
        "ts": np.asarray(timestamps, dtype=np.float64),
        "data": data,
        "topic": topic,
    }
    distinct = {frame for frame in frames if frame is not None}
    if len(distinct) == 1 and None not in frames:
        result["frame_id"] = next(iter(distinct))
    elif distinct:
        # Several frames (or some messages without one): keep them per row.
        result["frame_ids"] = np.asarray(frames, dtype=object)
    if lengths is not None:
        result["lengths"] = lengths
    return result


# --- commands -----------------------------------------------------------------------


def _cmd_topics(args) -> int:
    source = _source_for(args)
    for topic in source.get_topics():
        print(topic)
    return 0


def _cmd_info(args) -> int:
    source = _source_for(args)
    topics = source.get_topics()
    types = _topic_types(source)
    undecodable = [topic for topic in types if topic not in topics]
    print(f"source: {args.path}")
    try:
        duration = source.source.get_duration()
        print(f"duration: {duration:.3f} s")
    except (ValueError, AttributeError, TypeError):
        pass
    print(f"topics ({len(topics) + len(undecodable)}):")
    for topic in topics:
        print(f"  {topic}: {source.get_count(topic)} messages")
    for topic in undecodable:
        print(f"  {topic}: not decodable ({_undecodable_reason(source, types[topic])})")

    if args.messages:
        from .ops import describe_topic, format_describe

        summaries = {}
        skipped = []
        collected: dict[str, dict] = {
            topic: {"ts": [], "data": [], "counts": [], "frames": set()} for topic in topics
        }
        seen = 0
        exhausted = True
        for message in source.get_message():
            if seen >= args.messages:
                exhausted = False
                break
            slot = collected.get(message["topic"])
            if slot is not None:
                slot["ts"].append(message["timestamp"])
                slot["data"].append(message["data"])
                slot["counts"].append(message.get("point_count"))
                frame = _frame_text(message.get("frame_id"))
                if frame is not None:
                    slot["frames"].add(frame)
            seen += 1
        print(f"\nstats over the first {seen} messages:")
        for topic, slot in collected.items():
            if not slot["ts"]:
                continue
            try:
                data, lengths = _stack_ragged(slot["data"], slot["counts"])
            except ValueError:
                skipped.append(topic)
                continue
            topic_data = {"ts": np.asarray(slot["ts"], dtype=np.float64), "data": data, "topic": topic}
            if len(slot["frames"]) == 1:
                topic_data["frame_id"] = next(iter(slot["frames"]))
            summary = describe_topic(topic_data, name=topic)
            shape = _shape_text(data, lengths)
            if shape is not None:
                # Show the real row range rather than the zero-padded size.
                summary["shape"] = shape
            summaries[topic] = summary
        if summaries:
            print(format_describe(summaries))
        for topic in skipped:
            print(f"{topic}: stats skipped (messages have incompatible shapes)")
        if exhausted:
            # The whole recording was read: topics that yielded nothing were skipped.
            for topic, slot in collected.items():
                count = source.get_count(topic)
                if not slot["ts"] and count:
                    reason = _skip_reason(_topic_msgtype(source, topic), args.max_points)
                    print(f"{topic}: all {count} messages were skipped ({reason})")
    return 0


def _written_npz_path(returned, requested) -> Path:
    """The file actually written: numpy appends ``.npz`` to bare names."""

    path = Path(returned if returned is not None else requested)
    if not path.is_file() and not str(path).endswith(".npz"):
        candidate = Path(str(path) + ".npz")
        if candidate.is_file():
            return candidate
    return path


def _append_npz_array(path: Path, key: str, array: np.ndarray) -> bool:
    """Add one array to an existing ``.npz`` archive (as np.savez stores it)."""

    member = f"{key}.npy"
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        if member in archive.namelist():
            return False
        with archive.open(member, "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.asarray(array), allow_pickle=False)
    return True


def _topic_file_stem(topic: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", topic.strip("/")).strip("._")
    return stem or "topic"


def _output_file(out: str, topic: str, suffix: str) -> Path:
    """`out`, or ``out/<topic><suffix>`` when `out` names a directory."""

    path = Path(out)
    if path.is_dir() or out.endswith(("/", os.sep)):
        return path / (_topic_file_stem(topic) + suffix)
    return path


def _prepare_output_file(path: Path) -> None:
    """Create the output's parent directories up front, so a bad location
    fails before the input is streamed."""

    if path.is_dir():
        raise CLIError(f"output {path} is a directory")
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise CLIError(f"cannot write {path}: {parent} is not a directory")
    parent.mkdir(parents=True, exist_ok=True)


def _cmd_export(args) -> int:
    from .ops import save_topic_npz

    out = _output_file(args.out, args.topic, ".npz")
    source = _source_for(args)
    _require_topic(source, args.topic, args.max_points)
    _prepare_output_file(out)
    topic_data = _collect_topic(
        source, args.topic, limit=args.limit, stride=args.stride, max_points=args.max_points,
        stream=_topic_source(args, source, args.topic),
    )
    lengths = topic_data.get("lengths")
    out = _written_npz_path(save_topic_npz(out, topic_data), out)
    count = topic_data["ts"].shape[0]
    print(f"wrote {count} messages of {args.topic} to {out}")
    if lengths is not None:
        _append_npz_array(out, "lengths", lengths)
        rows = topic_data["data"].shape[1]
        if lengths.size and int(lengths.min()) < rows:
            print(
                f"  variable-size messages were zero-padded to {rows} rows (the largest "
                "message); per-message row counts are stored under 'lengths'"
            )
        else:
            print("  per-message point counts are stored under 'lengths'")
    return 0


class _CountingSource:
    """Forwards to a source, counting the messages it yields per topic, and
    stops before a point cloud that cannot fit the store's row capacity."""

    def __init__(self, source, capacity: int | None = None):
        self._source = source
        self._capacity = capacity
        self.yielded: dict[str, int] = {}
        self.exhausted = False

    def __getattr__(self, name):
        return getattr(self._source, name)

    def get_topics(self):
        return self._source.get_topics()

    def get_count(self, axis):
        return self._source.get_count(axis)

    def get_message(self):
        for message in self._source.get_message():
            topic = message["topic"]
            count = message.get("point_count")
            if self._capacity is not None and count is not None and int(count) > self._capacity:
                raise CLIError(
                    f"topic {topic!r} has a point cloud of {int(count)} points, more than the "
                    f"{self._capacity}-row capacity of new store topics; re-run into a new or "
                    f"empty directory with --max-points {int(count)} (or larger)"
                )
            self.yielded[topic] = self.yielded.get(topic, 0) + 1
            yield message
        self.exhausted = True


def _arrow_manifest(path: Path) -> dict | None:
    """The parsed manifest of an Arrow topic directory, or None."""

    try:
        manifest = json.loads((path / _ARROW_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(manifest, dict) or not _ARROW_MANIFEST_KEYS <= manifest.keys():
        return None
    if not isinstance(manifest["count"], int) or isinstance(manifest["count"], bool):
        return None
    if not isinstance(manifest["topic"], str):
        return None
    fragments = manifest.get("fragments")
    if fragments is not None and not isinstance(fragments, list):
        return None
    return manifest


def _is_arrow_topic_dir(entry: Path) -> bool:
    if not entry.is_dir():
        return False
    if (entry / _ARROW_MANIFEST).exists():
        manifest = _arrow_manifest(entry)
        if manifest is None:
            return False
        # Current stores name topic directories topic-<escaped>.data[.N];
        # 0.3.x stores used the topic name and wrote no fragment list.
        return bool(_ARROW_TOPIC_DIR.fullmatch(entry.name)) or "fragments" not in manifest
    # A topic directory whose first flush was interrupted has no manifest yet.
    return bool(_ARROW_TOPIC_DIR.fullmatch(entry.name))


def _is_tiledb_group(path: Path) -> bool:
    try:
        import tiledb
    except ImportError:
        # Without tiledb, trust the group markers (the backend import then
        # fails with an install hint).
        return (path / "__tiledb_group.tdb").is_file() or (path / "__group").is_dir()
    try:
        return tiledb.object_type(str(path)) == "group"
    except Exception:
        return False


def _store_entries(path: Path) -> list[Path]:
    return [
        entry for entry in path.iterdir()
        if entry.name not in _IGNORED_STORE_ENTRIES and entry.name != SOURCE_FINGERPRINT
    ]


def _store_kind(path: Path) -> str:
    """Classify an ingest destination: ``"missing"``, ``"empty"``,
    ``"tiledb"``, ``"arrow"``, ``"file"``, or ``"other"`` (a non-empty
    directory that is not a store)."""

    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "file"
    entries = _store_entries(path)
    if not entries:
        return "empty"
    names = {entry.name for entry in entries}
    if names & _TILEDB_MARKERS:
        return "tiledb" if _is_tiledb_group(path) else "other"
    # Every entry must be an Arrow topic directory: a project directory
    # that merely contains some <dir>/manifest.json is not a store.
    if all(_is_arrow_topic_dir(entry) for entry in entries):
        return "arrow"
    return "other"


def _check_writable(path: Path, kind: str) -> None:
    """Fail before reading the input when the store cannot be written."""

    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if not probe.is_dir():
        raise CLIError(f"cannot create {path}: {probe} is not a directory")
    targets = [probe]
    if kind == "arrow":
        targets.extend(entry for entry in _store_entries(path) if entry.is_dir())
    for target in targets:
        if not os.access(target, os.W_OK | os.X_OK):
            raise CLIError(f"cannot write to {path}: permission denied for {target}")


def _sample_digest(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(_SAMPLE_BYTES))
        if size > 2 * _SAMPLE_BYTES:
            handle.seek(size - _SAMPLE_BYTES)
            digest.update(handle.read(_SAMPLE_BYTES))
        elif size > _SAMPLE_BYTES:
            digest.update(handle.read())
    return digest.hexdigest()


def _source_fingerprint(path: str, source, max_points: int | None) -> dict:
    """Identify a recording by its files' sizes and sampled contents (not
    their names or locations, so a moved or copied recording still matches)."""

    files = []
    identity = []
    recording_files = _recording_files(path) if path != "DEM" else []
    for index, file_path in enumerate(recording_files):
        size = file_path.stat().st_size
        files.append({"name": file_path.name, "size": size})
        sampled = index < _SAMPLED_FILES // 2 or index >= len(recording_files) - _SAMPLED_FILES // 2
        identity.append([size, _sample_digest(file_path, size) if sampled else None])
    topics = {}
    for topic in source.get_topics():
        try:
            topics[topic] = int(source.get_count(topic))
        except Exception:
            topics[topic] = None
    if not identity:
        identity = sorted(topics.items())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    resolved = path if path == "DEM" else str(Path(path).absolute().resolve())
    return {
        "format": _FINGERPRINT_FORMAT,
        "source": resolved,
        "files": files,
        "topics": topics,
        "max_points": max_points,
        "digest": digest,
    }


def _read_fingerprint(store: Path) -> dict | None:
    path = store / SOURCE_FINGERPRINT
    if not path.exists():
        return None
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        recorded = None
    if not isinstance(recorded, dict) or "digest" not in recorded:
        return {"source": "an unknown source", "digest": None, "unreadable": True}
    return recorded


def _write_fingerprint(store: Path, fingerprint: dict) -> None:
    store.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{SOURCE_FINGERPRINT}.", suffix=".tmp", dir=store)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(fingerprint, handle, indent=1, sort_keys=True)
        os.chmod(temporary, 0o666 & ~_umask())
        os.replace(temporary, store / SOURCE_FINGERPRINT)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _store_counts(store: Path, kind: str) -> dict[str, int]:
    from .buffer import DataBuffer

    # Without a source the store is opened read-only and is never rewritten.
    reader = DataBuffer(None, data_uri=str(store), use_db=True, backend=kind, preload=0)
    try:
        return {topic: int(count) for topic, count in reader.counters.items()}
    finally:
        reader.close()


def _fingerprint_problem(recorded: dict, fingerprint: dict) -> str | None:
    if recorded.get("unreadable"):
        return f"its {SOURCE_FINGERPRINT} is unreadable"
    if recorded.get("digest") != fingerprint["digest"]:
        return f"it holds data ingested from a different recording ({recorded.get('source')})"
    if recorded.get("max_points") != fingerprint["max_points"]:
        previous = recorded.get("max_points")
        option = "no --max-points" if previous is None else f"--max-points {previous}"
        return f"it was ingested with {option}; resume it with the same setting"
    return None


def _counts_problem(stored: dict[str, int], fingerprint: dict) -> str | None:
    """Store/source consistency for stores written before fingerprints existed."""

    source_counts = fingerprint["topics"]
    for topic, count in stored.items():
        if not count:
            continue
        if topic not in source_counts:
            return f"it holds topic {topic!r}, which this recording does not have"
        expected = source_counts[topic]
        if expected is not None and count > expected:
            return f"it holds {count} messages of {topic!r}, more than this recording's {expected}"
    return None


def _cmd_ingest(args) -> int:
    from .buffer import DataBuffer
    from .sensors.pointcloud2_sensor import DEFAULT_MAX_POINTS

    out = Path(args.out)
    kind = _store_kind(out)
    if kind == "file":
        raise CLIError(f"output {out} exists and is not a directory")
    if kind == "other":
        raise CLIError(
            f"output {out} is a non-empty directory that is not an ArrayDataEngine store; "
            "choose a new or empty directory"
        )
    if args.backend is not None and kind in ("arrow", "tiledb") and kind != args.backend:
        raise CLIError(
            f"output {out} already holds an existing {kind} store; pass --backend {kind} to "
            "resume it or choose another directory"
        )
    _check_writable(out, kind)

    source = _source_for(args)
    topics = source.get_topics()
    if not topics:
        raise CLIError("source has no decodable topics; run `ade info` on it for details")

    fingerprint = _source_fingerprint(args.path, source, args.max_points)
    # A directory without data (new, empty, or holding only a fingerprint
    # left by a run that stopped before writing) is (re)recorded.
    recorded = _read_fingerprint(out) if kind in ("arrow", "tiledb") else None
    record = recorded is None
    if recorded is not None:
        problem = _fingerprint_problem(recorded, fingerprint)
    elif kind in ("arrow", "tiledb"):
        # A store from before fingerprints: adopt it only when its contents
        # are consistent with this recording.
        problem = _counts_problem(_store_counts(out, kind), fingerprint)
        record = problem is None
    else:
        problem = None
    if problem is not None and not args.force:
        raise CLIError(
            f"refusing to resume {out}: {problem}; choose another directory, or pass "
            "--force to append to it anyway"
        )

    backend = kind if kind in ("arrow", "tiledb") else (args.backend or "arrow")
    capacity = DEFAULT_MAX_POINTS if args.max_points is None else None
    counting = _CountingSource(source, capacity=capacity)
    try:
        with DataBuffer(
            data_source=counting,
            data_uri=str(out),
            axis=topics[0],
            use_db=True,
            backend=backend,
            preload=0,
        ) as buffer:
            before = dict(buffer.counters)
            if record:
                _write_fingerprint(out, fingerprint)
            buffer.load_data_db(topics[0])
            after = dict(buffer.counters)
            backend = buffer.backend
    except ValueError as exc:
        # Schema or capacity conflicts reported by the store.
        raise CLIError(f"cannot ingest into {out}: {_one_line(exc)}") from exc

    appended = {topic: after.get(topic, 0) - before.get(topic, 0) for topic in after}
    print(
        f"ingested {sum(appended.values())} new messages into {args.out} ({backend}); "
        f"the store holds {sum(after.values())}"
    )
    all_skipped = []
    for topic in dict.fromkeys([*topics, *after]):
        line = f"  {topic}: {appended.get(topic, 0)} new, {after.get(topic, 0)} total"
        try:
            expected = int(source.get_count(topic)) if topic in topics else 0
        except Exception:
            expected = 0
        # Only a fully read source tells how many messages were skipped.
        skipped = expected - counting.yielded.get(topic, 0) if counting.exhausted else 0
        if skipped > 0:
            line += f" ({skipped} of {expected} messages skipped)"
            if counting.yielded.get(topic, 0) == 0:
                all_skipped.append((topic, expected))
        print(line)
    if all_skipped:
        topic, expected = all_skipped[0]
        reason = _skip_reason(_topic_msgtype(source, topic), args.max_points)
        names = ", ".join(repr(name) for name, _ in all_skipped)
        raise CLIError(f"every message of {names} was skipped: {reason}")
    return 0


def _cmd_viewer(args) -> int:
    from .visualizers.point_cloud import VisTool

    out = _output_file(args.out, args.topic, ".html")
    source = _source_for(args)
    _require_topic(source, args.topic, args.max_points)
    _prepare_output_file(out)
    stream = _topic_source(args, source, args.topic)
    tool = None
    # Stream scans straight into the viewer (which keeps a bounded, uniform
    # subsample) instead of materializing the whole topic.
    for message in _iter_topic(stream, args.topic, limit=args.limit, stride=args.stride):
        data = np.asarray(message["data"])
        if data.ndim != 2 or data.shape[-1] < 3 or not np.issubdtype(data.dtype, np.number):
            raise CLIError(
                f"topic {args.topic!r} has messages of shape {data.shape} ({data.dtype}), "
                "not point clouds (N, 3+)"
            )
        if tool is None:
            tool = VisTool(embed=True, backend="html", output_path=out)
        tool.add_point_cloud(data, length=message.get("point_count"))
    if tool is None:
        raise _no_messages_error(source, args.topic, args.max_points)
    tool.show()
    return 0


def _cmd_demo(args) -> int:
    from .buffer import DataBuffer
    from .ops import (
        describe_dataset,
        format_describe,
        odometry_to_trajectory,
        pose_to_matrix,
        save_topic_npz,
        write_tum_trajectory,
    )
    from .sources.synthetic_source import SyntheticSource
    from .visualizers.point_cloud import VisTool

    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        raise CLIError(f"output {out_dir} exists and is not a directory")

    source = SyntheticSource(duration=args.duration, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Size the ring buffer to the largest topic so every message fits without
    # preallocating far more than the demo produces.
    depth = max([1, *(source.get_count(topic) for topic in source.get_topics())])
    buffer = DataBuffer(source, buffer_depth=depth, axis="/points", use_db=False, preload=0)
    for _ in range(source.get_count("/points")):
        buffer.roll_buffer("/points")

    print(format_describe(describe_dataset(buffer)))

    odom = buffer.get_index_range("/odom")
    trajectory = odometry_to_trajectory(odom)
    tum_path = write_tum_trajectory(out_dir / "trajectory.tum", trajectory)

    points = buffer.get_index_range("/points")
    npz_path = _written_npz_path(save_topic_npz(out_dir / "points.npz", points), out_dir / "points.npz")

    # Stitch scans with the odometry poses closest in time to each scan.
    viewer_path = out_dir / "viewer.html"
    tool = VisTool(embed=True, backend="html", output_path=viewer_path)
    odom_ts = odom["ts"]
    lengths = points.get("lengths")
    for index, (scan_ts, scan) in enumerate(zip(points["ts"], points["data"])):
        pose_row = odom["data"][int(np.argmin(np.abs(odom_ts - scan_ts)))]
        pose = np.concatenate((pose_row[0, :3], pose_row[2, :4]))
        matrix = pose_to_matrix(pose)
        tool.update(scan, matrix, length=None if lengths is None else int(lengths[index]))
        tool.add_pose_arrow(matrix)
    tool.show()

    print(f"wrote {tum_path}, {npz_path}, and {viewer_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="ade", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"arraydataengine {__version__}")
    debug_help = "show full tracebacks instead of one-line errors (or set ADE_DEBUG=1)"
    parser.add_argument("--debug", action="store_true", help=debug_help)

    # --debug is also accepted after the subcommand; SUPPRESS keeps the
    # subcommand from resetting a --debug given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=debug_help)

    reader = argparse.ArgumentParser(add_help=False)
    reader.add_argument(
        "--max-points", type=_positive_int, default=None, metavar="N",
        help="skip point clouds with more than N points (default: no limit)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show topics, counts, and duration", parents=[common, reader])
    p_info.add_argument("path")
    p_info.add_argument("--messages", type=_non_negative_int, default=0, metavar="N",
                        help="also stream the first N messages and print per-topic stats")
    p_info.set_defaults(fn=_cmd_info)

    p_topics = sub.add_parser("topics", help="list decodable topics, one per line",
                              parents=[common, reader])
    p_topics.add_argument("path")
    p_topics.set_defaults(fn=_cmd_topics)

    p_export = sub.add_parser(
        "export",
        help="export one topic to a .npz file",
        description="Export one topic to a .npz file. Variable-size messages (point "
        "clouds) are zero-padded to the largest exported message; their real row "
        "counts are stored in the archive's 'lengths' array.",
        parents=[common, reader],
    )
    p_export.add_argument("path")
    p_export.add_argument("-t", "--topic", required=True)
    p_export.add_argument("-o", "--out", required=True,
                          help="output file (numpy appends .npz when missing), or a directory "
                          "to write <topic>.npz into")
    p_export.add_argument("--limit", type=_positive_int, default=None,
                          help="max messages to export (positive)")
    p_export.add_argument("--stride", type=_positive_int, default=1,
                          help="keep every k-th message (positive)")
    p_export.set_defaults(fn=_cmd_export)

    p_ingest = sub.add_parser(
        "ingest",
        help="ingest a source into a persistent store",
        description="Ingest every topic of a source into a persistent store. An existing "
        "store is reopened with its own backend and resumed; a non-empty directory that is "
        "not a store, a store of a different backend than --backend, or a store written from "
        f"a different recording (recorded in its {SOURCE_FINGERPRINT}) is refused. "
        "Point-cloud topics store up to --max-points rows per message (30000 when omitted).",
        parents=[common, reader],
    )
    p_ingest.add_argument("path")
    p_ingest.add_argument("-o", "--out", required=True,
                          help="store directory (new, empty, or an existing store)")
    p_ingest.add_argument("--backend", choices=("arrow", "tiledb"), default=None,
                          help="storage engine for a new store (default: arrow); an existing "
                          "store keeps its detected backend")
    p_ingest.add_argument("--force", action="store_true",
                          help="append to an existing store even when it was written from a "
                          "different recording or with a different --max-points")
    p_ingest.set_defaults(fn=_cmd_ingest)

    p_viewer = sub.add_parser("viewer", help="write an interactive HTML point-cloud viewer",
                              parents=[common, reader])
    p_viewer.add_argument("path")
    p_viewer.add_argument("-t", "--topic", required=True)
    p_viewer.add_argument("-o", "--out", default="ade_pointcloud_viewer.html",
                          help="output HTML file, or a directory to write <topic>.html into")
    p_viewer.add_argument("--limit", type=_positive_int, default=None,
                          help="max scans to include (positive)")
    p_viewer.add_argument("--stride", type=_positive_int, default=1,
                          help="keep every k-th scan (positive)")
    p_viewer.set_defaults(fn=_cmd_viewer)

    p_demo = sub.add_parser("demo", help="run the synthetic-data showcase (no input files needed)",
                            parents=[common])
    p_demo.add_argument("-o", "--out", default="ade_demo", help="output directory")
    p_demo.add_argument("--duration", type=_positive_float, default=5.0,
                        help="seconds of synthetic data (positive)")
    p_demo.add_argument("--seed", type=_non_negative_int, default=0,
                        help="random seed (zero or a positive integer)")
    p_demo.set_defaults(fn=_cmd_demo)

    return parser


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        leaves = []
        for inner in exc.exceptions:
            leaves.extend(_leaf_exceptions(inner))
        return leaves
    return [exc]


def _error_message(exc: BaseException) -> str | None:
    """One-line message for common user errors; None means a real bug."""

    if isinstance(exc, BaseExceptionGroup):
        # e.g. every topic of a store failing to close on a read-only disk.
        messages = [_error_message(leaf) for leaf in _leaf_exceptions(exc)]
        if not messages or any(message is None for message in messages):
            return None
        unique = list(dict.fromkeys(messages))
        extra = f" (and {len(unique) - 1} more)" if len(unique) > 1 else ""
        return unique[0] + extra
    if isinstance(exc, CLIError):
        return str(exc)
    if isinstance(exc, ModuleNotFoundError):
        module = (exc.name or "").split(".")[0]
        if not module or module in ("arraydataengine", "ArrayDataEngine"):
            return None
        extra = _EXTRA_FOR_MODULE.get(module)
        hint = f'; install it with: pip install "arraydataengine[{extra}]"' if extra else ""
        return f"missing optional dependency {module!r}{hint}"
    if isinstance(exc, (FileNotFoundError, FileExistsError, NotADirectoryError,
                        IsADirectoryError, PermissionError)):
        return _one_line(exc)
    if isinstance(exc, OSError) and exc.errno is not None:
        # Environment failures such as a full disk or a read-only file system.
        return _one_line(exc)
    if _is_reader_error(exc):
        return f"cannot read the input: {_one_line(exc)}"
    return None


def _interrupted(exc: BaseException) -> bool:
    return any(isinstance(leaf, KeyboardInterrupt) for leaf in _leaf_exceptions(exc))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    debug = bool(getattr(args, "debug", False)) or _env_flag(os.environ.get("ADE_DEBUG"))
    try:
        return args.fn(args)
    except BaseException as exc:
        if isinstance(exc, (SystemExit, GeneratorExit)):
            raise
        if _interrupted(exc):
            # Stores were flushed and closed while the interrupt unwound.
            if debug:
                raise
            sys.stdout.flush()
            print("ade: interrupted", file=sys.stderr)
            return 130
        if not isinstance(exc, Exception):
            raise
        message = None if debug else _error_message(exc)
        if message is None:
            raise
        # Keep the error after any output already printed (pipes buffer stdout).
        sys.stdout.flush()
        print(f"ade: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
