"""Command-line interface for ArrayDataEngine.

Installed as the `ade` console script:

    ade info PATH               # topics, counts, duration, optional stats
    ade topics PATH             # one topic per line
    ade export PATH -t TOPIC -o out.npz
    ade ingest PATH -o /path/to/store/
    ade viewer PATH -t TOPIC -o viewer.html
    ade demo -o demo_dir/       # synthetic data end-to-end showcase

Common errors (missing files, missing optional dependencies, bad arguments)
print a one-line message and exit with status 2; pass --debug (or set
ADE_DEBUG=1) for the full traceback.
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
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


def _open_source(path: str):
    from .source import DataSources

    # Check up front: a missing bag path otherwise falls back to scanning its
    # parent directory, which can silently open a different recording.
    if path != "DEM" and not os.path.exists(path) and not glob.glob(path):
        raise FileNotFoundError(f"no such file or directory: {path}")
    try:
        return DataSources(path)
    except ValueError as exc:
        detail = str(exc).strip()
        if "not supported file type" in detail:
            detail = "unsupported file type; expected one of " + detail.rsplit(":", 1)[-1].strip()
        raise CLIError(f"cannot open {path}: {detail}") from exc


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


def _iter_topic(source, topic: str, limit: int | None = None, stride: int = 1):
    """Yield every `stride`-th message of `topic`, at most `limit` of them."""

    seen = 0
    kept = 0
    for message in source.get_message():
        if message["topic"] != topic:
            continue
        if seen % stride == 0:
            yield message
            kept += 1
            if limit is not None and kept >= limit:
                return
        seen += 1


def _topic_msgtype(source, topic: str) -> str | None:
    """Best-effort message type lookup for error messages."""

    inner = getattr(source, "source", None)
    for owner in (source, inner):
        getter = getattr(owner, "get_topic_types", None)
        if callable(getter):
            try:
                types = getter()
            except Exception:
                types = None
            if isinstance(types, dict) and types.get(topic):
                return str(types[topic])

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


def _no_messages_error(source, topic: str) -> CLIError:
    msgtype = _topic_msgtype(source, topic)
    if msgtype:
        return CLIError(
            f"topic {topic!r} has unsupported message type {msgtype} (no decodable messages)"
        )
    try:
        topics = list(source.get_topics())
    except Exception:
        topics = []
    if topics and topic not in topics:
        return CLIError(f"topic {topic!r} not found; available topics: {', '.join(topics)}")
    return CLIError(f"no messages found for topic {topic!r}")


def _collect_topic(source, topic: str, limit: int | None = None, stride: int = 1) -> dict:
    ids = []
    timestamps = []
    values = []
    frames = set()
    for message in _iter_topic(source, topic, limit=limit, stride=stride):
        ids.append(message.get("name"))
        timestamps.append(message["timestamp"])
        values.append(message["data"])
        frame = _frame_text(message.get("frame_id"))
        if frame is not None:
            frames.add(frame)
    if not values:
        raise _no_messages_error(source, topic)

    try:
        data, lengths = _stack_ragged(values)
    except ValueError as exc:
        raise CLIError(f"topic {topic!r}: {exc}") from exc
    result = {
        "id": np.asarray(ids, dtype=object),
        "name": np.asarray(ids, dtype=object),
        "ts": np.asarray(timestamps, dtype=np.float64),
        "data": data,
        "topic": topic,
    }
    if len(frames) == 1:
        result["frame_id"] = next(iter(frames))
    if lengths is not None:
        result["lengths"] = lengths
    return result


def _stack_ragged(values: list) -> tuple[np.ndarray, np.ndarray | None]:
    """Stack messages into one array, zero-padding a ragged leading dimension
    (variable-size point clouds) when the trailing dimensions agree.

    Returns ``(stacked, lengths)``; `lengths` holds each message's original
    leading size when padding was applied, else None. Raises ValueError when
    the shapes cannot be reconciled.
    """

    arrays = [np.asarray(value) for value in values]
    shapes = {arr.shape for arr in arrays}
    if len(shapes) == 1:
        return np.stack(arrays) if arrays else np.asarray(arrays), None

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


def _cmd_topics(args) -> int:
    source = _open_source(args.path)
    for topic in source.get_topics():
        print(topic)
    return 0


def _cmd_info(args) -> int:
    source = _open_source(args.path)
    topics = source.get_topics()
    print(f"source: {args.path}")
    try:
        duration = source.source.get_duration()
        print(f"duration: {duration:.3f} s")
    except (ValueError, AttributeError, TypeError):
        pass
    print(f"topics ({len(topics)}):")
    for topic in topics:
        print(f"  {topic}: {source.get_count(topic)} messages")

    if args.messages:
        from .ops import describe_topic, format_describe

        summaries = {}
        skipped = []
        collected: dict[str, dict] = {
            topic: {"ts": [], "data": [], "frames": set()} for topic in topics
        }
        seen = 0
        for message in source.get_message():
            slot = collected.get(message["topic"])
            if slot is not None:
                slot["ts"].append(message["timestamp"])
                slot["data"].append(message["data"])
                frame = _frame_text(message.get("frame_id"))
                if frame is not None:
                    slot["frames"].add(frame)
            seen += 1
            if seen >= args.messages:
                break
        print(f"\nstats over the first {seen} messages:")
        for topic, slot in collected.items():
            if not slot["ts"]:
                continue
            try:
                data, lengths = _stack_ragged(slot["data"])
            except ValueError:
                skipped.append(topic)
                continue
            topic_data = {"ts": np.asarray(slot["ts"], dtype=np.float64), "data": data, "topic": topic}
            if len(slot["frames"]) == 1:
                topic_data["frame_id"] = next(iter(slot["frames"]))
            summary = describe_topic(topic_data, name=topic)
            if lengths is not None:
                # Show the ragged row range rather than the zero-padded size.
                dims = [f"{lengths.min()}..{lengths.max()}", *(str(dim) for dim in data.shape[2:])]
                summary["shape"] = "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"
            summaries[topic] = summary
        if summaries:
            print(format_describe(summaries))
        for topic in skipped:
            print(f"{topic}: stats skipped (messages have incompatible shapes)")
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


def _cmd_export(args) -> int:
    from .ops import save_topic_npz

    source = _open_source(args.path)
    topic_data = _collect_topic(source, args.topic, limit=args.limit, stride=args.stride)
    lengths = topic_data.pop("lengths", None)
    out = _written_npz_path(save_topic_npz(args.out, topic_data), args.out)
    count = topic_data["ts"].shape[0]
    print(f"wrote {count} messages of {args.topic} to {out}")
    if lengths is not None and _append_npz_array(out, "lengths", lengths):
        print(
            f"  ragged messages were zero-padded to {topic_data['data'].shape[1]} rows; "
            "per-message row counts are stored under 'lengths'"
        )
    return 0


class _PaddedSource:
    """Zero-pads ragged topics (variable-size point clouds) to a fixed
    per-topic shape so they fit the fixed-shape persistent stores."""

    def __init__(self, source):
        self._source = source
        self._max_rows: dict[str, int] = {}
        ragged: dict[str, set] = {}
        for message in source.get_message():
            shape = np.asarray(message["data"]).shape
            ragged.setdefault(message["topic"], set()).add(shape)
        for topic, shapes in ragged.items():
            if len(shapes) > 1:
                trailing = {shape[1:] for shape in shapes}
                if len(trailing) != 1 or any(len(shape) == 0 for shape in shapes):
                    raise CLIError(
                        f"topic {topic!r} has incompatible message shapes; cannot ingest"
                    )
                self._max_rows[topic] = max(shape[0] for shape in shapes)

    def get_topics(self):
        return self._source.get_topics()

    def get_count(self, axis):
        return self._source.get_count(axis)

    def get_data_path(self):
        getter = getattr(self._source, "get_data_path", None)
        return getter() if callable(getter) else None

    def get_message(self):
        for message in self._source.get_message():
            rows = self._max_rows.get(message["topic"])
            if rows is not None:
                data = np.asarray(message["data"])
                if data.shape[0] < rows:
                    padded = np.zeros((rows,) + data.shape[1:], dtype=data.dtype)
                    padded[: data.shape[0]] = data
                    message = {**message, "data": padded}
            yield message


_TILEDB_MARKERS = {"__tiledb_group.tdb", "__group", "__meta"}


def _store_kind(path: Path) -> str:
    """Classify an ingest destination: ``"missing"``, ``"empty"``,
    ``"tiledb"``, ``"arrow"``, ``"file"``, or ``"other"`` (a non-empty
    directory that is not a store)."""

    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "file"
    entries = list(path.iterdir())
    if not entries:
        return "empty"
    if {entry.name for entry in entries} & _TILEDB_MARKERS:
        return "tiledb"
    if any(entry.is_dir() and (entry / "manifest.json").is_file() for entry in entries):
        return "arrow"
    return "other"


def _cmd_ingest(args) -> int:
    from .buffer import DataBuffer

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

    source = _PaddedSource(_open_source(args.path))
    topics = source.get_topics()
    if not topics:
        raise CLIError("source has no topics")
    with DataBuffer(
        data_source=source,
        data_uri=args.out,
        axis=topics[0],
        use_db=True,
        backend=args.backend,
        preload=0,
    ) as buffer:
        buffer.load_data_db(topics[0])
        counts = dict(buffer.buffer_impl.counters)
        backend = buffer.backend
    total = sum(counts.values())
    print(f"ingested {total} messages into {args.out} ({backend})")
    for topic, count in counts.items():
        print(f"  {topic}: {count}")
    return 0


def _cmd_viewer(args) -> int:
    from .visualizers.point_cloud import VisTool

    source = _open_source(args.path)
    tool = None
    # Stream scans straight into the viewer (which keeps a bounded, uniform
    # subsample) instead of materializing and padding the whole topic.
    for message in _iter_topic(source, args.topic, limit=args.limit, stride=args.stride):
        data = np.asarray(message["data"])
        if data.ndim != 2 or data.shape[-1] < 3 or not np.issubdtype(data.dtype, np.number):
            raise CLIError(
                f"topic {args.topic!r} has messages of shape {data.shape} ({data.dtype}), "
                "not point clouds (N, 3+)"
            )
        if tool is None:
            tool = VisTool(embed=True, backend="html", output_path=args.out)
        xyz = data[:, :3].astype(np.float64, copy=False)
        valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
        tool.add_point_cloud(xyz[valid])
    if tool is None:
        raise _no_messages_error(source, args.topic)
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
    for scan_ts, scan in zip(points["ts"], points["data"]):
        pose_row = odom["data"][int(np.argmin(np.abs(odom_ts - scan_ts)))]
        pose = np.concatenate((pose_row[0, :3], pose_row[2, :4]))
        matrix = pose_to_matrix(pose)
        valid = scan[np.abs(scan).sum(axis=1) > 0]
        world = valid @ matrix[:3, :3].T + matrix[:3, 3]
        tool.add_point_cloud(world)
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
    parser.add_argument("--debug", action="store_true",
                        help="show full tracebacks instead of one-line errors (or set ADE_DEBUG=1)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show topics, counts, and duration")
    p_info.add_argument("path")
    p_info.add_argument("--messages", type=_non_negative_int, default=0, metavar="N",
                        help="also stream the first N messages and print per-topic stats")
    p_info.set_defaults(fn=_cmd_info)

    p_topics = sub.add_parser("topics", help="list topics, one per line")
    p_topics.add_argument("path")
    p_topics.set_defaults(fn=_cmd_topics)

    p_export = sub.add_parser(
        "export",
        help="export one topic to a .npz file",
        description="Export one topic to a .npz file. Ragged messages (variable-size "
        "point clouds) are zero-padded to the largest message; their original row "
        "counts are stored in the archive's 'lengths' array.",
    )
    p_export.add_argument("path")
    p_export.add_argument("-t", "--topic", required=True)
    p_export.add_argument("-o", "--out", required=True,
                          help="output file (numpy appends .npz when missing)")
    p_export.add_argument("--limit", type=_positive_int, default=None,
                          help="max messages to export (positive)")
    p_export.add_argument("--stride", type=_positive_int, default=1,
                          help="keep every k-th message (positive)")
    p_export.set_defaults(fn=_cmd_export)

    p_ingest = sub.add_parser(
        "ingest",
        help="ingest a source into a persistent store",
        description="Ingest every topic of a source into a persistent store. An existing "
        "store is reopened with its own backend; a non-empty directory that is not a "
        "store (or a store of a different backend than --backend) is refused.",
    )
    p_ingest.add_argument("path")
    p_ingest.add_argument("-o", "--out", required=True,
                          help="store directory (new, empty, or an existing store)")
    p_ingest.add_argument("--backend", choices=("arrow", "tiledb"), default=None,
                          help="storage engine for a new store (default: arrow); an existing "
                          "store keeps its detected backend")
    p_ingest.set_defaults(fn=_cmd_ingest)

    p_viewer = sub.add_parser("viewer", help="write an interactive HTML point-cloud viewer")
    p_viewer.add_argument("path")
    p_viewer.add_argument("-t", "--topic", required=True)
    p_viewer.add_argument("-o", "--out", default="ade_pointcloud_viewer.html")
    p_viewer.add_argument("--limit", type=_positive_int, default=None,
                          help="max scans to include (positive)")
    p_viewer.add_argument("--stride", type=_positive_int, default=1,
                          help="keep every k-th scan (positive)")
    p_viewer.set_defaults(fn=_cmd_viewer)

    p_demo = sub.add_parser("demo", help="run the synthetic-data showcase (no input files needed)")
    p_demo.add_argument("-o", "--out", default="ade_demo", help="output directory")
    p_demo.add_argument("--duration", type=_positive_float, default=5.0,
                        help="seconds of synthetic data (positive)")
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.set_defaults(fn=_cmd_demo)

    return parser


def _error_message(exc: BaseException) -> str | None:
    """One-line message for common user errors; None means a real bug."""

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
        return str(exc)
    return None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    debug = args.debug or os.environ.get("ADE_DEBUG", "").strip() not in ("", "0")
    try:
        return args.fn(args)
    except Exception as exc:
        message = None if debug else _error_message(exc)
        if message is None:
            raise
        print(f"ade: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
