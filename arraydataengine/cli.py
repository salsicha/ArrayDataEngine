"""Command-line interface for ArrayDataEngine.

Installed as the `ade` console script:

    ade info PATH               # topics, counts, duration, optional stats
    ade topics PATH             # one topic per line
    ade export PATH -t TOPIC -o out.npz
    ade ingest PATH -o /path/to/tiledb_group/
    ade viewer PATH -t TOPIC -o viewer.html
    ade demo -o demo_dir/       # synthetic data end-to-end showcase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _open_source(path: str):
    from .source import DataSources

    return DataSources(path)


def _collect_topic(source, topic: str, limit: int | None = None, stride: int = 1) -> dict:
    ids = []
    timestamps = []
    values = []
    frame_id = None
    seen = 0
    for message in source.get_message():
        if message["topic"] != topic:
            continue
        if seen % stride == 0:
            ids.append(message.get("name"))
            timestamps.append(message["timestamp"])
            values.append(message["data"])
            if frame_id is None:
                frame_id = message.get("frame_id")
        seen += 1
        if limit is not None and len(values) >= limit:
            break
    if not values:
        raise SystemExit(f"no messages found for topic {topic!r}")

    result = {
        "id": np.asarray(ids, dtype=object),
        "name": np.asarray(ids, dtype=object),
        "ts": np.asarray(timestamps, dtype=np.float64),
        "data": _stack_messages(values),
        "topic": topic,
    }
    if frame_id is not None:
        result["frame_id"] = frame_id
    return result


def _stack_messages(values: list[np.ndarray]) -> np.ndarray:
    """Stack messages into one array, zero-padding a ragged leading dimension
    (variable-size point clouds) when the trailing dimensions agree."""

    shapes = {np.asarray(value).shape for value in values}
    if len(shapes) == 1:
        return np.asarray(values)

    trailing = {shape[1:] for shape in shapes}
    if len(trailing) != 1 or any(len(shape) == 0 for shape in shapes):
        raise SystemExit("topic messages have incompatible shapes; export them individually")

    rows = max(shape[0] for shape in shapes)
    first = np.asarray(values[0])
    stacked = np.zeros((len(values), rows) + first.shape[1:], dtype=first.dtype)
    for index, value in enumerate(values):
        arr = np.asarray(value)
        stacked[index, : arr.shape[0]] = arr
    return stacked


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
    except (ValueError, AttributeError):
        pass
    print(f"topics ({len(topics)}):")
    for topic in topics:
        print(f"  {topic}: {source.get_count(topic)} messages")

    if args.messages:
        from .ops import describe_topic, format_describe

        summaries = {}
        collected: dict[str, dict[str, list]] = {topic: {"ts": [], "data": []} for topic in topics}
        seen = 0
        for message in source.get_message():
            slot = collected.get(message["topic"])
            if slot is not None:
                slot["ts"].append(message["timestamp"])
                slot["data"].append(message["data"])
            seen += 1
            if seen >= args.messages:
                break
        print(f"\nstats over the first {seen} messages:")
        for topic, slot in collected.items():
            if not slot["ts"]:
                continue
            summaries[topic] = describe_topic(
                {"ts": np.asarray(slot["ts"]), "data": np.asarray(slot["data"]), "topic": topic},
                name=topic,
            )
        print(format_describe(summaries))
    return 0


def _cmd_export(args) -> int:
    from .ops import save_topic_npz

    source = _open_source(args.path)
    topic_data = _collect_topic(source, args.topic, limit=args.limit, stride=args.stride)
    out = save_topic_npz(args.out, topic_data)
    count = topic_data["ts"].shape[0]
    print(f"wrote {count} messages of {args.topic} to {out}")
    return 0


class _PaddedSource:
    """Zero-pads ragged topics (variable-size point clouds) to a fixed
    per-topic shape so they fit the fixed-dimension TileDB arrays."""

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
                    raise SystemExit(
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


def _cmd_ingest(args) -> int:
    from .buffer import DataBuffer

    source = _PaddedSource(_open_source(args.path))
    topics = source.get_topics()
    if not topics:
        raise SystemExit("source has no topics")
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
    total = sum(counts.values())
    print(f"ingested {total} messages into {args.out}")
    for topic, count in counts.items():
        print(f"  {topic}: {count}")
    return 0


def _cmd_viewer(args) -> int:
    from .visualizers.point_cloud import VisTool

    source = _open_source(args.path)
    topic_data = _collect_topic(source, args.topic, limit=args.limit, stride=args.stride)
    data = np.asarray(topic_data["data"], dtype=np.float64)
    if data.ndim != 3 or data.shape[-1] < 3:
        raise SystemExit(
            f"topic {args.topic!r} has shape {data.shape[1:]}, not point clouds (N, 3+)"
        )

    tool = VisTool(embed=True, backend="html", output_path=args.out)
    for scan in data:
        valid = scan[np.isfinite(scan[:, :3]).all(axis=1)]
        valid = valid[np.abs(valid[:, :3]).sum(axis=1) > 0]
        tool.add_point_cloud(valid)
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
    out_dir.mkdir(parents=True, exist_ok=True)

    source = SyntheticSource(duration=args.duration, seed=args.seed)
    buffer = DataBuffer(source, buffer_depth=100000, axis="/points", use_db=False, preload=0)
    for _ in range(source.get_count("/points")):
        buffer.roll_buffer("/points")

    print(format_describe(describe_dataset(buffer)))

    odom = buffer.get_index_range("/odom")
    trajectory = odometry_to_trajectory(odom)
    tum_path = write_tum_trajectory(out_dir / "trajectory.tum", trajectory)

    points = buffer.get_index_range("/points")
    npz_path = save_topic_npz(out_dir / "points.npz", points)

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

    parser = argparse.ArgumentParser(prog="ade", description=__doc__)
    parser.add_argument("--version", action="version", version=f"arraydataengine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show topics, counts, and duration")
    p_info.add_argument("path")
    p_info.add_argument("--messages", type=int, default=0,
                        help="also stream the first N messages and print per-topic stats")
    p_info.set_defaults(fn=_cmd_info)

    p_topics = sub.add_parser("topics", help="list topics, one per line")
    p_topics.add_argument("path")
    p_topics.set_defaults(fn=_cmd_topics)

    p_export = sub.add_parser("export", help="export one topic to a .npz file")
    p_export.add_argument("path")
    p_export.add_argument("-t", "--topic", required=True)
    p_export.add_argument("-o", "--out", required=True)
    p_export.add_argument("--limit", type=int, default=None, help="max messages to export")
    p_export.add_argument("--stride", type=int, default=1, help="keep every k-th message")
    p_export.set_defaults(fn=_cmd_export)

    p_ingest = sub.add_parser("ingest", help="ingest a source into a persistent store")
    p_ingest.add_argument("path")
    p_ingest.add_argument("-o", "--out", required=True, help="store directory")
    p_ingest.add_argument("--backend", choices=("arrow", "tiledb"), default="arrow",
                          help="storage engine (default: arrow)")
    p_ingest.set_defaults(fn=_cmd_ingest)

    p_viewer = sub.add_parser("viewer", help="write an interactive HTML point-cloud viewer")
    p_viewer.add_argument("path")
    p_viewer.add_argument("-t", "--topic", required=True)
    p_viewer.add_argument("-o", "--out", default="ade_pointcloud_viewer.html")
    p_viewer.add_argument("--limit", type=int, default=None)
    p_viewer.add_argument("--stride", type=int, default=1)
    p_viewer.set_defaults(fn=_cmd_viewer)

    p_demo = sub.add_parser("demo", help="run the synthetic-data showcase (no input files needed)")
    p_demo.add_argument("-o", "--out", default="ade_demo")
    p_demo.add_argument("--duration", type=float, default=5.0)
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.set_defaults(fn=_cmd_demo)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
