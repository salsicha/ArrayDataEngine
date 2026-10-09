"""Regression tests for the second review round of the `ade` CLI, the
point-cloud viewers, and the packaging/CI files."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arraydataengine import cli
from arraydataengine.cli import main

ROOT = Path(__file__).resolve().parents[1]


# --- helpers ------------------------------------------------------------------


class FakeSource:
    """DataSources stand-in: `counts` may claim more messages than are yielded
    (as a ROS source does when it skips messages)."""

    def __init__(self, messages, counts=None, types=None, max_points=None):
        self._messages = messages
        self._counts = counts or {}
        self._types = types or {}
        self.max_points = max_points
        self.topic_requests = []
        self.source = SimpleNamespace(get_duration=lambda: 1.0)

    def get_topics(self):
        topics = dict.fromkeys(message["topic"] for message in self._messages)
        topics.update(dict.fromkeys(self._counts))
        return list(topics)

    def get_count(self, topic):
        if topic in self._counts:
            return self._counts[topic]
        return sum(message["topic"] == topic for message in self._messages)

    def get_topic_types(self):
        return dict(self._types)

    def get_message(self, topics=None):
        self.topic_requests.append(topics)
        for message in self._messages:
            if topics is None or message["topic"] in topics:
                yield dict(message)


@pytest.fixture()
def fake(monkeypatch):
    def install(source):
        monkeypatch.setattr(cli, "_open_source", lambda path, **options: source)
        return source

    return install


def _padded_clouds(counts=(3, 5, 4), rows=10, frames=None):
    """Clouds zero-padded to `rows` rows with the real size in point_count."""
    messages = []
    for index, count in enumerate(counts):
        data = np.zeros((rows, 3), dtype=np.float32)
        data[:count] = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + 1.0
        message = {"topic": "/cloud", "timestamp": float(index), "data": data,
                   "name": "PointCloud2", "point_count": count}
        if frames is not None:
            message["frame_id"] = frames[index]
        messages.append(message)
    return messages


def _pose_messages(count=3, topic="/pose"):
    return [{"topic": topic, "timestamp": float(index), "data": np.full(7, float(index)),
             "name": "PoseStamped", "frame_id": "map"} for index in range(count)]


def _json_envelope(msg: dict) -> bytes:
    return json.dumps({"op": "publish", "topic": "/x", "msg": msg}).encode()


def _write_bag(path: Path, pose_count: int = 5, x_offset: float = 0.0, string_count: int = 0) -> Path:
    """A rosbag2 SQLite file with JSON PoseStamped messages (no extras needed)."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                            serialization_format TEXT, offered_qos_profiles TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER,
                              timestamp INTEGER, data BLOB);
        INSERT INTO topics VALUES (1, '/pose', 'geometry_msgs/msg/PoseStamped', 'cdr', '');
        INSERT INTO topics VALUES (2, '/chatter', 'std_msgs/msg/String', 'cdr', '');
        """
    )
    row = 0
    for index in range(pose_count):
        row += 1
        payload = _json_envelope({
            "header": {"stamp": {"sec": index + 1, "nanosec": 0}, "frame_id": "map"},
            "pose": {"position": {"x": x_offset + index, "y": 0.0, "z": 0.0},
                     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        })
        connection.execute("INSERT INTO messages VALUES (?, 1, ?, ?)", (row, (index + 1) * 10**9, payload))
    for index in range(string_count):
        row += 1
        connection.execute("INSERT INTO messages VALUES (?, 2, ?, ?)",
                           (row, (index + 1) * 10**9, _json_envelope({"data": f"hi {index}"})))
    connection.commit()
    connection.close()
    return path


def _tree(path: Path) -> dict:
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.rglob("*"))}


def _one_error_line(err: str) -> str:
    assert "Traceback" not in err
    assert err.startswith("ade: error: ") and err.count("\n") == 1, err
    return err


# --- store detection (finding 1) ---------------------------------------------------


def test_ingest_refuses_a_project_directory_with_a_web_manifest(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3")
    project = tmp_path / "webproj"
    (project / "public").mkdir(parents=True)
    manifest = project / "public" / "manifest.json"
    manifest.write_text('{"name": "My App", "icons": []}')
    before = _tree(project)

    assert main(["ingest", str(bag), "-o", str(project)]) == 2
    assert "not an ArrayDataEngine store" in _one_error_line(capsys.readouterr().err)
    assert _tree(project) == before
    assert manifest.read_text() == '{"name": "My App", "icons": []}'


def test_store_kind_requires_real_arrow_topic_directories(tmp_path):
    arrow = tmp_path / "arrow"
    topic = arrow / "topic-%2Fpose.data"
    topic.mkdir(parents=True)
    (topic / "manifest.json").write_text(json.dumps(
        {"topic": "/pose", "count": 0, "fragments": [], "shape": None, "dtype": None}))
    assert cli._store_kind(arrow) == "arrow"

    # 0.3.x stores: topic-name directories with manifests without fragments.
    legacy = tmp_path / "legacy"
    (legacy / "_pose").mkdir(parents=True)
    (legacy / "_pose" / "manifest.json").write_text(json.dumps(
        {"topic": "/pose", "count": 3, "closed": True, "name": "", "frame_id": None,
         "shape": [7], "dtype": "float64"}))
    assert cli._store_kind(legacy) == "arrow"

    # A valid topic directory next to unrelated files is not a store.
    (arrow / "README.md").write_text("notes")
    assert cli._store_kind(arrow) == "other"

    # Arrow-like directory names with a foreign manifest are not a store.
    foreign = tmp_path / "foreign"
    (foreign / "topic-x.data").mkdir(parents=True)
    (foreign / "topic-x.data" / "manifest.json").write_text('{"name": "x"}')
    assert cli._store_kind(foreign) == "other"


def test_placeholder_entries_count_as_an_empty_directory(tmp_path):
    out = tmp_path / "mount"
    out.mkdir()
    (out / ".gitkeep").write_text("")
    (out / "lost+found").mkdir()
    assert cli._store_kind(out) == "empty"


def test_stray_tiledb_marker_file_is_not_a_store(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3")
    out = tmp_path / "out"
    out.mkdir()
    (out / "__meta").write_text("not a group")
    assert cli._store_kind(out) == "other"
    assert main(["ingest", str(bag), "-o", str(out)]) == 2
    assert "not an ArrayDataEngine store" in _one_error_line(capsys.readouterr().err)
    assert sorted(path.name for path in out.iterdir()) == ["__meta"]


def test_ingest_into_a_directory_with_only_placeholders(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    bag = _write_bag(tmp_path / "bag.db3")
    out = tmp_path / "out"
    out.mkdir()
    (out / ".gitkeep").write_text("")
    assert main(["ingest", str(bag), "-o", str(out)]) == 0
    assert "ingested 5 new messages" in capsys.readouterr().out
    assert (out / ".gitkeep").exists() and (out / cli.SOURCE_FINGERPRINT).is_file()


# --- source fingerprint and counts (finding 3, ingest report) ------------------------


def test_ingest_refuses_to_resume_a_store_from_another_recording(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    from arraydataengine.buffer import DataBuffer

    bag_a = _write_bag(tmp_path / "a.db3", pose_count=5)
    bag_b = _write_bag(tmp_path / "b.db3", pose_count=10, x_offset=1000.0)
    store = tmp_path / "store"

    assert main(["ingest", str(bag_a), "-o", str(store)]) == 0
    out = capsys.readouterr().out
    assert "ingested 5 new messages" in out and "the store holds 5" in out
    recorded = json.loads((store / cli.SOURCE_FINGERPRINT).read_text())
    assert recorded["source"] == str(bag_a.resolve()) and recorded["topics"] == {"/pose": 5}
    before = _tree(store)

    assert main(["ingest", str(bag_b), "-o", str(store)]) == 2
    err = _one_error_line(capsys.readouterr().err)
    assert "different recording" in err and "--force" in err
    assert _tree(store) == before

    # Resuming the same recording appends nothing and reports the store total.
    assert main(["ingest", str(bag_a), "-o", str(store)]) == 0
    out = capsys.readouterr().out
    assert "ingested 0 new messages" in out and "the store holds 5" in out
    assert "/pose: 0 new, 5 total" in out

    # A moved copy of the same recording is still the same recording.
    moved = tmp_path / "moved.db3"
    moved.write_bytes(bag_a.read_bytes())
    assert main(["ingest", str(moved), "-o", str(store)]) == 0
    capsys.readouterr()

    # --force appends anyway (resume semantics), keeping the original record.
    assert main(["ingest", str(bag_b), "-o", str(store), "--force"]) == 0
    assert "ingested 5 new messages" in capsys.readouterr().out
    assert json.loads((store / cli.SOURCE_FINGERPRINT).read_text())["digest"] == recorded["digest"]
    reader = DataBuffer(None, data_uri=str(store), use_db=True, preload=0)
    assert reader.get_index_range("/pose")["data"].shape[0] == 10


def test_ingest_resume_requires_the_same_max_points(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    bag = _write_bag(tmp_path / "a.db3")
    store = tmp_path / "store"
    assert main(["ingest", str(bag), "-o", str(store), "--max-points", "500"]) == 0
    capsys.readouterr()
    assert main(["ingest", str(bag), "-o", str(store)]) == 2
    assert "--max-points 500" in _one_error_line(capsys.readouterr().err)
    assert main(["ingest", str(bag), "-o", str(store), "--max-points", "500"]) == 0


def test_store_without_fingerprint_is_checked_against_the_recording(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    big = _write_bag(tmp_path / "big.db3", pose_count=8)
    small = _write_bag(tmp_path / "small.db3", pose_count=3)
    store = tmp_path / "store"
    assert main(["ingest", str(big), "-o", str(store)]) == 0
    (store / cli.SOURCE_FINGERPRINT).unlink()  # as written by older versions
    capsys.readouterr()

    assert main(["ingest", str(small), "-o", str(store)]) == 2
    assert "more than this recording's 3" in _one_error_line(capsys.readouterr().err)
    assert not (store / cli.SOURCE_FINGERPRINT).exists()

    # The matching recording is adopted and recorded for later resumes.
    assert main(["ingest", str(big), "-o", str(store)]) == 0
    assert (store / cli.SOURCE_FINGERPRINT).is_file()


def test_fingerprint_file_is_ignored_by_the_backends(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    from arraydataengine.buffer import DataBuffer

    bag = _write_bag(tmp_path / "a.db3")
    store = tmp_path / "store"
    assert main(["ingest", str(bag), "-o", str(store)]) == 0
    reader = DataBuffer(None, data_uri=str(store), use_db=True, preload=0)
    assert list(reader.counters) == ["/pose"]


def test_fingerprint_with_tiledb_backend(tmp_path, capsys):
    pytest.importorskip("tiledb")
    from arraydataengine.buffer import DataBuffer

    bag = _write_bag(tmp_path / "a.db3")
    other = _write_bag(tmp_path / "b.db3", pose_count=6, x_offset=50.0)
    store = tmp_path / "store"
    assert main(["ingest", str(bag), "-o", str(store), "--backend", "tiledb"]) == 0
    assert (store / cli.SOURCE_FINGERPRINT).is_file()
    assert cli._store_kind(store) == "tiledb"
    capsys.readouterr()
    assert main(["ingest", str(other), "-o", str(store)]) == 2
    assert "different recording" in capsys.readouterr().err
    reader = DataBuffer(None, data_uri=str(store), use_db=True, preload=0)
    assert list(reader.counters) == ["/pose"] and reader.counters["/pose"] == 5


# --- --max-points and skip reasons (finding 2) ---------------------------------------


@pytest.mark.parametrize("command", [
    ["info", "x.db3"],
    ["topics", "x.db3"],
    ["export", "x.db3", "-t", "/a", "-o", "o.npz"],
    ["ingest", "x.db3", "-o", "store"],
    ["viewer", "x.db3", "-t", "/a"],
])
def test_every_source_command_accepts_max_points(command):
    parser = cli.build_parser()
    assert parser.parse_args(command).max_points is None
    assert parser.parse_args([*command, "--max-points", "40000"]).max_points == 40000
    with pytest.raises(SystemExit):
        parser.parse_args([*command, "--max-points", "0"])


def test_max_points_is_passed_to_the_source_only_when_given(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_open_source", lambda path, **options: calls.append(options))
    cli._source_for(SimpleNamespace(path="x", max_points=None))
    cli._source_for(SimpleNamespace(path="x", max_points=1234))
    assert calls == [{}, {"max_points": 1234}]


def test_export_names_the_max_points_skip_reason(fake, tmp_path, capsys):
    fake(FakeSource([], counts={"/os1/points": 3},
                    types={"/os1/points": "sensor_msgs/msg/PointCloud2"}))
    rc = main(["export", "bag", "-t", "/os1/points", "-o", str(tmp_path / "x.npz"),
               "--max-points", "1000"])
    assert rc == 2
    err = _one_error_line(capsys.readouterr().err)
    assert "all 3 messages of topic '/os1/points' were skipped" in err
    assert "--max-points 1000" in err
    assert "unsupported message type" not in err


def test_viewer_and_export_explain_skipped_messages_without_a_limit(fake, tmp_path, capsys):
    fake(FakeSource([], counts={"/pose": 4}, types={"/pose": "geometry_msgs/msg/PoseStamped"}))
    assert main(["viewer", "bag", "-t", "/pose", "-o", str(tmp_path / "v.html")]) == 2
    err = _one_error_line(capsys.readouterr().err)
    assert "all 4 messages" in err and "failed to decode" in err


def test_ingest_fails_when_every_message_of_a_topic_was_skipped(fake, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    fake(FakeSource(_pose_messages(3), counts={"/pose": 3, "/os1/points": 2},
                    types={"/os1/points": "sensor_msgs/msg/PointCloud2"}, max_points=1000))
    store = tmp_path / "store"
    assert main(["ingest", "bag", "-o", str(store), "--max-points", "1000"]) == 2
    captured = capsys.readouterr()
    assert "/pose: 3 new, 3 total" in captured.out
    assert "/os1/points: 0 new, 0 total (2 of 2 messages skipped)" in captured.out
    err = _one_error_line(captured.err)
    assert "'/os1/points'" in err and "--max-points 1000" in err


def test_ingest_stops_before_clouds_beyond_the_default_capacity(fake, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    from arraydataengine.sensors.pointcloud2_sensor import DEFAULT_MAX_POINTS

    cloud = {"topic": "/cloud", "timestamp": 0.0, "data": np.ones((4, 3)),
             "point_count": DEFAULT_MAX_POINTS + 1}
    fake(FakeSource([cloud]))
    assert main(["ingest", "bag", "-o", str(tmp_path / "store")]) == 2
    err = _one_error_line(capsys.readouterr().err)
    assert f"--max-points {DEFAULT_MAX_POINTS + 1}" in err


def test_info_reports_topics_whose_messages_were_all_skipped(fake, capsys):
    fake(FakeSource(_pose_messages(2), counts={"/pose": 2, "/big": 3},
                    types={"/big": "sensor_msgs/msg/PointCloud2"}))
    assert main(["info", "bag", "--messages", "50", "--max-points", "10"]) == 0
    out = capsys.readouterr().out
    assert "/big: all 3 messages were skipped" in out and "--max-points 10" in out


def test_export_and_viewer_decode_only_the_requested_topic(tmp_path, monkeypatch, capsys):
    calls = []
    real = cli._open_source

    def recording(path, **options):
        calls.append(options)
        return real(path, **options)

    monkeypatch.setattr(cli, "_open_source", recording)
    bag = _write_bag(tmp_path / "bag.db3", pose_count=3, string_count=2)
    assert main(["export", str(bag), "-t", "/pose", "-o", str(tmp_path / "p.npz")]) == 0
    # The unrestricted source answers metadata questions; the stream is
    # opened restricted to the exported topic.
    assert calls == [{}, {"topics": ["/pose"]}]
    calls.clear()
    assert main(["export", str(bag), "-t", "/nope", "-o", str(tmp_path / "n.npz")]) == 2
    assert calls == [{}]  # a missing topic fails before anything is streamed
    assert "/pose" in capsys.readouterr().err


def test_iter_topic_asks_the_source_for_one_topic(fake):
    source = FakeSource(_pose_messages(3) + _pose_messages(2, topic="/other"))
    messages = list(cli._iter_topic(source, "/pose"))
    assert len(messages) == 3 and source.topic_requests == [["/pose"]]


# --- export lengths, frames, output paths (findings 4 and 9) -------------------------


def test_export_uses_point_count_and_pads_to_the_topic_maximum(fake, tmp_path, capsys):
    fake(FakeSource(_padded_clouds((3, 5, 4), rows=10)))
    out = tmp_path / "clouds.npz"
    assert main(["export", "bag", "-t", "/cloud", "-o", str(out)]) == 0
    assert "padded to 5 rows" in capsys.readouterr().out
    with np.load(out, allow_pickle=False) as archive:
        assert archive["data"].shape == (3, 5, 3)
        assert archive["lengths"].tolist() == [3, 5, 4]
        assert archive["data"].dtype == np.float32
        assert np.all(archive["data"][0, 3:] == 0)


def test_export_writes_lengths_for_same_size_clouds(fake, tmp_path):
    fake(FakeSource(_padded_clouds((4, 4), rows=4)))
    out = tmp_path / "clouds.npz"
    assert main(["export", "bag", "-t", "/cloud", "-o", str(out)]) == 0
    with np.load(out, allow_pickle=False) as archive:
        assert archive["data"].shape == (2, 4, 3)
        assert archive["lengths"].tolist() == [4, 4]


def test_export_keeps_per_row_frame_ids(fake, tmp_path):
    from arraydataengine.ops import load_topic_npz

    fake(FakeSource(_padded_clouds((2, 3, 2), rows=3, frames=["lidar_a", "lidar_b", "lidar_a"])))
    out = tmp_path / "clouds.npz"
    assert main(["export", "bag", "-t", "/cloud", "-o", str(out)]) == 0
    loaded = load_topic_npz(out)
    assert list(loaded["frame_ids"]) == ["lidar_a", "lidar_b", "lidar_a"]


@pytest.mark.parametrize("existing", [True, False])
def test_export_into_a_directory_writes_topic_named_file(fake, tmp_path, capsys, existing):
    fake(FakeSource(_pose_messages(2, topic="/mapping/pose")))
    target = tmp_path / "exports"
    if existing:
        target.mkdir()
    out_arg = str(target) + ("" if existing else os.sep)
    assert main(["export", "bag", "-t", "/mapping/pose", "-o", out_arg]) == 0
    written = target / "mapping_pose.npz"
    assert written.is_file()
    assert not (tmp_path / "exports.npz").exists()
    assert str(written) in capsys.readouterr().out


def test_viewer_creates_parent_directories_and_trims_padding(fake, tmp_path):
    fake(FakeSource(_padded_clouds((3, 5, 4), rows=10)))
    out = tmp_path / "new" / "dir" / "view.html"
    assert main(["viewer", "bag", "-t", "/cloud", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    payload = json.loads(text[text.index("const data = ") + 13: text.index(";\nconst canvas")])
    points = np.asarray(payload["points"])
    assert points.shape == (12, 3)
    assert not np.any(np.all(points == 0, axis=1))


def test_output_parent_that_is_a_file_is_refused_before_streaming(fake, tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    fake(FakeSource(_pose_messages(2)))
    assert main(["export", "bag", "-t", "/pose", "-o", str(blocker / "out.npz")]) == 2
    assert "not a directory" in _one_error_line(capsys.readouterr().err)


# --- corrupt inputs (finding 5) ------------------------------------------------------


@pytest.mark.parametrize("name", ["rec.db3", "rec.bag", "rec.mcap"])
def test_git_lfs_pointer_inputs_are_explained(tmp_path, capsys, name):
    pointer = tmp_path / name
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    assert main(["info", str(pointer)]) == 2
    err = _one_error_line(capsys.readouterr().err)
    assert "Git LFS pointer" in err and "git lfs pull" in err


def test_git_lfs_pointer_inside_a_rosbag2_directory(tmp_path, capsys):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  relative_file_paths: [bag_0.db3]\n")
    (bag_dir / "bag_0.db3").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert main(["topics", str(bag_dir)]) == 2
    assert "Git LFS pointer" in _one_error_line(capsys.readouterr().err)


def test_empty_recording_is_a_one_line_error(tmp_path, capsys):
    empty = tmp_path / "empty.db3"
    empty.write_bytes(b"")
    assert main(["info", str(empty)]) == 2
    assert "is empty" in _one_error_line(capsys.readouterr().err)


@pytest.mark.parametrize("name", ["junk.db3", "junk.bag", "junk.mcap"])
def test_corrupt_recordings_are_one_line_errors(tmp_path, capsys, name):
    junk = tmp_path / name
    junk.write_bytes(b"\x00garbage that is not a recording" * 20)
    assert main(["info", str(junk)]) == 2
    _one_error_line(capsys.readouterr().err)


def test_incomplete_rosbag2_directory_is_a_one_line_error(tmp_path, capsys):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  version: 5\n  storage_identifier: sqlite3\n"
        "  relative_file_paths: [missing_0.db3]\n  duration: {nanoseconds: 1}\n"
        "  starting_time: {nanoseconds_since_epoch: 1}\n  message_count: 0\n"
        "  topics_with_message_count: []\n  compression_format: ''\n  compression_mode: ''\n"
    )
    assert main(["info", str(bag_dir)]) == 2
    _one_error_line(capsys.readouterr().err)


def test_reader_exceptions_are_recognized():
    class ReaderError(Exception):
        pass

    ReaderError.__module__ = "rosbags.rosbag1.reader"
    assert cli._error_message(ReaderError("File magic is invalid.")) == (
        "cannot read the input: File magic is invalid."
    )
    assert cli._error_message(sqlite3.DatabaseError("file is not a database")).endswith(
        "file is not a database")
    assert cli._error_message(RuntimeError("bug")) is None


# --- read-only output and exception groups (finding 6) --------------------------------


def test_exception_groups_are_unwrapped():
    group = ExceptionGroup("failed to close 2 Arrow topics", [
        PermissionError(13, "Permission denied", "/ro/a"),
        PermissionError(13, "Permission denied", "/ro/b"),
    ])
    message = cli._error_message(group)
    assert message.startswith("[Errno 13] Permission denied: '/ro/a'") and "1 more" in message
    assert cli._error_message(ExceptionGroup("x", [PermissionError("a"), ValueError("bug")])) is None


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_read_only_output_is_refused_before_reading(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        assert main(["ingest", str(bag), "-o", str(locked / "store")]) == 2
        assert "permission denied" in _one_error_line(capsys.readouterr().err)
        assert not (locked / "store").exists()
    finally:
        locked.chmod(0o755)


# --- undecodable topics (finding 7) ---------------------------------------------------


def test_info_lists_undecodable_topics_with_a_reason(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3", pose_count=2, string_count=2)
    assert main(["info", str(bag)]) == 0
    out = capsys.readouterr().out
    assert "topics (2):" in out
    assert "/pose: 2 messages" in out
    assert "/chatter: not decodable (unsupported message type std_msgs/msg/String)" in out


def test_compressed_images_without_opencv_name_the_extra(fake, tmp_path, capsys, monkeypatch):
    real = cli._module_available
    monkeypatch.setattr(cli, "_module_available", lambda name: False if name == "cv2" else real(name))
    fake(FakeSource(_pose_messages(1), types={
        "/pose": "geometry_msgs/msg/PoseStamped",
        "/camera/compressed": "sensor_msgs/msg/CompressedImage",
    }))
    assert main(["export", "bag", "-t", "/camera/compressed", "-o", str(tmp_path / "c.npz")]) == 2
    err = _one_error_line(capsys.readouterr().err)
    assert "needs OpenCV" in err and 'pip install "arraydataengine[image]"' in err
    assert main(["info", "bag"]) == 0
    out = capsys.readouterr().out
    assert "topics (2):" in out and "/camera/compressed: not decodable" in out and "OpenCV" in out


# --- argument and environment handling (finding 9) -------------------------------------


def test_negative_seed_is_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["demo", "--seed", "-1"])
    assert excinfo.value.code == 2
    assert "must be zero or a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("value, enabled", [
    ("1", True), ("true", True), ("YES", True), ("on", True), ("2", True),
    ("", False), ("0", False), ("false", False), ("no", False), ("off", False), ("False", False),
])
def test_ade_debug_environment_values(value, enabled, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ADE_DEBUG", value)
    missing = str(tmp_path / "missing.db3")
    if enabled:
        with pytest.raises(FileNotFoundError):
            main(["info", missing])
    else:
        assert main(["info", missing]) == 2
        _one_error_line(capsys.readouterr().err)


@pytest.mark.parametrize("argv", [
    ["--debug", "info", "{missing}"],
    ["info", "--debug", "{missing}"],
    ["info", "{missing}", "--debug"],
])
def test_debug_flag_before_or_after_the_subcommand(argv, tmp_path, monkeypatch):
    monkeypatch.delenv("ADE_DEBUG", raising=False)
    missing = str(tmp_path / "missing.db3")
    with pytest.raises(FileNotFoundError):
        main([arg.format(missing=missing) for arg in argv])


def test_keyboard_interrupt_is_one_line_and_exit_130(monkeypatch, capsys):
    monkeypatch.delenv("ADE_DEBUG", raising=False)

    def interrupted(path, **options):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_open_source", interrupted)
    assert main(["topics", "x.db3"]) == 130
    assert capsys.readouterr().err == "ade: interrupted\n"


# --- viewers (findings 8 and A) ---------------------------------------------------------


def test_html_update_drops_padding_before_transforming(tmp_path):
    from arraydataengine.visualizer import Visualizer

    vis = Visualizer("pointcloud", embed=True, backend="html", output_path=tmp_path / "v.html")
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 20.0, 30.0]
    scan = np.zeros((100, 3))
    scan[:4] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    vis.update(scan, transform)
    vis.update(scan, transform, length=2)
    points = np.vstack(vis.vis_tool.point_sets)
    assert points.shape == (6, 3)
    assert not np.any(np.all(np.isclose(points, [10.0, 20.0, 30.0]), axis=1))


def test_html_add_point_cloud_trims_with_length(tmp_path):
    from arraydataengine.visualizers.point_cloud import VisTool

    tool = VisTool(embed=True, backend="html", output_path=tmp_path / "v.html")
    cloud = np.zeros((50, 4))
    cloud[:3, :3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tool.add_point_cloud(cloud, length=2)
    tool.add_point_cloud(cloud)
    assert np.vstack(tool.point_sets).shape == (5, 3)


@pytest.fixture()
def fake_open3d(monkeypatch):
    """A stand-in open3d whose web_visualizer records being imported."""
    import arraydataengine.visualizers.point_cloud as point_cloud

    imported = []

    class WebVisualizer(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            imported.append(name)
            if name == "draw":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    open3d = types.ModuleType("open3d")
    open3d.web_visualizer = WebVisualizer("open3d.web_visualizer")
    monkeypatch.setitem(sys.modules, "open3d", open3d)
    monkeypatch.setitem(sys.modules, "open3d.web_visualizer", open3d.web_visualizer)
    monkeypatch.setattr(point_cloud, "o3d", None)
    monkeypatch.setattr(point_cloud, "draw", None)
    return point_cloud, imported


def test_web_visualizer_is_not_imported_outside_a_kernel(fake_open3d, tmp_path):
    point_cloud, imported = fake_open3d
    tool = point_cloud.VisTool(embed=True, output_path=tmp_path / "v.html")
    assert imported == []
    assert tool.show == tool._show_html  # falls back to the HTML viewer


def test_web_visualizer_is_imported_inside_a_kernel(fake_open3d, monkeypatch):
    point_cloud, imported = fake_open3d
    monkeypatch.setattr(point_cloud, "_in_ipython_kernel", lambda: True)
    assert point_cloud._ensure_open3d()
    assert "draw" in imported and point_cloud.draw is not None


def test_pointcloud_visualizer_lets_the_process_exit(tmp_path):
    pytest.importorskip("open3d")
    code = (
        "from arraydataengine.visualizer import Visualizer\n"
        f"Visualizer('pointcloud', output_path={str(tmp_path / 'x.html')!r})\n"
        "print('constructed', flush=True)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=120, cwd=tmp_path)
    assert result.returncode == 0 and "constructed" in result.stdout


# --- packaging and CI (findings B-E) ----------------------------------------------------


def _wheel(tmp_path: Path, entry_points: str | None) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "arraydataengine-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("arraydataengine/__init__.py", "")
        archive.writestr("ArrayDataEngine/__init__.py", "")
        archive.writestr("arraydataengine-1.2.3.dist-info/METADATA",
                         "Metadata-Version: 2.4\nName: arraydataengine\nVersion: 1.2.3\n")
        if entry_points is not None:
            archive.writestr("arraydataengine-1.2.3.dist-info/entry_points.txt", entry_points)
    return wheel


@pytest.mark.parametrize("entry_points, expected", [
    ("[console_scripts]\nade = arraydataengine.cli:main\n", None),
    (None, "missing entry_points.txt"),
    ("[console_scripts]\nother = x:y\n", "does not declare the 'ade' console script"),
    ("[console_scripts]\nade = arraydataengine.cli:other\n", "points to 'arraydataengine.cli:other'"),
])
def test_check_dist_requires_the_ade_console_script(tmp_path, monkeypatch, entry_points, expected):
    # scripts/ is not installed; import it from the checkout even when the
    # tests run against an installed wheel from another directory.
    monkeypatch.syspath_prepend(str(ROOT))
    from scripts import check_dist

    errors = check_dist._check_wheel(_wheel(tmp_path, entry_points), "1.2.3")
    if expected is None:
        assert errors == []
    else:
        assert any(expected in error for error in errors), errors


def test_dockerfile_copies_license_and_pins_ros2_numpy():
    dockerfile = (ROOT / "Dockerfile").read_text()
    copy_lines = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
    assert any("LICENSE" in line.split() for line in copy_lines)
    assert "archive/refs/tags" not in dockerfile and ".zip" not in dockerfile
    commit = re.search(r"ROS2_NUMPY_COMMIT=([0-9a-f]{40})\b", dockerfile)
    assert commit, "ros2_numpy must be pinned to a full commit hash"
    assert 'test "$(git rev-parse HEAD)" = "$ROS2_NUMPY_COMMIT"' in dockerfile


def _workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def test_workflows_pin_actions_and_key_pip_cache_on_pyproject():
    for name in ("tests.yml", "publish.yml"):
        workflow = _workflow(name)
        assert workflow["permissions"] == {"contents": "read"}
        for job in workflow["jobs"].values():
            for step in _steps(job):
                if "uses" in step:
                    assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
                if step.get("with", {}).get("cache") == "pip":
                    assert step["with"].get("cache-dependency-path") == "pyproject.toml"


def test_ci_tests_the_built_wheel():
    jobs = _workflow("tests.yml")["jobs"]
    base = "\n".join(step.get("run", "") for step in _steps(jobs["base-install"]))
    assert "pip install -e" not in base and "python -m build --wheel" in base
    assert "tests/test_fixes2_cli.py" in base
    dist = "\n".join(step.get("run", "") for step in _steps(jobs["dist"]))
    assert "python -m venv" in dist
    assert re.search(r"pip install .*\.whl|pip install .*wheels\[0\]", dist)
    assert "ade --help" in dist and "ade demo -o" in dist and "import arraydataengine" in dist


# --- RGB colour order in models and visualizers --------------------------------------


def test_to_gray_treats_colour_images_as_rgb():
    from arraydataengine.models.image.image import to_gray

    red = np.zeros((4, 5, 3), dtype=np.uint8)
    red[..., 0] = 100
    assert np.allclose(to_gray(red), 29.9, atol=1e-3)  # 0.299 * 100
    assert np.allclose(to_gray(red, color_order="bgr"), 11.4, atol=1e-3)  # read as blue
    rgba = np.concatenate([red, np.full((4, 5, 1), 255, dtype=np.uint8)], axis=2)
    assert np.allclose(to_gray(rgba), to_gray(red))
    gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
    assert np.array_equal(to_gray(gray), gray.astype(np.float32))
    with pytest.raises(ValueError):
        to_gray(red, color_order="hsv")
    try:
        import cv2
    except ImportError:
        return
    assert np.allclose(to_gray(red), cv2.cvtColor(red, cv2.COLOR_RGB2GRAY), atol=0.5)


@pytest.mark.parametrize("use_skimage", [True, False])
def test_dense_optical_flow_on_rgb_frames(use_skimage, monkeypatch):
    from arraydataengine.models.image.image import dense_optical_flow

    if use_skimage:
        pytest.importorskip("skimage.registration")
    else:
        pytest.importorskip("cv2")
        monkeypatch.setitem(sys.modules, "skimage.registration", None)  # force the OpenCV path
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, (64, 64))
    base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1) + np.roll(base, (1, 1), (0, 1))) / 4
    # Texture in the red channel only: BGR-style weighting would mostly ignore it.
    frame0 = np.zeros((64, 64, 3), dtype=np.float32)
    frame0[..., 0] = base
    frame1 = np.roll(frame0, 2, axis=1)
    flow_rgb = dense_optical_flow(frame0, frame1)
    flow_bgr = dense_optical_flow(frame0[..., ::-1], frame1[..., ::-1], color_order="bgr")
    assert np.allclose(flow_rgb, flow_bgr, atol=1e-4)
    assert np.isfinite(flow_rgb).all() and abs(flow_rgb[0]) > 0.5


def test_align_images_accepts_rgb_frames():
    pytest.importorskip("cv2")
    from arraydataengine.models.image.image import align_images

    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, (3, 48, 48, 3), dtype=np.uint8)
    aligned = align_images(frames)
    assert aligned.shape == frames.shape and aligned.dtype == np.uint8


class _FakeBoxes:
    """The parts of a maskrcnn-benchmark BoxList that frame_vis_generator uses."""

    def __init__(self, box, entity_id, label, size):
        self._box = np.asarray([box], dtype=np.float64)
        self._fields = {"ids": np.asarray([entity_id]), "labels": np.asarray([label])}
        self.size = size

    def get_field(self, name):
        return self._fields[name]

    def __getitem__(self, mask):
        return self

    def convert(self, mode):
        return self

    @property
    def bbox(self):
        box = self._box
        return SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: box)))


def test_box_colours_are_drawn_in_rgb_order(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    cv2 = pytest.importorskip("cv2")
    import matplotlib

    matplotlib.use("Agg")
    from arraydataengine.visualizers.video_segment import VisTool

    np.random.seed(7)
    colors = VisTool.get_n_colors(50)
    np.random.seed(7)
    expected = matplotlib.colormaps["gist_ncar"](np.linspace(0, 1, 50))
    np.random.shuffle(expected)
    assert np.allclose(colors, expected[:, :3] * 255)

    tool = VisTool.__new__(VisTool)
    tool._vis_height = None
    tool.num_colors = 50
    tool.colors = colors
    tool.class_names = ["person"]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    drawn = tool.frame_vis_generator(frame, _FakeBoxes((10, 10, 60, 60), 3, 1, (100, 100)))
    assert np.allclose(drawn[40, 60].astype(float), colors[3], atol=1.0)
