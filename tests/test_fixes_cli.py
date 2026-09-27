"""Regression tests for the `ade` CLI, visualizers, and packaging fixes."""

from __future__ import annotations

import json
import re
import sqlite3
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arraydataengine import cli
from arraydataengine.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example"


# --- helpers ------------------------------------------------------------------


class FakeSource:
    """Minimal DataSources stand-in with a fixed message list."""

    def __init__(self, messages, duration=1.0):
        self._messages = messages
        self.source = SimpleNamespace(get_duration=lambda: duration)

    def get_topics(self):
        return list(dict.fromkeys(message["topic"] for message in self._messages))

    def get_count(self, topic):
        return sum(message["topic"] == topic for message in self._messages)

    def get_message(self):
        for message in self._messages:
            yield dict(message)


def _cloud_messages(rows=(3, 5, 4), nan_row=False):
    messages = []
    for index, count in enumerate(rows):
        data = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + 1.0
        if nan_row:
            data[0] = np.nan
        messages.append({
            "topic": "/cloud", "timestamp": float(index), "data": data,
            "name": f"cloud{index}", "frame_id": "lidar",
        })
        messages.append({
            "topic": "/pose", "timestamp": index + 0.5, "data": np.full(7, float(index)),
            "name": f"pose{index}", "frame_id": "map",
        })
    return messages


@pytest.fixture()
def fake(monkeypatch):
    def install(messages):
        source = FakeSource(messages)
        monkeypatch.setattr(cli, "_open_source", lambda path: source)
        return source

    return install


def _json_envelope(msg: dict) -> bytes:
    return json.dumps({"op": "publish", "topic": "/x", "msg": msg}).encode()


def _write_bag(path: Path, pose_count: int = 5, string_count: int = 0) -> Path:
    """A tiny rosbag2 sqlite file: JSON PoseStamped plus an undecodable type."""
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
            "header": {"stamp": {"sec": index, "nanosec": 0}, "frame_id": "map"},
            "pose": {"position": {"x": float(index), "y": 0.0, "z": 0.0},
                     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        })
        connection.execute("INSERT INTO messages VALUES (?, 1, ?, ?)", (row, index * 10**9, payload))
    for index in range(string_count):
        row += 1
        connection.execute(
            "INSERT INTO messages VALUES (?, 2, ?, ?)",
            (row, index * 10**9, _json_envelope({"data": f"hello {index}"})),
        )
    connection.commit()
    connection.close()
    return path


def _viewer_payload(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8")
    start = text.index("const data = ") + len("const data = ")
    end = text.index(";\nconst canvas")

    def reject(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    return json.loads(text[start:end], parse_constant=reject)


# --- info -------------------------------------------------------------------


def test_info_messages_handles_ragged_clouds_and_frame_ids(fake, capsys):
    fake(_cloud_messages())
    assert main(["info", "any", "--messages", "100"]) == 0
    out = capsys.readouterr().out
    assert "/cloud: 3 msgs" in out
    assert "(3..5, 3)" in out
    assert "frame lidar" in out
    assert "/pose: 3 msgs" in out and "frame map" in out


def test_info_messages_skips_incompatible_shapes(fake, capsys):
    fake([
        {"topic": "/odd", "timestamp": 0.0, "data": np.zeros((2, 3))},
        {"topic": "/odd", "timestamp": 1.0, "data": np.zeros((2, 4))},
        {"topic": "/ok", "timestamp": 0.5, "data": np.zeros(2)},
    ])
    assert main(["info", "any", "--messages", "10"]) == 0
    out = capsys.readouterr().out
    assert "/odd: stats skipped" in out
    assert "/ok: 1 msgs" in out


@pytest.mark.parametrize("name", ["mapeverything_0.db3", "mapeverything_0.bag"])
def test_info_messages_on_example_recordings(name, capsys):
    path = EXAMPLE / name
    if not path.exists():
        pytest.skip(f"{path} not available")
    assert main(["info", str(path), "--messages", "2000"]) == 0
    out = capsys.readouterr().out
    stats = out.split("stats over", 1)[1]
    assert "/mapping/pointcloud/lidar: 113 msgs" in stats
    assert "frame map" in stats


# --- argument validation --------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["export", "x.db3", "-t", "/a", "-o", "o.npz", "--stride", "0"],
    ["export", "x.db3", "-t", "/a", "-o", "o.npz", "--limit", "0"],
    ["export", "x.db3", "-t", "/a", "-o", "o.npz", "--limit", "-3"],
    ["viewer", "x.db3", "-t", "/a", "--stride", "-1"],
    ["info", "x.db3", "--messages", "-5"],
    ["demo", "--duration", "0"],
    ["demo", "--duration", "nan"],
])
def test_invalid_numeric_arguments_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_ingest_backend_defaults_to_auto_detection():
    args = cli.build_parser().parse_args(["ingest", "x.db3", "-o", "store"])
    assert args.backend is None


# --- export -----------------------------------------------------------------


def test_export_ragged_topic_stores_lengths_and_reports_real_path(fake, tmp_path, capsys):
    from arraydataengine.ops import load_topic_npz

    fake(_cloud_messages())
    out = tmp_path / "clouds"
    assert main(["export", "any", "-t", "/cloud", "-o", str(out)]) == 0
    written = tmp_path / "clouds.npz"
    assert written.is_file()
    assert f"to {written}" in capsys.readouterr().out

    with np.load(written, allow_pickle=False) as archive:
        assert archive["lengths"].tolist() == [3, 5, 4]
        assert archive["data"].shape == (3, 5, 3)
    loaded = load_topic_npz(written)
    assert loaded["frame_id"] == "lidar"
    assert np.allclose(loaded["data"][0, 3:], 0.0)


def test_export_uniform_topic_has_no_lengths(fake, tmp_path):
    fake(_cloud_messages())
    out = tmp_path / "pose.npz"
    assert main(["export", "any", "-t", "/pose", "-o", str(out)]) == 0
    with np.load(out, allow_pickle=False) as archive:
        assert "lengths" not in archive.files


def test_export_limit_and_stride(fake, tmp_path):
    from arraydataengine.ops import load_topic_npz

    fake([{"topic": "/s", "timestamp": float(i), "data": np.array([i])} for i in range(10)])
    out = tmp_path / "s.npz"
    assert main(["export", "any", "-t", "/s", "-o", str(out), "--limit", "3", "--stride", "2"]) == 0
    assert load_topic_npz(out)["ts"].tolist() == [0.0, 2.0, 4.0]


def test_export_unsupported_message_type_is_explained(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3", pose_count=2, string_count=3)
    rc = main(["export", str(bag), "-t", "/chatter", "-o", str(tmp_path / "c.npz")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unsupported message type std_msgs/msg/String" in err
    assert err.count("\n") == 1


def test_export_unknown_topic_lists_available(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3")
    assert main(["export", str(bag), "-t", "/nope", "-o", str(tmp_path / "n.npz")]) == 2
    err = capsys.readouterr().err
    assert "'/nope' not found" in err and "/pose" in err


# --- user errors ------------------------------------------------------------------


def test_missing_input_is_a_one_line_error(tmp_path, capsys, monkeypatch):
    # A sibling recording must not be opened in place of the missing file.
    _write_bag(tmp_path / "other.db3")
    monkeypatch.chdir(tmp_path)
    assert main(["info", "missing.db3"]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ade: error: no such file or directory: missing.db3")
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_unsupported_input_type_is_a_one_line_error(tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    assert main(["topics", str(path)]) == 2
    assert "unsupported file type" in capsys.readouterr().err


def test_debug_flag_and_env_keep_tracebacks(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        main(["--debug", "info", str(tmp_path / "missing.db3")])
    monkeypatch.setenv("ADE_DEBUG", "1")
    with pytest.raises(FileNotFoundError):
        main(["info", str(tmp_path / "missing.db3")])


@pytest.mark.parametrize("module, extra", [("pyarrow", "arrow"), ("tiledb", "tiledb"), ("rosbags", "ros")])
def test_missing_optional_dependency_prints_install_hint(module, extra, monkeypatch, capsys):
    def boom(path):
        raise ModuleNotFoundError(f"No module named '{module}'", name=module)

    monkeypatch.setattr(cli, "_open_source", boom)
    assert main(["topics", "x.db3"]) == 2
    assert f'pip install "arraydataengine[{extra}]"' in capsys.readouterr().err


def test_internal_import_errors_are_not_hidden(monkeypatch):
    def boom(path):
        raise ModuleNotFoundError("No module named 'arraydataengine.nope'", name="arraydataengine.nope")

    monkeypatch.setattr(cli, "_open_source", boom)
    with pytest.raises(ModuleNotFoundError):
        main(["topics", "x.db3"])


def test_demo_output_pointing_at_a_file_is_rejected(tmp_path, capsys):
    target = tmp_path / "taken"
    target.write_text("keep me")
    assert main(["demo", "-o", str(target), "--duration", "0.5"]) == 2
    assert "not a directory" in capsys.readouterr().err
    assert target.read_text() == "keep me"


# --- demo ---------------------------------------------------------------------------


def test_demo_sizes_buffer_to_the_largest_topic(tmp_path, monkeypatch, capsys):
    import arraydataengine.buffer as buffer_module
    from arraydataengine.sources.synthetic_source import SyntheticSource

    depths = []
    original = buffer_module.DataBuffer

    class RecordingBuffer(original):
        def __init__(self, *args, **kwargs):
            depths.append(kwargs.get("buffer_depth"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(buffer_module, "DataBuffer", RecordingBuffer)
    assert main(["demo", "-o", str(tmp_path / "demo"), "--duration", "1.0"]) == 0
    source = SyntheticSource(duration=1.0)
    assert depths == [max(source.get_count(topic) for topic in source.get_topics())]
    payload = _viewer_payload(tmp_path / "demo" / "viewer.html")
    assert payload["points"] and payload["poses"]


# --- ingest -------------------------------------------------------------------------


def test_ingest_refuses_a_non_store_directory(tmp_path, capsys):
    bag = _write_bag(tmp_path / "bag.db3")
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("unrelated")
    assert main(["ingest", str(bag), "-o", str(out)]) == 2
    assert "not an ArrayDataEngine store" in capsys.readouterr().err
    assert sorted(path.name for path in out.iterdir()) == ["notes.txt"]


def test_ingest_reopens_an_existing_tiledb_store(tmp_path, capsys):
    pytest.importorskip("tiledb")
    bag = _write_bag(tmp_path / "bag.db3")
    out = tmp_path / "store"
    assert main(["ingest", str(bag), "-o", str(out), "--backend", "tiledb"]) == 0
    capsys.readouterr()

    # No --backend: the existing TileDB store is detected, not shadowed by
    # a second (arrow) store in the same directory.
    assert main(["ingest", str(bag), "-o", str(out)]) == 0
    assert "(tiledb)" in capsys.readouterr().out
    assert not list(out.rglob("manifest.json"))

    assert main(["ingest", str(bag), "-o", str(out), "--backend", "arrow"]) == 2
    assert "already holds an existing tiledb store" in capsys.readouterr().err


def test_ingest_new_store_defaults_to_arrow(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    bag = _write_bag(tmp_path / "bag.db3")
    out = tmp_path / "store"
    assert main(["ingest", str(bag), "-o", str(out)]) == 0
    assert "(arrow)" in capsys.readouterr().out
    assert list(out.rglob("manifest.json"))
    assert main(["ingest", str(bag), "-o", str(out), "--backend", "tiledb"]) == 2
    assert "already holds an existing arrow store" in capsys.readouterr().err


# --- viewer -------------------------------------------------------------------------


def test_viewer_streams_ragged_clouds_and_drops_nonfinite(fake, tmp_path):
    fake(_cloud_messages(nan_row=True))
    out = tmp_path / "view.html"
    assert main(["viewer", "any", "-t", "/cloud", "-o", str(out)]) == 0
    payload = _viewer_payload(out)
    assert len(payload["points"]) == (3 - 1) + (5 - 1) + (4 - 1)


def test_viewer_rejects_non_point_cloud_topics(fake, tmp_path, capsys):
    fake(_cloud_messages())
    assert main(["viewer", "any", "-t", "/pose", "-o", str(tmp_path / "v.html")]) == 2
    assert "not point clouds" in capsys.readouterr().err
    assert not (tmp_path / "v.html").exists()


def test_html_point_cloud_filters_nan_and_inf(tmp_path):
    from arraydataengine.visualizers.point_cloud import VisTool

    tool = VisTool(embed=True, backend="html", output_path=tmp_path / "v.html")
    points = np.ones((6, 3))
    points[1, 0] = np.nan
    points[2, 2] = np.inf
    tool.add_point_cloud(points)
    tool.add_point_cloud(np.full((2, 3), np.nan))
    bad_pose = np.eye(4)
    bad_pose[0, 3] = np.nan
    tool.add_pose_arrow(bad_pose)
    tool.add_pose_arrow(np.eye(4))
    tool.show()
    payload = _viewer_payload(tmp_path / "v.html")
    assert len(payload["points"]) == 4
    assert len(payload["poses"]) == 1


def test_html_point_cloud_memory_is_bounded_and_uniform(tmp_path, monkeypatch):
    from arraydataengine.visualizers import point_cloud

    monkeypatch.setattr(point_cloud, "HTML_MAX_POINTS", 100)
    tool = point_cloud.VisTool(embed=True, backend="html", output_path=tmp_path / "v.html")
    total = 0
    for count in (37, 91, 55, 120, 8, 64, 77, 150):
        index = np.arange(total, total + count, dtype=np.float64)
        tool.add_point_cloud(np.column_stack([index, np.ones(count), np.ones(count)]))
        total += count
        assert sum(len(chunk) for chunk in tool.point_sets) <= 200
    kept = np.vstack(tool.point_sets)[:, 0].astype(np.int64)
    stride = tool._html_stride
    assert stride > 1
    assert np.all(kept % stride == 0)
    assert kept.tolist() == list(range(0, total, stride))
    tool.show()
    assert len(_viewer_payload(tmp_path / "v.html")["points"]) == 100


def test_html_viewer_fits_scale_and_throttles_redraws(tmp_path):
    from arraydataengine.visualizers.point_cloud import VisTool

    tool = VisTool(embed=True, backend="html", output_path=tmp_path / "v.html")
    tool.add_point_cloud(np.random.default_rng(0).normal(scale=500.0, size=(50, 3)))
    tool.show()
    html = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert "scale = 120" not in html
    assert "requestAnimationFrame" in html


# --- image visualizer ---------------------------------------------------------------


@pytest.fixture()
def image_vis(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    matplotlib.use("Agg")
    from arraydataengine.visualizers import video_segment

    return video_segment


def test_image_visualizer_show_without_image_is_a_clear_error(image_vis):
    from arraydataengine.visualizer import Visualizer

    with pytest.raises(ValueError, match="needs an image"):
        Visualizer("image").show()
    with pytest.raises(TypeError, match="NumPy array or an image file path"):
        image_vis.VisTool().show(42)


def test_image_animation_never_writes_into_the_working_directory(image_vis, tmp_path, monkeypatch):
    pytest.importorskip("IPython")
    monkeypatch.chdir(tmp_path)
    tool = image_vis.VisTool(embed=True)
    for value in (0, 120, 240):
        tool.update(np.full((6, 6, 3), value, dtype=np.uint8))
    tool.show()
    assert list(tmp_path.iterdir()) == []

    target = tmp_path / "anim" / "clip.gif"
    target.parent.mkdir()
    tool = image_vis.VisTool(embed=True, output_path=target)
    for value in (0, 120):
        tool.append_img(np.full((6, 6, 3), value, dtype=np.uint8))
    assert tool.show_animation() == target
    assert target.read_bytes()[:3] == b"GIF"


def test_image_visualizer_colormap_uses_the_registry(image_vis):
    colors = image_vis.VisTool.get_n_colors(7)
    assert colors.shape == (7, 3)
    assert 0.0 <= colors.min() and colors.max() <= 255.0


# --- packaging ---------------------------------------------------------------------


def test_legacy_facade_exports_version():
    import ArrayDataEngine
    import arraydataengine

    assert ArrayDataEngine.__version__ == arraydataengine.__version__
    assert "__version__" in ArrayDataEngine.__all__


def test_extras_cover_imports_and_drop_unused_packages():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    assert "opencv-python" in extras["visualization"]
    declared = {name for group in extras.values() for name in group}
    for unused in ("navpy", "transforms3d", "rosbags-dataframe", "rosbags-image",
                   "pycocotools", "tensorboard", "ultralytics", "wandb"):
        assert unused not in declared


def test_container_files_have_no_insecure_defaults():
    requirements = (ROOT / "requirements.txt").read_text().split()
    assert "tk" not in requirements
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "--insecure" not in dockerfile
    assert not re.search(r"App\.token", dockerfile)
    assert "python3-tk" in dockerfile
    compose = (ROOT / "compose.yaml").read_text()
    assert not re.search(r"App\.token", compose)
