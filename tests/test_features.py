"""Tests for the 0.2.0 feature set: SyntheticSource, describe, TUM/KITTI,
NPZ IO, rosbridge JSON decoding, MCAP routing, and the CLI."""

import base64
import json
import sqlite3
import struct

import numpy as np
import pytest

from arraydataengine.buffer import DataBuffer
from arraydataengine.cli import main as cli_main
from arraydataengine.ops import (
    describe_dataset,
    describe_topic,
    format_describe,
    load_topic_npz,
    odometry_to_trajectory,
    read_tum_trajectory,
    save_topic_npz,
    to_tum_trajectory,
    write_kitti_trajectory,
    write_tum_trajectory,
)
from arraydataengine.sources.cdr import decode_supported_cdr_message
from arraydataengine.sources.synthetic_source import SyntheticSource


# --- SyntheticSource -----------------------------------------------------------


def test_synthetic_source_is_deterministic_and_ordered():
    first = list(SyntheticSource(seed=7, duration=2.0).messages())
    second = list(SyntheticSource(seed=7, duration=2.0).messages())
    assert len(first) == len(second)
    assert all(np.array_equal(a["data"], b["data"]) for a, b in zip(first, second))
    timestamps = [m["timestamp"] for m in first]
    assert timestamps == sorted(timestamps)


def test_synthetic_source_counts_and_buffer_roundtrip():
    source = SyntheticSource(duration=2.0)
    buffer = DataBuffer(source, buffer_depth=500, axis="/points", use_db=False, preload=0)
    for _ in range(source.get_count("/points")):
        buffer.roll_buffer("/points")
    points = buffer.get_index_range("/points")
    assert points["data"].shape == (source.get_count("/points"), 120, 3)
    odom = buffer.get_index_range("/odom")
    assert odom["data"].shape[1:] == (8, 4)


def test_synthetic_scans_stitch_back_to_landmarks():
    from arraydataengine.ops import pose_to_matrix

    source = SyntheticSource(duration=4.0)
    buffer = DataBuffer(source, buffer_depth=100000, axis="/points", use_db=False, preload=0)
    for _ in range(source.get_count("/points")):
        buffer.roll_buffer("/points")
    odom = buffer.get_index_range("/odom")
    points = buffer.get_index_range("/points")

    def world_points(index):
        scan = points["data"][index]
        row = odom["data"][int(np.argmin(np.abs(odom["ts"] - points["ts"][index])))]
        matrix = pose_to_matrix(np.concatenate((row[0, :3], row[2, :4])))
        valid = scan[np.abs(scan).sum(axis=1) > 0]
        return valid @ matrix[:3, :3].T + matrix[:3, 3]

    first = world_points(0)
    last = world_points(len(points["ts"]) - 1)
    # same landmarks observed from different poses land in the same place
    distances = np.linalg.norm(first[:, None, :] - last[None, :, :], axis=2).min(axis=1)
    assert np.median(distances) < 0.1


# --- describe -------------------------------------------------------------------


def test_describe_topic_and_dataset():
    topic = {
        "ts": np.arange(10, dtype=float) * 0.1,
        "data": np.ones((10, 3)),
        "topic": "/t",
        "frame_id": "map",
    }
    info = describe_topic(topic)
    assert info["count"] == 10
    assert np.isclose(info["rate_hz"], 10.0)
    assert info["shape"] == (3,)
    assert info["frame_id"] == "map"
    assert info["data_mean"] == 1.0

    summaries = describe_dataset({"/t": topic})
    assert "/t" in summaries
    assert "/t" in format_describe(summaries)


def test_describe_empty_topic():
    info = describe_topic({"ts": np.array([]), "data": np.empty((0, 3))})
    assert info["count"] == 0
    assert info["rate_hz"] is None


# --- trajectory IO ----------------------------------------------------------------


def _synthetic_trajectory():
    source = SyntheticSource(duration=2.0)
    buffer = DataBuffer(source, buffer_depth=500, axis="/odom", use_db=False, preload=0)
    for _ in range(source.get_count("/odom")):
        buffer.roll_buffer("/odom")
    return odometry_to_trajectory(buffer.get_index_range("/odom"))


def test_tum_round_trip(tmp_path):
    trajectory = _synthetic_trajectory()
    path = write_tum_trajectory(tmp_path / "traj.tum", trajectory)
    loaded = read_tum_trajectory(path)
    assert np.allclose(loaded["position"], trajectory["position"], atol=1e-6)
    dots = np.abs(np.sum(loaded["orientation"] * trajectory["orientation"], axis=1))
    assert np.allclose(dots, 1.0, atol=1e-6)


def test_tum_identity_for_missing_orientation():
    trajectory = {
        "ts": np.array([0.0, 1.0]),
        "position": np.zeros((2, 3)),
        "orientation": np.full((2, 4), np.nan),
    }
    rows = to_tum_trajectory(trajectory)
    assert np.allclose(rows[:, 4:8], [0.0, 0.0, 0.0, 1.0])


def test_kitti_export(tmp_path):
    trajectory = _synthetic_trajectory()
    path = write_kitti_trajectory(tmp_path / "traj.kitti", trajectory)
    rows = np.loadtxt(path, ndmin=2)
    assert rows.shape == (trajectory["ts"].shape[0], 12)
    rotation = rows[0].reshape(3, 4)[:, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)


# --- NPZ IO -------------------------------------------------------------------------


def test_topic_npz_round_trip(tmp_path):
    topic = {
        "id": np.array([b"a", "b", None], dtype=object),
        "ts": np.array([0.0, 0.1, 0.2]),
        "data": np.arange(9, dtype=np.float32).reshape(3, 3),
        "topic": "/t",
        "frame_id": "lidar",
    }
    path = save_topic_npz(tmp_path / "topic.npz", topic)
    loaded = load_topic_npz(path)
    assert np.allclose(loaded["ts"], topic["ts"])
    assert np.allclose(loaded["data"], topic["data"])
    assert loaded["data"].dtype == np.float32
    assert loaded["topic"] == "/t"
    assert loaded["frame_id"] == "lidar"
    assert loaded["id"].tolist() == ["a", "b", ""]


# --- rosbridge JSON decoding -----------------------------------------------------


def _json_envelope(msg: dict) -> bytes:
    return json.dumps({"op": "publish", "topic": "/x", "msg": msg}).encode()


def test_decode_rosbridge_json_pose():
    payload = _json_envelope({
        "header": {"stamp": {"sec": 100, "nanosec": 500000000}, "frame_id": "map"},
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    })
    decoded = decode_supported_cdr_message(payload, "geometry_msgs/msg/PoseStamped")
    assert decoded is not None
    assert np.allclose(decoded.data, [1, 2, 3, 0, 0, 0, 1])
    assert decoded.timestamp == 100.5
    assert decoded.frame_id == "map"


def test_decode_rosbridge_json_pointcloud():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    payload = _json_envelope({
        "header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "lidar"},
        "height": 1,
        "width": 2,
        "fields": [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ],
        "is_bigendian": False,
        "point_step": 12,
        "row_step": 24,
        "data": base64.b64encode(points.tobytes()).decode(),
        "is_dense": True,
    })
    decoded = decode_supported_cdr_message(payload, "sensor_msgs/msg/PointCloud2")
    assert decoded is not None
    assert np.allclose(decoded.data, points)


def test_malformed_payload_does_not_kill_stream(tmp_path):
    from arraydataengine.sources.db3_source import DB3Source

    db_path = tmp_path / "bag_0.db3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                            serialization_format TEXT, offered_qos_profiles TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER,
                              timestamp INTEGER, data BLOB);
        INSERT INTO topics VALUES (1, '/pose', 'geometry_msgs/msg/PoseStamped', 'cdr', '');
        """
    )
    good = _json_envelope({
        "header": {"stamp": {"sec": 2, "nanosec": 0}, "frame_id": "map"},
        "pose": {"position": {"x": 9.0, "y": 0.0, "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
    })
    garbage = b"\x00\x01\x00\x00" + b"\xff" * 8  # truncated CDR
    connection.execute("INSERT INTO messages VALUES (1, 1, 1000000000, ?)", (garbage,))
    connection.execute("INSERT INTO messages VALUES (2, 1, 2000000000, ?)", (good,))
    connection.commit()
    connection.close()

    messages = list(DB3Source(str(db_path)).messages())
    assert len(messages) == 1
    assert np.allclose(messages[0]["data"][:3], [9.0, 0.0, 0.0])


# --- MCAP ---------------------------------------------------------------------------


def _write_mcap_bag(bag_dir):
    rosbags = pytest.importorskip("rosbags")
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY if hasattr(Stores, "ROS2_JAZZY") else Stores.LATEST)
    Imu = store.types["sensor_msgs/msg/Imu"]
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Quaternion = store.types["geometry_msgs/msg/Quaternion"]
    Vector3 = store.types["geometry_msgs/msg/Vector3"]

    with Writer(bag_dir, version=8, storage_plugin=StoragePlugin.MCAP) as writer:
        connection = writer.add_connection("/imu", Imu.__msgtype__, typestore=store)
        for index in range(3):
            msg = Imu(
                header=Header(stamp=Time(sec=100 + index, nanosec=0), frame_id="imu_link"),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9),
                angular_velocity=Vector3(x=0.1, y=0.2, z=0.3),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vector3(x=0.0, y=0.0, z=9.8),
                linear_acceleration_covariance=np.zeros(9),
            )
            writer.write(connection, (100 + index) * 10**9, store.serialize_cdr(msg, Imu.__msgtype__))
    return bag_dir


def test_mcap_rosbag2_directory(tmp_path):
    from arraydataengine.source import DataSources

    bag_dir = _write_mcap_bag(tmp_path / "mcap_bag")
    source = DataSources(str(bag_dir))
    assert source.get_topics() == ["/imu"]
    assert source.get_count("/imu") == 3
    messages = list(source.get_message())
    assert len(messages) == 3
    assert messages[0]["data"].shape == (6, 4)


def test_mcap_file_path_routes_to_bag(tmp_path):
    from arraydataengine.source import DataSources

    bag_dir = _write_mcap_bag(tmp_path / "mcap_bag")
    mcap_file = next(bag_dir.glob("*.mcap"))
    source = DataSources(str(mcap_file))
    assert source.get_count("/imu") == 3
    assert len(list(source.get_message())) == 3


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture()
def json_pose_bag(tmp_path):
    db_path = tmp_path / "cli_bag_0.db3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                            serialization_format TEXT, offered_qos_profiles TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER,
                              timestamp INTEGER, data BLOB);
        INSERT INTO topics VALUES (1, '/pose', 'geometry_msgs/msg/PoseStamped', 'cdr', '');
        """
    )
    for index in range(5):
        payload = _json_envelope({
            "header": {"stamp": {"sec": index, "nanosec": 0}, "frame_id": "map"},
            "pose": {"position": {"x": float(index), "y": 0.0, "z": 0.0},
                     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        })
        connection.execute(
            "INSERT INTO messages VALUES (?, 1, ?, ?)", (index + 1, index * 10**9, payload)
        )
    connection.commit()
    connection.close()
    return db_path


def test_cli_topics_and_info(json_pose_bag, capsys):
    assert cli_main(["topics", str(json_pose_bag)]) == 0
    assert "/pose" in capsys.readouterr().out
    assert cli_main(["info", str(json_pose_bag), "--messages", "10"]) == 0
    out = capsys.readouterr().out
    assert "/pose: 5 messages" in out
    assert "1.0 Hz" in out


def test_cli_export_and_reload(json_pose_bag, tmp_path, capsys):
    out_path = tmp_path / "pose.npz"
    assert cli_main(["export", str(json_pose_bag), "-t", "/pose", "-o", str(out_path)]) == 0
    loaded = load_topic_npz(out_path)
    assert loaded["data"].shape == (5, 7)
    assert np.allclose(loaded["data"][:, 0], np.arange(5.0))


def test_cli_demo(tmp_path, capsys):
    out_dir = tmp_path / "demo"
    assert cli_main(["demo", "-o", str(out_dir), "--duration", "1.0"]) == 0
    assert (out_dir / "trajectory.tum").exists()
    assert (out_dir / "points.npz").exists()
    assert (out_dir / "viewer.html").exists()


def test_cli_stack_ragged_messages():
    from arraydataengine.cli import _stack_messages

    stacked = _stack_messages([np.ones((2, 3)), np.ones((4, 3))])
    assert stacked.shape == (2, 4, 3)
    assert np.allclose(stacked[0, 2:], 0.0)
