import sqlite3
from types import SimpleNamespace

import numpy as np


class PoseStamped(SimpleNamespace):
    pass

from ade.ops import apply_transform, pose_to_matrix, source_pipeline, valid_point_cloud_points
from ade.sensors.pose_sensor import PoseSensor


def test_pose_sensor_numpyifies_pose_stamped():
    msg = PoseStamped(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=250000000),
            frame_id="map",
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    sensor = PoseSensor(rawdata=b"", msgtype="geometry_msgs/msg/PoseStamped")
    sensor.deserialize = lambda: msg

    data, name, ts = sensor.numpyify()

    assert name == "PoseStamped"
    assert ts == 12.25
    assert sensor.frame_id == "map"
    assert np.allclose(data, np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))


def test_pose_to_matrix_transforms_point_cloud_points():
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    matrix = pose_to_matrix(pose)
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    transformed = apply_transform(points, matrix)

    assert np.allclose(transformed[0], np.array([1.0, 3.0, 3.0]))
    assert np.allclose(transformed[1], np.array([1.0, 2.0, 3.0]))


def test_valid_point_cloud_points_removes_padding():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [np.nan, 0.0, 1.0]])

    filtered = valid_point_cloud_points(points)

    assert np.allclose(filtered, np.array([[1.0, 2.0, 3.0]]))


def test_source_pipeline_nearest_topic_pairs():
    messages = [
        {"topic": "/pose", "timestamp": 0.0, "data": np.array([0])},
        {"topic": "/cloud", "timestamp": 0.1, "data": np.array([1])},
        {"topic": "/pose", "timestamp": 0.2, "data": np.array([2])},
        {"topic": "/cloud", "timestamp": 0.24, "data": np.array([3])},
    ]
    source = SimpleNamespace(
        get_topics=lambda: ["/pose", "/cloud"],
        get_message=lambda: iter(messages),
    )

    pairs = list(source_pipeline(source).nearest_topic_pairs("/cloud", "/pose"))

    assert len(pairs) == 2
    assert pairs[0][0]["timestamp"] == 0.1
    assert pairs[0][1]["timestamp"] == 0.0
    assert pairs[1][0]["timestamp"] == 0.24
    assert pairs[1][1]["timestamp"] == 0.2


def test_db3_source_sqlite_metadata_fallback_without_rosbags(monkeypatch, tmp_path):
    from ade.sources.db3_source import DB3Source
    from ade.sources import base_source

    monkeypatch.setattr(base_source, "AnyReader", None)

    db_path = tmp_path / "bag.db3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("create table topics(id integer primary key, name text, type text)")
        connection.execute("create table messages(id integer primary key, topic_id integer, timestamp integer, data blob)")
        connection.execute("insert into topics(id, name, type) values (1, '/pose', 'geometry_msgs/msg/PoseStamped')")
        connection.execute("insert into messages(topic_id, timestamp, data) values (1, 100, x'00')")
        connection.execute("insert into messages(topic_id, timestamp, data) values (1, 300, x'00')")

    source = DB3Source(str(db_path))

    assert source.get_topics() == ["/pose"]
    assert source.get_count("/pose") == 2
    assert np.isclose(source.get_duration(), 2e-7)
