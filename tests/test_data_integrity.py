"""Regressions for storage identity, resumed ingest, and row metadata."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from arraydataengine import DataBuffer
from arraydataengine.ops import CancellationToken, PipelineCancelled, source_pipeline, topic_pipeline


class Messages:
    def __init__(self, messages):
        self.messages = messages

    def get_topics(self):
        return list(dict.fromkeys(message["topic"] for message in self.messages))

    def get_count(self, topic):
        return sum(message["topic"] == topic for message in self.messages)

    def get_message(self):
        yield from self.messages


def message(value, topic="/sensor", frame_id=None):
    return {"topic": topic, "name": str(value), "timestamp": float(value),
            "data": np.array([float(value)]), "frame_id": frame_id}


@pytest.fixture(params=["arrow", "tiledb"])
def backend(request):
    pytest.importorskip("pyarrow" if request.param == "arrow" else "tiledb")
    return request.param


def topic_path(buffer, topic):
    impl = buffer.buffer_impl
    return Path(impl._topic_dir(topic) if buffer.backend == "arrow" else impl._get_array_uri(topic))


def test_colliding_topics_remain_separate_after_reopen(tmp_path, backend):
    topics = ["/a/b", "/a_b", "/a%2Fb", "/a/b__timestamps", "../outside"]
    source = Messages([message(i, topic) for i, topic in enumerate(topics)])
    with DataBuffer(source, data_uri=str(tmp_path / "store"), axis=topics[0], backend=backend, preload=0) as buffer:
        buffer.load_data_db(topics[0])
        paths = [topic_path(buffer, topic) for topic in topics]
        assert len(set(paths)) == len(topics)
        assert all(path.parent == tmp_path / "store" for path in paths)
        if backend == "tiledb":
            import tiledb
            with tiledb.Group(str(tmp_path / "store"), "r") as group:
                assert len(group) == 2 * len(topics)
    with DataBuffer(None, data_uri=str(tmp_path / "store"), backend=backend) as reopened:
        assert set(reopened.get_topics()) == set(topics)
        for i, topic in enumerate(topics):
            np.testing.assert_array_equal(reopened.get_index_range(topic)["data"], [[i]])


def test_legacy_topic_location_can_resume_alongside_colliding_topic(tmp_path, backend):
    source = Messages([message(0, "/a/b"), message(1, "/a_b"), message(2, "/a/b")])
    root = tmp_path / "store"
    first = DataBuffer(source, data_uri=str(root), backend=backend, axis="/a/b", preload=0)
    first.roll_buffer("/a/b")
    original = topic_path(first, "/a/b")
    first.close(closed=False)
    legacy = root / "_a_b"
    original.rename(legacy)
    if backend == "tiledb":
        Path(str(original) + "__timestamps").rename(str(legacy) + "__timestamps")
    # Release the old object's destructor before moving any more data.
    first.buffer_impl.read_only = True
    with DataBuffer(source, data_uri=str(root), backend=backend, axis="/a/b", preload=0) as resumed:
        assert topic_path(resumed, "/a/b") == legacy
        resumed.load_data_db("/a/b")
        np.testing.assert_array_equal(resumed.get_index_range("/a/b")["data"], [[0], [2]])
        np.testing.assert_array_equal(resumed.get_index_range("/a_b")["data"], [[1]])
    with DataBuffer(None, data_uri=str(root), backend=backend) as reopened:
        assert set(reopened.get_topics()) == {"/a/b", "/a_b"}
        np.testing.assert_array_equal(reopened.get_index_range("/a/b")["data"], [[0], [2]])


@pytest.mark.parametrize("filtered", [False, True])
def test_checkpoint_resume_matches_uninterrupted_multitopic_ingest(tmp_path, backend, filtered):
    source = Messages([message(i, "/a" if i % 3 else "/b") for i in range(12)])
    pipeline = source_pipeline(source)
    if filtered:
        pipeline = pipeline.filter(lambda msg: msg["timestamp"] % 2 == 0).index_range(0, 10, 2)
    expected = list(pipeline.iter_messages())
    checkpoint = {}
    token = CancellationToken()

    def cancel(progress):
        if progress.processed == 5:
            token.cancel()

    with pytest.raises(PipelineCancelled):
        pipeline.to_buffer(data_uri=str(tmp_path / "store"), backend=backend, use_db=True,
                           checkpoint=checkpoint, cancel_token=token, progress_callback=cancel)
    # Exercise the documented JSON checkpoint round trip.
    checkpoint = json.loads(json.dumps(checkpoint))
    with pipeline.to_buffer(data_uri=str(tmp_path / "store"), backend=backend, use_db=True,
                            checkpoint=checkpoint) as resumed:
        for topic in source.get_topics():
            values = [msg["data"] for msg in expected if msg["topic"] == topic]
            np.testing.assert_array_equal(resumed.get_index_range(topic)["data"], values)
    assert checkpoint["done"] is True


def test_write_pipeline_into_empty_persistent_buffer(tmp_path, backend):
    source = Messages([message(i) for i in range(3)])
    with DataBuffer(source, data_uri=str(tmp_path / "store"), backend=backend, preload=0) as buffer:
        source_pipeline(source).write_to_buffer(buffer)
    with DataBuffer(None, data_uri=str(tmp_path / "store"), backend=backend) as reopened:
        np.testing.assert_array_equal(reopened.get_index_range("/sensor")["data"], [[0], [1], [2]])


def test_arrow_append_owns_reused_array(tmp_path):
    pytest.importorskip("pyarrow")
    source = Messages([message(0)])
    with DataBuffer(source, data_uri=str(tmp_path / "store"), backend="arrow", preload=0) as buffer:
        shared = np.array([1.])
        for i in range(3):
            shared[:] = i
            buffer.append_buffer({**message(i), "data": shared})
        shared[:] = 99
    with DataBuffer(None, data_uri=str(tmp_path / "store"), backend="arrow") as reopened:
        np.testing.assert_array_equal(reopened.get_index_range("/sensor")["data"], [[0], [1], [2]])


@pytest.mark.parametrize("workers", [1, 2])
def test_memory_frame_filters_preserve_rows_across_maps_and_collection(workers):
    source = Messages([message(i, frame_id=frame) for i, frame in enumerate(["map", "odom", None, "map"])])
    buffer = DataBuffer(source, axis="/sensor", buffer_depth=4, preload=True)
    pipeline = buffer.topic("/sensor")
    np.testing.assert_array_equal(pipeline.frame_id("map").collect()["ts"], [0, 3])
    result = pipeline.map(lambda data: data + 10).frame_id("map").collect(chunk_size=2, max_workers=workers)
    np.testing.assert_array_equal(result["data"], [[10], [13]])
    assert result["frame_ids"].tolist() == ["map", "map"]
    all_rows = pipeline.collect(chunk_size=1, max_workers=workers)
    assert all_rows["frame_ids"].tolist() == ["map", "odom", None, "map"]
    np.testing.assert_array_equal(topic_pipeline(all_rows).frame_id("odom").collect()["ts"], [1])
    selected_view = buffer.topic_view("/sensor").select_indices(1, 3).map(lambda data: data)
    assert selected_view.frame_ids.tolist() == ["odom", None]
    windows = list(pipeline.window(size=2).iter_windows())
    assert windows[-1].frame_ids.tolist() == [None, "map"]


def test_memory_frame_ids_follow_ring_wrap_and_reset():
    source = Messages([message(0, frame_id="map"), message(1, frame_id="odom")])
    buffer = DataBuffer(source, axis="/sensor", buffer_depth=2, preload=True)
    buffer.append_buffer(message(2, frame_id="odom"))
    assert buffer.buffer_impl.frame_ids["/sensor"] == "odom"
    assert buffer.get_index_range("/sensor")["frame_ids"].tolist() == ["odom", "odom"]
    assert buffer.topic("/sensor").frame_id("map").collect()["ts"].size == 0
    buffer.reset(preload=True)
    np.testing.assert_array_equal(buffer.topic("/sensor").frame_id("map").collect()["ts"], [0])


@pytest.mark.parametrize("little_endian", [True, False])
def test_standalone_db3_decodes_standard_sensor_types(tmp_path, little_endian):
    pytest.importorskip("rosbags")
    from rosbags.typesys import Stores, get_typestore
    from arraydataengine.source import DataSources

    store = get_typestore(Stores.ROS2_JAZZY)
    types = store.types
    header = types["std_msgs/msg/Header"](types["builtin_interfaces/msg/Time"](7, 500000000), "sensor")
    vector = types["geometry_msgs/msg/Vector3"](1., 2., 3.)
    quat = types["geometry_msgs/msg/Quaternion"](0., 0., 0., 1.)
    pose = types["geometry_msgs/msg/Pose"](types["geometry_msgs/msg/Point"](4., 5., 6.), quat)
    twist = types["geometry_msgs/msg/Twist"](vector, vector)
    imu = types["sensor_msgs/msg/Imu"](header, quat, np.arange(9.), vector, np.arange(9.), vector, np.arange(9.))
    image = types["sensor_msgs/msg/Image"](header, 2, 2, "mono8", 0, 3, np.array([1,2,99,3,4,99], dtype=np.uint8))
    nav = types["sensor_msgs/msg/NavSatFix"](header, types["sensor_msgs/msg/NavSatStatus"](0, 1), 37., -122., 10., np.zeros(9), 0)
    odom = types["nav_msgs/msg/Odometry"](header, "base", types["geometry_msgs/msg/PoseWithCovariance"](pose, np.arange(36.)), types["geometry_msgs/msg/TwistWithCovariance"](twist, np.arange(36.)))
    path = tmp_path / "sensors.db3"
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);")
        for i, (topic, msg) in enumerate(zip(["/imu", "/image", "/gps", "/odom"], [imu, image, nav, odom]), 1):
            connection.execute("INSERT INTO topics VALUES (?, ?, ?)", (i, topic, msg.__msgtype__))
            raw = bytes(store.serialize_cdr(msg, msg.__msgtype__, little_endian=little_endian))
            connection.execute("INSERT INTO messages VALUES (?, ?, ?, ?)", (i, i, i * 10**9, raw))
    source = DataSources(str(path))
    rows = {row["topic"]: row for row in source.get_message()}
    assert set(rows) == set(source.get_topics())
    assert all(source.get_count(topic) == 1 for topic in rows)
    assert all(row["timestamp"] == 7.5 and row["frame_id"] == "sensor" for row in rows.values())
    np.testing.assert_array_equal(rows["/imu"]["data"][1], [0, 4, 8, 0])
    np.testing.assert_array_equal(rows["/image"]["data"], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(rows["/gps"]["data"], [37, -122, 10])
    np.testing.assert_array_equal(rows["/odom"]["data"][0], [4, 5, 6, 0])
    np.testing.assert_array_equal(rows["/odom"]["data"][3], [21, 28, 35, 0])
