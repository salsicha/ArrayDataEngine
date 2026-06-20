import os
import tempfile
import shutil
import pytest
import numpy as np
import tiledb

from ade.buffer import DataBuffer

# Mock Data Source
class MockDataSource:
    def __init__(self):
        self.count = 0

    def get_topics(self):
        return ["sensor_topic"]

    def get_count(self, topic):
        return 5

    def get_message(self):
        for i in range(5):
            yield {
                "topic": "sensor_topic",
                "timestamp": 100.0 + i * 0.1,
                "name": b"sensor_frame",
                "data": np.array([float(i), float(i) * 2.0], dtype=np.float64)
            }


class SyntheticMultiTopicSource:
    def get_topics(self):
        return ["/camera/image", "/imu"]

    def get_count(self, topic):
        return {
            "/camera/image": 3,
            "/imu": 2,
        }[topic]

    def get_message(self):
        messages = [
            ("/camera/image", 10.0, "image_0", [0.0, 0.0]),
            ("/imu", 10.05, "imu_0", [100.0, 100.5]),
            ("/camera/image", 10.1, "image_1", [1.0, 1.0]),
            ("/imu", 10.15, "imu_1", [101.0, 101.5]),
            ("/camera/image", 10.2, "image_2", [2.0, 2.0]),
        ]
        for topic, timestamp, name, data in messages:
            yield {
                "topic": topic,
                "timestamp": timestamp,
                "name": name,
                "data": np.asarray(data, dtype=np.float64),
            }


def test_numpy_buffer():
    source = MockDataSource()
    buf = DataBuffer(
        data_source=source,
        buffer_depth=3,
        topics=["sensor_topic"],
        axis="sensor_topic",
        use_db=False
    )

    # Initial roll during constructor should load the first message
    assert "sensor_topic" in buf.get_buffer()
    
    # Roll a few more times
    buf.roll_buffer("sensor_topic")
    buf.roll_buffer("sensor_topic")

    # The last element data should represent the third message (index 2: [2.0, 4.0])
    val = buf[-1]
    assert np.allclose(val, np.array([2.0, 4.0]))

    # Set item
    buf[-1] = np.array([10.0, 20.0], dtype=np.float64)
    assert np.allclose(buf[-1], np.array([10.0, 20.0]))


def test_numpy_buffer_synthetic_multitopic_rolls_non_axis_messages():
    source = SyntheticMultiTopicSource()
    buf = DataBuffer(
        data_source=source,
        buffer_depth=3,
        axis="/camera/image",
        use_db=False,
        preload=0,
    )

    buf.roll_buffer("/camera/image")
    assert list(buf.get_buffer()) == ["/camera/image"]

    buf.roll_buffer("/camera/image")
    buf.roll_buffer("/camera/image")

    image_range = buf.get_time_range("/camera/image", 10.0, 10.2)
    imu_range = buf.get_time_range("/imu", 10.0, 10.2)

    assert np.allclose(image_range["ts"], np.array([10.0, 10.1, 10.2]))
    assert np.allclose(image_range["data"], np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    assert np.allclose(imu_range["ts"], np.array([10.05, 10.15]))
    assert np.allclose(imu_range["data"], np.array([[100.0, 100.5], [101.0, 101.5]]))


def test_numpy_buffer_time_ranges():
    source = MockDataSource()
    buf = DataBuffer(
        data_source=source,
        buffer_depth=5,
        topics=["sensor_topic"],
        axis="sensor_topic",
        use_db=False
    )

    for _ in range(4):
        buf.roll_buffer("sensor_topic")

    time_range = buf.get_time_range("sensor_topic", 100.05, 100.25)
    assert np.allclose(time_range["ts"], np.array([100.1, 100.2]))
    assert np.allclose(time_range["data"], np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert time_range["id"].tolist() == [b"sensor_frame", b"sensor_frame"]
    assert time_range["topic"] == "sensor_topic"

    last_window = buf.get_last_seconds("sensor_topic", 0.15)
    assert np.allclose(last_window["ts"], np.array([100.3, 100.4]))
    assert np.allclose(last_window["data"], np.array([[3.0, 6.0], [4.0, 8.0]]))


def test_tiledb_buffer():
    temp_dir = tempfile.mkdtemp()
    group_uri = os.path.join(temp_dir, "tiledb_test_group/")
    
    try:
        source = MockDataSource()
        buf = DataBuffer(
            data_source=source,
            buffer_depth=5,
            data_uri=group_uri,
            topics=["sensor_topic"],
            axis="sensor_topic",
            use_db=True
        )

        # Roll to load all messages
        buf.roll_buffer("sensor_topic")
        buf.roll_buffer("sensor_topic")
        buf.roll_buffer("sensor_topic")

        # Get buffer data
        data_dict = buf.get_buffer()
        assert "sensor_topic" in data_dict
        # The data should contain the appended messages features
        assert len(data_dict["sensor_topic"]["data"]) > 0

        # Retrieve a slice
        sliced = buf[0:2]
        assert len(sliced) == 2

        # Retrieve integer index
        val = buf[0]
        assert np.allclose(val, np.array([0.0, 0.0]))

        # Cleanup internal writers
        buf.buffer_impl.close()

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_tiledb_buffer_time_ranges():
    temp_dir = tempfile.mkdtemp()
    group_uri = os.path.join(temp_dir, "tiledb_time_range_group/")

    try:
        source = MockDataSource()
        buf = DataBuffer(
            data_source=source,
            buffer_depth=5,
            data_uri=group_uri,
            topics=["sensor_topic"],
            axis="sensor_topic",
            use_db=True
        )

        for _ in range(4):
            buf.roll_buffer("sensor_topic")

        time_range = buf.get_time_range("sensor_topic", 100.05, 100.25)
        assert np.allclose(time_range["ts"], np.array([100.1, 100.2]))
        assert np.allclose(time_range["data"], np.array([[1.0, 2.0], [2.0, 4.0]]))

        last_window = buf.get_last_seconds("sensor_topic", 0.15)
        assert np.allclose(last_window["ts"], np.array([100.3, 100.4]))
        assert np.allclose(last_window["data"], np.array([[3.0, 6.0], [4.0, 8.0]]))

        buf.buffer_impl.close()

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_tiledb_buffer_context_manager_and_timestamp_sidecar():
    temp_dir = tempfile.mkdtemp()
    group_uri = os.path.join(temp_dir, "tiledb_sidecar_group/")

    try:
        source = MockDataSource()
        with DataBuffer(
            data_source=source,
            buffer_depth=5,
            data_uri=group_uri,
            topics=["sensor_topic"],
            axis="sensor_topic",
            use_db=True
        ) as buf:
            for _ in range(4):
                buf.roll_buffer("sensor_topic")

            data_dict = buf.get_buffer()
            assert np.allclose(data_dict["sensor_topic"]["ts"], np.array([100.0, 100.1, 100.2, 100.3, 100.4]))

        array_uri = group_uri + "sensor_topic"
        timestamp_uri = group_uri + "sensor_topic__timestamps"
        assert os.path.exists(timestamp_uri)

        with tiledb.open(array_uri, "r") as array:
            assert "timestamp" not in array.meta
            assert bool(array.meta["closed"]) is True

        with tiledb.open(timestamp_uri, "r") as array:
            assert bool(array.meta["closed"]) is True
            assert np.allclose(array[0:5]["timestamp"], np.array([100.0, 100.1, 100.2, 100.3, 100.4]))

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_tiledb_buffer_synthetic_multitopic_persistence(tmp_path):
    group_uri = str(tmp_path / "synthetic_tiledb_group") + "/"

    with DataBuffer(
        data_source=SyntheticMultiTopicSource(),
        data_uri=group_uri,
        axis="/camera/image",
        use_db=True,
        preload=0,
    ) as buf:
        buf.load_data_db("/camera/image")

        buffer = buf.get_buffer()
        assert set(buffer) == {"/camera/image", "/imu"}
        assert np.allclose(buffer["/camera/image"]["ts"], np.array([10.0, 10.1, 10.2]))
        assert np.allclose(buffer["/imu"]["ts"], np.array([10.05, 10.15]))
        assert np.allclose(buffer["/camera/image"]["data"], np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        assert np.allclose(buffer["/imu"]["data"], np.array([[100.0, 100.5], [101.0, 101.5]]))

        imu_range = buf.get_time_range("/imu", 10.1, 10.2)
        assert np.allclose(imu_range["ts"], np.array([10.15]))
        assert np.allclose(imu_range["data"], np.array([[101.0, 101.5]]))

    image_ts_uri = group_uri + "_camera_image__timestamps"
    imu_ts_uri = group_uri + "_imu__timestamps"
    assert os.path.exists(image_ts_uri)
    assert os.path.exists(imu_ts_uri)

    with tiledb.open(image_ts_uri, "r") as array:
        assert bool(array.meta["closed"]) is True
        assert np.allclose(array[0:3]["timestamp"], np.array([10.0, 10.1, 10.2]))

    with tiledb.open(imu_ts_uri, "r") as array:
        assert bool(array.meta["closed"]) is True
        assert np.allclose(array[0:2]["timestamp"], np.array([10.05, 10.15]))
