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
