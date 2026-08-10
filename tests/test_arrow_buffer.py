"""Tests for the Apache Arrow / Parquet persistent buffer backend."""

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from arraydataengine.buffer import DataBuffer
from arraydataengine.buffers.arrow_buffer import ArrowBuffer


class StreamSource:
    def __init__(self, count=5):
        self.count = count

    def get_topics(self):
        return ["sensor_topic"]

    def get_count(self, topic):
        return self.count

    def get_message(self):
        for i in range(self.count):
            yield {
                "topic": "sensor_topic",
                "timestamp": 100.0 + i * 0.1,
                "name": b"sensor_frame",
                "data": np.array([float(i), float(i) * 2.0], dtype=np.float64),
                "frame_id": "map",
            }


class MultiTopicSource:
    def get_topics(self):
        return ["/camera/image", "/imu"]

    def get_count(self, topic):
        return {"/camera/image": 3, "/imu": 2}[topic]

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


class SpatialFrameSource:
    def get_topics(self):
        return ["sensor_topic"]

    def get_count(self, topic):
        return 4

    def get_message(self):
        messages = [
            ("map", "frame_0", [0.0, 0.0, 0.0]),
            ("odom", "frame_1", [5.0, 5.0, 0.0]),
            ("map", "frame_2", [2.0, 2.0, 0.0]),
            ("base", "frame_3", [10.0, 10.0, 0.0]),
        ]
        for index, (frame_id, name, data) in enumerate(messages):
            yield {
                "topic": "sensor_topic",
                "timestamp": float(index),
                "name": name,
                "data": np.asarray(data, dtype=np.float64),
                "frame_id": frame_id,
            }


def test_use_db_defaults_to_arrow_for_new_stores(tmp_path):
    buf = DataBuffer(
        StreamSource(), data_uri=str(tmp_path / "grp") + "/", axis="sensor_topic",
        use_db=True, preload=0,
    )
    try:
        assert buf.backend == "arrow"
        assert isinstance(buf.buffer_impl, ArrowBuffer)
    finally:
        buf.close()


def test_existing_tiledb_store_is_sniffed(tmp_path):
    pytest.importorskip("tiledb")
    group_uri = str(tmp_path / "tdb_grp") + "/"
    first = DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic",
        use_db=True, backend="tiledb", preload=0,
    )
    first.load_data_db("sensor_topic")
    first.close()

    reopened = DataBuffer(None, data_uri=group_uri, axis="sensor_topic", use_db=True)
    try:
        assert reopened.backend == "tiledb"
        assert reopened.get_size() == 5
    finally:
        reopened.close()


def test_arrow_reopens_without_original_source(tmp_path):
    group_uri = str(tmp_path / "arrow_grp") + "/"
    with DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    ) as buf:
        buf.load_data_db("sensor_topic")

    reopened = DataBuffer(None, data_uri=group_uri, axis="sensor_topic", use_db=True)
    try:
        assert reopened.backend == "arrow"
        assert reopened.get_topics() == ["sensor_topic"]
        assert reopened.get_size() == 5
        assert reopened.topic("sensor_topic").metadata.frame_id == "map"

        time_range = reopened.get_time_range("sensor_topic", 100.1, 100.3)
        assert np.allclose(time_range["ts"], np.array([100.1, 100.2, 100.3]))
        assert np.allclose(time_range["data"], np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))
        assert time_range["name"].tolist() == [b"sensor_frame"] * 3

        lazy = reopened.topic("sensor_topic").index_range(2, 5).collect(chunk_size=1)
        assert np.allclose(lazy["ts"], np.array([100.2, 100.3, 100.4]))
        assert np.allclose(lazy["data"], np.array([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]))

        last = reopened.get_last_seconds("sensor_topic", 0.15)
        assert np.allclose(last["ts"], np.array([100.3, 100.4]))

        assert np.allclose(reopened[-1], np.array([4.0, 8.0]))
    finally:
        reopened.close()


def test_arrow_resumes_partial_ingest(tmp_path):
    group_uri = str(tmp_path / "resume_grp") + "/"
    first = DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    )
    first.roll_buffer("sensor_topic")
    first.roll_buffer("sensor_topic")
    first.close(closed=False)

    resumed = DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    )
    try:
        assert resumed.buffer_impl.counters["sensor_topic"] == 2
        resumed.load_data_db("sensor_topic")
        data = resumed.get_buffer()["sensor_topic"]
        assert np.allclose(data["ts"], np.array([100.0, 100.1, 100.2, 100.3, 100.4]))
        assert np.allclose(
            data["data"],
            np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
        )
    finally:
        resumed.close()


def test_arrow_multitopic_persistence_and_closed_flags(tmp_path):
    group_uri = str(tmp_path / "multi_grp") + "/"
    with DataBuffer(
        MultiTopicSource(), data_uri=group_uri, axis="/camera/image", use_db=True, preload=0
    ) as buf:
        buf.load_data_db("/camera/image")

        buffer = buf.get_buffer()
        assert set(buffer) == {"/camera/image", "/imu"}
        assert np.allclose(buffer["/camera/image"]["ts"], np.array([10.0, 10.1, 10.2]))
        assert np.allclose(buffer["/imu"]["ts"], np.array([10.05, 10.15]))

    manifest = json.loads(
        (tmp_path / "multi_grp" / "_camera_image" / "manifest.json").read_text()
    )
    assert manifest["closed"] is True
    assert manifest["count"] == 3


def test_arrow_frame_id_and_spatial_pushdown(tmp_path):
    group_uri = str(tmp_path / "frame_grp") + "/"
    with DataBuffer(
        SpatialFrameSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    ) as buf:
        buf.load_data_db("sensor_topic")

        frames = buf.topic("sensor_topic").frame_id("map").collect(chunk_size=2)
        assert frames["name"].tolist() == [b"frame_0", b"frame_2"]

        spatial = buf.topic("sensor_topic").spatial_bounds(
            [1.0, 1.0, -1.0], [6.0, 6.0, 1.0]
        ).collect(chunk_size=2)
        assert spatial["name"].tolist() == [b"frame_1", b"frame_2"]


def test_arrow_readonly_reopen_does_not_write(tmp_path):
    group_uri = str(tmp_path / "ro_grp") + "/"
    first = DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    )
    first.roll_buffer("sensor_topic")
    first.roll_buffer("sensor_topic")
    first.close(closed=False)

    with DataBuffer(None, data_uri=group_uri, axis="sensor_topic", use_db=True) as reopened:
        assert reopened.buffer_impl.read_only is True
        reopened.get_time_range("sensor_topic", 0.0, 1e12)

    manifest = json.loads(
        (tmp_path / "ro_grp" / "sensor_topic" / "manifest.json").read_text()
    )
    assert manifest["closed"] is False
    assert manifest["count"] == 2


def test_arrow_reopen_append_persists(tmp_path):
    group_uri = str(tmp_path / "append_grp") + "/"
    first = DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    )
    first.roll_buffer("sensor_topic")
    first.roll_buffer("sensor_topic")
    first.close(closed=False)

    reopened = DataBuffer(None, data_uri=group_uri, axis="sensor_topic", use_db=True)
    reopened.append_buffer(
        {"topic": "sensor_topic", "timestamp": 200.0, "name": b"n", "data": np.array([9.0, 9.0])}
    )
    reopened.close()

    final = DataBuffer(None, data_uri=group_uri, axis="sensor_topic", use_db=True)
    try:
        assert final.buffer_impl.counters["sensor_topic"] == 3
        rows = final.get_index_range("sensor_topic")
        assert rows["ts"].tolist() == [100.0, 100.1, 200.0]
    finally:
        final.close()


def test_arrow_setitem_rejected(tmp_path):
    group_uri = str(tmp_path / "imm_grp") + "/"
    with DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    ) as buf:
        buf.load_data_db("sensor_topic")
        with pytest.raises(NotImplementedError, match="tiledb"):
            buf[0] = np.array([1.0, 1.0])


def test_arrow_rejects_shape_changes(tmp_path):
    group_uri = str(tmp_path / "shape_grp") + "/"
    with DataBuffer(
        StreamSource(), data_uri=group_uri, axis="sensor_topic", use_db=True, preload=0
    ) as buf:
        buf.roll_buffer("sensor_topic")
        with pytest.raises(ValueError, match="fixed shape"):
            buf.append_buffer(
                {"topic": "sensor_topic", "timestamp": 1.0, "name": b"n",
                 "data": np.zeros((3, 3))}
            )


def test_arrow_options_and_many_fragments_preserve_order(tmp_path):
    group_uri = str(tmp_path / "frag_grp") + "/"
    with DataBuffer(
        StreamSource(50),
        data_uri=group_uri,
        axis="sensor_topic",
        use_db=True,
        preload=0,
        backend="arrow",
        backend_options={"flush_bytes": 1, "row_group_bytes": 1, "compression": "snappy"},
    ) as buf:
        buf.load_data_db("sensor_topic")
        rows = buf.get_index_range("sensor_topic")
        assert np.allclose(rows["ts"], 100.0 + np.arange(50) * 0.1)
        chunks = list(buf.iter_topic_chunks("sensor_topic", 7))
        streamed = np.concatenate([chunk.ts for chunk in chunks])
        assert np.allclose(streamed, rows["ts"])

    fragments = list((tmp_path / "frag_grp" / "sensor_topic").glob("part-*.parquet"))
    assert len(fragments) == 50  # flush_bytes=1 forces one fragment per message


def test_backend_options_rejected_for_other_backends(tmp_path):
    with pytest.raises(ValueError, match="arrow"):
        DataBuffer(
            StreamSource(), data_uri=str(tmp_path / "x") + "/", axis="sensor_topic",
            use_db=True, backend="tiledb", backend_options={"flush_bytes": 1},
        )


def test_source_pipeline_to_buffer_defaults_to_arrow(tmp_path):
    from arraydataengine.ops import source_pipeline

    group_uri = str(tmp_path / "pipe_grp") + "/"
    buf = source_pipeline(StreamSource()).to_buffer(data_uri=group_uri, use_db=True)
    try:
        assert buf.backend == "arrow"
        assert buf.get_size() == 5
    finally:
        buf.close()
