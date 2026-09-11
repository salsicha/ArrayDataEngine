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
        buf.buffer_impl._manifest_path("/camera/image").read_text()
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
        first.buffer_impl._manifest_path("sensor_topic").read_text()
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

    fragments = list(buf.buffer_impl._topic_dir("sensor_topic").glob("part-*.parquet"))
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


@pytest.mark.parametrize("committed_rows", [0, 1])
@pytest.mark.parametrize("failure", ["fragment", "manifest"])
def test_arrow_recovers_after_process_exit(tmp_path, committed_rows, failure):
    import subprocess
    import sys

    script = r"""
import os
import sys
import numpy as np
from arraydataengine.buffers.arrow_buffer import ArrowBuffer
root, committed, failure = sys.argv[1], int(sys.argv[2]), sys.argv[3]
buf = ArrowBuffer(None, None, root, flush_bytes=1)
def message(i):
    return dict(topic='sensor_topic', timestamp=100.0+i*0.1, name=b'sensor_frame',
                frame_id='map', data=np.array([float(i), float(i)*2]))
for i in range(committed):
    buf.append_buffer(message(i))
replace = os.replace
def crash_before_publish(src, dst):
    if failure == 'fragment' and str(dst).endswith('.parquet'):
        os._exit(73)
    if failure == 'manifest' and len(buf._fragments['sensor_topic']) > committed:
        os._exit(73)
    replace(src, dst)
os.replace = crash_before_publish
buf.append_buffer(message(committed))
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path),
                             str(committed_rows), failure], capture_output=True, text=True)
    assert result.returncode == 73, result.stderr
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="sensor_topic") as reader:
        assert reader.get_size() == committed_rows
        assert len(reader.topic("sensor_topic").collect()["ts"]) == committed_rows
    with DataBuffer(StreamSource(3), data_uri=tmp_path, backend="arrow",
                    axis="sensor_topic", preload=0, backend_options={"flush_bytes": 1}) as resumed:
        resumed.load_data_db("sensor_topic")
        assert resumed.get_size() == 3
        expected = [[0, 0], [1, 2], [2, 4]]
        np.testing.assert_array_equal(resumed.get_index_range("sensor_topic")["data"], expected)
        np.testing.assert_array_equal(resumed.topic("sensor_topic").collect()["data"], expected)


def test_arrow_reconciles_legacy_fragment_count(tmp_path):
    with DataBuffer(StreamSource(2), data_uri=tmp_path, backend="arrow",
                    axis="sensor_topic", preload=0, backend_options={"flush_bytes": 1}) as buf:
        buf.load_data_db("sensor_topic")
    manifest_path = buf.buffer_impl._manifest_path("sensor_topic")
    manifest = json.loads(manifest_path.read_text())
    del manifest["fragments"]
    manifest["count"] = 1
    manifest_path.write_text(json.dumps(manifest))
    # Freeze the old instance so its destructor cannot update our legacy fixture.
    buf.buffer_impl.read_only = True
    with DataBuffer(StreamSource(3), data_uri=tmp_path, backend="arrow",
                    axis="sensor_topic", preload=0) as resumed:
        assert resumed.get_size() == 2
        resumed.load_data_db("sensor_topic")
        np.testing.assert_array_equal(resumed.get_index_range("sensor_topic")["data"],
                                      [[0, 0], [1, 2], [2, 4]])


@pytest.mark.parametrize("flush_bytes", [1, 1000000])
def test_arrow_rejects_dtype_changes_without_modifying_data(tmp_path, flush_bytes):
    with DataBuffer(None, data_uri=tmp_path, backend="arrow",
                    backend_options={"flush_bytes": flush_bytes}) as buf:
        message = dict(topic="t", timestamp=0., name="first", frame_id="map",
                       data=np.array([1, 2], dtype=np.int32))
        buf.append_buffer(message)
        with pytest.raises(ValueError, match="must keep dtype"):
            buf.append_buffer({**message, "data": np.array([1.5, 2.5])})
        assert buf.counters["t"] == 1
        buf.append_buffer({**message, "timestamp": 1.})
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="t") as reader:
        rows = reader.get_index_range("t")
        assert rows["data"].dtype == np.int32
        np.testing.assert_array_equal(rows["data"], [[1, 2], [1, 2]])
        with pytest.raises(ValueError, match="must keep dtype"):
            reader.append_buffer({**message, "data": np.array([1., 2.])})


@pytest.mark.parametrize("frames", [["map", "odom", None, "map"], ["map", None], [None, "map"]])
@pytest.mark.parametrize("workers", [1, 2])
def test_arrow_preserves_row_frames_through_reads_and_maps(tmp_path, frames, workers):
    from arraydataengine.ops import topic_pipeline

    with DataBuffer(None, data_uri=tmp_path, backend="arrow") as buf:
        for i, frame in enumerate(frames):
            buf.append_buffer(dict(topic="t", timestamp=float(i), name=str(i),
                                   frame_id=frame, data=np.array([float(i)])))
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="t") as reader:
        expected = [i for i, frame in enumerate(frames) if frame == "map"]
        assert reader.get_index_range("t")["frame_ids"].tolist() == frames
        assert reader.get_time_range("t", 0, len(frames))["frame_ids"].tolist() == frames
        assert reader.topic_view("t").frame_ids.tolist() == frames
        pipeline = reader.topic("t")
        assert pipeline.frame_id("map").collect()["ts"].tolist() == expected
        mapped = pipeline.map(lambda data: data + 10).frame_id("map").collect(
            chunk_size=1, max_workers=workers)
        assert mapped["ts"].tolist() == expected
        assert mapped["frame_ids"].tolist() == ["map"] * len(expected)
        collected = pipeline.collect(chunk_size=1, max_workers=workers)
        assert collected["frame_ids"].tolist() == frames
        assert topic_pipeline(collected).frame_id("map").collect()["ts"].tolist() == expected


def test_arrow_ignores_stale_topic_frame_metadata(tmp_path):
    with DataBuffer(None, data_uri=tmp_path, backend="arrow") as buf:
        for i, frame in enumerate(["map", None]):
            buf.append_buffer(dict(topic="t", timestamp=float(i), name=str(i),
                                   frame_id=frame, data=np.array([float(i)])))
    manifest_path = buf.buffer_impl._manifest_path("t")
    manifest = json.loads(manifest_path.read_text())
    manifest["frame_id"] = "map"  # Metadata written by older versions.
    manifest_path.write_text(json.dumps(manifest))
    buf.buffer_impl.read_only = True
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="t") as reader:
        assert reader.topic("t").frame_id("map").collect()["ts"].tolist() == [0.]



def test_arrow_failed_manifest_commit_can_retry(tmp_path, monkeypatch):
    import os

    buf = ArrowBuffer(None, None, tmp_path, flush_bytes=1)
    message = dict(topic="t", timestamp=0., name="first", data=np.array([1.]))
    replace = os.replace

    def fail_commit(src, dst):
        if str(dst).endswith("manifest.json") and buf._persisted.get("t", 0):
            raise OSError("simulated manifest write failure")
        replace(src, dst)

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", fail_commit)
        with pytest.raises(OSError, match="simulated"):
            buf.append_buffer(message)
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="t") as reader:
        assert reader.get_size() == 0
    buf.close()
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="t") as reader:
        assert reader.get_size() == 1
        np.testing.assert_array_equal(reader.get_index_range("t")["data"], [[1.]])


def test_arrow_reports_unreadable_manifest_without_rewriting(tmp_path):
    topic_dir = tmp_path / "topic-t.data"
    topic_dir.mkdir()
    manifest = topic_dir / "manifest.json"
    manifest.write_text('{"topic":')
    with pytest.raises(ValueError, match="Unreadable Arrow manifest"):
        DataBuffer(StreamSource(), data_uri=tmp_path, backend="arrow", preload=0)
    assert manifest.read_text() == '{"topic":'
