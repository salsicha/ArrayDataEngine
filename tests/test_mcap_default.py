import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap")
from arraydataengine import DataBuffer, DataSources, McapRecorder, mcap_to_parquet, parquet_to_mcap
from arraydataengine.ops import verify_recording_loop_closures


def msg(i, topic="t", shape=(3,), **extra):
    return dict(topic=topic, timestamp=float(i), name="Array", data=np.full(shape, i, dtype=np.float32), **extra)


class Source:
    def __init__(self, n):
        self.n = n
    def get_topics(self):
        return ["t"]
    def get_count(self, topic):
        return self.n
    def get_message(self):
        for i in range(self.n):
            yield msg(i)


def test_default_segments_resume_metadata_and_store_source(tmp_path):
    with DataBuffer(Source(5), data_uri=tmp_path, use_db=True, axis="t", preload=0,
                    backend_options={"segment_bytes": 1}) as buffer:
        assert buffer.backend == "mcap"
        buffer.roll_buffer("t")
        buffer.roll_buffer("t")
        assert buffer[-1][0] == 1
    assert len(list(tmp_path.glob("*.mcap"))) == 2
    with DataBuffer(Source(5), data_uri=tmp_path, use_db=True, axis="t", preload=0) as buffer:
        buffer.load_data_db("t")
        assert buffer.get_size() == 5
        assert [len(c.data) for c in buffer.topic("t").index_range(1, 4).iter_chunks(2)] == [2, 1]
    assert len(list(DataSources(tmp_path).get_message())) == 5
    assert DataSources(tmp_path).get_count("t") == 5
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    with DataBuffer(None, data_uri=tmp_path, use_db=True, axis="t") as reader:
        assert reader[-2][0] == 3
        assert reader.get_time_range("t", 2, 3)["ts"].tolist() == [2, 3]
    assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


def test_mcap_store_rejects_conflicts_and_stale_writers(tmp_path):
    store = tmp_path / "store"
    one = DataBuffer(None, data_uri=store, backend="mcap")
    stale = DataBuffer(None, data_uri=store, backend="mcap")
    one.append_buffer(msg(0))
    with pytest.raises((BlockingIOError, ValueError)):
        stale.append_buffer(msg(1))
    one.close()
    with pytest.raises(ValueError, match="changed"):
        stale.append_buffer(msg(1))
    stale.close()
    bad = tmp_path / "not-a-store"
    bad.mkdir(); (bad / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="not an MCAP"):
        DataBuffer(Source(1), data_uri=bad, use_db=True)
    assert (bad / "keep.txt").read_text() == "keep"
    for backend in ("arrow", "tiledb"):
        with pytest.raises(ValueError, match="MCAP store"):
            DataBuffer(None, data_uri=store, backend=backend)


def test_mcap_store_accepts_shape_changes_and_retains_metadata(tmp_path):
    with DataBuffer(None, data_uri=tmp_path, backend="mcap") as b:
        b.append_buffer(msg(0, shape=(2,3), point_count=2, encoding="xyz", timestamp_ns=12))
        b.append_buffer(msg(1, shape=(5,3), point_count=5, encoding="xyz", timestamp_ns=13))
        values = b.get_topic("t") if hasattr(b, "get_topic") else b.buffer_impl.get_topic("t")
        assert values["data"].shape == (2,5,3)
        assert values["lengths"].tolist() == [2,5]
        assert [chunk.data.shape for chunk in b.iter_topic_chunks("t", 100)] == [(1,2,3), (1,5,3)]
    rows = list(DataSources(tmp_path).get_message())
    assert rows[0]["timestamp_ns"] == 12 and rows[1]["encoding"] == "xyz"
    assert rows[0]["data"].shape == (2,3)


def test_arrow_global_staging_budget_and_oversized_validation(tmp_path):
    pytest.importorskip("pyarrow")
    budget = 5000
    with DataBuffer(None, data_uri=tmp_path, backend="arrow",
                    backend_options={"memory_budget_bytes":budget,"flush_bytes":100000}) as b:
        for i in range(20):
            b.append_buffer(msg(i, topic=f"t{i%5}", shape=(300,)))
            assert sum(s["bytes"] for s in b.buffer_impl._staged.values()) <= budget
        before = dict(b.counters)
        with pytest.raises(ValueError, match="exceeds"):
            b.append_buffer(msg(30, topic="large", shape=(2000,)))
        assert b.counters == before
    with DataBuffer(None, data_uri=tmp_path, use_db=True) as b:
        assert b.backend == "arrow" and sum(b.counters.values()) == 20


def test_schema_cache_global_budget_and_exact_restore(tmp_path):
    pytest.importorskip("pyarrow")
    raw = tmp_path / "source.mcap"
    with McapRecorder(raw) as w:
        w.write_encoded("unknown",b"original wire data",message_encoding="custom",timestamp_ns=1)
        for i in range(20):
            w.write(msg(i, topic=f"t{i%3}", shape=(2+i%2,3), timestamp_ns=100+i, sequence=i))
    original = raw.read_bytes()
    cache = mcap_to_parquet(raw, tmp_path / "cache", topics=["t0","t1","t2"],
                            fidelity="copy", memory_budget_bytes=6000, row_group_bytes=4000)
    assert len(cache.manifest["partitions"]) == 6
    assert cache.manifest["peak_staged_bytes"] <= 6000
    assert sum(cache.manifest["counts"].values()) == 20
    assert cache.message("t0",0)["timestamp_ns"] == 100
    rows = list(cache.rows("t0",columns=["record_index"]))
    assert min(r["record_index"] for r in rows) == 1  # Unknown raw channel remains in the original.
    raw.unlink()
    restored = parquet_to_mcap(cache, tmp_path / "restored.mcap")
    assert restored.read_bytes() == original


def test_reference_cache_reuse_invalidation_and_store_restore(tmp_path):
    pytest.importorskip("pyarrow")
    store = tmp_path / "store"
    with DataBuffer(Source(2),data_uri=store,use_db=True,axis="t") as b:
        b.load_data_db("t")
    cache = mcap_to_parquet(store,tmp_path / "cache")
    again = mcap_to_parquet(store,tmp_path / "cache")
    assert again.manifest == cache.manifest
    parquet_to_mcap(cache,tmp_path / "restored")
    assert DataSources(tmp_path / "restored").get_count("t") == 2
    with DataBuffer(None,data_uri=store,backend="mcap") as b:
        b.append_buffer(msg(2))
    with pytest.raises(ValueError,match="stale"):
        mcap_to_parquet(store,tmp_path / "cache")
    first = Path(cache.manifest["sources"][0]["path"])
    first.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):
        parquet_to_mcap(cache,tmp_path / "bad.mcap")


def test_loop_closure_uses_parquet_and_only_loads_candidate_pairs(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from arraydataengine.ops import point_cloud
    raw = tmp_path / "loop.mcap"
    with McapRecorder(raw) as w:
        for i, x in enumerate([0.,10.,20.,0.1]):
            w.write(dict(topic="pose", timestamp=float(i), name="PoseStamped",
                         data=np.array([x,0,0,0,0,0,1.])))
            w.write(dict(topic="points", timestamp=float(i)+0.01, name="PointCloud2",
                         data=np.ones((10+i,3)),point_count=10+i))
    pairs = []
    def verify(clouds, poses, candidates, **kwargs):
        pairs.append([len(c) for c in clouds])
        return {"source_index":np.array([1]),"target_index":np.array([0]),
                "pose_distance":np.array([.1]),"accepted":np.array([True]),"fitness":np.array([1.]),
                "inlier_rmse":np.array([0.]),"correspondence_count":np.array([10]),
                "transform":np.eye(4)[None],"odometry_seed":np.eye(4)[None]}
    monkeypatch.setattr(point_cloud,"verify_loop_closures",verify)
    result = verify_recording_loop_closures(raw,trajectory_topic="pose",points_topic="points",
                                           radius=.5,min_separation=2,cache_dir=tmp_path / "analysis")
    assert result["source_index"].tolist() == [3]
    assert result["target_index"].tolist() == [0]
    assert pairs == [[10,13]]
    assert list((tmp_path / "analysis").rglob("*.parquet"))


def test_failed_conversion_never_publishes_partial_cache(tmp_path):
    pytest.importorskip("pyarrow")
    raw=tmp_path/'input.mcap'
    with McapRecorder(raw) as w:
        w.write(msg(0,shape=(10000,)))
    with pytest.raises(ValueError,match='exceeds'):
        mcap_to_parquet(raw,tmp_path/'bad',memory_budget_bytes=1024)
    assert not (tmp_path/'bad').exists()
    assert not list(tmp_path.glob('.parquet-build-*'))


def test_loop_closure_real_icp(tmp_path):
    pytest.importorskip("pyarrow")
    landmarks = np.random.default_rng(5).normal(size=(100, 3))
    raw = tmp_path / "loop.mcap"
    with McapRecorder(raw) as writer:
        for i, x in enumerate([0., 10., 20., .1]):
            writer.write(dict(topic="pose", timestamp=float(i), data=np.array([x,0,0,0,0,0,1.])))
            writer.write(dict(topic="points", timestamp=float(i), data=landmarks-[x,0,0], point_count=100))
    result = verify_recording_loop_closures(raw, trajectory_topic="pose", points_topic="points",
                                           radius=.5, min_separation=2, max_correspondence_distance=.5)
    assert result["source_index"].tolist() == [3]
    assert result["target_index"].tolist() == [0]
    assert result["fitness"][0] > .99
    np.testing.assert_allclose(result["transform"][0,:3,3], [.1,0,0], atol=1e-8)


def test_crash_replays_only_uncommitted_messages(tmp_path):
    import subprocess
    import sys
    code = '''
import os, sys
import numpy as np
from arraydataengine import DataBuffer
b = DataBuffer(None, data_uri=sys.argv[1], backend="mcap")
b.append_buffer(dict(topic="t", timestamp=0., data=np.zeros(3)))
b.buffer_impl.flush()
b.append_buffer(dict(topic="t", timestamp=1., data=np.ones(3)))
os._exit(0)
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
    assert DataSources(tmp_path).get_count("t") == 1
    with DataBuffer(Source(3), data_uri=tmp_path, use_db=True, axis="t", preload=0) as b:
        b.load_data_db("t")
    rows = list(DataSources(tmp_path).get_message())
    assert [row["timestamp"] for row in rows] == [0., 1., 2.]


def test_completing_one_source_keeps_unrelated_topics_open(tmp_path):
    with DataBuffer(None, data_uri=tmp_path, backend="mcap") as b:
        b.append_buffer(msg(0, topic="other"))
    with DataBuffer(Source(2), data_uri=tmp_path, use_db=True, axis="t", preload=0) as b:
        b.load_data_db("t")
    closed = json.loads((tmp_path / "mcap_store.json").read_text())["closed"]
    assert closed == {"t": True, "other": False}
