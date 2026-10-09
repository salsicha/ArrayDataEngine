"""Regressions for the second storage review (backends, hydration, ragged topics)."""

import json
import logging
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arraydataengine import DataBuffer

pa = pytest.importorskip("pyarrow")
tiledb = pytest.importorskip("tiledb")

from arraydataengine.buffers import arrow_buffer, tiledb_buffer  # noqa: E402

PERSISTENT = ["arrow", "tiledb"]
ALL_BACKENDS = ["memory", "arrow", "tiledb"]


class Source:
    def __init__(self, n=5, topics=("t",), shape=(2,), dtype=np.float64, max_points=None, counts=None):
        self.n, self.topics, self.shape, self.dtype = n, list(topics), shape, dtype
        self.max_points = max_points
        self.counts = counts

    def get_topics(self):
        return list(self.topics)

    def get_count(self, topic):
        return self.n if self.counts is None else self.counts.get(topic, 0)

    def get_message(self):
        for i in range(self.n):
            for topic in self.topics:
                yield {"topic": topic, "timestamp": float(i), "name": f"{topic}{i}", "frame_id": "map",
                       "data": np.full(self.shape, i, dtype=self.dtype)}


class Clouds:
    """Point clouds of varying size, as sources yield them: unpadded with point_count."""

    def __init__(self, sizes, topic="c", max_points=None, offset=10.0):
        self.sizes, self.topic, self.max_points, self.offset = list(sizes), topic, max_points, offset

    def get_topics(self):
        return [self.topic]

    def get_count(self, topic):
        return len(self.sizes)

    def get_message(self):
        for i, size in enumerate(self.sizes):
            yield {"topic": self.topic, "timestamp": float(i), "name": f"c{i}", "frame_id": "lidar",
                   "data": np.full((size, 3), self.offset + i, dtype=np.float32), "point_count": size}


def message(i, topic="t", data=None, **extra):
    return {"topic": topic, "timestamp": float(i), "name": f"m{i}",
            "data": np.array([float(i), float(i)]) if data is None else data, **extra}


def ingest(backend, tmp_path, source, axis, **kwargs):
    if backend == "memory":
        kwargs.setdefault("buffer_depth", 16)
        kwargs.setdefault("preload", 16)
        return DataBuffer(source, axis=axis, **kwargs)
    buf = DataBuffer(source, data_uri=str(tmp_path / backend), backend=backend, axis=axis, preload=0, **kwargs)
    buf.load_data_db(axis)
    return buf


def op(kind, *args, **kwargs):
    return SimpleNamespace(kind=kind, args=args, kwargs=kwargs)


# -- RAGGED_SPEC section 2: variable-size topics --------------------------------------


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_ragged_topics_store_padded_rows_with_lengths(tmp_path, backend):
    buf = ingest(backend, tmp_path, Clouds([3, 0, 5, 2]), "c", max_points=8)
    rows = buf.get_index_range("c")
    assert rows["data"].shape == (4, 8, 3) and rows["data"].dtype == np.float32
    assert rows["lengths"].dtype == np.int64 and rows["lengths"].tolist() == [3, 0, 5, 2]
    assert np.all(rows["data"][0, 3:] == 0) and np.all(rows["data"][2, :5] == 12)
    assert buf.get_time_range("c", 1, 2)["lengths"].tolist() == [0, 5]
    assert buf.get_last_seconds("c", 1)["lengths"].tolist() == [5, 2]
    chunks = list(buf.buffer_impl.iter_topic_chunks("c", 3))
    assert [chunk["lengths"].tolist() for chunk in chunks] == [[3, 0, 5], [2]]
    assert buf[2].shape == (5, 3) and buf[1].shape == (0, 3)  # one message: its real rows
    topic = buf.get_buffer()["c"]
    lengths = topic["length"] if isinstance(topic, np.ndarray) else topic["lengths"]
    assert list(lengths) == [3, 0, 5, 2]
    with pytest.raises(ValueError, match=r"c message has 9 rows.*capacity of 8.*ragged_topics"):
        buf.append_buffer({"topic": "c", "timestamp": 9.0, "name": "x", "point_count": 9,
                           "data": np.zeros((9, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="trailing shape"):
        buf.append_buffer({"topic": "c", "timestamp": 9.0, "name": "x", "point_count": 2,
                           "data": np.zeros((2, 4), dtype=np.float32)})
    assert buf.get_size() == 4
    buf.close()
    if backend != "memory":
        with DataBuffer(None, data_uri=str(tmp_path / backend), backend=backend, axis="c") as reader:
            assert reader.get_index_range("c")["lengths"].tolist() == [3, 0, 5, 2]
            assert reader.topic("c").collect()["data"].shape == (4, 8, 3)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_fixed_shape_topics_have_no_lengths(tmp_path, backend):
    buf = ingest(backend, tmp_path, Source(3), "t")
    assert "lengths" not in buf.get_index_range("t")
    assert all("lengths" not in chunk for chunk in buf.buffer_impl.iter_topic_chunks("t", 2))
    topic = buf.get_buffer()["t"]
    assert "length" not in (topic.dtype.names if isinstance(topic, np.ndarray) else topic)
    buf.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_ragged_capacity_priority(tmp_path, backend):
    source = Clouds([2, 1], max_points=6)
    buf = ingest(backend, tmp_path / "a", source, "c")
    assert buf.get_index_range("c")["data"].shape[1] == 6  # the source's max_points
    buf.close()
    buf = ingest(backend, tmp_path / "b", source, "c", max_points=5)
    assert buf.get_index_range("c")["data"].shape[1] == 5  # DataBuffer(max_points) wins
    buf.close()
    buf = ingest(backend, tmp_path / "c", source, "c", max_points=5, ragged_topics={"c": 4})
    assert buf.get_index_range("c")["data"].shape[1] == 4  # ragged_topics wins
    buf.close()
    from arraydataengine.sensors.pointcloud2_sensor import DEFAULT_MAX_POINTS

    buf = ingest(backend, tmp_path / "d", Clouds([2]), "c")
    assert buf.get_index_range("c")["data"].shape[1] == DEFAULT_MAX_POINTS
    buf.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_ragged_topics_option_without_point_count(tmp_path, backend):
    class Rows(Source):
        def get_message(self):
            for i, size in enumerate([1, 3]):
                yield {"topic": "t", "timestamp": float(i), "name": "t", "data": np.ones((size, 2))}

    buf = ingest(backend, tmp_path, Rows(2), "t", ragged_topics={"t": 4})
    rows = buf.get_index_range("t")
    assert rows["data"].shape == (2, 4, 2) and rows["lengths"].tolist() == [1, 3]
    buf.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_ragged_spatial_pushdown_ignores_padding(tmp_path, backend):
    # Clouds far from the origin: zero padding must never make the origin "inside".
    buf = ingest(backend, tmp_path, Clouds([2, 3], offset=50.0), "c", max_points=4)
    near_origin = op("spatial_bounds", np.full(3, -1.0), np.full(3, 1.0), columns=(0, 1, 2))
    chunks = list(buf.buffer_impl.iter_topic_chunks("c", 4, operations=(near_origin,)))
    assert sum(len(chunk["ts"]) for chunk in chunks) == 0
    near_cloud = op("spatial_bounds", np.full(3, 50.5), np.full(3, 51.5), columns=(0, 1, 2))
    chunks = list(buf.buffer_impl.iter_topic_chunks("c", 4, operations=(near_cloud,)))
    assert np.concatenate([chunk["ts"] for chunk in chunks]).tolist() == [1.0]
    buf.close()


@pytest.mark.parametrize("backend", PERSISTENT)
def test_point_clouds_append_to_legacy_fixed_shape_topics(tmp_path, backend):
    # Stores written before lengths existed hold padded (max_points, 3) clouds
    # without a length; new unpadded clouds keep being padded to that shape.
    uri = str(tmp_path / "store")
    with DataBuffer(None, data_uri=uri, backend=backend) as buf:
        buf.append_buffer({"topic": "c", "timestamp": 0.0, "name": "c", "data": np.ones((6, 3))})
    with DataBuffer(None, data_uri=uri, backend=backend, axis="c") as buf:
        buf.append_buffer({"topic": "c", "timestamp": 1.0, "name": "c", "point_count": 2,
                           "data": np.full((2, 3), 7.0)})
        rows = buf.get_index_range("c")
    assert "lengths" not in rows and rows["data"].shape == (2, 6, 3)
    assert rows["data"][1, :2].tolist() == [[7.0] * 3] * 2 and not rows["data"][1, 2:].any()


def test_memory_get_buffer_ragged_structured_field():
    buf = DataBuffer(Clouds([1, 2]), axis="c", buffer_depth=4, preload=2, max_points=3)
    topic = buf.get_buffer()["c"]
    assert topic.dtype.names == ("ts", "id", "frame_id", "data", "length")
    assert topic["data"].shape == (2, 3, 3) and topic["length"].tolist() == [1, 2]


# -- finding 1: legacy Arrow bool payloads --------------------------------------------


def test_arrow_resume_keeps_legacy_bool_tensor_encoding(tmp_path, monkeypatch):
    uri = str(tmp_path / "store")
    # Older versions stored bool payloads as bool tensors, not raw bytes.
    monkeypatch.setattr(arrow_buffer, "_stores_raw_bytes", lambda dtype: dtype.kind not in "iufb")
    with DataBuffer(Source(4, shape=(3,), dtype=np.bool_), data_uri=uri, backend="arrow", axis="t",
                    preload=0) as buf:
        for _ in range(2):
            buf.roll_buffer("t")
    monkeypatch.undo()
    with DataBuffer(Source(4, shape=(3,), dtype=np.bool_), data_uri=uri, backend="arrow", axis="t",
                    preload=0) as buf:
        buf.load_data_db("t")
    with DataBuffer(None, data_uri=uri, backend="arrow", axis="t") as reader:
        expected = [[False] * 3, [True] * 3, [True] * 3, [True] * 3]
        assert reader.get_index_range("t")["data"].tolist() == expected
        assert reader.get_time_range("t", 0, 10)["data"].tolist() == expected
        assert reader.topic("t").collect()["data"].tolist() == expected
        assert reader[3].tolist() == [True] * 3


# -- finding 2: TileDB never claims Arrow stores --------------------------------------


def test_tiledb_open_of_arrow_store_creates_nothing(tmp_path):
    uri = tmp_path / "store"
    with DataBuffer(None, data_uri=str(uri), backend="arrow") as buf:
        buf.append_buffer(message(0))
    before = sorted(os.listdir(uri))
    for source in (None, Source(2)):
        with pytest.raises(ValueError, match="Arrow store"):
            DataBuffer(source, data_uri=str(uri), backend="tiledb", preload=0)
    assert sorted(os.listdir(uri)) == before
    with DataBuffer(None, data_uri=str(uri), use_db=True) as reader:
        assert reader.backend == "arrow" and reader.get_topics() == ["t"]
    # Older versions created an empty group here; the store stays an Arrow store.
    tiledb.group_create(str(uri))
    with DataBuffer(None, data_uri=str(uri), use_db=True) as buf:
        assert buf.backend == "arrow" and buf.get_topics() == ["t"]
        buf.append_buffer(message(1))
    with pytest.raises(ValueError, match="Arrow store"):
        DataBuffer(Source(2), data_uri=str(uri), backend="tiledb", preload=0)


def test_tiledb_read_only_open_of_missing_directory_writes_nothing(tmp_path):
    uri = tmp_path / "missing"
    with DataBuffer(None, data_uri=str(uri), backend="tiledb") as reader:
        assert reader.get_topics() == []
    assert not uri.exists()


def test_tiledb_refuses_non_empty_foreign_directory(tmp_path):
    (tmp_path / "notes.txt").write_text("not a store")
    buf = DataBuffer(None, data_uri=str(tmp_path), backend="tiledb")
    with pytest.raises(ValueError, match="not a TileDB store"):
        buf.append_buffer(message(0))
    assert sorted(os.listdir(tmp_path)) == ["notes.txt"]


def test_arrow_refuses_writing_into_tiledb_store(tmp_path):
    uri = str(tmp_path / "store")
    with DataBuffer(Source(2), data_uri=uri, backend="tiledb", axis="t", preload=0) as buf:
        buf.load_data_db("t")
    with pytest.raises(ValueError, match="TileDB store"):
        DataBuffer(Source(2), data_uri=uri, backend="arrow", axis="t", preload=0)


# -- finding 3: TileDB unicode topics resume -------------------------------------------


def test_tiledb_unicode_topic_resumes(tmp_path):
    uri = str(tmp_path / "store")
    with DataBuffer(None, data_uri=uri, backend="tiledb") as buf:
        buf.append_buffer(message(0, data=np.array(["ab", "xyz"])))
    with DataBuffer(None, data_uri=uri, backend="tiledb", axis="t") as buf:
        buf.append_buffer(message(1, data=np.array(["abcd", "e"])))
        with pytest.raises(ValueError, match="must keep dtype"):
            buf.append_buffer(message(2, data=np.array([1, 2])))
    with DataBuffer(None, data_uri=uri, backend="tiledb", axis="t") as reader:
        assert reader.get_index_range("t")["data"].tolist() == [["ab", "xyz"], ["abcd", "e"]]


# -- finding 4: TileDB crash between creating the data array and its index --------------


def test_tiledb_resumes_after_crash_before_index_creation(tmp_path, monkeypatch):
    uri = str(tmp_path / "store")
    real_create = tiledb.Array.create
    calls = []

    def crashing_create(array_uri, schema, *args, **kwargs):
        calls.append(array_uri)
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated crash")
        return real_create(array_uri, schema, *args, **kwargs)

    monkeypatch.setattr(tiledb.Array, "create", crashing_create)
    buf = DataBuffer(Source(3), data_uri=uri, backend="tiledb", axis="t", preload=0)
    with pytest.raises(KeyboardInterrupt):
        buf.roll_buffer("t")
    buf.buffer_impl.read_only = True
    monkeypatch.undo()
    with DataBuffer(None, data_uri=uri, backend="tiledb") as reader:
        assert reader.get_topics() == []
    with DataBuffer(Source(3), data_uri=uri, backend="tiledb", axis="t", preload=0) as buf:
        buf.load_data_db("t")
        assert buf.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0]


# -- findings 5, 6: TileDB close is reader-safe and never rewrites payloads -------------


def _fragments(uri):
    return len(tiledb.array_fragments(uri))


def test_tiledb_close_keeps_concurrent_readers_working(tmp_path):
    uri = str(tmp_path / "store")
    options = {"flush_bytes": 24 * 10, "row_overhead_bytes": 0}  # a fragment per 10 rows
    writer = DataBuffer(None, data_uri=uri, backend="tiledb", backend_options=options)
    for i in range(20):
        writer.append_buffer(message(i, data=np.full(3, float(i))))
    writer.close()
    writer = DataBuffer(None, data_uri=uri, backend="tiledb", backend_options=options)
    for i in range(20, 32):
        writer.append_buffer(message(i, data=np.full(3, float(i))))
    reader = DataBuffer(None, data_uri=uri, backend="tiledb", axis="t")
    chunks = reader.buffer_impl.iter_topic_chunks("t", 4)
    first = next(chunks)
    writer.close()
    rest = [chunk["ts"] for chunk in chunks]
    assert np.concatenate([first["ts"], *rest]).tolist() == [float(i) for i in range(30)]
    assert reader.get_size() == 30
    reader.close()


def test_tiledb_close_does_not_rewrite_payload_and_compact_does(tmp_path):
    uri = str(tmp_path / "store")
    options = {"flush_bytes": 3 * 800, "row_overhead_bytes": 0}
    with DataBuffer(None, data_uri=uri, backend="tiledb", backend_options=options) as buf:
        for i in range(9):
            buf.append_buffer(message(i, data=np.full(100, float(i))))
        data_uri = buf.buffer_impl._get_array_uri("t")
    assert _fragments(data_uri) == 3
    with DataBuffer(None, data_uri=uri, backend="tiledb", backend_options=options) as buf:
        buf.append_buffer(message(9, data=np.full(100, 9.0)))
    assert _fragments(data_uri) == 4  # appended, not merged into the existing payload
    with DataBuffer(None, data_uri=uri, backend="tiledb", backend_options={"consolidate_max_bytes": 1}) as buf:
        buf.compact()
    assert _fragments(data_uri) == 4  # larger than consolidate_max_bytes: metadata only
    with DataBuffer(None, data_uri=uri, backend="tiledb", axis="t") as buf:
        buf.compact()
        assert _fragments(data_uri) == 1
        assert buf.get_index_range("t")["data"][:, 0].tolist() == [float(i) for i in range(10)]


# -- finding 7: two handles creating the same Arrow topic --------------------------------


def test_arrow_second_creator_of_a_topic_is_refused(tmp_path):
    uri = str(tmp_path / "store")
    first = DataBuffer(None, data_uri=uri, backend="arrow")
    second = DataBuffer(None, data_uri=uri, backend="arrow")
    first.append_buffer(message(0))
    second.append_buffer(message(100))
    first.close()
    with pytest.raises(RuntimeError, match="created at .* by another writer"):
        second.close()
    second.buffer_impl.read_only = True
    assert sorted(os.listdir(uri)) == ["topic-t.data"]


def test_arrow_duplicate_topic_directories_keep_the_fuller_one(tmp_path, caplog):
    uri = tmp_path / "store"
    with DataBuffer(Source(3), data_uri=str(uri), backend="arrow", axis="t", preload=0) as buf:
        buf.load_data_db("t")
    with DataBuffer(None, data_uri=str(tmp_path / "other"), backend="arrow") as buf:
        buf.append_buffer(message(9))
    os.rename(tmp_path / "other" / "topic-t.data", uri / "topic-t.data.1")
    with caplog.at_level(logging.WARNING), DataBuffer(None, data_uri=str(uri), backend="arrow", axis="t") as reader:
        assert reader.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0]
    assert "several Arrow directories" in caplog.text


# -- finding 8: truncated fragment of a 0.3.1 ingest --------------------------------------


def _legacy_arrow_store(tmp_path, rows=6):
    """A 0.3.1-style store: <root>/<topic>/part-*.parquet, manifest without fragments."""
    uri = tmp_path / "legacy"
    with DataBuffer(None, data_uri=str(uri), backend="arrow", backend_options={"flush_bytes": 1}) as buf:
        for i in range(rows):
            buf.append_buffer(message(i))
    os.rename(uri / "topic-t.data", uri / "t")
    manifest = json.loads((uri / "t" / "manifest.json").read_text())
    del manifest["fragments"]
    (uri / "t" / "manifest.json").write_text(json.dumps(manifest))
    return uri


def test_arrow_legacy_store_with_truncated_fragment_opens_and_resumes(tmp_path, caplog):
    uri = _legacy_arrow_store(tmp_path, rows=4)
    data = (uri / "t" / "part-00003.parquet").read_bytes()
    (uri / "t" / "part-00004.parquet").write_bytes(data[: len(data) // 2])
    with caplog.at_level(logging.WARNING), DataBuffer(None, data_uri=str(uri), backend="arrow", axis="t") as reader:
        assert reader.get_size() == 4
    assert "part-00004.parquet" in caplog.text

    class Six(Source):
        def get_message(self):
            for i in range(6):
                yield message(i)

    with DataBuffer(Six(6), data_uri=str(uri), backend="arrow", axis="t", preload=0) as buf:
        buf.load_data_db("t")
    with DataBuffer(None, data_uri=str(uri), backend="arrow", axis="t") as reader:
        assert reader.get_index_range("t")["ts"].tolist() == [float(i) for i in range(6)]


# -- finding A: only Arrow topic directories are adopted ----------------------------------


def test_arrow_never_adopts_or_rewrites_foreign_manifests(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    web_manifest = '{"name": "My App", "short_name": "app", "icons": []}'
    (public / "manifest.json").write_text(web_manifest)
    with DataBuffer(Source(2), data_uri=str(tmp_path), backend="arrow", axis="t", preload=0) as buf:
        buf.load_data_db("t")
        assert buf.get_topics() == ["t"]
    assert (public / "manifest.json").read_text() == web_manifest
    with DataBuffer(None, data_uri=str(tmp_path), backend="arrow") as reader:
        assert reader.get_topics() == ["t"]


def test_arrow_still_adopts_legacy_topic_directories(tmp_path):
    uri = _legacy_arrow_store(tmp_path, rows=3)
    with DataBuffer(None, data_uri=str(uri), backend="arrow", axis="t") as reader:
        assert reader.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0]


# -- finding 9: direct appends of topics the source does not list ---------------------------


@pytest.mark.parametrize("backend", PERSISTENT)
def test_derived_topic_appends_are_not_capped(tmp_path, backend):
    source = Source(4, topics=("imu",), counts={"imu": 4})
    with DataBuffer(source, data_uri=str(tmp_path / backend), backend=backend, axis="imu", preload=0) as buf:
        for i in range(4):
            buf.roll_buffer("imu")
            buf.append_buffer(message(i, topic="imu_norm", data=np.array([float(i)])))
        assert buf.counters == {"imu": 4, "imu_norm": 4}


# -- finding 10 / improvement: Arrow index ranges and point reads ----------------------------


def test_arrow_index_reads_reuse_decoded_row_groups(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    uri = str(tmp_path / "store")
    with DataBuffer(Source(400), data_uri=uri, backend="arrow", axis="t", preload=0,
                    backend_options={"row_group_bytes": 16 * 100}) as buf:
        buf.load_data_db("t")
    reader = DataBuffer(None, data_uri=uri, backend="arrow", axis="t")
    groups_read = []
    original = pq.ParquetFile.read_row_groups
    monkeypatch.setattr(pq.ParquetFile, "read_row_groups",
                        lambda self, groups, *a, **k: (groups_read.extend(groups), original(self, groups, *a, **k))[1])
    assert [float(reader[i][0]) for i in range(150, 160)] == [float(i) for i in range(150, 160)]
    assert groups_read == [1]
    groups_read.clear()
    sparse = reader.topic("t").index_range(100, 150).collect(chunk_size=4)
    assert sparse["ts"].tolist() == [float(i) for i in range(100, 150)]
    assert len(groups_read) <= 2
    monkeypatch.setattr(reader.buffer_impl, "_take", lambda *a, **k: pytest.fail("dense range read row by row"))
    dense = reader.topic("t").index_range(0, 400, 2).collect(chunk_size=16)
    assert dense["ts"].tolist() == [float(i) for i in range(0, 400, 2)]
    reader.close()


# -- finding 11: NaN timestamps --------------------------------------------------------------


def test_tiledb_nan_timestamps_disable_bisection(tmp_path):
    with DataBuffer(None, data_uri=str(tmp_path / "s"), backend="tiledb") as buf:
        for i, ts in enumerate([0.0, 1.0, float("nan"), 3.0, 4.0]):
            buf.append_buffer({"topic": "t", "timestamp": ts, "name": "t", "data": np.array([float(i)])})
        assert buf.get_time_range("t", 3, 6)["ts"].tolist() == [3.0, 4.0]
    with DataBuffer(None, data_uri=str(tmp_path / "s"), backend="tiledb", axis="t") as reader:
        assert reader.get_time_range("t", 0.5, 3.5)["ts"].tolist() == [1.0, 3.0]
        assert reader.topic("t").time_range(0.5, 3.5).collect()["ts"].tolist() == [1.0, 3.0]


# -- finding 12: unbounded TileDB topics end at the int32 domain -----------------------------


def test_tiledb_unbounded_topic_reports_full_at_domain_end(tmp_path):
    buf = DataBuffer(None, data_uri=str(tmp_path / "s"), backend="tiledb")
    buf.append_buffer(message(0))
    impl = buf.buffer_impl
    impl.counters["t"] = tiledb_buffer.UNBOUNDED_CAPACITY
    with pytest.raises(ValueError, match="already full"):
        buf.append_buffer(message(1))
    impl.counters["t"] = 1
    buf.close()


# -- finding 13, E: published Arrow files ---------------------------------------------------


def test_arrow_files_follow_umask_and_fsync_writable_handles(tmp_path, monkeypatch):
    import fcntl

    modes = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            modes.append(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(arrow_buffer, "_UMASK", 0o022)
    with DataBuffer(None, data_uri=str(tmp_path / "s"), backend="arrow") as buf:
        buf.append_buffer(message(0))
    assert modes and all(mode == os.O_RDWR for mode in modes)
    topic_dir = tmp_path / "s" / "topic-t.data"
    for path in topic_dir.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o644, path


def test_arrow_writable_open_removes_stale_temporaries(tmp_path):
    uri = tmp_path / "s"
    with DataBuffer(None, data_uri=str(uri), backend="arrow") as buf:
        buf.append_buffer(message(0))
    topic_dir = uri / "topic-t.data"
    stale, fresh = topic_dir / ".part-00001.parquet.abc.tmp", topic_dir / ".manifest.json.def.tmp"
    for path in (stale, fresh):
        path.write_bytes(b"partial")
    os.utime(stale, (0, 0))
    with DataBuffer(None, data_uri=str(uri), backend="arrow", axis="t"):
        pass  # read-only opens never delete anything
    assert stale.exists()
    with DataBuffer(Source(1), data_uri=str(uri), backend="arrow", axis="t", preload=0):
        pass
    assert not stale.exists() and fresh.exists()


def test_arrow_new_topic_directory_entry_is_fsynced(tmp_path, monkeypatch):
    synced = []
    real = arrow_buffer._fsync_directory
    monkeypatch.setattr(arrow_buffer, "_fsync_directory", lambda path: (synced.append(Path(path)), real(path)))
    with DataBuffer(None, data_uri=str(tmp_path / "s"), backend="arrow") as buf:
        buf.append_buffer(message(0))
    assert synced[0] == tmp_path / "s"


# -- improvements: flush accounting, context exit, completion, options, casts -------------------


@pytest.mark.parametrize("backend", PERSISTENT)
def test_flush_threshold_counts_per_row_overhead(tmp_path, backend):
    options = {"flush_bytes": 10 * (8 + 256)}
    buf = DataBuffer(None, data_uri=str(tmp_path / backend), backend=backend, backend_options=options)
    for i in range(25):
        buf.append_buffer(message(i, data=np.array([float(i)])))
    assert buf.buffer_impl._persisted["t"] == 20  # two flushes of 10 tiny rows
    buf.close()


@pytest.mark.parametrize("backend", PERSISTENT)
def test_context_exit_on_error_keeps_closed_flags_and_the_error(tmp_path, backend):
    uri = str(tmp_path / "store")
    with DataBuffer(Source(2, topics=("a",)), data_uri=uri, backend=backend, axis="a", preload=0) as buf:
        buf.load_data_db("a")
    with pytest.raises(KeyError, match="user error"):
        with DataBuffer(Source(3, topics=("b",)), data_uri=uri, backend=backend, axis="b", preload=0) as buf:
            buf.roll_buffer("b")
            raise KeyError("user error")
    with DataBuffer(None, data_uri=uri, backend=backend) as reader:
        assert reader.buffer_impl.closed_topics == {"a": True, "b": False}
        assert reader.counters == {"a": 2, "b": 1}

    buf = DataBuffer(None, data_uri=uri, backend=backend, axis="b")
    buf.append_buffer(message(5, topic="b", data=np.full(2, 5.0)))
    monkeypatch_close = buf.buffer_impl.close

    def failing_close(*args, **kwargs):
        raise RuntimeError("close failed")

    buf.buffer_impl.close = failing_close
    with pytest.raises(KeyError, match="user error") as raised:
        with buf:
            raise KeyError("user error")
    assert any("close failed" in note for note in raised.value.__notes__)
    buf.buffer_impl.close = monkeypatch_close
    buf.close()


@pytest.mark.parametrize("backend", PERSISTENT)
def test_exhausted_source_closes_only_topics_it_delivered(tmp_path, backend):
    uri = str(tmp_path / "store")
    with DataBuffer(None, data_uri=uri, backend=backend) as buf:
        buf.append_buffer(message(0, topic="derived"))
    factory = lambda: Source(2).get_message()  # noqa: E731
    with DataBuffer(factory, topics=["t"], data_uri=uri, backend=backend, axis="t", preload=0) as buf:
        buf.load_data_db("t")
        buf.reset(preload=0)
        assert buf._source_exhausted is False
        with pytest.raises(StopIteration):
            for _ in range(3):
                buf.roll_buffer("t")
    with DataBuffer(None, data_uri=uri, backend=backend) as reader:
        assert reader.buffer_impl.closed_topics == {"derived": False, "t": True}


def test_tiledb_backend_options(tmp_path):
    buf = DataBuffer(None, data_uri=str(tmp_path / "s"), backend="tiledb",
                     backend_options={"flush_bytes": 123, "consolidate_max_bytes": 456})
    assert (buf.buffer_impl.flush_bytes, buf.buffer_impl.consolidate_max_bytes) == (123, 456)
    with pytest.raises(ValueError, match="not supported by the 'memory' backend"):
        DataBuffer(Source(1), axis="t", backend_options={"flush_bytes": 1})


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_safe_widening_casts_are_applied(tmp_path, backend):
    if backend == "memory":
        buf = DataBuffer(Source(1), axis="t", buffer_depth=4, preload=0)
    else:
        buf = DataBuffer(None, data_uri=str(tmp_path / backend), backend=backend)
    buf.append_buffer(message(0, data=np.zeros(2)))
    buf.append_buffer(message(1, data=np.array([2, 3], dtype=np.float32)))
    buf.append_buffer(message(2, data=np.array([4, 5], dtype=np.int16)))
    rows = buf.get_index_range("t")["data"]
    assert rows.dtype == np.float64 and rows.tolist() == [[0, 0], [2, 3], [4, 5]]
    with pytest.raises(ValueError, match="must keep dtype"):
        buf.append_buffer(message(3, data=np.array([1j, 2j])))
    buf.close()


# -- findings B, C, D: topics=, get_buffer(), get_size() ----------------------------------------


def test_memory_buffer_honors_topics_and_grows_lazily():
    source = Source(3, topics=("pose", "camera"), shape=(4,))
    buf = DataBuffer(source, buffer_depth=1000, axis="pose", topics=["pose"], preload=3)
    assert buf.get_topics() == ["pose"]
    assert list(buf.get_buffer()) == ["pose"]
    assert buf.buffer_impl._data_buffer["pose"].shape[0] < 1000  # storage grows with messages
    assert "camera" not in buf.buffer_impl._data_buffer


@pytest.mark.parametrize("depth, rolls", [(5, 3), (3, 7), (20, 17)])
def test_memory_get_buffer_returns_filled_rows_in_order(depth, rolls):
    buf = DataBuffer(Source(rolls), buffer_depth=depth, axis="t", preload=0)
    for _ in range(rolls):
        buf.roll_buffer("t")
    expected = [float(i) for i in range(max(0, rolls - depth), rolls)]
    window = buf.get_buffer()["t"]
    assert window["ts"].tolist() == expected
    assert buf.get_size() == len(expected)
    assert float(buf[buf.get_size() - 1][0]) == expected[-1] and float(buf[0][0]) == expected[0]
    buf[0] = np.array([-1.0, -1.0])
    assert buf.get_index_range("t")["data"][0].tolist() == [-1.0, -1.0]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_get_size_counts_rows_available_on_every_backend(tmp_path, backend):
    kwargs = {"buffer_depth": 10} if backend == "memory" else {"data_uri": str(tmp_path / backend)}
    buf = DataBuffer(Source(7), backend=backend, axis="t", preload=0, **kwargs)
    for _ in range(4):
        buf.roll_buffer("t")
    assert buf.get_size() == 4
    assert buf[buf.get_size() - 1].tolist() == [3.0, 3.0]
    buf.close(closed=False)
