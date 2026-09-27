"""Regressions for storage-backend fixes (resume, concurrency, schemas, reads)."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from arraydataengine import DataBuffer
from arraydataengine.buffers.common import spatial_bounds_for_data

pytest.importorskip("pyarrow")
tiledb = pytest.importorskip("tiledb")

PERSISTENT = ["arrow", "tiledb"]
ALL_BACKENDS = ["memory", "arrow", "tiledb"]


class Source:
    def __init__(self, n=5, topic="t", frames=None, shape=(2,), dtype=np.float64, ts=None, counted=True):
        self.n, self.topic, self.frames, self.shape, self.dtype, self.ts = n, topic, frames, shape, dtype, ts
        if not counted:
            self.get_count = None

    def get_topics(self):
        return [self.topic]

    def get_count(self, topic):
        return self.n

    def get_message(self):
        for i in range(self.n):
            yield {
                "topic": self.topic,
                "timestamp": float(i if self.ts is None else self.ts[i]),
                "name": f"m{i}",
                "frame_id": None if self.frames is None else self.frames[i],
                "data": np.full(self.shape, i, dtype=self.dtype),
            }


def open_buffer(backend, tmp_path, source, **kwargs):
    if backend == "memory":
        kwargs.setdefault("buffer_depth", source.n)
        kwargs.setdefault("preload", source.n)
        return DataBuffer(source, axis=source.topic, **kwargs)
    kwargs.setdefault("preload", 0)
    buf = DataBuffer(source, data_uri=str(tmp_path / backend), backend=backend, axis=source.topic, **kwargs)
    buf.load_data_db(source.topic)
    return buf


def message(i, topic="t", data=None, **extra):
    return {"topic": topic, "timestamp": float(i), "name": f"m{i}",
            "data": np.array([float(i), float(i)]) if data is None else data, **extra}


# 1: leaving a `with` block only marks complete topics closed ------------------

@pytest.mark.parametrize("backend", PERSISTENT)
def test_context_exit_keeps_partial_topics_resumable(tmp_path, backend):
    uri = str(tmp_path / "store")
    with DataBuffer(Source(5), data_uri=uri, backend=backend, axis="t", preload=0) as buf:
        buf.roll_buffer("t")
        buf.roll_buffer("t")
    with DataBuffer(None, data_uri=uri, backend=backend, axis="t") as reader:
        assert reader.buffer_impl.closed_topics["t"] is False
    with DataBuffer(Source(5), data_uri=uri, backend=backend, axis="t", preload=0) as buf:
        buf.load_data_db("t")
        assert buf.get_index_range("t")["data"][:, 0].tolist() == [0, 1, 2, 3, 4]
    assert buf.buffer_impl.closed_topics["t"] is True


def test_context_exit_closes_topics_once_the_source_is_exhausted(tmp_path):
    uri = str(tmp_path / "store")
    with DataBuffer(lambda: Source(2).get_message(), topics=["t"], data_uri=uri, backend="arrow",
                    axis="t", preload=0) as buf:
        buf.roll_buffer("t")
        buf.roll_buffer("t")
        with pytest.raises(StopIteration):
            buf.roll_buffer("t")
    assert buf.buffer_impl.closed_topics["t"] is True


def test_arrow_resumes_partial_topic_that_older_versions_marked_closed(tmp_path):
    uri = str(tmp_path / "store")
    with DataBuffer(Source(5), data_uri=uri, backend="arrow", axis="t", preload=0) as buf:
        buf.roll_buffer("t")
        buf.roll_buffer("t")
    manifest_path = buf.buffer_impl._manifest_path("t")
    manifest = json.loads(manifest_path.read_text())
    manifest["closed"] = True  # what the old __exit__ wrote
    manifest_path.write_text(json.dumps(manifest))
    with DataBuffer(Source(5), data_uri=uri, backend="arrow", axis="t", preload=0) as buf:
        buf.load_data_db("t")
        assert buf.get_size() == 5


# 2: stale Arrow handles never roll back another writer ------------------------

def test_arrow_stale_handle_cannot_roll_back_committed_fragments(tmp_path):
    uri = str(tmp_path / "store")
    stale = DataBuffer(Source(3), data_uri=uri, backend="arrow", axis="t", preload=0)
    stale.load_data_db("t")
    other = DataBuffer(None, data_uri=uri, backend="arrow", axis="t")
    other.append_buffer(message(9))
    other.close()

    stale.get_index_range("t")
    stale.get_time_range("t", 0, 10)
    stale.close()
    with DataBuffer(None, data_uri=uri, backend="arrow", axis="t") as reader:
        assert reader.get_size() == 4

    stale.append_buffer(message(10))
    with pytest.raises(RuntimeError, match="another writer"):
        stale.close()
    with DataBuffer(None, data_uri=uri, backend="arrow", axis="t") as reader:
        assert reader.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0, 9.0]
    stale.buffer_impl.read_only = True


# 3: Arrow payload validation and all-topic close ------------------------------

@pytest.mark.parametrize("payload", [
    np.float64(1.5),
    np.array([True, False]),
    np.array([b"ab", b"cd"]),
    np.array(["2020-01-01"], dtype="datetime64[s]"),
    np.array([1.0, 2.0], dtype=">f4"),
    np.array([1 + 2j]),
], ids=["scalar", "bool", "bytes", "datetime64", "big-endian", "complex"])
def test_arrow_round_trips_non_tensor_payloads(tmp_path, payload):
    with DataBuffer(None, data_uri=tmp_path, backend="arrow") as buf:
        buf.append_buffer(message(0, data=payload))
        buf.append_buffer(message(1, data=payload))
        staged = buf.get_index_range("t")["data"]
    with DataBuffer(None, data_uri=tmp_path, backend="arrow", axis="t") as reader:
        rows = reader.get_index_range("t")["data"]
        chunked = reader.topic("t").collect(chunk_size=1)["data"]
    expected = np.stack([np.asarray(payload).astype(np.asarray(payload).dtype.newbyteorder("="))] * 2)
    for out in (staged, rows, chunked):
        assert out.dtype == expected.dtype and out.shape == expected.shape
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("payload", [np.zeros((0, 3)), np.array([object()], dtype=object)])
def test_arrow_rejects_unstorable_payloads_on_append(tmp_path, payload):
    with DataBuffer(None, data_uri=tmp_path, backend="arrow") as buf:
        with pytest.raises(ValueError):
            buf.append_buffer(message(0, data=payload))
        buf.append_buffer(message(0, topic="ok"))
    with DataBuffer(None, data_uri=tmp_path, backend="arrow") as reader:
        assert reader.get_topics() == ["ok"]


def test_arrow_close_attempts_every_topic(tmp_path, monkeypatch):
    buf = DataBuffer(None, data_uri=tmp_path, backend="arrow")
    for topic in ("a", "b", "c"):
        buf.append_buffer(message(0, topic=topic))
    impl = buf.buffer_impl
    original = impl._flush_topic

    def failing(topic):
        if topic != "b":
            raise OSError(f"disk full for {topic}")
        original(topic)

    monkeypatch.setattr(impl, "_flush_topic", failing)
    with pytest.raises(ExceptionGroup) as raised:
        buf.close()
    assert len(raised.value.exceptions) == 2
    monkeypatch.undo()
    impl.read_only = True
    with DataBuffer(None, data_uri=tmp_path, backend="arrow") as reader:
        assert reader.get_topics() == ["b"]


# 4, 13: TileDB empty results and subscripts ------------------------------------

def test_tiledb_queries_matching_no_rows_return_empty_results(tmp_path):
    buf = open_buffer("tiledb", tmp_path, Source(3))
    for result in (buf.get_index_range("t", 2, 2), buf.get_time_range("t", 100.0, 200.0),
                   buf.get_time_range("t", 0.2, 0.8)):
        assert result["data"].shape == (0, 2) and result["data"].dtype == np.float64
        assert result["ts"].size == 0 and result["frame_ids"].size == 0
    buf.close()


@pytest.mark.parametrize("backend", PERSISTENT)
def test_persistent_subscripts_use_message_count(tmp_path, backend):
    buf = DataBuffer(Source(5), data_uri=str(tmp_path / backend), backend=backend, axis="t", preload=0)
    for _ in range(3):
        buf.roll_buffer("t")
    np.testing.assert_array_equal(buf[np.int64(1)], [1, 1])
    np.testing.assert_array_equal(buf[:], [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(buf[-2:], [[1, 1], [2, 2]])
    np.testing.assert_array_equal(buf[::-2], [[2, 2], [0, 0]])
    assert buf[3:3].shape == (0, 2)
    with pytest.raises(IndexError):
        buf[3]
    with pytest.raises(TypeError):
        buf[1.5]
    buf.close(closed=False)


def test_tiledb_setitem_uses_message_count_and_refreshes_spatial_index(tmp_path):
    buf = DataBuffer(Source(5, shape=(1, 3)), data_uri=str(tmp_path / "s"), backend="tiledb", axis="t", preload=0)
    for _ in range(3):
        buf.roll_buffer("t")
    buf[-1] = np.array([[50.0, 50.0, 50.0]])
    buf[:2] = 7.0
    np.testing.assert_array_equal(buf[:], [[7, 7, 7], [7, 7, 7], [50, 50, 50]])
    near = buf.topic("t").spatial_bounds([49, 49, 49], [51, 51, 51]).collect()["ts"].tolist()
    assert near == [2.0]
    buf.close(closed=False)


# 5: unsorted timestamps ---------------------------------------------------------

@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_time_queries_filter_out_of_order_timestamps(tmp_path, backend):
    source = Source(6, shape=(1,), ts=[0.0, 1.0, 5.0, 2.0, 3.0, 4.0])
    buf = open_buffer(backend, tmp_path, source)
    assert buf.get_time_range("t", 1.5, 3.5)["ts"].tolist() == [2.0, 3.0]
    assert buf.topic("t").time_range(1.5, 3.5).collect()["ts"].tolist() == [2.0, 3.0]
    assert buf.get_last_seconds("t", 1.0)["ts"].tolist() == [3.0, 4.0]
    buf.close()
    if backend == "tiledb":
        with DataBuffer(None, data_uri=str(tmp_path / backend), backend=backend, axis="t") as reader:
            assert reader.get_time_range("t", 1.5, 3.5)["ts"].tolist() == [2.0, 3.0]


def test_tiledb_legacy_store_without_sortedness_metadata_is_checked(tmp_path):
    buf = open_buffer("tiledb", tmp_path, Source(6, shape=(1,), ts=[0.0, 1.0, 5.0, 2.0, 3.0, 4.0]))
    buf.close()
    with tiledb.open(buf.buffer_impl._get_timestamp_array_uri("t"), "w") as array:
        del array.meta["ts_sorted"]
    with DataBuffer(None, data_uri=str(tmp_path / "tiledb"), backend="tiledb", axis="t") as reader:
        assert reader.get_time_range("t", 1.5, 3.5)["ts"].tolist() == [2.0, 3.0]


# 6: TileDB per-row frame ids -----------------------------------------------------

@pytest.mark.parametrize("frames", [["map", None, "map"], ["map", "odom", "map"], [None, "map", "map"]])
def test_tiledb_preserves_row_frames_like_other_backends(tmp_path, frames):
    expected = [i for i, frame in enumerate(frames) if frame == "map"]
    for backend in ALL_BACKENDS:
        buf = open_buffer(backend, tmp_path, Source(3, frames=frames, shape=(1,)))
        rows = buf.get_index_range("t")
        assert rows["frame_ids"].tolist() == frames, backend
        assert rows.get("frame_id") is None, backend
        assert buf.topic("t").frame_id("map").collect()["ts"].tolist() == expected, backend
        mapped = buf.topic("t").map(lambda data: data + 1).frame_id("map").collect()
        assert mapped["ts"].tolist() == expected, backend
        buf.close()


def test_tiledb_uniform_frame_survives_reopen(tmp_path):
    open_buffer("tiledb", tmp_path, Source(3, frames=["map"] * 3)).close()
    with DataBuffer(None, data_uri=str(tmp_path / "tiledb"), backend="tiledb", axis="t") as reader:
        assert reader.get_index_range("t")["frame_id"] == "map"
        assert reader.topic("t").metadata.frame_id == "map"


# 7: spatial bounds for organized clouds -----------------------------------------

def test_spatial_bounds_cover_organized_clouds():
    cloud = np.zeros((2, 2, 3))
    cloud[1, 1] = [4.0, 5.0, 6.0]
    valid, mins, maxs = spatial_bounds_for_data(cloud)
    assert valid
    np.testing.assert_array_equal(mins, [0, 0, 0])
    np.testing.assert_array_equal(maxs, [4, 5, 6])
    valid, mins, _ = spatial_bounds_for_data(np.arange(24, dtype=np.uint8).reshape(2, 4, 3))
    assert valid and mins.tolist() == [0, 1, 2]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_spatial_pushdown_on_organized_clouds_matches_memory(tmp_path, backend):
    buf = open_buffer(backend, tmp_path, Source(2, shape=(2, 2, 3)))
    got = buf.topic("t").spatial_bounds([-0.5] * 3, [0.5] * 3).collect()["ts"].tolist()
    assert got == [0.0]
    buf.close()


@pytest.mark.parametrize("backend", PERSISTENT)
def test_spatial_pushdown_keeps_rows_without_valid_bounds(tmp_path, backend, monkeypatch):
    # Older versions stored invalid bounds for organized clouds; such rows
    # must reach the exact filter instead of being pruned.
    from arraydataengine.buffers import arrow_buffer, tiledb_buffer

    invalid = lambda data: (False, np.full(3, np.nan), np.full(3, np.nan))  # noqa: E731
    monkeypatch.setattr(arrow_buffer, "spatial_bounds_for_data", invalid)
    monkeypatch.setattr(tiledb_buffer, "_spatial_bounds_for_data", invalid)
    open_buffer(backend, tmp_path, Source(2, shape=(2, 2, 3))).close()
    monkeypatch.undo()
    with DataBuffer(None, data_uri=str(tmp_path / backend), backend=backend, axis="t") as reader:
        got = reader.topic("t").spatial_bounds([-0.5] * 3, [0.5] * 3).collect()["ts"].tolist()
    assert got == [0.0]


# 8, 12: memory/TileDB validation ---------------------------------------------------

@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("bad", [np.array([[7.0, 8.0, 9.0]]), np.zeros((4, 3), dtype=np.float32)],
                         ids=["smaller-cloud", "float32"])
def test_backends_reject_shape_and_dtype_changes(tmp_path, backend, bad):
    buf = open_buffer(backend, tmp_path, Source(2, shape=(4, 3)), **({"buffer_depth": 3} if backend == "memory" else {}))
    with pytest.raises(ValueError, match="must keep"):
        buf.append_buffer(message(9, data=bad))
    assert buf.get_index_range("t")["ts"].tolist() == [0.0, 1.0]
    buf.close()


def test_tiledb_rejects_narrowing_casts(tmp_path):
    with DataBuffer(None, data_uri=str(tmp_path / "s"), backend="tiledb") as buf:
        buf.append_buffer(message(0, data=np.array([1, 2], dtype=np.int16)))
        with pytest.raises(ValueError, match="must keep dtype"):
            buf.append_buffer(message(1, data=np.array([1.7, 70000.0])))
        assert buf.buffer_impl.counters["t"] == 1


def test_memory_non_ascii_names_and_rejected_rows_leave_no_partial_state():
    buf = DataBuffer(Source(1, topic="img"), axis="img", buffer_depth=2, preload=0)
    buf.append_buffer({"topic": "img", "timestamp": 0.0, "name": "câmera_001.png", "data": np.zeros(2)})
    assert buf.get_index_range("img")["name"].tolist() == ["câmera_001.png".encode()]
    with pytest.raises(ValueError):
        buf.append_buffer({"topic": "img", "timestamp": 1.0, "name": "x", "data": np.zeros(3)})
    rows = buf.get_index_range("img")
    assert rows["ts"].tolist() == [0.0] and buf.get_size() == 1


# 9: Arrow single-row reads read only the needed row groups -----------------------

def test_arrow_point_reads_touch_only_needed_row_groups(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    with DataBuffer(Source(60, shape=(8,)), data_uri=str(tmp_path / "s"), backend="arrow", axis="t", preload=0,
                    backend_options={"flush_bytes": 64 * 20, "row_group_bytes": 64 * 4}) as buf:
        buf.load_data_db("t")
    reader = DataBuffer(None, data_uri=str(tmp_path / "s"), backend="arrow", axis="t")
    assert len(reader.buffer_impl._fragments["t"]) == 3
    rows_read = []
    original = pq.ParquetFile.read_row_groups

    def recording(self, row_groups, *args, **kwargs):
        table = original(self, row_groups, *args, **kwargs)
        rows_read.append(table.num_rows)
        return table

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", recording)
    np.testing.assert_array_equal(reader[-1], np.full(8, 59))
    np.testing.assert_array_equal(reader.get_index_range("t", 33, 34)["data"], [np.full(8, 33)])
    assert reader.get_last_seconds("t", 0.0)["ts"].tolist() == [59.0]
    np.testing.assert_array_equal(reader.get_index_range("t", 5, 60, 25)["data"][:, 0], [5, 30, 55])
    assert max(rows_read) <= 4 and sum(rows_read) <= 4 * 6
    reader.close()


def test_arrow_leading_index_range_does_not_scan(tmp_path, monkeypatch):
    buf = open_buffer("arrow", tmp_path, Source(50), backend_options={"flush_bytes": 16 * 10})
    impl = buf.buffer_impl
    monkeypatch.setattr(impl, "_iter_batches", lambda *a, **k: pytest.fail("scanned the topic"))
    assert buf.topic("t").index_range(3, 6).collect()["ts"].tolist() == [3.0, 4.0, 5.0]
    monkeypatch.undo()

    scanned = []
    original = impl._iter_batches

    def counting(*args, **kwargs):
        for batch in original(*args, **kwargs):
            scanned.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(impl, "_iter_batches", counting)
    got = buf.topic("t").time_range(0.0, 100.0).index_range(0, 2).collect(chunk_size=5)
    assert got["ts"].tolist() == [0.0, 1.0] and sum(scanned) <= 10
    buf.close()


# 10: reads on writer handles neither flush nor rewrite metadata ------------------

def test_arrow_reads_on_writer_handle_do_not_flush(tmp_path, monkeypatch):
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append(str(dst)), real_replace(src, dst)))
    buf = DataBuffer(Source(40), data_uri=str(tmp_path / "s"), backend="arrow", axis="t", preload=0)
    seen = [chunk["t"]["ts"].size for chunk, _ in buf.get_data("t")]
    assert seen == list(range(1, 41))
    buf.close()
    fragments = list(buf.buffer_impl._topic_dir("t").glob("part-*.parquet"))
    assert len(fragments) == 1
    assert sum(path.endswith("manifest.json") for path in replaced) <= 3


def test_tiledb_reads_on_writer_handle_do_not_write(tmp_path):
    open_buffer("tiledb", tmp_path, Source(3)).close()
    buf = DataBuffer(Source(3), data_uri=str(tmp_path / "tiledb"), backend="tiledb", axis="t", preload=0)
    path = Path(buf.buffer_impl._get_array_uri("t"))
    before = {p: sorted(os.listdir(p / "__meta")) for p in (path, Path(str(path) + "__timestamps"))}
    for _ in range(20):
        buf.get_index_range("t", 0, 1)
        buf.get_time_range("t", 0, 1)
        buf[0]
    buf.close()
    assert {p: sorted(os.listdir(p / "__meta")) for p in before} == before


@pytest.mark.parametrize("backend", PERSISTENT)
def test_staged_rows_are_readable_before_flush(tmp_path, backend):
    buf = DataBuffer(Source(4), data_uri=str(tmp_path / "s"), backend=backend, axis="t", preload=0)
    for _ in range(3):
        buf.roll_buffer("t")
    assert buf.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0]
    assert buf.get_time_range("t", 1.0, 2.0)["ts"].tolist() == [1.0, 2.0]
    assert buf.get_last_seconds("t", 0.5)["ts"].tolist() == [2.0]
    assert buf.topic("t").frame_id("map").collect()["ts"].size == 0
    assert buf.topic("t").collect(chunk_size=2)["ts"].tolist() == [0.0, 1.0, 2.0]
    buf.close(closed=False)


# 11: memory topics listed but not yet received -------------------------------------

def test_memory_unreceived_topic_returns_empty_results():
    class Two:
        def get_topics(self):
            return ["a", "b"]

        def get_message(self):
            for i in range(3):
                yield {"topic": "a", "timestamp": float(i), "name": "a", "data": np.array([i])}

    buf = DataBuffer(Two(), axis="a", buffer_depth=3, preload=1)
    assert buf.get_index_range("b")["ts"].size == 0
    assert buf.get_time_range("b", 0, 10)["ts"].size == 0
    assert buf.get_last_seconds("b", 1.0)["frame_ids"].size == 0
    assert len(buf.topic_view("b")) == 0


# 14, 15: TileDB store creation paths ---------------------------------------------------

def test_tiledb_opens_existing_empty_directory(tmp_path):
    uri = tmp_path / "made-by-mkdtemp"
    uri.mkdir()
    with DataBuffer(Source(3), data_uri=str(uri), backend="tiledb", axis="t", preload=0) as buf:
        buf.load_data_db("t")
        assert buf.get_size() == 3


@pytest.mark.parametrize("backend", PERSISTENT)
def test_direct_appends_and_callable_sources_on_fresh_stores(tmp_path, backend):
    with DataBuffer(None, data_uri=str(tmp_path / "direct"), backend=backend) as buf:
        for i in range(3):
            buf.append_buffer(message(i))
    with DataBuffer(None, data_uri=str(tmp_path / "direct"), backend=backend, axis="t") as reader:
        assert reader.get_size() == 3
        assert reader.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0]

    factory = lambda: Source(3).get_message()  # noqa: E731
    with DataBuffer(factory, topics=["t"], data_uri=str(tmp_path / "call"), backend=backend,
                    preload=0) as buf:
        buf.load_data_db("t")
        assert buf.get_size() == 3


# 16: directory fsync after atomic publish ---------------------------------------

def test_arrow_atomic_output_fsyncs_parent_directory(tmp_path, monkeypatch):
    import stat
    from arraydataengine.buffers import arrow_buffer

    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(stat.S_ISDIR(os.fstat(fd).st_mode)), real_fsync(fd)))
    with arrow_buffer._atomic_output(tmp_path / "file.json") as temporary:
        temporary.write_text("{}")
    assert synced == [False, True]
    assert (tmp_path / "file.json").read_text() == "{}"


# 17: TileDB hydration is strict and non-destructive --------------------------------

def test_tiledb_unreadable_array_raises_instead_of_being_replaced(tmp_path):
    open_buffer("tiledb", tmp_path, Source(3)).close()
    store = tmp_path / "tiledb"
    array_dir = next(p for p in store.iterdir() if p.name.endswith(".data"))
    for schema_file in (array_dir / "__schema").iterdir():
        if schema_file.is_file():
            schema_file.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Unreadable TileDB topic array"):
        DataBuffer(Source(3), data_uri=str(store), backend="tiledb", axis="t", preload=0)
    assert sorted(p.name for p in store.iterdir() if p.name.startswith("topic-")) == [
        array_dir.name, array_dir.name + "__timestamps"]


def test_tiledb_never_deletes_written_array_missing_its_index(tmp_path):
    buf = open_buffer("tiledb", tmp_path, Source(3))
    buf.close()
    impl = buf.buffer_impl
    shutil.rmtree(impl._get_timestamp_array_uri("t"))
    with pytest.raises(ValueError, match="refusing to replace"):
        impl._init_tdb(message(0))
    assert len(tiledb.array_fragments(impl._get_array_uri("t"))) == 1
    impl.read_only = True


def test_tiledb_duplicate_topic_arrays_keep_the_fuller_one(tmp_path):
    uri = str(tmp_path / "store")
    first = DataBuffer(Source(5), data_uri=uri, backend="tiledb", axis="t", preload=0)
    first.roll_buffer("t")
    first.close(closed=False)
    first.buffer_impl.read_only = True
    array = first.buffer_impl._get_array_uri("t")
    for suffix in ("", "__timestamps"):
        shutil.copytree(array + suffix, str(tmp_path / ("stale" + suffix)))
    with DataBuffer(Source(5), data_uri=uri, backend="tiledb", axis="t", preload=0) as resumed:
        resumed.load_data_db("t")
    # Older versions could leave a stale one-row array under the original
    # name, sorting before the complete array a failed hydration re-created.
    for suffix in ("", "__timestamps"):
        os.rename(array + suffix, array + ".1" + suffix)
        os.rename(str(tmp_path / ("stale" + suffix)), array + suffix)
    with DataBuffer(None, data_uri=uri, backend="tiledb", axis="t") as reader:
        assert reader.get_size() == 5
        assert reader.buffer_impl._get_array_uri("t") == array + ".1"
        assert reader.get_index_range("t")["ts"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


# Coordinator finding: TileDB ingest bloat ------------------------------------------

def test_tiledb_ingest_of_many_small_messages_stays_compact(tmp_path):
    uri = tmp_path / "store"
    with DataBuffer(Source(200, shape=(7,)), data_uri=str(uri), backend="tiledb", axis="t", preload=0) as buf:
        buf.load_data_db("t")
        data_uri = Path(buf.buffer_impl._get_array_uri("t"))
    for path in (data_uri, Path(str(data_uri) + "__timestamps")):
        assert len(tiledb.array_fragments(str(path))) == 1
        assert len(os.listdir(path / "__meta")) == 1
    size = sum(f.stat().st_size for f in uri.rglob("*") if f.is_file())
    assert size < 512 * 1024, size
    with DataBuffer(None, data_uri=str(uri), backend="tiledb", axis="t") as reader:
        rows = reader.get_index_range("t", 0, 200, 50)
        assert rows["ts"].tolist() == [0.0, 50.0, 100.0, 150.0]
        assert rows["name"].tolist() == [b"m0", b"m50", b"m100", b"m150"]
        np.testing.assert_array_equal(rows["data"][:, 0], [0, 50, 100, 150])


def test_tiledb_fragment_consolidation_uses_bounded_buffers(tmp_path, monkeypatch):
    # TileDB's 50 MB-per-attribute default cost ~1.5 GB RSS on the index array.
    import arraydataengine.buffers.tiledb_buffer as tiledb_buffer

    seen = []
    consolidate = tiledb.consolidate

    def recording_consolidate(uri, config=None, **kwargs):
        if config is not None and config["sm.consolidation.mode"] == "fragments":
            seen.append((len(tiledb.array_fragments(uri)), int(config["sm.consolidation.buffer_size"])))
        return consolidate(uri, config=config, **kwargs)

    monkeypatch.setattr(tiledb_buffer.tiledb, "consolidate", recording_consolidate)
    frames = Source(12, shape=(64, 64, 3), dtype=np.uint8)
    uri = str(tmp_path / "store")
    buf = DataBuffer(frames, data_uri=uri, backend="tiledb", axis="t", preload=0)
    buf.buffer_impl.flush_bytes = 64 * 64 * 3 * 4  # four rows per fragment
    buf.load_data_db("t")
    buf.close()
    assert seen and all(fragments > 1 for fragments, _ in seen)
    assert all(size <= 8 * 1024 * 1024 for _, size in seen)
    with DataBuffer(None, data_uri=uri, backend="tiledb", axis="t") as reader:
        np.testing.assert_array_equal(reader.get_index_range("t")["data"][:, 0, 0, 0], np.arange(12))


def test_tiledb_flushes_in_batches_and_resumes_after_last_flush(tmp_path):
    uri = str(tmp_path / "store")
    buf = DataBuffer(Source(10), data_uri=uri, backend="tiledb", axis="t", preload=0)
    buf.buffer_impl.flush_bytes = 16 * 4
    for _ in range(9):
        buf.roll_buffer("t")
    fragments = len(tiledb.array_fragments(buf.buffer_impl._get_array_uri("t")))
    assert fragments == 2  # rows 0-3 and 4-7; row 8 is still staged
    buf.buffer_impl.read_only = True  # simulate a crash: staged row 8 is lost
    with DataBuffer(Source(10), data_uri=uri, backend="tiledb", axis="t", preload=0) as resumed:
        assert resumed.buffer_impl.counters["t"] == 8
        resumed.load_data_db("t")
        assert resumed.get_index_range("t")["ts"].tolist() == [float(i) for i in range(10)]


@pytest.mark.parametrize("backend", PERSISTENT)
def test_reads_merge_persisted_and_staged_rows(tmp_path, backend):
    source = Source(25, frames=["map", "odom"] * 13, shape=(3,))
    buf = DataBuffer(source, data_uri=str(tmp_path / backend), backend=backend, axis="t", preload=0)
    buf.buffer_impl.flush_bytes = 24 * 10  # rows 0-19 flushed, 20-22 staged
    for _ in range(23):
        buf.roll_buffer("t")
    assert buf.buffer_impl._persisted["t"] == 20
    assert buf.get_time_range("t", 17.5, 21)["ts"].tolist() == [18.0, 19.0, 20.0, 21.0]
    assert buf.get_index_range("t", 5, 23, 4)["ts"].tolist() == [5.0, 9.0, 13.0, 17.0, 21.0]
    assert buf[::-7][:, 0].tolist() == [22.0, 15.0, 8.0, 1.0]
    lazy = buf.topic("t").time_range(8, 22).frame_id("map").index_range(1, 5).collect(chunk_size=3)
    assert lazy["ts"].tolist() == [10.0, 12.0, 14.0, 16.0]
    assert buf.topic("t").spatial_bounds([19.5] * 3, [21.5] * 3).collect()["ts"].tolist() == [20.0, 21.0]
    assert buf.get_last_seconds("t", 2)["ts"].tolist() == [20.0, 21.0, 22.0]
    buf.close(closed=False)
