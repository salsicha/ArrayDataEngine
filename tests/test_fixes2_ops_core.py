"""Second-round regressions for ops core: callable dispatch, ragged topics,
row stacking, checkpoints/progress, NPZ IO, alignment, and trajectory IO."""

import functools
import json
import operator

import numpy as np
import pytest

from arraydataengine import DataBuffer
from arraydataengine.ops import (
    CancellationToken,
    PipelineCancelled,
    dataset_query,
    describe_topic,
    load_topic_npz,
    nearest_time_index,
    read_tum_trajectory,
    resample_topic,
    save_topic_npz,
    source_pipeline,
    topic_lengths,
    topic_pipeline,
    topic_rows,
    topic_view,
    valid_point_cloud_points,
    voxel_downsample,
    write_tum_trajectory,
)


class Messages:
    def __init__(self, messages, max_points=None):
        self.messages = messages
        self.max_points = max_points

    def get_topics(self):
        return list(dict.fromkeys(message["topic"] for message in self.messages))

    def get_count(self, topic):
        return sum(message["topic"] == topic for message in self.messages)

    def get_message(self):
        yield from self.messages


def _messages(count, topics=("/s",)):
    return [
        {"topic": topics[i % len(topics)], "name": str(i), "timestamp": float(i), "data": np.array([float(i)])}
        for i in range(count)
    ]


def _range_topic(count):
    return {"ts": np.arange(count, dtype=np.float64), "data": np.arange(count, dtype=np.float64)}


def _object_rows(rows):
    array = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        array[index] = row
    return array


def _ragged_topic():
    lengths = np.array([2, 5, 0, 3])
    data = np.zeros((4, 5, 3))
    for index, length in enumerate(lengths):
        data[index, :length] = 100.0 * (index + 1) + np.arange(length * 3).reshape(length, 3)
    return {"ts": np.arange(4.0), "data": data, "lengths": lengths, "id": np.array(["a", "b", "c", "d"])}


@pytest.fixture(params=["arrow", "tiledb"])
def backend(request):
    pytest.importorskip("pyarrow" if request.param == "arrow" else "tiledb")
    return request.param


def _stored(buffer, topic):
    return buffer.get_index_range(topic)["data"].ravel().astype(int).tolist()


# --- 1: callable dispatch ---------------------------------------------------------


@pytest.mark.parametrize(
    "fn, expected",
    [
        (functools.partial(np.multiply, 2), [[0.0, 2.0], [4.0, 6.0]]),
        (functools.partial(np.matmul, np.eye(2) * 3), [[0.0, 3.0], [6.0, 9.0]]),
        (operator.itemgetter(0), [0.0, 2.0]),
        (operator.methodcaller("sum"), [1.0, 5.0]),
        (np.vectorize(lambda value: value + 1), [[1.0, 2.0], [3.0, 4.0]]),
        (lambda *args: len(args), [1, 1]),
    ],
)
def test_star_args_callables_get_data_only(fn, expected):
    topic = {"ts": np.array([10.0, 20.0]), "data": np.arange(4.0).reshape(2, 2)}
    assert np.asarray(topic_view(topic).map(fn).data).tolist() == expected
    assert np.asarray(topic_pipeline(topic).map(fn).collect()["data"]).tolist() == expected
    assert np.asarray(topic_pipeline(topic).map(fn).collect(max_workers=2, chunk_size=1)["data"]).tolist() == expected


def test_star_args_reducers_and_source_item_getters():
    topic = {"ts": np.array([10.0, 20.0]), "data": np.array([1.0, 2.0])}
    assert topic_view(topic).reduce(functools.partial(np.add)) == 3.0
    assert topic_pipeline(topic).reduce(np.vectorize(lambda a, b: a + b)) == 3.0

    pipeline = source_pipeline(Messages(_messages(3)))
    assert [m["data"].tolist() for m in pipeline.map(operator.itemgetter("data")).iter_messages()] == [[0.0], [1.0], [2.0]]
    assert [m["timestamp"] for m in pipeline.filter(lambda data: data[0] > 0).iter_messages()] == [1.0, 2.0]


def test_metadata_still_reaches_callables_that_ask_for_it():
    topic = {"ts": np.array([10.0, 20.0]), "data": np.array([1.0, 2.0]), "id": np.array(["a", "b"], dtype=object)}
    assert topic_view(topic).map(lambda d, t: d + t).data.tolist() == [11.0, 22.0]
    assert topic_view(topic).map(lambda x, y, z: f"{z}{y}").data.tolist() == ["a10.0", "b20.0"]

    class Scaler:
        def __call__(self, points, stamp):
            return points * 0 + stamp

    assert topic_pipeline(topic).map(Scaler()).collect()["data"].tolist() == [10.0, 20.0]
    assert topic_view(topic).reduce(lambda acc, data, ts, name: acc + [name], initial=[]) == ["a", "b"]


def test_package_functions_never_get_timestamps():
    clouds = np.random.default_rng(0).normal(size=(2, 20, 3))
    topic = {"ts": np.array([0.5, 1.5]), "data": clouds}
    with pytest.raises(TypeError, match="voxel_size"):
        topic_pipeline(topic).map(voxel_downsample)
    with pytest.raises(TypeError, match="voxel_size"):
        topic_view(topic).map(voxel_downsample)
    with pytest.raises(TypeError, match="voxel_size"):
        source_pipeline(Messages(_messages(2))).map(voxel_downsample)

    bound = functools.partial(voxel_downsample, voxel_size=0.5)
    expected = [voxel_downsample(cloud, 0.5).shape for cloud in clouds]
    assert [row.shape for row in topic_pipeline(topic).map(bound).collect()["data"]] == expected


def test_source_message_dispatch_falls_back_to_data_consistently():
    clouds = [
        {"topic": "/c", "name": str(i), "timestamp": float(i), "data": np.ones((4, 3)) * (i + 1)} for i in range(2)
    ]
    pipeline = source_pipeline(Messages(clouds))
    assert [m["data"].shape for m in pipeline.map(valid_point_cloud_points).iter_messages()] == [(4, 3), (4, 3)]
    assert [float(m["data"]) for m in pipeline.map(np.linalg.norm).iter_messages()] == pytest.approx(
        [np.sqrt(12.0), 2 * np.sqrt(12.0)]
    )
    assert [m["data"].shape for m in pipeline.map(lambda m: m["data"][:1]).iter_messages()] == [(1, 3), (1, 3)]

    def broken(message):
        raise ValueError("real message bug")

    with pytest.raises(ValueError, match="real message bug"):
        list(pipeline.map(broken).iter_messages())


# --- 2: row stacking keeps values and matches the pass-through schema -------------


def test_row_path_keeps_bytes_and_object_rows():
    blobs = [b"\x89PNG\x00\x00", b"\xff\xd8\x00"]
    topic = {"ts": np.arange(2.0), "data": _object_rows(blobs)}
    pipeline = topic_pipeline(topic)
    for result in (
        pipeline.collect(progress_callback=lambda p: None)["data"],
        pipeline.filter(lambda x: True).collect()["data"],
        pipeline.map(lambda x: x).collect()["data"],
        topic_view(topic).map(lambda x: x).data,
    ):
        assert result.dtype == object and list(result) == blobs

    rows = _object_rows([np.zeros((2, 3)), np.ones((2, 3)), np.ones((4, 3)), np.ones((4, 3))])
    ragged = topic_pipeline({"ts": np.arange(4.0), "data": rows})
    fast = [(c.data.dtype, c.data.shape) for c in ragged.iter_chunks(chunk_size=2)]
    slow = [(c.data.dtype, c.data.shape) for c in ragged.filter(lambda x: True).iter_chunks(chunk_size=2)]
    assert fast == slow == [(np.dtype(object), (2,)), (np.dtype(object), (2,))]

    labels = {"ts": np.arange(2.0), "data": np.array([b"ab", b"c"], dtype="S8")}
    assert topic_pipeline(labels).filter(lambda x: True).collect()["data"].dtype == np.dtype("S8")


def test_mixed_map_outputs_are_not_coerced_to_strings():
    def label(value):
        return "big" if value >= 3 else int(value)

    for result in (
        topic_pipeline(_range_topic(6)).map(label).collect(chunk_size=3)["data"],
        topic_pipeline(_range_topic(6)).map(label).collect(chunk_size=6)["data"],
        topic_view(_range_topic(6)).map(label).data,
    ):
        assert result.tolist() == [0, 1, 2, "big", "big", "big"]
    assert topic_view(_range_topic(3)).map(float).data.dtype == np.float64


# --- 3: ragged topics (RAGGED_SPEC section 3) -----------------------------------------


def test_topic_view_rows_and_row_operations_are_trimmed():
    topic = _ragged_topic()
    view = topic_view(topic)
    assert view.lengths.tolist() == [2, 5, 0, 3]
    assert [row.shape for row in view.rows()] == [(2, 3), (5, 3), (0, 3), (3, 3)]
    assert view.map(len).data.tolist() == [2, 5, 0, 3]
    kept = view.filter(lambda points: len(points) >= 3)
    assert kept.data.shape == (2, 5, 3) and kept.lengths.tolist() == [5, 3]
    assert view.reduce(lambda acc, points: acc + len(points), initial=0) == 10
    assert [w.lengths.tolist() for w in view.window(size=2)] == [[2], [2, 5], [5, 0], [0, 3]]
    assert topic_view(view).lengths.tolist() == [2, 5, 0, 3]
    assert view.as_dict()["lengths"].tolist() == [2, 5, 0, 3]

    structured = np.zeros(4, dtype=[("ts", "f8"), ("data", "f8", (5, 3)), ("length", "i8")])
    structured["ts"], structured["data"], structured["length"] = topic["ts"], topic["data"], topic["lengths"]
    assert [row.shape for row in topic_rows(structured)] == [(2, 3), (5, 3), (0, 3), (3, 3)]
    assert topic_lengths(structured).tolist() == [2, 5, 0, 3]

    with pytest.raises(ValueError, match="lengths"):
        topic_view({"ts": np.arange(2.0), "data": np.zeros((2, 3, 3)), "lengths": np.array([1, 4])})


@pytest.mark.parametrize("options", [{}, {"progress_callback": lambda p: None}, {"max_workers": 2}])
def test_pipelines_carry_lengths_and_trim_rows(options):
    topic = _ragged_topic()
    pipeline = topic_pipeline(topic)
    for selected in (pipeline, pipeline.filter(lambda points: True)):
        collected = selected.collect(chunk_size=3, **options)
        assert collected["data"].shape == (4, 5, 3)
        assert collected["lengths"].tolist() == [2, 5, 0, 3]
        chunks = list(selected.iter_chunks(chunk_size=3, **options))
        assert [c.lengths.tolist() for c in chunks] == [[2, 5, 0], [3]]
    mapped = pipeline.map(lambda points: points.mean(axis=0) if len(points) else np.zeros(3)).collect(**options)
    assert "lengths" not in mapped and mapped["data"].shape == (4, 3)
    assert [row["data"].shape for row in pipeline.iter_rows(**options)] == [(2, 3), (5, 3), (0, 3), (3, 3)]
    assert [r.shape for r in pipeline.rows(**options)] == [(2, 3), (5, 3), (0, 3), (3, 3)]
    assert pipeline.reduce(lambda acc, points: acc + len(points), initial=0) == 10
    windows = list(pipeline.filter(lambda p: True).window(size=2).iter_windows(chunk_size=2))
    assert [w.lengths.tolist() for w in windows] == [[2], [2, 5], [5, 0], [0, 3]]


def test_alignment_keeps_ragged_lengths():
    from arraydataengine.ops import align_nearest, rolling_window_join

    topic = _ragged_topic()
    aligned = align_nearest({"ts": np.array([0.9, 2.2, 10.0]), "data": np.zeros(3)}, topic, tolerance=0.5)
    assert aligned["lengths"].tolist() == [5, 0, 0] and aligned["valid"].tolist() == [True, True, False]
    joined = rolling_window_join({"ts": np.array([1.0, 3.0]), "data": np.zeros(2)}, topic, seconds=1.0)
    assert [w.lengths.tolist() for w in joined["windows"]] == [[2, 5], [0, 3]]
    assert resample_topic(topic, rate_hz=1.0, method="nearest")["lengths"].tolist() == [2, 5, 0, 3]
    with pytest.raises(ValueError, match="ragged"):
        resample_topic(topic, rate_hz=1.0)


def test_spatial_filters_ignore_padding():
    topic = _ragged_topic()
    near_origin = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    assert topic_pipeline(topic).spatial_bounds(*near_origin).collect()["ts"].tolist() == []
    assert topic_pipeline(topic).filter(lambda p: True).spatial_bounds(*near_origin).collect()["ts"].tolist() == []
    assert dataset_query({"/c": topic}).spatial_bounds(*near_origin).collect()["/c"]["ts"].tolist() == []
    assert topic_pipeline(topic).spatial_bounds([100, 100, 100], [110, 110, 110]).collect()["ts"].tolist() == [0.0]


def test_ragged_npz_round_trip_and_summary(tmp_path):
    topic = _ragged_topic()
    loaded = load_topic_npz(save_topic_npz(tmp_path / "padded", topic))
    assert np.array_equal(loaded["data"], topic["data"])
    assert loaded["lengths"].tolist() == [2, 5, 0, 3]
    assert [r.shape for r in topic_rows(loaded)] == [(2, 3), (5, 3), (0, 3), (3, 3)]

    rows = _object_rows([np.ones((2, 3), np.float32), np.empty((0,)), np.ones((4, 3), np.float32)])
    loaded = load_topic_npz(save_topic_npz(tmp_path / "object", {"ts": np.arange(3.0), "data": rows}))
    assert [(r.shape, r.dtype) for r in loaded["data"]] == [((2, 3), np.float32), ((0, 3), np.float32), ((4, 3), np.float32)]

    summary = describe_topic(topic)
    assert summary["data_min"] == 100.0 and summary["length_min"] == 0 and summary["length_max"] == 5


def test_ragged_npz_trims_storage_padding_to_longest_row(tmp_path):
    topic = _ragged_topic()
    capacity_padded = {**topic, "data": np.concatenate([topic["data"], np.zeros((4, 95, 3))], axis=1)}
    loaded = load_topic_npz(save_topic_npz(tmp_path / "trimmed", capacity_padded))
    assert loaded["data"].shape == (4, 5, 3)
    assert [r.tolist() for r in topic_rows(loaded)] == [r.tolist() for r in topic_rows(topic)]


def test_source_messages_are_trimmed_to_point_count():
    def message(index, count, rows=6):
        data = np.zeros((rows, 3))
        data[:count] = index + 1
        return {"topic": "/c", "name": str(index), "timestamp": float(index), "data": data, "point_count": count}

    source = Messages([message(0, 2), message(1, 4), message(2, 3)], max_points=8)
    pipeline = source_pipeline(source)
    assert [(m["data"].shape[0], m["point_count"]) for m in pipeline.iter_messages()] == [(2, 2), (4, 4), (3, 3)]
    doubled = pipeline.map(lambda m: {**m, "data": np.concatenate([m["data"], m["data"]])})
    assert [m["point_count"] for m in doubled.iter_messages()] == [4, 8, 6]
    assert [m["point_count"] for m in pipeline.map(lambda data, ts: data[:1]).iter_messages()] == [1, 1, 1]
    assert ["point_count" in m for m in pipeline.map(lambda d, t: d.mean(axis=0)).iter_messages()] == [False] * 3

    buffer = pipeline.to_buffer(buffer_depth=3)
    stored = buffer.get_index_range("/c")
    assert stored["lengths"].tolist() == [2, 4, 3]
    assert stored["data"].shape[1] == 8  # sized from the source's max_points
    assert [row.shape[0] for row in buffer.topic("/c").rows()] == [2, 4, 3]


# --- 3 (npz): bytes, ids, metadata ----------------------------------------------------


def test_npz_bytes_rows_int_ids_and_bytes_metadata(tmp_path):
    blobs = [b"\xff\xd8\xff\xe0JFIF\x00\x01", b"\x89PNG\r\n\x1a\n\x00\x00"]
    topic = {"ts": np.arange(2.0), "data": _object_rows(blobs), "id": np.array([7, 8]),
             "frame_id": b"camera", "topic": b"/images"}
    loaded = load_topic_npz(save_topic_npz(tmp_path / "blobs", topic))
    assert list(loaded["data"]) == blobs
    assert loaded["id"].tolist() == [7, 8] and loaded["id"].dtype.kind == "i"
    assert loaded["frame_id"] == "camera" and loaded["topic"] == "/images"
    empty = load_topic_npz(save_topic_npz(tmp_path / "empty", {"ts": np.array([]), "data": np.empty(0, dtype=object)}))
    assert empty["data"].shape == (0,)


# --- 4/6/9: checkpoints and progress ---------------------------------------------------


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_progress_callback_errors_do_not_lose_rows(filtered, error):
    pipeline = topic_pipeline(_range_topic(10))
    if filtered:
        pipeline = pipeline.filter(lambda x: True)
    armed = {"on": True}

    def callback(progress):
        if armed["on"] and progress.processed == 5:
            armed["on"] = False
            raise error("stop")

    checkpoint, got = {}, []
    with pytest.raises(error):
        for row in pipeline.iter_rows(checkpoint=checkpoint, progress_callback=callback):
            got.append(int(row["data"]))
    rest = [int(x) for x in pipeline.collect(checkpoint=checkpoint)["data"]]
    assert got + rest == list(range(10))


def test_source_progress_callback_errors_do_not_lose_messages():
    pipeline = source_pipeline(Messages(_messages(10, topics=("/a", "/b")))).index_range(0, None, 1)
    armed = {"on": True}

    def callback(progress):
        if armed["on"] and progress.processed == 5:
            armed["on"] = False
            raise RuntimeError("stop")

    checkpoint, got = {}, []
    with pytest.raises(RuntimeError):
        for message in pipeline.iter_messages(checkpoint=checkpoint, progress_callback=callback):
            got.append(int(message["timestamp"]))
    checkpoint = json.loads(json.dumps(checkpoint))
    assert checkpoint["topic_emitted"] == {"/a": 2, "/b": 2}
    assert checkpoint["operation_counters"] == [{"/a": 2, "/b": 2}]
    got += [int(m["timestamp"]) for m in pipeline.iter_messages(checkpoint=checkpoint)]
    assert got == list(range(10))


def test_source_index_counters_roll_back_when_a_later_operation_fails():
    state = {"fail": 4}

    def fn(message):
        if message["timestamp"] == state["fail"]:
            raise RuntimeError("transient")
        return message

    pipeline = source_pipeline(Messages(_messages(10))).index_range(0, None, 2).map(fn)
    checkpoint, got = {}, []
    with pytest.raises(RuntimeError):
        for message in pipeline.iter_messages(checkpoint=checkpoint):
            got.append(int(message["timestamp"]))
    state["fail"] = -1
    got += [int(m["timestamp"]) for m in pipeline.iter_messages(checkpoint=json.loads(json.dumps(checkpoint)))]
    assert got == [0, 2, 4, 6, 8]


def test_source_progress_snapshots_are_independent_and_counts_stay_live():
    seen = []
    checkpoint = {}
    list(source_pipeline(Messages(_messages(6, topics=("/a", "/b")))).iter_messages(
        checkpoint=checkpoint, progress_callback=lambda p: seen.append(p.checkpoint), progress_interval=2,
    ))
    assert [snapshot["topic_emitted"] for snapshot in seen] == [
        {"/a": 1, "/b": 1}, {"/a": 2, "/b": 2}, {"/a": 3, "/b": 3}, {"/a": 3, "/b": 3}
    ]
    assert checkpoint["topic_emitted"] == {"/a": 3, "/b": 3}


@pytest.mark.parametrize("workers", [1, 2])
def test_iter_chunks_progress_checkpoint_never_runs_ahead_of_delivered_rows(workers):
    class Crash(BaseException):
        pass

    for crash_at in (5, 15, 25):
        persisted = {}

        def callback(progress):
            persisted.clear()
            persisted.update(json.loads(json.dumps(progress.checkpoint)))
            if progress.processed >= crash_at and not progress.done:
                raise Crash

        pipeline = topic_pipeline(_range_topic(30)).filter(lambda x: True)
        received = []
        with pytest.raises(Crash):
            for chunk in pipeline.iter_chunks(chunk_size=10, checkpoint={}, progress_callback=callback,
                                              max_workers=workers):
                received += chunk.data.tolist()
        assert persisted.get("processed", 0) <= len(received)
        rest = pipeline.collect(checkpoint=persisted)["data"].tolist()
        assert received + rest == list(range(30))

    seen = []
    topic_pipeline(_range_topic(30)).filter(lambda x: True).collect(
        chunk_size=10, checkpoint={"processed": 0}, progress_callback=lambda p: seen.append(p)
    )
    assert all(p.checkpoint["processed"] == 0 for p in seen if not p.done)
    assert seen[-1].done and seen[-1].checkpoint["processed"] == 30


def test_fast_path_keeps_reporting_progress_per_chunk():
    seen = []
    result = topic_pipeline(_range_topic(25)).collect(chunk_size=10, progress_callback=seen.append, checkpoint={})
    assert result["data"].tolist() == list(range(25))
    # Whole chunks pass through, so progress arrives once per chunk.
    assert [(p.processed, p.done) for p in seen] == [(10, False), (20, False), (25, False), (25, True)]


# --- 5: to_buffer(buffer=persistent) reconciles the checkpoint --------------------------


def _partial_ingest(pipeline, uri, backend, stop_at):
    token = CancellationToken()

    def cancel(progress):
        if progress.processed == stop_at:
            token.cancel()

    with pytest.raises(PipelineCancelled):
        pipeline.to_buffer(data_uri=uri, use_db=True, backend=backend, checkpoint={},
                           cancel_token=token, progress_callback=cancel)


def _checkpoint_after(pipeline, processed):
    saved = {}

    def keep(progress):
        if progress.processed == processed and not progress.done:
            saved.update(progress.checkpoint)

    list(pipeline.iter_messages(checkpoint={}, progress_callback=keep))
    return json.loads(json.dumps(saved))


@pytest.mark.parametrize("stop_at, saved_at", [(2, 5), (5, 2)])
def test_to_buffer_into_existing_store_reconciles_checkpoint(tmp_path, backend, stop_at, saved_at):
    pipeline = source_pipeline(Messages(_messages(7)))
    uri = str(tmp_path / "store")
    _partial_ingest(pipeline, uri, backend, stop_at=stop_at)
    checkpoint = _checkpoint_after(pipeline, saved_at)
    store = DataBuffer(data_source=None, data_uri=uri, use_db=True, backend=backend, topics=["/s"])
    resumed = pipeline.to_buffer(buffer=store, checkpoint=checkpoint)
    assert _stored(resumed, "/s") == list(range(7))
    resumed.close()


# --- 7: JSON-safe checkpoints ----------------------------------------------------------------


@pytest.mark.parametrize("ids", [np.arange(5), np.array([b"a", b"b", b"c", b"d", b"e"])])
@pytest.mark.parametrize("mode", ["rows", "passthrough", "parallel"])
def test_checkpoints_are_json_serializable(ids, mode):
    topic = {"ts": np.arange(5.0), "data": np.arange(5.0), "id": ids}
    pipeline = topic_pipeline(topic)
    checkpoint = {}
    if mode == "rows":
        pipeline.collect(checkpoint=checkpoint, progress_callback=lambda p: None, chunk_size=2)
        pipeline.filter(lambda x: True).collect(checkpoint={}, chunk_size=2)
    elif mode == "passthrough":
        pipeline.collect(checkpoint=checkpoint)
    else:
        pipeline.filter(lambda x: True).collect(checkpoint=checkpoint, max_workers=2, chunk_size=2)
    restored = json.loads(json.dumps(checkpoint))
    assert restored["processed"] == 5
    assert pipeline.collect(checkpoint=restored)["ts"].tolist() == []


# --- 8: nearest_time_index fast path ---------------------------------------------------------


def test_nearest_time_index_assume_sorted_and_view_cache():
    ts = np.array([0.0, 1.0, 1.0, 2.5, 4.0, np.nan])
    view = topic_view({"ts": ts[:-1], "data": np.zeros(5)})
    for query in (-1.0, 0.4, 0.5, 1.0, 1.75, 3.25, 9.0, np.nan):
        for tolerance in (None, 0.3):
            expected = nearest_time_index(ts, query, tolerance=tolerance)
            assert nearest_time_index(ts, query, tolerance=tolerance, assume_sorted=True) == expected
            assert view.nearest_time_index(query, tolerance=tolerance) == expected
            assert nearest_time_index(view, query, tolerance=tolerance) == expected
    assert nearest_time_index(np.array([]), 1.0, assume_sorted=True) is None
    unsorted = topic_view({"ts": np.array([3.0, 1.0, 2.0]), "data": np.zeros(3)})
    assert not unsorted.is_time_sorted() and unsorted.nearest_time_index(2.9) == 0


# --- 10: collect from a live in-memory ring -----------------------------------------------------


def test_collect_does_not_alias_a_live_ring():
    def stream():
        for i in range(100):
            yield {"topic": "/a", "name": f"m{i}", "timestamp": float(i), "data": np.full(2, float(i))}

    buffer = DataBuffer(data_source=stream, buffer_depth=6, topics=["/a"], axis="/a", preload=0)
    for _ in range(6):
        buffer.roll_buffer("/a")

    class ProducerTick:
        """Appends to the ring between chunk deliveries."""

        def cancelled(self):
            buffer.roll_buffer("/a")
            buffer.roll_buffer("/a")
            return False

    result = buffer.topic("/a").collect(chunk_size=2, cancel_token=ProducerTick())
    assert result["data"][:, 0].tolist() == result["ts"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


# --- improvements ---------------------------------------------------------------------------


def test_parallel_iteration_stops_promptly_when_the_consumer_breaks():
    import time

    calls = []

    def work(value):
        calls.append(value)
        if value >= 50:  # the first chunk is fast, the rest are slow
            time.sleep(0.001)
        return value

    rows = topic_pipeline(_range_topic(200)).map(work).iter_rows(chunk_size=50, max_workers=2)
    next(rows)
    rows.close()
    # Chunks still queued when the consumer stops are cancelled, not run.
    assert len(calls) < 200


def test_parallel_topic_collect_enforces_the_shared_budget():
    calls = []

    def count(value):
        calls.append(value)
        return value

    topic = _range_topic(1000)
    query = dataset_query({"/a": topic, "/b": topic, "/c": topic}).map(count)
    with pytest.raises(MemoryError, match="max_rows=100"):
        query.collect(chunk_size=10, max_rows=100, topic_workers=3)
    # Every topic stops once the shared total passes the budget.
    assert len(calls) < 200
    small = dataset_query({"/a": _range_topic(40), "/b": _range_topic(40), "/c": _range_topic(40)})
    assert sum(len(v["ts"]) for v in small.collect(chunk_size=10, max_rows=120, topic_workers=3).values()) == 120


def test_windows_keep_topic_frame_when_rows_have_none():
    topic = {"ts": np.arange(3.0), "data": np.arange(3.0)}
    windows = list(topic_pipeline(topic, frame_id="map").filter(lambda x: True).window(size=2).iter_windows())
    assert all(w.metadata.frame_id == "map" for w in windows)
    bare = list(topic_pipeline(topic).filter(lambda x: True).window(size=2).iter_windows())
    assert all(w.frame_ids is None and "frame_ids" not in w.as_dict() for w in bare)


def test_resample_nearest_without_tolerance_marks_out_of_range_invalid():
    topic = {"ts": np.array([0.0, 1.0, 2.0]), "data": np.array([0.0, 1.0, 2.0])}
    result = resample_topic(topic, rate_hz=1.0, start=-1.0, end=3.0, method="nearest")
    assert result["valid"].tolist() == [False, True, True, True, False]
    assert np.isnan(result["data"][[0, 4]]).all()
    tolerant = resample_topic(topic, rate_hz=2.0, start=-0.5, end=2.5, method="nearest", tolerance=0.5)
    assert tolerant["valid"].all()


# --- A: TUM export of pose topics -------------------------------------------------------------


def test_write_tum_accepts_pose_topics(tmp_path):
    poses = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]], dtype=np.float64)
    topic = {"ts": np.array([0.5, 1.5]), "data": poses}
    for i, source in enumerate((topic, topic_view(topic))):
        loaded = read_tum_trajectory(write_tum_trajectory(tmp_path / f"pose{i}.tum", source))
        assert np.allclose(loaded["position"], poses[:, :3])
        assert np.allclose(loaded["orientation"], poses[:, 3:])
        assert np.allclose(loaded["ts"], topic["ts"])


def test_tum_export_of_odometry_and_missing_orientations():
    from arraydataengine.ops import to_tum_trajectory

    poses = {"ts": np.arange(2.0), "data": np.array([[1, 2, 3, 0, 0, 0, 0], [4, 5, 6, 0, 0, 0, 2]], dtype=float)}
    assert to_tum_trajectory(poses)[:, 4:].tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
    odometry = np.zeros((2, 8, 4))
    odometry[:, 0, :3] = [[1, 2, 3], [4, 5, 6]]
    odometry[:, 2, 3] = 1.0
    rows = to_tum_trajectory({"ts": np.arange(2.0), "data": odometry})
    assert rows[:, 1:4].tolist() == [[1, 2, 3], [4, 5, 6]] and rows[:, 7].tolist() == [1.0, 1.0]


def test_new_numeric_helpers_are_exported():
    import arraydataengine.ops as ops
    from arraydataengine.ops import pose_to_trajectory, sample_elevation_at_latlon

    assert callable(pose_to_trajectory) and callable(sample_elevation_at_latlon)
    assert {"pose_to_trajectory", "sample_elevation_at_latlon", "topic_rows", "topic_lengths"} <= set(ops.__all__)
    assert all(hasattr(ops, name) for name in ops.__all__)


@pytest.mark.parametrize("backend", ["memory", "arrow", "tiledb"])
def test_source_pipeline_to_buffer_forwards_capacity_options(tmp_path, backend):
    def clouds():
        rng = np.random.default_rng(0)
        for i, n in enumerate((40, 120, 80)):  # larger than the capacity the source advertises
            yield {"topic": "/c", "timestamp": float(i), "name": str(i),
                   "data": rng.normal(size=(n, 3)), "point_count": n}

    class Source:
        max_points = 50

        def get_topics(self):
            return ["/c"]

        def get_count(self, topic):
            return 3

        def get_message(self):
            return clouds()

    kwargs = {} if backend == "memory" else {"use_db": True, "backend": backend, "data_uri": str(tmp_path / backend)}
    small = {**kwargs, "data_uri": str(tmp_path / "small")} if kwargs else {}
    with pytest.raises(ValueError, match="capacity|max_points"):  # the source's own limit (50) is too small
        source_pipeline(Source()).to_buffer(buffer_depth=3, **small)
    buffer = source_pipeline(Source()).to_buffer(buffer_depth=3, max_points=128, **kwargs)
    stored = buffer.get_index_range("/c")
    assert stored["lengths"].tolist() == [40, 120, 80]
    assert stored["data"].shape[1] == 128
