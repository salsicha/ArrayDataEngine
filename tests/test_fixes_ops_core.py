"""Regressions for ops core pipelines, alignment, summaries, and NPZ IO."""

import json

import numpy as np
import pytest

from arraydataengine.ops import (
    CancellationToken,
    PipelineCancelled,
    align_exact,
    align_nearest,
    dataset_query,
    describe_topic,
    format_describe,
    load_topic_npz,
    nearest_time_index,
    resample_topic,
    rolling_window_join,
    save_topic_npz,
    source_pipeline,
    topic_pipeline,
    topic_view,
)


class Messages:
    def __init__(self, messages):
        self.messages = messages

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


def _range_topic(count, width=None):
    data = np.arange(count, dtype=np.float64)
    if width is not None:
        data = np.repeat(data[:, None], width, axis=1)
    return {"ts": np.arange(count, dtype=np.float64), "data": data}


@pytest.fixture(params=["arrow", "tiledb"])
def backend(request):
    pytest.importorskip("pyarrow" if request.param == "arrow" else "tiledb")
    return request.param


def _stored(buffer, topic):
    return buffer.get_index_range(topic)["data"].ravel().astype(int).tolist()


# --- 1: persistent to_buffer resume trusts the store over the checkpoint --------


def _partial_ingest(pipeline, uri, backend, stop_at):
    token = CancellationToken()

    def cancel(progress):
        if progress.processed == stop_at:
            token.cancel()

    with pytest.raises(PipelineCancelled):
        pipeline.to_buffer(data_uri=uri, use_db=True, backend=backend, checkpoint={},
                           cancel_token=token, progress_callback=cancel)


def _checkpoint_after(pipeline, processed):
    """A checkpoint as saved by a progress callback at `processed` messages."""
    saved = {}

    def keep(progress):
        if progress.processed == processed and not progress.done:
            saved.update(progress.checkpoint)

    list(pipeline.iter_messages(checkpoint={}, progress_callback=keep))
    return json.loads(json.dumps(saved))


@pytest.mark.parametrize("legacy", [False, True])
def test_to_buffer_resume_with_checkpoint_ahead_of_store_replays(tmp_path, backend, legacy):
    # The store holds 2 rows, but the checkpoint claims 5 (the process died
    # before staged rows were flushed): resuming must not skip rows 2..4.
    pipeline = source_pipeline(Messages(_messages(7)))
    uri = str(tmp_path / "store")
    _partial_ingest(pipeline, uri, backend, stop_at=2)
    ahead = _checkpoint_after(pipeline, 5)
    if legacy:
        ahead.pop("topic_emitted")
    with pipeline.to_buffer(data_uri=uri, use_db=True, backend=backend, checkpoint=ahead) as resumed:
        assert _stored(resumed, "/s") == list(range(7))
    assert ahead["done"] is True


def test_to_buffer_resume_with_checkpoint_behind_store_skips_stored_rows(tmp_path, backend):
    pipeline = source_pipeline(Messages(_messages(8, topics=("/a", "/b"))))
    uri = str(tmp_path / "store")
    _partial_ingest(pipeline, uri, backend, stop_at=6)
    behind = _checkpoint_after(pipeline, 3)
    with pipeline.to_buffer(data_uri=uri, use_db=True, backend=backend, checkpoint=behind) as resumed:
        assert _stored(resumed, "/a") == list(range(0, 8, 2))
        assert _stored(resumed, "/b") == list(range(1, 8, 2))


def test_to_buffer_flushes_appended_rows_when_a_callable_fails(tmp_path, backend):
    state = {"fail": 4}

    def fn(message):
        if message["timestamp"] == state["fail"]:
            raise RuntimeError("bad message")
        return message

    pipeline = source_pipeline(Messages(_messages(7))).map(fn)
    uri = str(tmp_path / "store")
    checkpoint = {}
    with pytest.raises(RuntimeError, match="bad message"):
        pipeline.to_buffer(data_uri=uri, use_db=True, backend=backend, checkpoint=checkpoint)
    # Resume immediately (the failed buffer may still be referenced).
    state["fail"] = -1
    with pipeline.to_buffer(data_uri=uri, use_db=True, backend=backend, checkpoint=checkpoint) as resumed:
        assert _stored(resumed, "/s") == list(range(7))


# --- 2: callable dispatch by signature -------------------------------------------


def test_map_does_not_feed_metadata_into_optional_parameters():
    topic = {"data": np.array([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]]), "ts": np.array([2.0, 100.0, 200.0])}
    expected = [5.0, 10.0, 5.0]
    assert topic_view(topic).map(np.linalg.norm).data.tolist() == expected
    assert topic_pipeline(topic).map(np.linalg.norm).collect()["data"].tolist() == expected
    assert topic_pipeline(topic).map(np.linalg.norm).collect(max_workers=2, chunk_size=1)["data"].tolist() == expected

    def scale(data, factor=2.0):
        return data * factor

    assert np.allclose(topic_view(topic).map(scale).data, topic["data"] * 2.0)
    assert np.allclose(topic_view(topic).map(np.sqrt).data, np.sqrt(topic["data"]))


def test_callables_receive_required_metadata_arguments():
    topic = {"data": np.array([1.0, 2.0]), "ts": np.array([10.0, 20.0]), "id": np.array(["a", "b"], dtype=object)}
    assert topic_view(topic).map(lambda data, ts: data + ts).data.tolist() == [11.0, 22.0]
    assert topic_pipeline(topic).map(lambda data, ts, name: f"{name}{data}").collect()["data"].tolist() == ["a1.0", "b2.0"]
    assert topic_pipeline(topic).filter(lambda data, ts: ts > 15).collect()["ts"].tolist() == [20.0]


def test_reduce_with_builtin_and_metadata_reducers():
    topic = {"ts": np.array([100.0, 200.0]), "data": np.array([1.0, 2.0]), "id": np.array([7, 8])}
    assert topic_view(topic).reduce(max) == 2.0
    assert topic_pipeline(topic).reduce(max) == 2.0
    assert topic_view(topic).reduce(lambda acc, data, ts: acc + ts, initial=0.0) == 300.0
    assert topic_pipeline(topic).reduce(lambda acc, data, ts, name: acc + [int(name)], initial=[]) == [7, 8]


def test_source_map_with_data_style_callable():
    pipeline = source_pipeline(Messages(_messages(3))).map(lambda data, ts: data + ts)
    assert [message["data"].tolist() for message in pipeline.iter_messages()] == [[0.0], [2.0], [4.0]]


# --- 3/4/12: checkpoints track delivered rows ------------------------------------


@pytest.mark.parametrize("workers", [1, 2])
def test_iter_chunks_exception_mid_chunk_does_not_skip_rows(workers):
    fail = {"value": 25}

    def fn(data):
        if data == fail["value"]:
            raise RuntimeError("transient")
        return data

    checkpoint = {}
    received = []
    with pytest.raises(RuntimeError, match="transient"):
        for chunk in topic_pipeline(_range_topic(30)).map(fn).iter_chunks(
            chunk_size=10, checkpoint=checkpoint, max_workers=workers
        ):
            received.extend(chunk.data.tolist())
    assert checkpoint["processed"] == len(received) == 20

    fail["value"] = -1
    checkpoint = json.loads(json.dumps(checkpoint))
    for chunk in topic_pipeline(_range_topic(30)).map(fn).iter_chunks(
        chunk_size=10, checkpoint=checkpoint, max_workers=workers
    ):
        received.extend(chunk.data.tolist())
    assert received == list(range(30))


@pytest.mark.parametrize("workers", [1, 2])
def test_consumer_stopping_early_resumes_without_duplicates(workers):
    pipeline = topic_pipeline(_range_topic(40)).filter(lambda value: value % 2 == 0)
    checkpoint = {}
    chunks = pipeline.iter_chunks(chunk_size=10, checkpoint=checkpoint, max_workers=workers)
    received = next(chunks).data.tolist()
    chunks.close()
    for chunk in pipeline.iter_chunks(chunk_size=10, checkpoint=dict(checkpoint), max_workers=workers):
        received.extend(chunk.data.tolist())
    assert received == list(range(0, 40, 2))

    checkpoint = {}
    rows = pipeline.iter_rows(chunk_size=10, checkpoint=checkpoint, max_workers=workers)
    received = [next(rows)["data"] for _ in range(3)]
    rows.close()
    received.extend(row["data"] for row in pipeline.iter_rows(chunk_size=10, checkpoint=checkpoint, max_workers=workers))
    assert received == list(range(0, 40, 2))


def test_progress_checkpoint_matches_progress():
    seen = []
    checkpoint = {}
    list(topic_pipeline(_range_topic(5)).iter_rows(
        checkpoint=checkpoint,
        progress_callback=lambda p: seen.append((p.processed, p.checkpoint["processed"], p.done, p.checkpoint["done"])),
    ))
    assert all(processed == saved for processed, saved, _, _ in seen)
    assert seen[-1] == (5, 5, True, True)

    saved = {}

    def keep(progress):
        if progress.processed == 3 and not progress.done:
            saved.update(progress.checkpoint)

    list(topic_pipeline(_range_topic(5)).iter_rows(checkpoint={}, progress_callback=keep))
    resumed = [row["ts"] for row in topic_pipeline(_range_topic(5)).iter_rows(checkpoint=saved)]
    assert resumed == [3.0, 4.0]


# --- 5/7: non-array row values and ragged outputs -------------------------------


def test_map_to_python_scalars_and_object_topics():
    topic = {"ts": np.arange(5.0), "data": np.arange(15.0).reshape(5, 3)}
    result = topic_pipeline(topic).map(lambda d: float(d.sum())).filter(lambda v: v > 10).collect()
    assert result["data"].tolist() == [12.0, 21.0, 30.0, 39.0]

    paths = {"ts": np.arange(2.0), "data": np.array(["a.png", "b.png"], dtype=object)}
    assert topic_view(paths).map(str.upper).data.tolist() == ["A.PNG", "B.PNG"]
    assert topic_pipeline(paths).collect()["data"].tolist() == ["a.png", "b.png"]
    assert topic_pipeline(paths).filter(lambda p: p.startswith("b")).collect()["data"].tolist() == ["b.png"]
    assert topic_view(paths).reduce(lambda acc, p: acc + p) == "a.pngb.png"


def test_ragged_map_outputs_become_object_arrays():
    clouds = np.random.default_rng(0).normal(size=(4, 50, 3))
    topic = {"ts": np.arange(4.0), "data": clouds}

    def above(points):
        return points[points[:, 2] > 0]

    expected = [len(above(cloud)) for cloud in clouds]
    mapped = topic_view(topic).map(above)
    assert mapped.data.dtype == object and [len(c) for c in mapped.data] == expected
    chunks = list(topic_pipeline(topic).map(above).iter_chunks(chunk_size=2))
    assert [len(c) for chunk in chunks for c in chunk.data] == expected
    collected = topic_pipeline(topic).map(above).collect(chunk_size=1)
    assert collected["data"].dtype == object and [len(c) for c in collected["data"]] == expected


# --- 8: collect + cancellation -----------------------------------------------------


def test_collect_cancel_attaches_partial_and_resumes():
    token = CancellationToken()
    checkpoint = {}

    def cancel(progress):
        if progress.processed == 15:
            token.cancel()

    with pytest.raises(PipelineCancelled) as info:
        topic_pipeline(_range_topic(30)).collect(chunk_size=10, checkpoint=checkpoint,
                                                  cancel_token=token, progress_callback=cancel)
    partial = info.value.partial
    rest = topic_pipeline(_range_topic(30)).collect(chunk_size=10, checkpoint=checkpoint)
    assert partial["ts"].tolist() + rest["ts"].tolist() == [float(i) for i in range(30)]


def test_collect_failure_rolls_checkpoint_back():
    checkpoint = {"processed": 5, "emitted": 5, "skipped": 0}
    with pytest.raises(MemoryError):
        topic_pipeline(_range_topic(30)).map(lambda d: d).collect(chunk_size=4, checkpoint=checkpoint, max_rows=10)
    assert checkpoint == {"processed": 5, "emitted": 5, "skipped": 0}


# --- 6/13: alignment and resampling -----------------------------------------------


def test_alignment_with_unsorted_target_keeps_caller_order():
    reference = {"ts": np.array([2.0, 1.0, 3.0]), "data": np.zeros(3)}
    target = {"ts": np.array([3.0, 1.0, 2.0]), "data": np.array([30.0, 10.0, 20.0])}
    nearest = align_nearest(reference, target)
    assert nearest["data"].tolist() == [20.0, 10.0, 30.0]
    assert nearest["target_index"].tolist() == [2, 1, 0]
    exact = align_exact(reference, target)
    assert exact["data"].tolist() == [20.0, 10.0, 30.0]
    assert exact["target_index"].tolist() == [2, 1, 0]
    assert nearest_time_index(target["ts"], 2.9) == 0
    assert nearest_time_index(np.array([0.0, 2.0]), 1.0) == 1  # ties prefer the later sample

    linear = resample_topic(target, rate_hz=1.0, start=1.0, end=3.0)
    assert linear["data"].tolist() == [10.0, 20.0, 30.0]
    nearest_grid = resample_topic(target, rate_hz=1.0, method="nearest")
    assert nearest_grid["data"].tolist() == [10.0, 20.0, 30.0]
    assert nearest_grid["target_index"].tolist() == [1, 2, 0]

    joined = rolling_window_join({"ts": np.array([2.0]), "data": np.zeros(1)}, target, seconds=1.0)
    assert joined["windows"][0].data.tolist() == [10.0, 20.0]


def test_nearest_alignment_ignores_nan_timestamps():
    target = {"ts": np.array([0.0, np.nan, 2.0]), "data": np.array([0.0, 99.0, 2.0])}
    aligned = align_nearest({"ts": np.array([np.nan, 1.9]), "data": np.zeros(2)}, target)
    assert aligned["target_index"].tolist() == [-1, 2]


def test_seconds_window_on_unsorted_timestamps_matches_lazy_window():
    topic = {"ts": np.array([0.0, 5.0, 1.0, 6.0, 2.0, 7.0]), "data": np.arange(6.0)}
    eager = [window.data.tolist() for window in topic_view(topic).window(seconds=1.5, size=4)]
    lazy = [window.data.tolist() for window in topic_pipeline(topic).window(seconds=1.5, size=4).collect()]
    assert eager == lazy


def test_linear_resample_honors_tolerance_and_range():
    gap = {"ts": np.array([0.0, 1.0, 10.0]), "data": np.array([0.0, 1.0, 10.0])}
    result = resample_topic(gap, rate_hz=1.0, tolerance=0.5)
    assert result["valid"].tolist() == [True, True] + [False] * 8 + [True]
    assert np.isnan(result["data"][2:10]).all()

    outside = resample_topic(gap, rate_hz=1.0, start=-2.0, end=11.0)
    assert outside["valid"][:2].tolist() == [False, False] and not outside["valid"][-1]
    assert np.isnan(outside["data"][:2]).all() and np.isnan(outside["data"][-1])
    assert outside["data"][2] == 0.0


def test_fixed_rate_grid_keeps_last_sample_at_epoch_scale():
    base = 1.7e9
    topic = {"ts": np.array([base, base + 0.3]), "data": np.array([0.0, 3.0])}
    result = resample_topic(topic, rate_hz=10.0)
    assert result["ts"].shape == (4,)
    assert result["valid"].all()
    assert np.allclose(result["data"], [0.0, 1.0, 2.0, 3.0], atol=1e-5)


# --- 9: NPZ IO -------------------------------------------------------------------------


def test_npz_path_suffix_object_data_and_frame_ids(tmp_path):
    path = save_topic_npz(tmp_path / "imu", {"ts": np.arange(3.0), "data": np.arange(3.0)})
    assert path.name == "imu.npz" and path.exists()
    assert load_topic_npz(path)["data"].tolist() == [0.0, 1.0, 2.0]

    clouds = np.empty(2, dtype=object)
    clouds[0] = np.arange(15.0).reshape(5, 3)
    clouds[1] = np.arange(21.0).reshape(7, 3)
    loaded = load_topic_npz(save_topic_npz(tmp_path / "pc.npz", {"ts": np.arange(2.0), "data": clouds}))
    assert loaded["data"].dtype == object
    assert all(np.array_equal(a, b) for a, b in zip(loaded["data"], clouds))

    text = {"ts": np.arange(2.0), "data": np.array(["a.png", "b.png"], dtype=object),
            "frame_ids": np.array(["map", None], dtype=object)}
    loaded = load_topic_npz(save_topic_npz(tmp_path / "text.npz", text))
    assert loaded["data"].tolist() == ["a.png", "b.png"]
    assert loaded["frame_ids"].tolist() == ["map", None]

    with pytest.raises(TypeError, match="pickle"):
        save_topic_npz(tmp_path / "bad.npz", {"ts": np.arange(1.0), "data": np.array([{"a": 1}], dtype=object)})


# --- 10/11: dataset collect and parallel progress --------------------------------------


def test_dataset_collect_keeps_frame_ids_and_global_limits():
    topic = {"ts": np.arange(4.0), "data": np.arange(4.0), "frame_ids": np.array(["map", "odom"] * 2, dtype=object)}
    query = dataset_query({"/a": topic, "/b": topic})
    sequential = query.collect()
    assert sequential["/a"]["frame_ids"].tolist() == ["map", "odom", "map", "odom"]
    assert sorted(sequential["/a"]) == sorted(query.collect(topic_workers=2)["/a"])
    with pytest.raises(MemoryError):
        query.collect(max_rows=6)


def test_parallel_progress_reports_each_interval_crossing():
    calls = {1: [], 2: []}
    for workers in (1, 2):
        topic_pipeline(_range_topic(1000)).filter(lambda v: True).collect(
            chunk_size=100, max_workers=workers, progress_interval=50,
            progress_callback=lambda p, w=workers: calls[w].append(p.processed),
        )
    assert len(calls[1]) == 21  # every 50 rows plus the final done report
    assert calls[2][:-1] == list(range(100, 1001, 100))


# --- 14: describe_topic ------------------------------------------------------------


def test_describe_topic_with_unsorted_and_nan_timestamps():
    info = describe_topic({"ts": np.array([0.4, 0.1, 0.2, 0.3, 0.0]), "data": np.zeros(5)})
    assert (info["start_time"], info["end_time"]) == (0.0, 0.4)
    assert np.isclose(info["duration"], 0.4) and np.isclose(info["rate_hz"], 10.0)
    assert info["dt_min"] > 0 and info["non_monotonic_count"] == 2
    assert "2 out-of-order" in format_describe({"t": info})

    info = describe_topic({"ts": np.array([0.0, np.nan, 0.2]), "data": np.zeros(3)})
    assert np.isclose(info["dt_mean"], 0.2) and np.isclose(info["rate_hz"], 5.0)
    assert info["nonfinite_ts_count"] == 1


# --- improvements: fast path, copies, lazy frame metadata ---------------------------


def test_passthrough_collect_matches_row_path():
    topic = {
        "ts": np.arange(25.0),
        "data": np.arange(75, dtype=np.float32).reshape(25, 3),
        "id": np.array([f"m{i}" for i in range(25)], dtype=object),
        "frame_ids": np.array(["map", "odom"] * 12 + ["map"], dtype=object),
    }
    fast = topic_pipeline(topic).time_range(3.0, 20.0).collect(chunk_size=4)
    rows = topic_pipeline(topic).time_range(3.0, 20.0).map(lambda d: d).collect(chunk_size=4)
    assert sorted(fast) == sorted(rows)
    for key in ("ts", "data", "id", "frame_ids"):
        assert fast[key].tolist() == rows[key].tolist()
        assert fast[key].dtype == rows[key].dtype
    chunks = list(topic_pipeline(topic).time_range(3.0, 20.0).iter_chunks(chunk_size=4))
    assert [len(chunk) for chunk in chunks] == [4, 4, 4, 4, 2]


def test_pipeline_outputs_never_alias_the_source():
    topic = _range_topic(8, width=2)
    original = topic["data"].copy()
    collected = topic_pipeline(topic).collect(copy=False)
    collected["data"][:] = -1
    for chunk in topic_pipeline(topic).iter_chunks(chunk_size=3):
        chunk.data[:] = -1
    for row in topic_pipeline(topic).iter_rows(copy=True):
        row["data"][:] = -1
    for row in topic_pipeline(topic).filter(lambda d: True).iter_rows(copy=True, max_workers=2, chunk_size=3):
        row["data"][:] = -1

    def mutate(data):
        data[:] = -1
        return data

    topic_pipeline(topic).map(mutate, copy=False).collect(copy=True)
    assert np.array_equal(topic["data"], original)


def test_sequential_resume_skips_covered_chunks():
    topic = _range_topic(100)
    resumed = topic_pipeline(topic).filter(lambda v: True).collect(chunk_size=7, checkpoint={"processed": 50})
    assert resumed["ts"].tolist() == [float(i) for i in range(50, 100)]
    fast = topic_pipeline(topic).collect(chunk_size=7, checkpoint={"processed": 50})
    assert fast["ts"].tolist() == resumed["ts"].tolist()


def test_lazy_frame_metadata():
    topic = {"ts": np.arange(4.0), "data": np.arange(4.0), "frame_ids": np.array(["map", "map", "odom", "map"], dtype=object)}
    view = topic_view(topic)
    assert view.metadata.frame_id is None
    assert view.select_indices(0, 2).metadata.frame_id == "map"
    assert [window.metadata.frame_id for window in view.window(size=2)] == ["map", "map", None, None]
