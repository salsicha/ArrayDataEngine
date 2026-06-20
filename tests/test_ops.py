import numpy as np

from ade.buffer import DataBuffer
from ade.ops import (
    align_bounded,
    align_exact,
    align_nearest,
    align_topic,
    apply_transform,
    cluster_dbscan,
    crop_bounds,
    crop_raster,
    dataset_query,
    dem_grid_to_points,
    depth_to_points,
    DatasetQuery,
    enu_to_navsat,
    estimate_normals,
    filter_topic,
    hillshade,
    interpolate_timeseries,
    iter_chunks,
    knn_search,
    map_topic,
    mosaic_tiles,
    navsat_to_enu,
    normalize_image,
    normalize_quaternion,
    pad_image,
    radius_outlier_filter,
    reduce_topic,
    resample_topic,
    rolling_window_join,
    resize_nearest,
    rgb_to_gray,
    sample_grid,
    segment_plane,
    select_indices,
    select_time_range,
    slerp,
    slope_aspect,
    statistical_outlier_filter,
    TopicPipeline,
    TopicView,
    topic_pipeline,
    topic_view,
    transform_dem_grid,
    transform_navsat,
    transform_odometry,
    transform_poses,
    transform_vectors,
    trajectory_speed,
    valid_depth_mask,
    voxel_downsample,
    window_topic,
)


def _topic():
    return {
        "id": np.array(["a", "b", "c", "d"], dtype=object),
        "ts": np.array([0.0, 0.5, 1.0, 1.5]),
        "data": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    }


def test_topic_operations_select_map_filter_reduce_window_and_align():
    topic = _topic()

    selected = select_time_range(topic, 0.25, 1.0)
    assert selected["id"].tolist() == ["b", "c"]
    assert np.allclose(selected["data"], np.array([[1.0, 1.0], [2.0, 2.0]]))

    by_index = select_indices(topic, 1, 4, 2)
    assert by_index["id"].tolist() == ["b", "d"]

    mapped = map_topic(topic, lambda data, ts: data + ts)
    assert np.allclose(mapped["data"], np.array([[0.0, 0.0], [1.5, 1.5], [3.0, 3.0], [4.5, 4.5]]))

    filtered = filter_topic(topic, lambda data: data[0] >= 2.0)
    assert filtered["id"].tolist() == ["c", "d"]

    reduced = reduce_topic(topic, lambda acc, data: acc + data, initial=np.zeros(2))
    assert np.allclose(reduced, np.array([6.0, 6.0]))

    fixed_windows = list(window_topic(topic, size=2))
    assert [window["id"].tolist() for window in fixed_windows] == [["a"], ["a", "b"], ["b", "c"], ["c", "d"]]

    time_windows = list(window_topic(topic, seconds=0.6))
    assert [window["id"].tolist() for window in time_windows] == [["a"], ["a", "b"], ["b", "c"], ["c", "d"]]

    target = {
        "id": np.array(["x", "y"], dtype=object),
        "ts": np.array([0.1, 1.4]),
        "data": np.array([[10.0], [20.0]]),
    }
    aligned = align_nearest(topic, target, tolerance=0.2)
    assert aligned["target_index"].tolist() == [0, -1, -1, 1]
    assert aligned["valid"].tolist() == [True, False, False, True]


def test_topic_alignment_modes():
    reference = {
        "id": np.array(["r0", "r1", "r2", "r3"], dtype=object),
        "ts": np.array([0.0, 0.5, 1.0, 1.5]),
        "data": np.array([[0.0], [1.0], [2.0], [3.0]]),
    }
    target = {
        "id": np.array(["t0", "t1", "t2"], dtype=object),
        "ts": np.array([0.0, 0.9, 1.5]),
        "data": np.array([[10.0], [20.0], [30.0]]),
        "topic": "/target",
    }

    exact = align_exact(reference, target)
    assert exact["mode"] == "exact"
    assert exact["target_index"].tolist() == [0, -1, -1, 2]
    assert exact["id"].tolist() == ["t0", None, None, "t2"]
    assert np.allclose(exact["data"][[0, 3]], np.array([[10.0], [30.0]]))
    assert np.isnan(exact["data"][1]).all()

    nearest = align_topic(reference, target, mode="nearest")
    assert nearest["target_index"].tolist() == [0, 1, 1, 2]

    bounded = align_bounded(reference, target, tolerance=0.15)
    assert bounded["mode"] == "bounded_tolerance"
    assert bounded["target_index"].tolist() == [0, -1, 1, 2]

    linear = resample_topic(reference, period=0.75)
    assert linear["mode"] == "fixed_rate"
    assert np.allclose(linear["ts"], np.array([0.0, 0.75, 1.5]))
    assert np.allclose(linear["data"], np.array([[0.0], [1.5], [3.0]]))

    nearest_resampled = align_topic(None, target, mode="fixed_rate", period=0.5, interpolation="nearest", tolerance=0.11)
    assert nearest_resampled["mode"] == "fixed_rate_nearest"
    assert nearest_resampled["target_index"].tolist() == [0, -1, 1, 2]

    joined = rolling_window_join(reference, target, seconds=0.6)
    assert joined["mode"] == "rolling_window"
    assert joined["counts"].tolist() == [1, 1, 1, 2]
    assert [window.ids.tolist() for window in joined["windows"]] == [["t0"], ["t0"], ["t1"], ["t1", "t2"]]

    joined_by_dispatch = align_topic(reference, target, mode="rolling_window", seconds=0.05)
    assert joined_by_dispatch["counts"].tolist() == [1, 0, 0, 1]


def test_topic_view_preserves_metadata_and_supports_chunked_operations():
    topic = _topic()
    view = topic_view(topic, topic="/points", source_uri="/data/run", frame_id="map")

    assert isinstance(view, TopicView)
    assert view.metadata.topic == "/points"
    assert view.metadata.source_uri == "/data/run"
    assert view.metadata.frame_id == "map"
    assert view.metadata.shape == (2,)
    assert view.metadata.dtype == topic["data"].dtype
    assert view.metadata.count == 4
    assert view.metadata.start_time == 0.0
    assert view.metadata.end_time == 1.5
    assert view.metadata.names.tolist() == ["a", "b", "c", "d"]

    chunks = list(iter_chunks(view, chunk_size=2))
    assert [chunk.metadata.count for chunk in chunks] == [2, 2]
    assert [chunk.metadata.start_time for chunk in chunks] == [0.0, 1.0]

    mapped = view.map(lambda data, ts, name: data + ts + len(name), chunk_size=2)
    assert mapped.metadata.topic == "/points"
    assert mapped.metadata.count == 4
    assert np.allclose(mapped.data[1], np.array([2.5, 2.5]))

    out = np.empty_like(topic["data"])
    mapped_dict = map_topic(topic, lambda data: data * 3, out=out, chunk_size=3)
    assert mapped_dict["data"] is out
    assert np.allclose(out, topic["data"] * 3)
    assert mapped_dict["metadata"].names.tolist() == ["a", "b", "c", "d"]

    filtered = filter_topic(topic, lambda data, ts, name: name in {"b", "d"}, chunk_size=1)
    assert filtered["id"].tolist() == ["b", "d"]

    reduced = reduce_topic(topic, lambda acc, data, ts, name: acc + data + ts, initial=np.zeros(2), chunk_size=2)
    assert np.allclose(reduced, np.array([9.0, 9.0]))


def test_topic_pipeline_executes_lazily_and_collects_explicitly():
    calls = []
    pipeline = (
        topic_pipeline(_topic(), topic="/points")
        .time_range(0.5, 1.5)
        .map(lambda data, ts, name: calls.append((name, ts)) or data + ts)
        .filter(lambda data: data[0] >= 3.0)
    )

    assert isinstance(pipeline, TopicPipeline)
    assert calls == []

    chunks = list(pipeline.iter_chunks(chunk_size=1))
    assert [chunk.ids.tolist() for chunk in chunks] == [["c"], ["d"]]
    assert np.allclose(chunks[0].data, np.array([[3.0, 3.0]]))
    assert calls == [("b", 0.5), ("c", 1.0), ("d", 1.5)]

    calls.clear()
    collected = pipeline.collect(chunk_size=2)
    assert collected["id"].tolist() == ["c", "d"]
    assert np.allclose(collected["data"], np.array([[3.0, 3.0], [4.5, 4.5]]))
    assert calls == [("b", 0.5), ("c", 1.0), ("d", 1.5)]

    try:
        topic_pipeline(_topic()).collect(max_rows=2)
    except MemoryError as exc:
        assert "max_rows=2" in str(exc)
    else:
        raise AssertionError("collect() should enforce max_rows")

    try:
        topic_pipeline(_topic()).collect(max_bytes=8)
    except MemoryError as exc:
        assert "max_bytes=8" in str(exc)
    else:
        raise AssertionError("collect() should enforce max_bytes")

    out = np.empty((4, 2), dtype=np.float64)
    with_out = topic_pipeline(_topic()).collect(out=out, max_bytes=64)
    assert with_out["data"] is out
    assert np.allclose(out, _topic()["data"])

    large_allowed = topic_pipeline(_topic()).collect(max_rows=2, max_bytes=8, allow_large=True)
    assert large_allowed["data"].shape == (4, 2)

    rows = list(topic_pipeline(_topic()).index_range(1, 4, 2).iter_rows(chunk_size=2))
    assert [row["id"] for row in rows] == ["b", "d"]

    reduced = pipeline.reduce(lambda acc, data: acc + data, initial=np.zeros(2), chunk_size=1)
    assert np.allclose(reduced, np.array([7.5, 7.5]))

    windows = topic_pipeline(_topic()).time_range(0.0, 1.0).window(size=2).collect(chunk_size=1)
    assert [window.ids.tolist() for window in windows] == [["a"], ["a", "b"], ["b", "c"]]


def test_dataset_query_selects_topics_time_index_frame_and_bounds():
    nav = {
        "id": np.array(["n0", "n1", "n2"], dtype=object),
        "ts": np.array([0.0, 1.0, 2.0]),
        "data": np.array([
            [37.0, -122.0, 10.0],
            [38.0, -123.0, 11.0],
            [37.1, -122.1, 12.0],
        ]),
        "topic": "/navsat",
        "frame_id": "earth",
    }
    points = {
        "id": np.array(["p0", "p1", "p2"], dtype=object),
        "ts": np.array([0.0, 1.0, 2.0]),
        "data": np.array([
            [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
            [[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]],
            [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]),
        "topic": "/points",
        "frame_id": "map",
    }
    camera = {
        "id": np.array(["c0", "c1"], dtype=object),
        "ts": np.array([0.0, 1.0]),
        "data": np.array([[0.0], [1.0]]),
        "topic": "/camera",
        "frame_id": "camera",
    }

    query = dataset_query({
        "/navsat": nav,
        "/points": points,
        "/camera": camera,
    })
    assert isinstance(query, DatasetQuery)

    selected = query.select_topics("/navsat", "/points").time_range(0.5, 2.0).index_range(0, 2).collect(chunk_size=1)
    assert list(selected) == ["/navsat", "/points"]
    assert selected["/navsat"]["id"].tolist() == ["n1", "n2"]
    assert selected["/points"]["id"].tolist() == ["p1", "p2"]

    earth = query.frame_id("earth").collect()
    assert list(earth) == ["/navsat"]
    assert earth["/navsat"]["metadata"].frame_id == "earth"

    geo = query.select_topic("/navsat").geographic_bounds(36.9, -122.2, 37.2, -121.9).collect()
    assert geo["/navsat"]["id"].tolist() == ["n0", "n2"]

    spatial = query.select_topic("/points").spatial_bounds([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]).collect()
    assert spatial["/points"]["id"].tolist() == ["p1"]

    rows = list(query.select_topics(["/camera"]).iter_rows(chunk_size=1))
    assert [row["topic"] for row in rows] == ["/camera", "/camera"]


class _SmallSource:
    def get_topics(self):
        return ["axis"]

    def get_count(self, topic):
        return 3

    def get_message(self):
        for i in range(3):
            yield {
                "topic": "axis",
                "timestamp": float(i),
                "name": f"frame_{i}",
                "data": np.array([float(i)]),
                "frame_id": "map",
            }


def test_data_buffer_operation_wrappers():
    buffer = DataBuffer(_SmallSource(), buffer_depth=3, axis="axis", preload=True)

    view = buffer.topic_view("axis")
    assert view.metadata.topic == "axis"
    assert view.metadata.frame_id == "map"
    assert view.metadata.count == 3
    assert view.metadata.names.tolist() == [b"frame_0", b"frame_1", b"frame_2"]

    mapped = buffer.map_topic("axis", lambda data: data * 2)
    assert np.allclose(mapped["data"], np.array([[0.0], [2.0], [4.0]]))

    out = np.empty((3, 1), dtype=np.float64)
    chunked_mapped = buffer.map_topic("axis", lambda data, ts: data + ts, out=out, chunk_size=2)
    assert chunked_mapped["data"] is out
    assert np.allclose(out, np.array([[0.0], [2.0], [4.0]]))

    filtered = buffer.filter_topic("axis", lambda data, ts: ts >= 1.0)
    assert np.allclose(filtered["ts"], np.array([1.0, 2.0]))

    chunked_filtered = buffer.filter_topic("axis", lambda data, ts: ts != 1.0, chunk_size=1)
    assert np.allclose(chunked_filtered["ts"], np.array([0.0, 2.0]))

    reduced = buffer.reduce_topic("axis", lambda acc, data: acc + data, initial=np.zeros(1))
    assert np.allclose(reduced, np.array([3.0]))

    chunked_reduced = buffer.reduce_topic("axis", lambda acc, data: acc + data, initial=np.zeros(1), chunk_size=2)
    assert np.allclose(chunked_reduced, np.array([3.0]))

    chunks = list(buffer.iter_topic_chunks("axis", chunk_size=2))
    assert [chunk.metadata.count for chunk in chunks] == [2, 1]
    assert [chunk.metadata.topic for chunk in chunks] == ["axis", "axis"]

    windows = list(buffer.window_topic("axis", size=2))
    assert [window["ts"].tolist() for window in windows] == [[0.0], [0.0, 1.0], [1.0, 2.0]]

    dataset = buffer.dataset().frame_id("map").time_range(1.0, 2.0).collect(chunk_size=1)
    assert list(dataset) == ["axis"]
    assert np.allclose(dataset["axis"]["ts"], np.array([1.0, 2.0]))


def test_data_buffer_topic_pipeline_streams_without_get_buffer():
    buffer = DataBuffer(_SmallSource(), buffer_depth=3, axis="axis", preload=True)

    def fail_get_buffer(*args, **kwargs):
        raise AssertionError("lazy topic pipeline should not call get_buffer")

    buffer.get_buffer = fail_get_buffer
    calls = []
    pipeline = buffer.topic("axis").time_range(1.0, 2.0).map(
        lambda data, ts: calls.append(ts) or data + 1.0
    )

    assert calls == []
    collected = pipeline.collect(chunk_size=1)
    assert calls == [1.0, 2.0]
    assert np.allclose(collected["ts"], np.array([1.0, 2.0]))
    assert np.allclose(collected["data"], np.array([[2.0], [3.0]]))


def test_geometry_and_point_cloud_operations():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.1, 1.0, 0.0],
        [10.0, 10.0, 10.0],
    ])

    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    transformed = apply_transform(points, transform)
    assert np.allclose(transformed[0], np.array([1.0, 2.0, 3.0]))

    vectors = transform_vectors(np.array([[1.0, 0.0, 0.0]]), transform)
    assert np.allclose(vectors, np.array([[1.0, 0.0, 0.0]]))

    cropped, mask = crop_bounds(points, min_bound=[0, 0, -1], max_bound=[1.2, 1.2, 1], return_mask=True)
    assert cropped.shape == (4, 3)
    assert mask.tolist() == [True, True, True, True, False]

    downsampled = voxel_downsample(points[:4], voxel_size=0.5)
    assert downsampled.shape == (2, 3)
    assert np.allclose(downsampled, np.array([[0.05, 0.0, 0.0], [1.05, 1.0, 0.0]]))

    distances, indices = knn_search(points, np.array([0.0, 0.0, 0.0]), k=2)
    assert indices.shape == (1, 2)
    assert np.allclose(distances[0, 0], 0.0)

    normals = estimate_normals(points[:4], k=3, orient_toward=np.array([0.0, 0.0, 1.0]))
    assert np.allclose(normals[:, 2], np.ones(4))

    filtered, stat_mask = statistical_outlier_filter(points, k=2, std_ratio=1.0, return_mask=True)
    assert filtered.shape[0] == 4
    assert stat_mask.tolist() == [True, True, True, True, False]

    radius_filtered, radius_mask = radius_outlier_filter(points, radius=0.2, min_neighbors=1, return_mask=True)
    assert radius_filtered.shape[0] == 4
    assert radius_mask.tolist() == [True, True, True, True, False]

    labels = cluster_dbscan(points, eps=0.25, min_points=2)
    assert labels.tolist() == [0, 0, 1, 1, -1]

    plane, inliers = segment_plane(points[:4], distance_threshold=1.0e-6, iterations=10, seed=1)
    assert np.isclose(abs(plane[2]), 1.0)
    assert inliers.all()


def test_coordinate_frame_transforms_for_poses_navsat_odometry_and_dem():
    theta = np.pi / 2.0
    transform = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 1.0],
        [np.sin(theta), np.cos(theta), 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    poses = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    transformed_poses = transform_poses(poses, transform)
    assert np.allclose(transformed_poses[0, :3], np.array([1.0, 3.0, 3.0]))
    assert np.allclose(transformed_poses[0, 3:7], np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]))

    odom = np.zeros((8, 4), dtype=np.float64)
    odom[0] = [1.0, 0.0, 0.0, 0.0]
    odom[1] = [1.0, 4.0, 9.0, 0.0]
    odom[2] = [0.0, 0.0, 0.0, 1.0]
    odom[4] = [1.0, 0.0, 0.0, 0.0]
    odom[5] = [1.0, 4.0, 9.0, 0.0]
    odom[6] = [0.0, 1.0, 0.0, 0.0]
    transformed_odom = transform_odometry(odom, transform)
    assert np.allclose(transformed_odom[0, :3], np.array([1.0, 3.0, 3.0]))
    assert np.allclose(transformed_odom[2, :4], np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]))
    assert np.allclose(transformed_odom[4, :3], np.array([0.0, 1.0, 0.0]))
    assert np.allclose(transformed_odom[6, :3], np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(transformed_odom[1, :3], np.array([4.0, 1.0, 9.0]))

    navsat = np.array([[37.0, -122.0, 10.0]])
    nav_transform = np.eye(4)
    nav_transform[:3, 3] = [10.0, 20.0, 5.0]
    transformed_navsat = transform_navsat(navsat, nav_transform, 37.0, -122.0, 10.0)
    transformed_enu = navsat_to_enu(
        transformed_navsat[:, 0],
        transformed_navsat[:, 1],
        transformed_navsat[:, 2],
        37.0,
        -122.0,
        10.0,
    )
    assert np.allclose(transformed_enu, np.array([[10.0, 20.0, 5.0]]))

    elevation = np.array([[1.0, 2.0], [3.0, 4.0]])
    dem_points = dem_grid_to_points(elevation, resolution=2.0, origin=(10.0, 20.0))
    assert np.allclose(dem_points[1, 1], np.array([12.0, 22.0, 4.0]))
    transformed_dem = transform_dem_grid(elevation, nav_transform, resolution=2.0, origin=(10.0, 20.0))
    assert transformed_dem.shape == (2, 2, 3)
    assert np.allclose(transformed_dem[0, 0], np.array([20.0, 40.0, 6.0]))


def test_image_depth_operations():
    image = np.array([[0, 5], [10, 15]], dtype=np.uint8)
    normalized = normalize_image(image)
    assert normalized.dtype == np.float32
    assert np.allclose(normalized, np.array([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=np.float32))

    padded = pad_image(image, 1, value=9)
    assert padded.shape == (4, 4)
    assert padded[0, 0] == 9

    resized = resize_nearest(image, (3, 3))
    assert resized.shape == (3, 3)
    assert resized[0, 0] == 0
    assert resized[-1, -1] == 15

    rgb = np.dstack([image, image, image])
    gray = rgb_to_gray(rgb)
    assert np.allclose(gray, image)

    depth = np.array([[1.0, 0.0], [2.0, np.nan]])
    mask = valid_depth_mask(depth)
    assert mask.tolist() == [[True, False], [True, False]]

    points = depth_to_points(depth, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert np.allclose(points, np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0]]))


def test_navigation_operations():
    q = normalize_quaternion(np.array([0.0, 0.0, 0.0, 2.0]))
    assert np.allclose(q, np.array([0.0, 0.0, 0.0, 1.0]))

    halfway = slerp(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 0.0]), 0.5)
    assert np.isclose(np.linalg.norm(halfway), 1.0)
    assert np.allclose(abs(halfway[2]), np.sqrt(0.5))

    interp = interpolate_timeseries(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]]),
        np.array([0.5, 1.5]),
    )
    assert np.allclose(interp, np.array([[5.0, 10.0], [15.0, 30.0]]))

    enu = navsat_to_enu(np.array([37.0001]), np.array([-122.0001]), np.array([12.0]), 37.0, -122.0, 10.0)
    llh = enu_to_navsat(enu, 37.0, -122.0, 10.0)
    assert np.allclose(llh, np.array([[37.0001, -122.0001, 12.0]]))

    speed = trajectory_speed(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]))
    assert np.allclose(speed, np.array([1.0, 1.5, 2.0]))


def test_dem_raster_operations():
    tile_a = np.array([[1, 2], [3, 4]])
    tile_b = np.array([[5, 6], [7, 8]])
    mosaic = mosaic_tiles({(0, 0): tile_a, (0, 1): tile_b})
    assert np.array_equal(mosaic, np.array([[1, 2, 5, 6], [3, 4, 7, 8]]))

    cropped = crop_raster(mosaic, 0, 2, 1, 3)
    assert np.array_equal(cropped, np.array([[2, 5], [4, 7]]))

    elevation = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    slope, aspect = slope_aspect(elevation, resolution=1.0)
    assert np.allclose(slope, np.full((3, 3), np.arctan(1.0)))
    assert np.allclose(aspect, np.full((3, 3), -np.pi / 2.0))

    shaded = hillshade(elevation)
    assert shaded.shape == elevation.shape
    assert np.all((shaded >= 0.0) & (shaded <= 1.0))

    samples = sample_grid(mosaic, np.array([0.5]), np.array([0.5]))
    assert np.allclose(samples, np.array([2.5]))
