import sys
import types

import numpy as np

from ade.buffer import DataBuffer
from ade.ops import (
    align_bounded,
    align_exact,
    align_image,
    align_images,
    align_nearest,
    align_topic,
    angular_velocity_from_quaternions,
    add_trajectory_quality_mask,
    apply_image_mask,
    apply_transform,
    backproject_pixels,
    bounds_mask,
    CameraModel,
    camera_matrix,
    camera_model,
    cluster_dbscan,
    close_mask,
    connected_components,
    convert_color,
    convert_image_dtype,
    colorize_points,
    compensate_gravity,
    compensate_imu_gravity,
    correct_bias,
    correct_imu_bias,
    crop_camera_matrix,
    crop_bounds,
    crop_geographic_bounds,
    crop_image,
    crop_images,
    crop_oriented_bounds,
    crop_raster,
    curvature_descriptors,
    dataset_query,
    dead_reckon_trajectory,
    dem_grid_to_points,
    depth_to_normals,
    depth_to_point_grid,
    depth_to_points,
    DatasetQuery,
    differentiate_timeseries,
    differentiate_trajectory,
    dilate_mask,
    distort_normalized_points,
    distort_pixels,
    enu_to_navsat,
    enu_to_ned,
    erode_mask,
    estimate_bias,
    estimate_image_shift,
    estimate_normals,
    euler_to_quaternion,
    farthest_point_downsample,
    filter_topic,
    FrameGraph,
    frame_optical_flow,
    frame_to_frame_optical_flow,
    from_open3d_point_cloud,
    fuse_rgbd_frames,
    geographic_bounds_mask,
    hillshade,
    hybrid_search,
    image_gradients,
    image_mask,
    image_pyramid,
    iter_aligned_images,
    iter_frame_optical_flow,
    iter_motion_compensated_windows,
    iter_rgbd_frame_points,
    imu_to_trajectory,
    interpolate_quaternions,
    interpolate_timeseries,
    interpolate_trajectory,
    integrate_orientations,
    integrate_timeseries,
    integrate_trajectory,
    iter_chunks,
    knn_search,
    local_covariances,
    local_mean,
    local_statistics,
    local_std,
    local_to_navsat,
    map_topic,
    mask_trajectory,
    mosaic_tiles,
    motion_compensated_rolling_windows,
    multi_scale_icp,
    navsat_to_enu,
    navsat_to_local,
    navsat_to_ned,
    navsat_to_trajectory,
    ned_to_enu,
    ned_to_navsat,
    nearest_neighbor_distance_stats,
    nearest_neighbor_distances,
    normalized_points_to_pixels,
    normalize_image,
    normalize_images,
    normalize_quaternion,
    odometry_to_trajectory,
    odometry_seeded_icp,
    open_mask,
    oriented_bounds_mask,
    pad_image,
    pad_images,
    pixels_to_normalized_points,
    points_to_depth_image,
    point_to_plane_icp,
    point_to_point_icp,
    propagate_trajectory_covariance,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    project_camera_points,
    project_dem_to_image,
    project_points_to_image,
    random_downsample,
    radius_outlier_filter,
    reduce_topic,
    resample_imu,
    resample_navsat,
    resample_odometry,
    resample_topic,
    resample_trajectory,
    rectification_map,
    rectify_image,
    rolling_window_join,
    resize_images,
    resize_images_nearest,
    resize_nearest,
    rgbd_to_points,
    rgb_to_gray,
    rotate_vectors_by_quaternion,
    sample_grid,
    sample_image_at_pixels,
    sample_image_at_points,
    scale_camera_matrix,
    segment_ground,
    segment_plane,
    select_indices,
    select_mask,
    select_time_range,
    sensor_to_trajectory,
    slerp,
    smooth_timeseries,
    smooth_trajectory,
    slope_aspect,
    statistical_outlier_filter,
    threshold_image,
    TopicPipeline,
    TopicView,
    to_open3d_point_cloud,
    topic_pipeline,
    topic_view,
    translate_image,
    transform_dem_grid,
    transform_navsat,
    transform_odometry,
    transform_poses,
    transform_vectors,
    trajectory_quality_mask,
    trajectory_speed,
    uniform_downsample,
    undistort_normalized_points,
    undistort_pixels,
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

    xy_mask = bounds_mask(points, min_bound=[0.0, 0.0], max_bound=[1.0, 1.0], columns=(0, 1))
    assert xy_mask.tolist() == [True, True, True, False, False]

    selected_rows = select_mask(points, np.array([True, False, True, False, False]))
    assert np.allclose(selected_rows, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))

    image = np.arange(12).reshape(2, 2, 3)
    selected_pixels = select_mask(image, np.array([[True, False], [False, True]]))
    assert np.allclose(selected_pixels, np.array([[0, 1, 2], [9, 10, 11]]))

    theta = np.pi / 4.0
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    local_points = np.array([
        [0.0, 0.0, 0.0],
        [0.9, 0.4, 0.0],
        [1.1, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.0, 0.6],
    ])
    oriented_points = local_points @ rotation.T
    oriented_mask = oriented_bounds_mask(
        oriented_points,
        center=[0.0, 0.0, 0.0],
        extent=[2.0, 1.0, 1.0],
        rotation=rotation,
    )
    assert oriented_mask.tolist() == [True, True, False, False, False]
    oriented_crop = crop_oriented_bounds(
        oriented_points,
        center=[0.0, 0.0, 0.0],
        extent=[2.0, 1.0, 1.0],
        rotation=rotation,
    )
    assert np.allclose(oriented_crop, oriented_points[:2])

    downsampled = voxel_downsample(points[:4], voxel_size=0.5)
    assert downsampled.shape == (2, 3)
    assert np.allclose(downsampled, np.array([[0.05, 0.0, 0.0], [1.05, 1.0, 0.0]]))

    attributed_points = np.column_stack((np.arange(6, dtype=np.float64), np.zeros((6, 2)), np.arange(100, 106)))
    uniform, uniform_indices = uniform_downsample(attributed_points, every_k=2, return_indices=True)
    assert uniform_indices.tolist() == [0, 2, 4]
    assert np.allclose(uniform[:, 3], np.array([100.0, 102.0, 104.0]))

    random_sample, random_indices = random_downsample(attributed_points, count=3, seed=4, return_indices=True)
    expected_random = np.sort(np.random.default_rng(4).choice(attributed_points.shape[0], size=3, replace=False))
    assert random_indices.tolist() == expected_random.tolist()
    assert np.allclose(random_sample, attributed_points[expected_random])

    ratio_sample = random_downsample(attributed_points, ratio=0.5, seed=1)
    assert ratio_sample.shape == (3, 4)

    line = np.column_stack((np.arange(5, dtype=np.float64), np.zeros((5, 2))))
    farthest, farthest_indices = farthest_point_downsample(line, count=3, start_index=0, return_indices=True)
    assert farthest_indices.tolist() == [0, 4, 2]
    assert np.allclose(farthest[:, 0], np.array([0.0, 4.0, 2.0]))

    distances, indices = knn_search(points, np.array([0.0, 0.0, 0.0]), k=2)
    assert indices.shape == (1, 2)
    assert np.allclose(distances[0, 0], 0.0)

    hybrid_distances, hybrid_indices, hybrid_counts = hybrid_search(
        points,
        np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [100.0, 0.0, 0.0]]),
        radius=0.3,
        max_neighbors=2,
    )
    assert hybrid_counts.tolist() == [2, 1, 0]
    assert hybrid_indices[0].tolist() == [0, 1]
    assert np.allclose(hybrid_distances[0], np.array([0.0, 0.1]))
    assert hybrid_indices[1].tolist() == [4, -1]
    assert np.isinf(hybrid_distances[1, 1])
    assert hybrid_indices[2].tolist() == [-1, -1]

    normals = estimate_normals(points[:4], k=3, orient_toward=np.array([0.0, 0.0, 1.0]))
    assert np.allclose(normals[:, 2], np.ones(4))

    covariances, covariance_indices = local_covariances(points[:4], k=3, return_indices=True)
    assert covariances.shape == (4, 3, 3)
    assert covariance_indices.shape == (4, 3)
    assert np.allclose(covariances[:, 2, :], 0.0)
    assert np.allclose(covariances[:, :, 2], 0.0)

    descriptors = curvature_descriptors(points[:4], k=3)
    assert descriptors["eigenvalues"].shape == (4, 3)
    assert np.allclose(descriptors["curvature"], np.zeros(4))
    assert np.all(descriptors["linearity"] >= 0.0)
    assert np.all(descriptors["planarity"] >= 0.0)

    neighbor_distances = nearest_neighbor_distances(line, k=2)
    assert neighbor_distances.shape == (5, 2)
    assert np.allclose(neighbor_distances[0], np.array([1.0, 2.0]))
    assert np.allclose(neighbor_distances[2], np.array([1.0, 1.0]))

    distance_stats = nearest_neighbor_distance_stats(line, k=2)
    assert np.allclose(distance_stats["per_point_mean"][0], 1.5)
    assert np.allclose(distance_stats["per_point_mean"][2], 1.0)
    assert np.isclose(distance_stats["global_min"], 1.0)
    assert np.isclose(distance_stats["global_max"], 2.0)

    filtered, stat_mask = statistical_outlier_filter(points, k=2, std_ratio=1.0, return_mask=True)
    assert filtered.shape[0] == 4
    assert stat_mask.tolist() == [True, True, True, True, False]

    radius_filtered, radius_mask = radius_outlier_filter(points, radius=0.2, min_neighbors=1, return_mask=True)
    assert radius_filtered.shape[0] == 4
    assert radius_mask.tolist() == [True, True, True, True, False]

    labels = cluster_dbscan(points, eps=0.25, min_points=2)
    assert labels.tolist() == [0, 0, 1, 1, -1]

    component_points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    component_labels, component_counts = connected_components(component_points, radius=0.2, return_counts=True)
    assert component_labels.tolist() == [0, 0, 1, 1, 2]
    assert component_counts.tolist() == [2, 2, 1]
    filtered_components = connected_components(component_points, radius=0.2, min_component_size=2)
    assert filtered_components.tolist() == [0, 0, 1, 1, -1]

    plane, inliers = segment_plane(points[:4], distance_threshold=1.0e-6, iterations=10, seed=1)
    assert np.isclose(abs(plane[2]), 1.0)
    assert inliers.all()

    scene = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.02],
        [0.0, 2.0, -0.01],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.2],
    ])
    ground, non_ground, ground_mask, ground_plane = segment_ground(
        scene,
        distance_threshold=0.05,
        iterations=50,
        seed=3,
        return_plane=True,
    )
    assert ground.shape[0] == 6
    assert non_ground.shape[0] == 2
    assert ground_mask.tolist() == [True, True, True, True, True, True, False, False]
    assert ground_plane[2] > 0.0


def test_point_cloud_registration_helpers():
    source = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 2.0, 0.0],
        [2.0, 0.0, 2.0],
    ])
    translation = np.array([0.25, -0.2, 0.1])
    target = source + translation

    point_result = point_to_point_icp(
        source,
        target,
        max_iterations=10,
        max_correspondence_distance=1.0,
        return_correspondences=True,
    )
    assert np.allclose(point_result["transform"][:3, 3], translation)
    assert np.allclose(point_result["transform"][:3, :3], np.eye(3))
    assert point_result["correspondence_count"] == source.shape[0]
    assert point_result["inlier_rmse"] < 1.0e-12
    assert point_result["correspondences"]["source_indices"].shape == (source.shape[0],)

    plane_target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    plane_source = plane_target + np.array([0.0, 0.0, 0.25])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (plane_target.shape[0], 1))
    plane_result = point_to_plane_icp(
        plane_source,
        plane_target,
        target_normals=normals,
        max_iterations=10,
        max_correspondence_distance=1.0,
        min_correspondences=3,
    )
    assert np.allclose(plane_result["transform"][:3, 3], np.array([0.0, 0.0, -0.25]))
    assert plane_result["inlier_rmse"] < 1.0e-12

    multi_result = multi_scale_icp(
        source,
        target,
        voxel_sizes=(0.5, 0.0),
        max_iterations=(5, 5),
        max_correspondence_distances=(1.0, 1.0),
    )
    assert np.allclose(multi_result["transform"][:3, 3], translation)
    assert len(multi_result["levels"]) == 2

    source_pose = np.eye(4)
    source_pose[:3, 3] = translation
    target_pose = np.eye(4)
    odom_result = odometry_seeded_icp(
        source,
        target,
        source_pose=source_pose,
        target_pose=target_pose,
        max_iterations=3,
        max_correspondence_distance=1.0,
    )
    assert np.allclose(odom_result["odometry_seed"], source_pose)
    assert np.allclose(odom_result["transform"][:3, 3], translation)


def test_open3d_point_cloud_adapters_use_optional_dependency():
    class FakePointCloud:
        def __init__(self):
            self.points = np.empty((0, 3), dtype=np.float64)
            self.colors = np.empty((0, 3), dtype=np.float64)
            self.normals = np.empty((0, 3), dtype=np.float64)

    fake_open3d = types.SimpleNamespace(
        geometry=types.SimpleNamespace(PointCloud=FakePointCloud),
        utility=types.SimpleNamespace(Vector3dVector=lambda values: np.asarray(values, dtype=np.float64).copy()),
    )
    sentinel = object()
    previous = sys.modules.get("open3d", sentinel)
    sys.modules["open3d"] = fake_open3d
    try:
        points = np.array([
            [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 128.0, 255.0, 0.0, 1.0, 0.0],
        ])
        point_cloud = to_open3d_point_cloud(
            points,
            color_columns=(3, 4, 5),
            normal_columns=(6, 7, 8),
        )

        assert np.allclose(point_cloud.points, points[:, :3])
        assert np.allclose(point_cloud.colors, np.array([[1.0, 0.0, 0.0], [0.0, 128.0 / 255.0, 1.0]]))
        assert np.allclose(point_cloud.normals, points[:, 6:9])

        combined = from_open3d_point_cloud(point_cloud)
        assert combined.shape == (2, 9)
        assert np.allclose(combined[:, :3], points[:, :3])
        assert np.allclose(combined[:, 3:6], point_cloud.colors)
        assert np.allclose(combined[:, 6:9], point_cloud.normals)

        separated = from_open3d_point_cloud(point_cloud, as_dict=True)
        assert set(separated) == {"points", "colors", "normals"}
        assert np.allclose(separated["points"], points[:, :3])
    finally:
        if previous is sentinel:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = previous


def test_geographic_crop_and_bounds_masks():
    navsat = np.array([
        [37.0, -122.0, 10.0],
        [36.8, -122.0, 11.0],
        [37.1, -121.8, 12.0],
        [37.0, 170.0, 0.0],
        [37.0, -175.0, 0.0],
        [np.nan, -122.0, 0.0],
    ])

    mask = geographic_bounds_mask(navsat, min_lat=36.9, min_lon=-122.2, max_lat=37.2, max_lon=-121.9)
    assert mask.tolist() == [True, False, False, False, False, False]

    cropped, returned_mask = crop_geographic_bounds(
        navsat,
        min_lat=36.9,
        min_lon=-122.2,
        max_lat=37.2,
        max_lon=-121.9,
        return_mask=True,
    )
    assert np.allclose(cropped, np.array([[37.0, -122.0, 10.0]]))
    assert returned_mask.tolist() == mask.tolist()

    wrapped = crop_geographic_bounds(navsat, min_lat=36.0, min_lon=170.0, max_lat=38.0, max_lon=-170.0)
    assert np.allclose(wrapped, np.array([[37.0, 170.0, 0.0], [37.0, -175.0, 0.0]]))


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


def test_frame_graph_static_and_time_varying_transforms():
    graph = FrameGraph()
    base_to_odom = np.eye(4)
    base_to_odom[:3, 3] = [1.0, 0.0, 0.0]
    odom_to_map = np.eye(4)
    odom_to_map[:3, 3] = [0.0, 2.0, 0.0]

    graph.add_static_transform("base", "odom", base_to_odom)
    graph.add_static_transform("odom", "map", odom_to_map)

    base_to_map = graph.lookup_transform("base", "map")
    assert np.allclose(base_to_map[:3, 3], np.array([1.0, 2.0, 0.0]))
    assert np.allclose(graph.transform_points(np.array([[0.0, 0.0, 0.0]]), "base", "map"), np.array([[1.0, 2.0, 0.0]]))
    assert np.allclose(graph.transform_points(np.array([[1.0, 2.0, 0.0]]), "map", "base"), np.array([[0.0, 0.0, 0.0]]))
    assert graph.has_frame("odom")

    theta = np.pi / 2.0
    sensor_to_base_start = np.eye(4)
    sensor_to_base_end = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 10.0],
        [np.sin(theta), np.cos(theta), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    graph.add_time_varying_transform(
        "sensor",
        "base",
        timestamps=np.array([0.0, 10.0]),
        transforms=np.stack([sensor_to_base_start, sensor_to_base_end]),
    )

    midpoint = graph.lookup_transform("sensor", "base", timestamp=5.0)
    assert np.allclose(midpoint[:3, 3], np.array([5.0, 0.0, 0.0]))
    sensor_point = np.array([[1.0, 0.0, 0.0]])
    base_point = graph.transform_points(sensor_point, "sensor", "base", timestamp=5.0)
    assert np.allclose(base_point, np.array([[5.0 + np.sqrt(0.5), np.sqrt(0.5), 0.0]]))

    map_point = graph.transform_points(sensor_point, "sensor", "map", timestamp=5.0)
    assert np.allclose(map_point, np.array([[6.0 + np.sqrt(0.5), 2.0 + np.sqrt(0.5), 0.0]]))
    assert np.allclose(graph.transform_points(base_point, "base", "sensor", timestamp=5.0), sensor_point)

    try:
        graph.lookup_transform("sensor", "base")
    except ValueError as exc:
        assert "timestamp is required" in str(exc)
    else:
        raise AssertionError("dynamic frame lookup should require timestamp")


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

    assert np.allclose(crop_image(image, 0, 2, 1, 2), np.array([[5], [15]], dtype=np.uint8))

    sequence = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    resized_sequence = resize_images_nearest(sequence, (3, 2))
    assert resized_sequence.shape == (2, 3, 2)
    assert np.allclose(resized_sequence[:, -1], sequence[:, -1][:, [0, 2]])
    assert np.allclose(resize_images(sequence, (3, 2)), resized_sequence)

    cropped_sequence = crop_images(sequence, 0, 2, 1, 3)
    assert np.allclose(cropped_sequence, sequence[:, :, 1:3])

    padded_sequence = pad_images(sequence, ((1, 0), (0, 1)), value=99)
    assert padded_sequence.shape == (2, 3, 4)
    assert np.all(padded_sequence[:, 0, :] == 99)
    assert np.all(padded_sequence[:, :, -1] == 99)

    normalized_sequence = normalize_images(sequence, per_image=True)
    assert normalized_sequence.dtype == np.float32
    assert np.isclose(normalized_sequence[0].min(), 0.0)
    assert np.isclose(normalized_sequence[0].max(), 1.0)
    assert np.isclose(normalized_sequence[1].min(), 0.0)
    assert np.isclose(normalized_sequence[1].max(), 1.0)

    rgb = np.dstack([image, image, image])
    gray = rgb_to_gray(rgb)
    assert np.allclose(gray, image)

    rgb_sequence = np.stack((rgb, rgb + 1), axis=0)
    gray_sequence = convert_color(rgb_sequence, "rgb_to_gray")
    assert gray_sequence.shape == (2, 2, 2)
    assert np.allclose(gray_sequence[0], image)

    bgr_sequence = convert_color(rgb_sequence, "rgb_to_bgr")
    assert np.allclose(bgr_sequence[..., 0], rgb_sequence[..., 2])
    assert np.allclose(bgr_sequence[..., 2], rgb_sequence[..., 0])

    rgba_sequence = convert_color(rgb_sequence, "rgb_to_rgba")
    assert rgba_sequence.shape == (2, 2, 2, 4)
    assert np.all(rgba_sequence[..., 3] == 255)

    gray_rgb = convert_color(image, "gray_to_rgb")
    assert gray_rgb.shape == (2, 2, 3)
    assert np.allclose(gray_rgb[..., 0], image)

    float_image = convert_image_dtype(image, np.float32)
    assert float_image.dtype == np.float32
    assert np.allclose(float_image, image.astype(np.float32) / 255.0)
    assert np.array_equal(convert_image_dtype(float_image, np.uint8), image)

    bounded_mask = image_mask(image, min_value=5, max_value=10)
    assert bounded_mask.tolist() == [[False, True], [True, False]]
    masked_image = apply_image_mask(image, bounded_mask, fill_value=99)
    assert np.array_equal(masked_image, np.array([[99, 5], [10, 99]], dtype=np.uint8))

    thresholded = threshold_image(image, threshold=5, high=255, low=0, dtype=np.uint8)
    assert np.array_equal(thresholded, np.array([[0, 255], [255, 255]], dtype=np.uint8))

    single_pixel = np.zeros((3, 3), dtype=bool)
    single_pixel[1, 1] = True
    assert dilate_mask(single_pixel, size=3).sum() == 9
    assert not open_mask(single_pixel, size=3).any()

    hole = np.ones((5, 5), dtype=bool)
    hole[2, 2] = False
    assert close_mask(hole, size=3)[2, 2]
    assert erode_mask(np.ones((3, 3), dtype=bool), size=3).sum() == 1

    ramp = np.tile(np.arange(5, dtype=np.float64), (5, 1))
    gradients = image_gradients(ramp, method="sobel")
    assert np.isclose(gradients["dx"][2, 2], 1.0)
    assert np.isclose(gradients["dy"][2, 2], 0.0)
    assert np.isclose(gradients["magnitude"][2, 2], 1.0)

    pyramid = image_pyramid(np.arange(16).reshape(4, 4), levels=3)
    assert [level.shape for level in pyramid] == [(4, 4), (2, 2), (1, 1)]

    stats_image = np.arange(9, dtype=np.float64).reshape(3, 3)
    stats = local_statistics(stats_image, size=3, statistics=("mean", "std", "min", "max"))
    assert np.isclose(stats["mean"][1, 1], 4.0)
    assert np.isclose(stats["std"][1, 1], np.std(stats_image))
    assert stats["min"][1, 1] == 0.0
    assert stats["max"][1, 1] == 8.0
    assert np.allclose(local_mean(stats_image, size=3), stats["mean"])
    assert np.allclose(local_std(stats_image, size=3), stats["std"])

    depth = np.array([[1.0, 0.0], [2.0, np.nan]])
    mask = valid_depth_mask(depth)
    assert mask.tolist() == [[True, False], [True, False]]

    points = depth_to_points(depth, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert np.allclose(points, np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0]]))

    grid = depth_to_point_grid(np.ones((3, 3)), fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    assert grid.shape == (3, 3, 3)
    assert np.allclose(grid[1, 1], np.array([0.0, 0.0, 1.0]))

    normals = depth_to_normals(np.ones((3, 3)), fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    assert normals.shape == (3, 3, 3)
    assert np.allclose(normals[1, 1], np.array([0.0, 0.0, -1.0]))
    assert np.isnan(normals[0, 0]).all()

    invalid_center = np.ones((3, 3))
    invalid_center[1, 1] = 0.0
    invalid_normals = depth_to_normals(invalid_center, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    assert np.isnan(invalid_normals[1, 1]).all()


def test_image_motion_operations():
    base = np.zeros((24, 24), dtype=np.float64)
    base[8:12, 9:14] = 1.0
    base[15, 7] = 2.0
    base[6, 16] = 3.0
    shifted = translate_image(base, (2, -3))
    shifted_again = translate_image(base, (3, -2))

    shift = estimate_image_shift(base, shifted, max_shift=5)
    assert np.allclose(shift, np.array([2.0, -3.0]))

    flow = frame_optical_flow(base, shifted, max_shift=5)
    assert flow.shape == (24, 24, 2)
    assert np.allclose(flow[..., 0], 2.0)
    assert np.allclose(flow[..., 1], -3.0)

    sequence = np.stack([base, shifted, shifted_again])
    flows = frame_to_frame_optical_flow(sequence, max_shift=5)
    assert flows.shape == (2, 24, 24, 2)
    assert np.allclose(flows[0, ..., 0], 2.0)
    assert np.allclose(flows[0, ..., 1], -3.0)
    assert np.allclose(flows[1, ..., 0], 1.0)
    assert np.allclose(flows[1, ..., 1], 1.0)

    streamed_flows = list(iter_frame_optical_flow(sequence, max_shift=5))
    assert len(streamed_flows) == 2
    assert np.allclose(streamed_flows[1][0, 0], np.array([1.0, 1.0]))

    aligned_single, single_shift = align_image(shifted, reference=base, max_shift=5, return_shift=True)
    assert np.allclose(single_shift, np.array([2.0, -3.0]))
    assert np.allclose(aligned_single, base)

    aligned, shifts = align_images(sequence, max_shift=5, return_shifts=True)
    assert np.allclose(shifts, np.array([[0.0, 0.0], [2.0, -3.0], [3.0, -2.0]]))
    assert np.allclose(aligned, np.stack([base, base, base]))

    streamed_aligned = list(iter_aligned_images(sequence, max_shift=5))
    assert len(streamed_aligned) == 3
    assert np.allclose(streamed_aligned[-1], base)

    windows = list(iter_motion_compensated_windows(sequence, window_size=2, max_shift=5, return_shifts=True))
    assert len(windows) == 3
    first_window, first_shifts = windows[0]
    assert first_window.shape == (1, 24, 24)
    assert np.allclose(first_shifts, np.array([[0.0, 0.0]]))
    second_window, second_shifts = windows[1]
    assert second_window.shape == (2, 24, 24)
    assert np.allclose(second_shifts, np.array([[-2.0, 3.0], [0.0, 0.0]]))
    assert np.allclose(second_window[0], shifted)
    assert np.allclose(second_window[1], shifted)

    lazy_windows = motion_compensated_rolling_windows(sequence, window_size=3, max_shift=5, min_periods=2)
    collected = list(lazy_windows)
    assert [window.shape for window in collected] == [(2, 24, 24), (3, 24, 24)]
    assert np.allclose(collected[-1][-1], shifted_again)


def test_camera_model_utilities():
    intrinsics = camera_matrix(fx=2.0, fy=4.0, cx=1.0, cy=2.0)
    model = camera_model(
        intrinsics=intrinsics,
        image_shape=(5, 6),
        distortion=np.array([0.05, -0.01, 0.001, -0.002, 0.003]),
        rectification=np.eye(3),
    )
    assert isinstance(model, CameraModel)
    assert model.image_shape == (5, 6)
    assert model.distortion.shape == (8,)

    scaled = scale_camera_matrix(intrinsics, scale_x=2.0, scale_y=0.5)
    assert np.allclose(scaled, camera_matrix(fx=4.0, fy=2.0, cx=2.0, cy=1.0))
    cropped = crop_camera_matrix(scaled, row_offset=1.0, col_offset=2.0)
    assert np.allclose(cropped, camera_matrix(fx=4.0, fy=2.0, cx=0.0, cy=0.0))

    pixels = np.array([[1.5, 2.0], [2.0, 3.0], [0.5, 1.0]])
    normalized = pixels_to_normalized_points(pixels, intrinsics)
    assert np.allclose(normalized_points_to_pixels(normalized, intrinsics), pixels)

    coefficients = np.array([0.05, -0.01, 0.001, -0.002, 0.003])
    distorted_normalized = distort_normalized_points(normalized, coefficients)
    undistorted_normalized = undistort_normalized_points(distorted_normalized, coefficients, iterations=10)
    assert np.allclose(undistorted_normalized, normalized, atol=1e-8)

    distorted_pixels = distort_pixels(pixels, intrinsics, coefficients)
    undistorted_pixels = undistort_pixels(distorted_pixels, intrinsics, coefficients, iterations=10)
    assert np.allclose(undistorted_pixels, pixels, atol=1e-8)

    points = np.array([[0.0, 0.0, 1.0], [0.5, 0.25, 1.0], [0.0, 0.0, -1.0]])
    projected, depth, valid = project_camera_points(
        points,
        camera_model(intrinsics=intrinsics, image_shape=(5, 6), distortion=np.zeros(5)),
        return_depth=True,
    )
    assert np.allclose(projected[:2], np.array([[1.0, 2.0], [2.0, 3.0]]))
    assert np.allclose(depth, np.array([1.0, 1.0, -1.0]))
    assert valid.tolist() == [True, True, False]

    projected_with_distortion, distorted_valid = project_points_to_image(
        points[:2],
        camera_matrix=intrinsics,
        distortion=np.zeros(5),
        image_shape=(5, 6),
    )
    assert distorted_valid.tolist() == [True, True]
    assert np.allclose(projected_with_distortion, projected[:2])

    backprojected = backproject_pixels(projected[:2], depth[:2], intrinsics, distortion=np.zeros(5))
    assert np.allclose(backprojected, points[:2])

    image = np.arange(25, dtype=np.float64).reshape(5, 5)
    rect_map = rectification_map(image.shape, intrinsics, distortion=np.zeros(5), rectification=np.eye(3))
    assert rect_map.shape == (5, 5, 2)
    assert np.allclose(rect_map[2, 1], np.array([1.0, 2.0]))
    rectified = rectify_image(image, intrinsics, distortion=np.zeros(5), rectification=np.eye(3), bilinear=False)
    assert np.array_equal(rectified, image)


def test_projection_helpers_between_sensor_arrays():
    intrinsics = camera_matrix(fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    points = np.array([
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [0.0, 1.0, 2.0],
        [0.0, 0.0, -1.0],
    ])

    pixels, depth, mask = project_points_to_image(
        points,
        image_shape=(3, 3),
        camera_matrix=intrinsics,
        return_depth=True,
    )
    assert np.allclose(pixels[:3], np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]))
    assert np.allclose(depth, np.array([1.0, 1.0, 2.0, -1.0]))
    assert mask.tolist() == [True, True, True, False]

    gray = np.array([[0.0, 10.0], [20.0, 30.0]])
    interpolated = sample_image_at_pixels(gray, np.array([[0.5, 0.5]]))
    assert np.allclose(interpolated, np.array([15.0]))

    rgb = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    sampled, sampled_mask = sample_image_at_points(
        rgb,
        points,
        camera_matrix=intrinsics,
        bilinear=False,
        return_mask=True,
    )
    assert sampled_mask.tolist() == [True, True, True, False]
    assert np.allclose(sampled[0], rgb[1, 1])
    assert np.allclose(sampled[1], rgb[1, 2])
    assert np.allclose(sampled[2], rgb[2, 1])

    colorized, color_mask = colorize_points(
        points,
        rgb,
        camera_matrix=intrinsics,
        bilinear=False,
        return_mask=True,
    )
    assert color_mask.tolist() == [True, True, True, False]
    assert colorized.shape == (4, 6)
    assert np.allclose(colorized[1, :3], points[1])
    assert np.allclose(colorized[1, 3:], rgb[1, 2])
    assert np.isnan(colorized[3, 3:]).all()

    depth_image, index_image = points_to_depth_image(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]),
        image_shape=(3, 3),
        camera_matrix=intrinsics,
        return_indices=True,
    )
    assert np.allclose(depth_image[1, 1], 1.0)
    assert index_image[1, 1] == 1
    assert np.allclose(depth_image[1, 2], 1.0)

    aligned_depth = np.array([[1.0, 0.0], [2.0, 3.0]])
    aligned_rgb = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    rgbd_points = rgbd_to_points(aligned_depth, aligned_rgb, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert np.allclose(rgbd_points[:, :3], np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    assert np.allclose(rgbd_points[2, 3:], aligned_rgb[1, 1])

    depth_stack = np.stack([np.ones((2, 2)), np.full((2, 2), 2.0)])
    rgb_stack = np.stack([np.zeros((2, 2, 3)), np.full((2, 2, 3), 10.0)])
    translation = np.eye(4)
    translation[:3, 3] = np.array([10.0, 0.0, 0.0])
    fused_chunks = list(iter_rgbd_frame_points(
        depth_stack,
        rgb_stack,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        transforms=np.stack([np.eye(4), translation]),
    ))
    assert [chunk.shape for chunk in fused_chunks] == [(4, 6), (4, 6)]
    assert np.allclose(fused_chunks[1][0, :3], np.array([10.0, 0.0, 2.0]))

    fused = fuse_rgbd_frames(
        depth_stack,
        rgb_stack,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        transforms=np.stack([np.eye(4), translation]),
    )
    assert fused.shape == (8, 6)
    assert np.allclose(fused[0, :3], np.array([0.0, 0.0, 1.0]))
    assert np.allclose(fused[4, :3], np.array([10.0, 0.0, 2.0]))
    assert np.allclose(fused[:4, 3:], 0.0)
    assert np.allclose(fused[4:, 3:], 10.0)

    dem_pixels, dem_depth, dem_mask = project_dem_to_image(
        np.ones((2, 2), dtype=np.float64),
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        image_shape=(3, 3),
        return_depth=True,
    )
    assert dem_pixels.shape == (2, 2, 2)
    assert np.allclose(dem_pixels[1, 1], np.array([1.0, 1.0]))
    assert np.allclose(dem_depth, np.ones((2, 2)))
    assert dem_mask.all()


def test_navigation_operations():
    q = normalize_quaternion(np.array([0.0, 0.0, 0.0, 2.0]))
    assert np.allclose(q, np.array([0.0, 0.0, 0.0, 1.0]))

    halfway = slerp(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 0.0]), 0.5)
    assert np.isclose(np.linalg.norm(halfway), 1.0)
    assert np.allclose(abs(halfway[2]), np.sqrt(0.5))

    euler = np.array([0.1, -0.2, 0.3])
    quaternion = euler_to_quaternion(euler)
    assert np.allclose(quaternion_to_euler(quaternion), euler)
    yaw_quaternion = euler_to_quaternion(0.0, 0.0, 90.0, degrees=True)
    rotated = rotate_vectors_by_quaternion(np.array([1.0, 0.0, 0.0]), yaw_quaternion)
    assert np.allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)
    rotation = quaternion_to_rotation_matrix(yaw_quaternion)
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-12)

    stationary_accel = np.array([[0.0, 0.0, 9.80665]])
    compensated = compensate_gravity(stationary_accel, np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert np.allclose(compensated, np.zeros((1, 3)))

    biased = np.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    bias = estimate_bias(biased, axis=0)
    assert np.allclose(bias, np.array([3.0, 4.0, 5.0]))
    corrected, returned_bias = correct_bias(biased, return_bias=True)
    assert np.allclose(returned_bias, bias)
    assert np.allclose(corrected, np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))

    interp = interpolate_timeseries(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]]),
        np.array([0.5, 1.5]),
    )
    assert np.allclose(interp, np.array([[5.0, 10.0], [15.0, 30.0]]))

    enu = navsat_to_enu(np.array([37.0001]), np.array([-122.0001]), np.array([12.0]), 37.0, -122.0, 10.0)
    llh = enu_to_navsat(enu, 37.0, -122.0, 10.0)
    assert np.allclose(llh, np.array([[37.0001, -122.0001, 12.0]]))

    navsat_samples = np.array([
        [37.0, -122.0, 10.0],
        [
            37.0 + np.rad2deg(20.0 / 6378137.0),
            -122.0 + np.rad2deg(10.0 / (6378137.0 * np.cos(np.deg2rad(37.0)))),
            8.0,
        ],
    ])
    local_enu, reference = navsat_to_local(navsat_samples, frame="enu", return_reference=True)
    assert reference == {"lat": 37.0, "lon": -122.0, "alt": 10.0, "frame": "enu"}
    assert np.allclose(local_enu, np.array([[0.0, 0.0, 0.0], [10.0, 20.0, -2.0]]))
    local_ned = navsat_to_local({"data": navsat_samples}, frame="ned")
    assert np.allclose(local_ned, np.array([[0.0, 0.0, -0.0], [20.0, 10.0, 2.0]]))
    assert np.allclose(enu_to_ned(local_enu), local_ned)
    assert np.allclose(ned_to_enu(local_ned), local_enu)
    assert np.allclose(navsat_to_ned(navsat_samples[:, 0], navsat_samples[:, 1], navsat_samples[:, 2], 37.0, -122.0, 10.0), local_ned)
    assert np.allclose(local_to_navsat(local_enu, 37.0, -122.0, 10.0, frame="enu"), navsat_samples)
    assert np.allclose(ned_to_navsat(local_ned, 37.0, -122.0, 10.0), navsat_samples)
    assert np.allclose(local_to_navsat(local_ned, 37.0, -122.0, 10.0, frame="ned"), navsat_samples)

    speed = trajectory_speed(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]))
    assert np.allclose(speed, np.array([1.0, 1.5, 2.0]))

    orientations = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    interpolated_orientation = interpolate_quaternions(
        np.array([0.0, 1.0]),
        orientations,
        np.array([0.5]),
    )
    assert np.allclose(abs(interpolated_orientation[0, 2]), np.sqrt(0.5))
    assert np.allclose(abs(interpolated_orientation[0, 3]), np.sqrt(0.5))

    trajectory = {
        "ts": np.array([0.0, 1.0]),
        "position": np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]),
        "orientation": orientations,
        "linear_velocity": np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]),
        "angular_velocity": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        "linear_acceleration": np.array([[1.0, 1.0, 1.0], [3.0, 5.0, 7.0]]),
        "position_covariance": np.array([[0.0, 0.2, 0.4], [1.0, 1.2, 1.4]]),
        "orientation_covariance": np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]),
        "linear_velocity_covariance": np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]),
        "angular_velocity_covariance": np.array([[0.3, 0.3, 0.3], [0.5, 0.5, 0.5]]),
        "linear_acceleration_covariance": np.array([[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]]),
        "source": "odometry",
        "topic": "/odom",
        "frame_id": "odom",
    }
    interpolated = interpolate_trajectory(trajectory, np.array([0.0, 0.5, 1.0]))
    assert interpolated["topic"] == "/odom"
    assert interpolated["frame_id"] == "odom"
    assert np.allclose(interpolated["position"][1], np.array([5.0, 10.0, 15.0]))
    assert np.allclose(interpolated["linear_velocity"][1], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(interpolated["angular_velocity"][1], np.array([0.0, 0.0, 1.0]))
    assert np.allclose(interpolated["linear_acceleration"][1], np.array([2.0, 3.0, 4.0]))
    assert np.allclose(interpolated["position_covariance"][1], np.array([0.5, 0.7, 0.9]))
    assert np.allclose(abs(interpolated["orientation"][1, 2]), np.sqrt(0.5))
    fixed_rate = resample_trajectory(trajectory, period=0.5)
    assert np.allclose(fixed_rate["ts"], np.array([0.0, 0.5, 1.0]))

    smoothed = smooth_timeseries(np.array([[0.0], [9.0], [0.0]]), window_size=3)
    assert np.allclose(smoothed, np.array([[3.0], [3.0], [3.0]]))

    motion_ts = np.array([0.0, 1.0, 2.0])
    motion_position = np.column_stack((motion_ts ** 2, np.zeros((motion_ts.size, 2))))
    differentiated_position = differentiate_timeseries(motion_ts, motion_position)
    assert np.allclose(differentiated_position[:, 0], np.array([1.0, 2.0, 3.0]))

    constant_velocity = np.tile(np.array([1.0, 0.0, 0.0]), (motion_ts.size, 1))
    integrated_position = integrate_timeseries(motion_ts, constant_velocity, initial=np.zeros(3))
    assert np.allclose(integrated_position, np.column_stack((motion_ts, np.zeros((motion_ts.size, 2)))))

    yaw = np.array([0.0, 0.5, 1.0])
    motion_orientation = euler_to_quaternion(np.column_stack((
        np.zeros(motion_ts.size),
        np.zeros(motion_ts.size),
        yaw,
    )))
    angular_velocity = angular_velocity_from_quaternions(motion_ts, motion_orientation)
    assert np.allclose(angular_velocity, np.tile(np.array([0.0, 0.0, 0.5]), (motion_ts.size, 1)))
    integrated_orientation = integrate_orientations(
        motion_ts,
        angular_velocity,
        initial_orientation=motion_orientation[0],
    )
    assert np.allclose(quaternion_to_euler(integrated_orientation)[:, 2], yaw)

    trajectory_motion = {
        "ts": motion_ts,
        "position": motion_position,
        "orientation": motion_orientation,
        "linear_velocity": constant_velocity,
        "angular_velocity": angular_velocity,
        "linear_acceleration": np.zeros((motion_ts.size, 3)),
        "position_covariance": np.zeros((motion_ts.size, 3)),
        "orientation_covariance": np.zeros((motion_ts.size, 3)),
        "linear_velocity_covariance": np.zeros((motion_ts.size, 3)),
        "angular_velocity_covariance": np.zeros((motion_ts.size, 3)),
        "linear_acceleration_covariance": np.zeros((motion_ts.size, 3)),
        "source": "synthetic",
    }
    smoothed_trajectory = smooth_trajectory(
        {**trajectory_motion, "position": np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0]])},
        window_size=3,
        fields=("position",),
        smooth_orientation=False,
    )
    assert np.allclose(smoothed_trajectory["position"][:, 0], np.array([3.0, 3.0, 3.0]))

    differentiated_trajectory = differentiate_trajectory(trajectory_motion)
    assert np.allclose(differentiated_trajectory["linear_velocity"][:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(differentiated_trajectory["angular_velocity"][:, 2], np.array([0.5, 0.5, 0.5]))
    assert np.allclose(differentiated_trajectory["linear_acceleration"][:, 0], np.array([1.0, 1.0, 1.0]))

    dead_reckoned = dead_reckon_trajectory(trajectory_motion, initial_position=np.zeros(3))
    assert np.allclose(dead_reckoned["position"], np.column_stack((motion_ts, np.zeros((motion_ts.size, 2)))))
    assert np.allclose(quaternion_to_euler(dead_reckoned["orientation"])[:, 2], yaw)
    integrated_trajectory = integrate_trajectory(trajectory_motion, initial_position=np.zeros(3))
    assert np.allclose(integrated_trajectory["position"], dead_reckoned["position"])

    body_frame_input = {**trajectory_motion, "angular_velocity": np.zeros((motion_ts.size, 3))}
    body_frame = dead_reckon_trajectory(
        body_frame_input,
        initial_position=np.zeros(3),
        initial_orientation=euler_to_quaternion(0.0, 0.0, 90.0, degrees=True),
        body_frame_velocity=True,
    )
    assert np.allclose(body_frame["position"][:, 0], np.zeros(motion_ts.size), atol=1e-12)
    assert np.allclose(body_frame["position"][:, 1], motion_ts, atol=1e-12)

    propagated = propagate_trajectory_covariance(
        trajectory_motion,
        process_noise={"position": np.array([0.1, 0.2, 0.3]), "orientation_covariance": 0.01},
    )
    assert np.allclose(propagated["position_covariance"][2], np.array([0.2, 0.4, 0.6]))
    assert np.allclose(propagated["orientation_covariance"][2], np.array([0.02, 0.02, 0.02]))

    quality_input = {
        **trajectory_motion,
        "position_covariance": np.array([
            [0.1, 0.1, 0.1],
            [5.0, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ]),
        "status": np.array([0, 0, -1]),
    }
    quality_mask = trajectory_quality_mask(
        quality_input,
        covariance_limits={"position": 1.0},
        valid_statuses={0},
    )
    assert quality_mask.tolist() == [True, False, False]

    annotated = add_trajectory_quality_mask(
        quality_input,
        covariance_limits={"position": 1.0},
        valid_statuses={0},
    )
    assert annotated["quality_mask"].tolist() == [True, False, False]

    masked = mask_trajectory(quality_input, quality_mask)
    assert np.allclose(masked["ts"], motion_ts)
    assert np.isnan(masked["position"][1:]).all()
    assert masked["quality_mask"].tolist() == [True, False, False]

    dropped = mask_trajectory(quality_input, quality_mask, drop=True)
    assert np.allclose(dropped["ts"], np.array([0.0]))
    assert np.allclose(dropped["position"], quality_input["position"][:1])
    assert dropped["quality_mask"].tolist() == [True]


def test_sensor_streams_convert_to_common_trajectory_arrays():
    timestamps = np.array([0.0, 1.0])
    imu = np.zeros((2, 6, 4), dtype=np.float64)
    imu[:, 0] = np.array([0.0, 0.0, 0.0, 1.0])
    imu[:, 1, :3] = np.array([0.1, 0.2, 0.3])
    imu[:, 2, :3] = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    imu[:, 3, :3] = np.array([0.01, 0.02, 0.03])
    imu[:, 4, :3] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    imu[:, 5, :3] = np.array([0.4, 0.5, 0.6])

    imu_traj = imu_to_trajectory({"ts": timestamps, "data": imu, "topic": "/imu", "frame_id": "base_link"})
    assert imu_traj["source"] == "imu"
    assert imu_traj["topic"] == "/imu"
    assert imu_traj["frame_id"] == "base_link"
    assert imu_traj["pose"].shape == (2, 7)
    assert imu_traj["trajectory"].shape == (2, 13)
    assert np.isnan(imu_traj["position"]).all()
    assert np.allclose(imu_traj["orientation"], np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1)))
    assert np.allclose(imu_traj["angular_velocity"][:, 2], np.array([0.1, 0.2]))
    assert np.allclose(imu_traj["linear_acceleration"][1], np.array([4.0, 5.0, 6.0]))

    corrected_imu, imu_biases = correct_imu_bias({"ts": timestamps, "data": imu}, sample_slice=slice(0, 1), return_bias=True)
    assert np.allclose(imu_biases["angular_velocity"], np.array([0.0, 0.0, 0.1]))
    assert np.allclose(imu_biases["linear_acceleration"], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(corrected_imu["data"][1, 2, :3], np.array([0.0, 0.0, 0.1]))
    assert np.allclose(corrected_imu["data"][1, 4, :3], np.array([3.0, 3.0, 3.0]))

    gravity_imu = imu.copy()
    gravity_imu[:, 4, :3] = np.array([0.0, 0.0, 9.80665])
    gravity_corrected = compensate_imu_gravity(gravity_imu)
    assert np.allclose(gravity_corrected[:, 4, :3], np.zeros((2, 3)))

    odom = np.zeros((2, 8, 4), dtype=np.float64)
    odom[:, 0, :3] = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    odom[:, 1, :3] = np.array([0.1, 0.2, 0.3])
    odom[:, 2, :4] = np.array([0.0, 0.0, 0.0, 1.0])
    odom[:, 4, :3] = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    odom[:, 6, :3] = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    odom_traj = odometry_to_trajectory(odom, timestamps=timestamps)
    assert odom_traj["source"] == "odometry"
    assert np.allclose(odom_traj["position"][1], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(odom_traj["linear_velocity"][1], np.array([1.0, 1.0, 0.0]))
    assert np.allclose(odom_traj["trajectory"][1, :7], odom_traj["pose"][1])

    navsat = np.array([
        [37.0, -122.0, 10.0],
        [37.0 + np.rad2deg(20.0 / 6378137.0), -122.0, 12.0],
    ])
    nav_traj = navsat_to_trajectory(
        {
            "ts": timestamps,
            "data": navsat,
            "status": np.array([0, -1]),
            "position_covariance": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
        ref_lat=37.0,
        ref_lon=-122.0,
        ref_alt=10.0,
    )
    assert nav_traj["source"] == "navsat"
    assert nav_traj["reference"] == {"lat": 37.0, "lon": -122.0, "alt": 10.0}
    assert np.allclose(nav_traj["position"], np.array([[0.0, 0.0, 0.0], [0.0, 20.0, 2.0]]))
    assert np.allclose(nav_traj["linear_velocity"], np.array([[0.0, 20.0, 2.0], [0.0, 20.0, 2.0]]))
    assert nav_traj["status"].tolist() == [0, -1]
    assert np.allclose(nav_traj["position_covariance"][1], np.array([4.0, 5.0, 6.0]))

    dispatched = sensor_to_trajectory(odom, kind="odometry", timestamps=timestamps)
    assert np.allclose(dispatched["pose"], odom_traj["pose"])

    imu_resampled = resample_imu({"ts": timestamps, "data": imu}, target_timestamps=np.array([0.5]))
    assert np.allclose(imu_resampled["angular_velocity"][0, 2], 0.15)
    assert np.allclose(imu_resampled["linear_acceleration"][0], np.array([2.5, 3.5, 4.5]))
    assert np.allclose(imu_resampled["angular_velocity_covariance"][0], np.array([0.01, 0.02, 0.03]))

    odom_resampled = resample_odometry(odom, timestamps=timestamps, target_timestamps=np.array([0.5]))
    assert np.allclose(odom_resampled["position"][0], np.array([0.5, 1.0, 1.5]))
    assert np.allclose(odom_resampled["linear_velocity"][0], np.array([1.0, 0.5, 0.0]))
    assert np.allclose(odom_resampled["angular_velocity"][0, 2], 0.15)

    nav_resampled = resample_navsat(
        {"ts": timestamps, "data": navsat},
        target_timestamps=np.array([0.5]),
        ref_lat=37.0,
        ref_lon=-122.0,
        ref_alt=10.0,
    )
    assert np.allclose(nav_resampled["position"][0], np.array([0.0, 10.0, 1.0]))
    assert np.allclose(nav_resampled["linear_velocity"][0], np.array([0.0, 20.0, 2.0]))
    assert np.allclose(nav_resampled["navsat"][0, 2], 11.0)


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
