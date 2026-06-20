import numpy as np

from ade.buffer import DataBuffer
from ade.ops import (
    align_nearest,
    apply_transform,
    cluster_dbscan,
    crop_bounds,
    crop_raster,
    depth_to_points,
    enu_to_navsat,
    estimate_normals,
    filter_topic,
    hillshade,
    interpolate_timeseries,
    knn_search,
    map_topic,
    mosaic_tiles,
    navsat_to_enu,
    normalize_image,
    normalize_quaternion,
    pad_image,
    radius_outlier_filter,
    reduce_topic,
    resize_nearest,
    rgb_to_gray,
    sample_grid,
    segment_plane,
    select_indices,
    select_time_range,
    slerp,
    slope_aspect,
    statistical_outlier_filter,
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
            }


def test_data_buffer_operation_wrappers():
    buffer = DataBuffer(_SmallSource(), buffer_depth=3, axis="axis", preload=True)

    mapped = buffer.map_topic("axis", lambda data: data * 2)
    assert np.allclose(mapped["data"], np.array([[0.0], [2.0], [4.0]]))

    filtered = buffer.filter_topic("axis", lambda data, ts: ts >= 1.0)
    assert np.allclose(filtered["ts"], np.array([1.0, 2.0]))

    reduced = buffer.reduce_topic("axis", lambda acc, data: acc + data, initial=np.zeros(1))
    assert np.allclose(reduced, np.array([3.0]))

    windows = list(buffer.window_topic("axis", size=2))
    assert [window["ts"].tolist() for window in windows] == [[0.0], [0.0, 1.0], [1.0, 2.0]]


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
