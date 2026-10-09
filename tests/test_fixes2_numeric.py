"""Regression tests for the second numeric review round (ops: geometry, point
cloud, nav, DEM, image, ML)."""

from __future__ import annotations

import tracemalloc
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from arraydataengine import DataBuffer
from arraydataengine.ops import dem, image, ml, nav
from arraydataengine.ops import geometry as g
from arraydataengine.ops import point_cloud as pc
from arraydataengine.sources.synthetic_source import SyntheticSource


@pytest.fixture(params=["scipy", "numpy"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(pc, "_scipy_ckdtree", lambda: None)
    elif pc._scipy_ckdtree() is None:
        pytest.skip("SciPy not installed")
    return request.param


# --- ML -----------------------------------------------------------------------------


def test_torch_iterable_dataset_transforms_only_the_worker_share(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def transform(sample):
        calls.append(int(sample["x"][0]))
        return sample

    samples = [{"x": np.full(2, i, dtype=np.float32)} for i in range(40)]
    dataset = ml.to_torch_dataset(samples, transform=transform)
    seen = []
    for worker_id in range(4):
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda w=worker_id: SimpleNamespace(id=w, num_workers=4))
        calls.clear()
        rows = [int(sample["x"][0]) for sample in dataset]
        assert rows == list(range(worker_id, 40, 4))
        assert calls == rows  # transform ran on this worker's share only
        seen.extend(rows)
    assert sorted(seen) == list(range(40))

    # Transforms of a wrapped IteratorDataset also run after sharding.
    inner = ml.to_iterator_dataset(samples, transform=transform)
    calls.clear()
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=4))
    assert len(list(ml.to_torch_dataset(inner))) == 10
    assert len(calls) == 10


def test_generator_sources_fail_loudly_on_second_pass():
    def generate():
        for i in range(3):
            yield {"x": np.full(2, i)}

    dataset = ml.to_iterator_dataset(generate())
    assert len(list(dataset)) == 3
    with pytest.raises(RuntimeError, match="one-shot iterator"):
        list(dataset)
    # Factories and re-iterables stay re-iterable.
    factory = ml.to_iterator_dataset(generate)
    assert len(list(factory)) == len(list(factory)) == 3
    assert len(ml.to_numpy_dataset(generate())) == 3


# --- undistortion -------------------------------------------------------------------


def _radial_reference(rho, k1, k2):
    optimize = pytest.importorskip("scipy.optimize")
    return np.array([
        optimize.brentq(lambda r, d=d: r * (1 + k1 * r * r + k2 * r ** 4) - d, 0.0, 10.0, xtol=1e-15)
        for d in rho
    ])


def test_undistortion_converges_where_the_fixed_point_iteration_cycles():
    K = g.camera_matrix(300, 300, 320, 240)
    D = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])  # monotonic radial map: unique inverse
    assert g._radial_distortion_limit(g._distortion_coefficients(D)) is None
    corners = np.array([[0.0, 0.0], [639.0, 479.0], [600.0, 50.0], [320.0, 240.0]])
    undistorted = g.undistort_pixels(corners, K, D)
    assert np.abs(g.distort_pixels(undistorted, K, D) - corners).max() < 1e-6

    # Independent reference: a bracketing root finder on the radial equation.
    normalized = g.pixels_to_normalized_points(corners, K)
    rho = np.linalg.norm(normalized, axis=1)
    radius = _radial_reference(rho, -0.3, 0.1)
    expected = normalized * np.divide(radius, rho, out=np.ones_like(rho), where=rho > 0)[:, None]
    assert np.allclose(g.pixels_to_normalized_points(undistorted, K), expected, atol=1e-10)

    # Random points across the whole lens (with tangential terms) round-trip.
    rng = np.random.default_rng(0)
    radius = np.sqrt(rng.uniform(0, 1, 2000)) * 1.96
    angle = rng.uniform(0, 2 * np.pi, 2000)
    truth = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    D = np.array([-0.3, 0.1, 0.001, -0.001, 0.0])
    recovered = g.undistort_normalized_points(g.distort_normalized_points(truth, D), D)
    assert np.abs(recovered - truth).max() < 1e-9


def test_undistortion_matches_opencv_and_marks_unsolvable_points_nan():
    cv2 = pytest.importorskip("cv2")
    K = g.camera_matrix(500, 480, 320, 240)
    D = np.array([-0.28, 0.07, 0.001, -0.0015, 0.0])
    uv = np.stack(np.meshgrid(np.linspace(0, 639, 9), np.linspace(0, 479, 7)), -1).reshape(-1, 2)
    reference = cv2.undistortPoints(uv[:, None, :], K, D, P=K, criteria=(cv2.TERM_CRITERIA_COUNT, 300, 0))[:, 0]
    assert np.abs(g.undistort_pixels(uv, K, D) - reference).max() < 1e-6

    # k1 = -0.5 folds at r^2 = 2/3 (max distorted radius 0.544): beyond it
    # there is no solution; just inside it the result is exact.
    D = np.array([-0.5, 0.0, 0.0, 0.0])
    points = np.array([[0.6, 0.0], [0.54, 0.0], [0.0, 0.3]])
    result = g.undistort_normalized_points(points, D)
    assert np.isnan(result[0]).all()
    assert np.abs(g.distort_normalized_points(result[1:], D) - points[1:]).max() < 1e-12
    assert (np.sum(result[1:] ** 2, axis=1) <= 2.0 / 3.0).all()
    # Count-only iteration keeps OpenCV's semantics (no fallback).
    coarse = g.undistort_normalized_points(points[1:], D, iterations=1, tolerance=None)
    assert np.abs(coarse - result[1:]).max() > 1e-3


# --- loop closures / ICP ------------------------------------------------------------


def _lidar_scan(origin, seed, n_beams=16, step_deg=0.5, max_range=60.0):
    """Sensor-frame scan of a ground plane and box-room walls (|x|=40, |y|=25)."""

    elevation = np.deg2rad(np.linspace(-25, 5, n_beams))
    azimuth = np.deg2rad(np.arange(0, 360, step_deg))
    e, a = np.meshgrid(elevation, azimuth, indexing="ij")
    rays = np.stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)), -1).reshape(-1, 3)
    sensor = np.array([origin[0], origin[1], 1.8])
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = [np.where(rays[:, 2] < 0, -sensor[2] / rays[:, 2], np.inf)]
        for axis, bound in ((0, 40), (0, -40), (1, 25), (1, -25)):
            t = (bound - sensor[axis]) / rays[:, axis]
            hits.append(np.where(t > 0, t, np.inf))
    distance = np.min(hits, axis=0)
    points = sensor + rays * np.where(distance < max_range, distance, np.nan)[:, None]
    points = points[np.isfinite(points).all(axis=1)]
    points += np.random.default_rng(seed).normal(0, 0.01, points.shape)
    return points - sensor + [0.0, 0.0, 1.8]


def test_default_loop_closure_gate_accepts_drifted_loops_and_rejects_unrelated_scans():
    target = _lidar_scan((0.0, 0.0), seed=1)
    source = _lidar_scan((2.0, 1.0), seed=2)
    # Odometry drifted by (0.3, -0.2) m: ~4x the default scoring gate.
    trajectory = np.array([[0, 0, 0, 0, 0, 0, 1], [2.3, 0.8, 0, 0, 0, 0, 1]], float)
    assert 3 * pc._median_point_spacing(target) < 0.3
    result = pc.verify_loop_closures([target, source], trajectory, candidates=[(1, 0)], return_all=True)
    assert result["accepted"].tolist() == [True]
    assert np.linalg.norm(result["transform"][0][:3, 3] - [2.0, 1.0, 0.0]) < 0.1
    assert np.allclose(result["odometry_seed"][0][:3, 3], [2.3, 0.8, 0.0])

    blob = np.random.default_rng(3).normal(size=(3000, 3)) * [3, 3, 1] + [5, 0, 1]
    result = pc.verify_loop_closures([target, blob], trajectory, candidates=[(1, 0)], return_all=True)
    assert result["accepted"].tolist() == [False]

    # voxel_size=0 means "no voxel grid", exactly like None.
    plane = np.column_stack([np.random.default_rng(4).uniform(0, 10, (400, 2)), np.zeros(400)])
    poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (2, 1))
    none = pc.verify_loop_closures([plane, plane], poses, candidates=[(1, 0)], return_all=True)
    zero = pc.verify_loop_closures([plane, plane], poses, candidates=[(1, 0)], return_all=True, voxel_size=0)
    assert zero["accepted"].tolist() == none["accepted"].tolist() == [True]
    assert zero["fitness"][0] == none["fitness"][0] == 1.0


def test_icp_without_correspondences_reports_zero_iterations():
    plane = np.column_stack([np.random.default_rng(0).uniform(0, 5, (200, 2)), np.zeros(200)])
    far = plane + 100.0
    for icp in (pc.point_to_point_icp, pc.point_to_plane_icp):
        result = icp(far, plane, max_correspondence_distance=1.0)
        assert result["iterations"] == 0 and result["fitness"] == 0.0
    result = pc.multi_scale_icp(far, plane, voxel_sizes=(1.0, 0.0), max_correspondence_distances=(2.0, 1.0))
    assert result["iterations"] == 0
    aligned = pc.point_to_point_icp(plane + [0.05, 0, 0], plane, max_correspondence_distance=0.5, max_iterations=3)
    assert aligned["iterations"] >= 1


# --- neighbour searches with duplicates --------------------------------------------------


def _cloud_with_duplicates(seed=0):
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(4.0), np.arange(4.0), [0.0]), -1).reshape(-1, 3) * 0.5
    cloud = np.concatenate([
        rng.uniform(0, 3, (120, 3)),
        np.zeros((60, 3)),            # many exact duplicates
        grid, grid, grid[:5],         # equidistant ties and duplicated ties
        np.full((2, 3), np.nan),
    ])
    return cloud[rng.permutation(cloud.shape[0])]


def _distances(points, queries):
    with np.errstate(invalid="ignore"):
        return np.sqrt(((queries[:, None, :3] - points[None, :, :3]) ** 2).sum(axis=-1))


def _reference_dbscan(points, eps, min_points):
    d = _distances(points, points)
    neighbors = [np.flatnonzero(row <= eps) for row in d]
    labels = np.full(len(points), -1)
    visited = np.zeros(len(points), bool)
    cluster = 0
    for index in range(len(points)):
        if visited[index]:
            continue
        visited[index] = True
        if neighbors[index].size < min_points:
            continue
        labels[index] = cluster
        queue = deque(int(i) for i in neighbors[index] if i != index)
        while queue:
            current = queue.popleft()
            if not visited[current]:
                visited[current] = True
                if neighbors[current].size >= min_points:
                    queue.extend(int(i) for i in neighbors[current] if labels[i] < 0)
            if labels[current] < 0:
                labels[current] = cluster
        cluster += 1
    return labels


def _reference_components(points, radius):
    d = _distances(points, points)
    labels = np.full(len(points), -1)
    finite = np.isfinite(points).all(axis=1)
    component = 0
    for start in range(len(points)):
        if labels[start] >= 0 or not finite[start]:
            continue
        labels[start] = component
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in np.flatnonzero((d[current] <= radius) & (labels < 0)):
                labels[neighbor] = component
                queue.append(neighbor)
        component += 1
    return labels


def test_neighbor_ops_match_brute_force_with_duplicates_and_ties(backend):
    cloud = _cloud_with_duplicates()
    queries = np.concatenate([cloud[:40], np.zeros((3, 3)), [[0.25, 0.25, 0.0], [0.5, 0.5, 0.0]]])
    d = _distances(cloud, queries)
    for radius, k in ((0.0, 3), (0.5, 1), (0.5, 4), (0.71, 6), (2.0, 70)):
        distances, indices, counts = pc.hybrid_search(cloud, queries, radius, k)
        for row in range(len(queries)):
            inside = np.flatnonzero(d[row] <= radius)
            expected = inside[np.lexsort((inside, d[row, inside]))][:k]
            assert counts[row] == expected.size
            assert indices[row, :counts[row]].tolist() == expected.tolist()
            assert np.allclose(distances[row, :counts[row]], d[row, expected])

    finite = np.isfinite(cloud).all(axis=1)
    d_self = _distances(cloud, cloud)
    for radius, min_neighbors in ((0.0, 1), (0.3, 3), (0.5, 10)):
        _, mask = pc.radius_outlier_filter(cloud, radius, min_neighbors, return_mask=True)
        assert mask.tolist() == (finite & ((d_self <= radius).sum(axis=1) - 1 >= min_neighbors)).tolist()

    for eps, min_points in ((0.3, 4), (0.6, 10), (0.2, 70)):
        assert pc.cluster_dbscan(cloud, eps, min_points).tolist() == _reference_dbscan(cloud, eps, min_points).tolist()

    for radius in (0.0, 0.2, 0.5):
        labels, counts = pc.connected_components(cloud, radius, return_counts=True)
        assert labels.tolist() == _reference_components(cloud, radius).tolist()
        assert counts.tolist() == np.bincount(labels[labels >= 0]).tolist()
        filtered = pc.connected_components(cloud, radius, min_component_size=3)
        reference = _reference_components(cloud, radius)
        sizes = np.bincount(reference[reference >= 0])
        kept = np.flatnonzero(sizes >= 3)
        remap = np.full(sizes.size, -1)
        remap[kept] = np.arange(kept.size)
        assert filtered.tolist() == np.where(reference >= 0, remap[np.maximum(reference, 0)], -1).tolist()

    exact = pc.radius_search(cloud, queries, 0.0)
    for row, found in enumerate(exact):
        assert found.tolist() == np.flatnonzero(d[row] == 0.0).tolist()


def test_neighbor_ops_stay_memory_bounded_with_many_duplicates(backend):
    rng = np.random.default_rng(0)
    real = rng.uniform(1, 5, size=(800, 3))
    cloud = np.concatenate([real, np.zeros((20000, 3))])  # e.g. un-trimmed padding
    shifted = np.concatenate([real + 0.05, np.zeros((20000, 3))])
    tracemalloc.start()
    try:
        _, mask = pc.radius_outlier_filter(cloud, radius=0.5, min_neighbors=3, return_mask=True)
        labels = pc.cluster_dbscan(cloud, eps=0.3, min_points=5)
        components = pc.connected_components(cloud, radius=0.3)
        _, indices, counts = pc.hybrid_search(cloud, cloud[-5:], radius=0.5, max_neighbors=3)
        result = pc.point_to_point_icp(shifted, cloud, max_correspondence_distance=0.5, max_iterations=3)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 150 * 2**20
    assert mask[-20000:].all()
    assert (labels[-20000:] == labels[-1]).all() and labels[-1] >= 0
    assert (components[-20000:] == components[-1]).all()
    assert indices[:, :3].tolist() == [[800, 801, 802]] * 5 and (counts == 3).all()
    assert result["fitness"] > 0.9


# --- nav ------------------------------------------------------------------------------


def test_slerp_is_accurate_for_tiny_angles():
    rotation = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(0)
    worst = 0.0
    for angle in (1e-9, 1e-6, 1e-3, 0.03, 0.5, 3.0):
        for _ in range(20):
            start = rotation.Rotation.random(random_state=rng)
            axis = rng.normal(size=3)
            end = rotation.Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle) * start
            fraction = rng.uniform()
            expected = rotation.Slerp([0, 1], rotation.Rotation.concatenate([start, end]))([fraction]).as_quat()[0]
            q0, q1 = start.as_quat(), -end.as_quat()
            for result in (
                nav.slerp(q0, q1, fraction),
                g._slerp_quaternion(q0, q1, fraction),
                nav.interpolate_quaternions(np.array([0.0, 1.0]), np.stack([q0, q1]), np.array([fraction]))[0],
            ):
                result = result if np.dot(result, expected) >= 0 else -result
                worst = max(worst, float(np.abs(result - expected).max()))
    assert worst < 1e-12


def test_smooth_trajectory_aligns_signs_across_nan_rows():
    yaw = np.deg2rad(np.linspace(0, 20, 9))
    quaternions = np.stack([np.zeros(9), np.zeros(9), np.sin(yaw / 2), np.cos(yaw / 2)], -1)
    quaternions[5:] *= -1.0  # same rotations, opposite sign
    quaternions[4] = 0.0     # no orientation estimate
    imu = np.zeros((9, 6, 4))
    imu[:, 0] = quaternions
    trajectory = nav.imu_to_trajectory({"ts": np.arange(9.0), "data": imu})
    smoothed = nav.smooth_trajectory(trajectory, window_size=3)["orientation"]
    assert np.isnan(smoothed[4]).all()
    valid = np.delete(np.arange(9), 4)
    smoothed_yaw = 2 * np.arctan2(smoothed[valid, 2], smoothed[valid, 3])
    smoothed_yaw = np.rad2deg((smoothed_yaw + np.pi) % (2 * np.pi) - np.pi)
    window_means = [np.mean([np.rad2deg(yaw[j]) for j in range(max(0, i - 1), min(9, i + 2)) if j != 4] +
                            ([np.rad2deg(yaw[0])] if i == 0 else []) + ([np.rad2deg(yaw[8])] if i == 8 else []))
                    for i in valid]
    assert np.allclose(smoothed_yaw, window_means, atol=0.05)


def test_pose_to_trajectory_and_sensor_kind():
    yaw = np.array([0.0, 0.5, 1.0])
    poses = np.column_stack([np.arange(3.0), np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3),
                             np.sin(yaw / 2), np.cos(yaw / 2)])
    poses[2, 3:7] = 0.0  # no orientation
    dtype = np.dtype([("ts", "<f8"), ("id", "S8"), ("data", "<f8", (7,))])
    topic = np.zeros(3, dtype=dtype)
    topic["ts"] = [10.0, 10.5, 11.0]
    topic["data"] = poses
    for value in (topic, {"ts": topic["ts"], "data": poses}):
        trajectory = nav.pose_to_trajectory(value, compute_velocity=True)
        assert trajectory["source"] == "pose"
        assert np.allclose(trajectory["ts"], [10.0, 10.5, 11.0])
        assert np.allclose(trajectory["position"], poses[:, :3])
        assert np.allclose(trajectory["orientation"][:2], poses[:2, 3:7]) and np.isnan(trajectory["orientation"][2]).all()
        assert np.allclose(trajectory["linear_velocity"][:, 0], 2.0)
    single = nav.sensor_to_trajectory(poses[0], kind="PoseStamped")
    assert single["pose"].shape == (1, 7) and np.isnan(single["linear_velocity"]).all()


def test_nav_helpers_accept_partially_filled_get_buffer_topics():
    source = SyntheticSource(
        topics=[{"name": "/odom", "kind": "odometry", "rate": 20.0}, {"name": "/gps", "kind": "navsat", "rate": 20.0}],
        duration=1.0,
    )
    buffer = DataBuffer(source, buffer_depth=50, axis="/odom", use_db=False, preload=0)
    for _ in range(5):  # 5 of 50 ring slots filled
        buffer.roll_buffer("/odom")
    topics = buffer.get_buffer()
    assert len(topics["/odom"]) == 5  # unfilled ring slots are not returned
    odometry = nav.odometry_to_trajectory(topics["/odom"])
    assert odometry["ts"].size == 5 and np.all(np.diff(odometry["ts"]) > 0)
    nav.dead_reckon_trajectory(odometry)
    navsat = nav.navsat_to_trajectory(topics["/gps"])
    assert navsat["reference"]["lat"] != 0.0
    transformed = g.transform_odometry(topics["/odom"], np.eye(4))
    assert transformed.dtype == topics["/odom"].dtype
    assert np.allclose(transformed["data"][:, 0], topics["/odom"]["data"][:, 0])


# --- geometry: odometry and geographic topics ---------------------------------------------


def test_transform_odometry_keeps_unknown_variances_local_and_maps_zero_quaternions_to_nan():
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = yaw90
    odometry = np.zeros((2, 8, 4))
    odometry[:, 1, :3] = [0.1, 0.2, np.nan]
    odometry[:, 2, 3] = 1.0
    odometry[:, 5, :3] = [0.5, 0.7, np.nan]
    odometry[1, 2] = 0.0
    odometry[:, 3, :3] = [-1.0, 0.0, 0.0]  # ROS "unknown" flag stays untouched
    out = g.transform_odometry(odometry, transform, twist_frame="parent")
    assert np.allclose(out[0, 1, :3], [0.2, 0.1, np.nan], equal_nan=True)
    assert np.allclose(out[0, 5, :3], [0.7, 0.5, np.nan], equal_nan=True)
    assert np.allclose(out[0, 3, :3], [-1.0, 0.0, 0.0])
    assert np.allclose(out[0, 2], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    assert np.isnan(out[1, 2]).all()
    # Topic dicts come back as topic dicts.
    topic = g.transform_odometry({"ts": np.arange(2.0), "data": odometry}, transform)
    assert np.allclose(topic["ts"], [0, 1]) and np.allclose(topic["data"][0, 1, :3], [0.2, 0.1, np.nan], equal_nan=True)


def test_crop_geographic_bounds_accepts_navsat_topics():
    dtype = np.dtype([("ts", "<f8"), ("id", "S8"), ("data", "<f8", (3,))])
    topic = np.zeros(4, dtype=dtype)
    topic["ts"] = np.arange(4.0)
    topic["data"] = [[37.1, -122.1, 0], [38.5, -122.1, 0], [37.5, -121.5, 0], [np.nan, np.nan, 0]]
    cropped, mask = g.crop_geographic_bounds(topic, 37, -123, 38, -121, return_mask=True)
    assert mask.tolist() == [True, False, True, False]
    assert cropped.dtype == dtype and cropped["ts"].tolist() == [0.0, 2.0]
    as_dict = g.crop_geographic_bounds({"ts": topic["ts"], "data": topic["data"], "id": topic["id"]}, 37, -123, 38, -121)
    assert as_dict["ts"].tolist() == [0.0, 2.0] and len(as_dict["id"]) == 2


def test_dem_geometry_helpers_follow_dem_conventions():
    terrain = np.arange(12.0).reshape(3, 4)
    for north_up in (False, True):
        points = g.dem_grid_to_points(terrain, resolution=(2.0, 3.0), origin=(10.0, 20.0), north_up=north_up)
        expected = dem.dem_to_point_cloud(terrain, resolution=(2.0, 3.0), origin=(10.0, 20.0), north_up=north_up)
        assert np.array_equal(points.reshape(-1, 3), expected)
    north_up = g.dem_grid_to_points(terrain, north_up=True)
    assert north_up[0, 0, 1] == 2.0 and north_up[-1, 0, 1] == 0.0  # row 0 is the northern edge
    shifted = g.transform_dem_grid(terrain, np.eye(4), resolution=(2.0, 3.0), north_up=True)
    assert np.array_equal(shifted, g.dem_grid_to_points(terrain, resolution=(2.0, 3.0), north_up=True))
    K = g.camera_matrix(100, 100, 50, 50)
    camera = np.eye(4)
    camera[2, 3] = 50.0
    pixels, valid = g.project_dem_to_image(terrain, camera_matrix=K, transform=camera, north_up=True)
    reference, _ = g.project_points_to_image(north_up.reshape(-1, 3), camera_matrix=K, transform=camera)
    assert valid.all() and np.allclose(pixels.reshape(-1, 2), reference)


# --- DEM ------------------------------------------------------------------------------------


def _linear_tiles(samples=121):
    def elevation(lat, lon):
        return 10000.0 * (lat - 37.0) + 1000.0 * (lon + 122.0)

    tiles = {}
    for name, lat0, lon0 in (("N37W122", 37, -122), ("N37W121", 37, -121), ("N36W122", 36, -122), ("N36W121", 36, -121)):
        rows = np.arange(samples)[:, None]
        cols = np.arange(samples)[None, :]
        tiles[name] = elevation(lat0 + 1 - rows / (samples - 1), lon0 + cols / (samples - 1))
    return tiles, elevation


def test_sample_elevation_at_latlon_locates_geographic_mosaics():
    tiles, elevation = _linear_tiles()
    mosaic, lats, lons = dem.mosaic_dem_tiles(tiles, overlap=1, return_index=True)
    rng = np.random.default_rng(0)
    lat = rng.uniform(36.0, 38.0, 200)
    lon = rng.uniform(-122.0, -120.0, 200)
    fixes = np.column_stack((lat, lon, np.zeros(200)))
    by_index = dem.sample_elevation_at_latlon(mosaic, lat, lon, index=(lats, lons))
    assert np.allclose(by_index, elevation(lat, lon), atol=1e-6)
    by_bounds = dem.sample_elevation_at_latlon(mosaic, lat, lon, bounds=(-122, 36, -120, 38), north_up=True)
    assert np.allclose(by_bounds, by_index)
    assert np.allclose(dem.sample_elevation_at_navsat(mosaic, fixes, index=(lats, lons)), by_index)
    assert np.allclose(dem.sample_elevation_at_navsat(mosaic, fixes, bounds=(-122, 36, -120, 38), north_up=True), by_index)
    south_origin = dem.sample_elevation_at_latlon(mosaic[::-1], lat, lon, bounds=(-122, 36, -120, 38))
    assert np.allclose(south_origin, by_index)
    nearest = dem.sample_elevation_at_latlon(mosaic, lat, lon, index=(lats, lons), method="nearest")
    assert np.abs(nearest - by_index).max() <= 0.5 * (10000.0 + 1000.0) / 120 + 1e-6
    assert np.isnan(dem.sample_elevation_at_latlon(mosaic, [39.0], [-121.0], index=(lats, lons))).all()

    # A raster crossing the antimeridian (east < west).
    wrap = np.tile(np.arange(5.0), (3, 1))  # columns at 179, 179.5, 180, -179.5, -179
    values = dem.sample_elevation_at_latlon(wrap, [0.5, 0.5, 0.5], [179.25, -179.75, 180.0], bounds=(179, 0, -179, 1))
    assert np.allclose(values, [0.5, 2.5, 2.0])
    with pytest.raises(ValueError, match="ref_lat"):
        dem.sample_elevation_at_navsat(mosaic, fixes)


def test_dem_nearest_half_cell_ties_pick_the_same_cell_in_both_row_conventions():
    terrain = np.arange(6.0)[:, None] * np.ones((1, 3))  # row index == elevation
    for y in (0.5, 1.5, 2.5, 3.5, 4.5):
        native = dem.sample_elevation(terrain, [1.0], [y], bilinear=False)
        flipped = dem.sample_elevation(terrain[::-1], [1.0], [y], bilinear=False, north_up=True)
        assert native[0] == flipped[0] == np.floor(y + 0.5)
        patch = dem.terrain_patch(terrain, (1.0, y), 1)
        patch_flipped = dem.terrain_patch(terrain[::-1], (1.0, y), 1, north_up=True)
        assert patch[0, 0] == patch_flipped[0, 0] == np.floor(y + 0.5)
    assert dem.sample_grid(terrain, [2.5], [0.5], bilinear=False)[0] == 3.0
    assert dem.sample_grid(terrain, [5.5], [0.0], bilinear=False)[0] == 5.0  # half-cell border clamps


def test_mosaic_auto_overlap_only_for_tiles_identified_as_srtm():
    srtm = np.zeros((1201, 1201), dtype=np.int16)
    assert dem.mosaic_dem_tiles({"N37W122": srtm, "N37W121.hgt": srtm}).shape == (1201, 2401)
    assert dem.mosaic_dem_tiles({(37, -122): srtm, (37, -121): srtm}).shape == (1201, 2402)
    assert dem.mosaic_dem_tiles({"N37W122_dem.tif": srtm, "N37W121_dem.tif": srtm}).shape == (1201, 2402)
    assert dem.mosaic_dem_tiles({(37, -122): srtm, (37, -121): srtm}, overlap=1).shape == (1201, 2401)


# --- image ------------------------------------------------------------------------------


def test_convert_image_dtype_saturates_64_bit_targets():
    assert image.convert_image_dtype(np.array([255], np.uint8), np.int64)[0] == np.iinfo(np.int64).max
    assert image.convert_image_dtype(np.array([255], np.uint8), np.uint64)[0] == np.iinfo(np.uint64).max
    assert image.convert_image_dtype(np.array([32767, -32768], np.int16), np.int64).tolist() == [
        np.iinfo(np.int64).max, np.iinfo(np.int64).min]
    floats = image.convert_image_dtype(np.array([1.0, -1.0, 0.0]), np.int64)
    assert floats.tolist() == [np.iinfo(np.int64).max, -np.iinfo(np.int64).max, 0]
    assert image.convert_image_dtype(np.array([1.0, 0.0]), np.uint64).tolist() == [np.iinfo(np.uint64).max, 0]
    extreme = np.array([np.iinfo(np.int64).max, 0], np.int64)
    assert image.convert_image_dtype(image.convert_image_dtype(extreme, np.float64), np.int64).tolist() == extreme.tolist()
    assert image.convert_image_dtype(np.array([np.iinfo(np.uint64).max], np.uint64), np.int64)[0] == np.iinfo(np.int64).max


def test_convert_image_dtype_matches_skimage_img_as_for_every_pair():
    util = pytest.importorskip("skimage.util")
    converters = {
        np.dtype(np.float32): util.img_as_float32,
        np.dtype(np.float64): util.img_as_float64,
        np.dtype(np.uint8): util.img_as_ubyte,
        np.dtype(np.uint16): util.img_as_uint,
        np.dtype(np.int16): util.img_as_int,
        np.dtype(np.bool_): util.img_as_bool,
    }
    sources = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64, np.bool_]
    for source in map(np.dtype, sources):
        if source.kind in "ui":
            info = np.iinfo(source)
            values = np.unique(np.array([info.min, info.min + 1, -1, 0, 1, 2, 3, info.max // 3, info.max // 2,
                                         info.max // 2 + 1, info.max - 1, info.max], dtype=object)
                               .clip(info.min, info.max).astype(source))
        elif source.kind == "b":
            values = np.array([True, False])
        else:
            values = np.array([-1.0, -0.6, -1e-3, 0.0, 1e-3, 0.25, 0.5, 0.75, 0.999, 1.0], dtype=source)
        for target, convert in converters.items():
            if target == source:
                continue
            ours = image.convert_image_dtype(values, target)
            # skimage scales float16 inputs in float16, which overflows for
            # 16-bit targets; feed it float64 values (same numbers) instead.
            theirs = convert(values.astype(np.float64) if source == np.float16 else values)
            if source == np.float16 and target.kind == "f":
                theirs = convert(values)
            assert ours.dtype == theirs.dtype, (source, target)
            if source.kind == "f" and target.kind == "i":
                # Documented difference: symmetric `x * max` (round-trips)
                # instead of skimage's `x * (max + 0.5) - 0.5`.
                assert np.abs(ours.astype(np.int64) - theirs.astype(np.int64)).max() <= 1, (source, target)
            elif target.kind == "f":
                assert np.allclose(ours, theirs, rtol=1e-6, atol=0), (source, target)
            else:
                assert np.array_equal(ours, theirs), (source, target, values, ours, theirs)


def test_image_gradients_spatial_axes_refer_to_the_input_array():
    rng = np.random.default_rng(0)
    sequence = rng.uniform(size=(2, 6, 8, 3))
    reference = image.image_gradients(sequence)["dx"]
    assert np.allclose(image.image_gradients(sequence, spatial_axes=(-3, -2))["dx"], reference)
    assert np.allclose(image.image_gradients(sequence, spatial_axes=(1, 2))["dx"], reference)
    channels_first = np.moveaxis(sequence, -1, 1)
    assert np.allclose(image.image_gradients(channels_first, channel_axis=1, spatial_axes=(2, 3))["dx"], reference)
    assert np.allclose(image.image_gradients(channels_first, channel_axis=1)["dx"], reference)
    with pytest.raises(ValueError, match="channel axis"):
        image.image_gradients(channels_first, channel_axis=1, spatial_axes=(1, 2))


def test_ml_splits_and_windows_keep_ragged_lengths_and_frame_ids():
    from arraydataengine.ops.core import topic_view

    lengths = np.array([1, 4, 2, 3, 4], dtype=np.int64)
    data = np.zeros((5, 4, 3))
    for row, length in enumerate(lengths):
        data[row, :length] = row + 1
    frames = np.array([b"cam", b"lidar", b"cam", b"imu", b"cam"], dtype=object)
    ids = np.array([b"m0", b"m1", b"m2", b"m3", b"m4"])
    as_dict = {"ts": np.array([4.0, 0.0, 3.0, 1.0, 2.0]), "data": data, "id": ids, "lengths": lengths,
               "frame_ids": frames}
    structured = np.zeros(5, dtype=[("ts", "<f8"), ("id", "S4"), ("frame_id", "S8"), ("length", "<i8"),
                                    ("data", "<f8", (4, 3))])
    structured["ts"], structured["id"], structured["frame_id"] = as_dict["ts"], ids, frames.astype("S8")
    structured["length"], structured["data"] = lengths, data
    for topic in (as_dict, structured, topic_view(as_dict)):
        splits = ml.split_topic(topic, fractions=(0.6, 0.4), names=("train", "val"))
        for split in splits.values():
            rows = [int(name.decode().lstrip("m")) for name in split["id"]]
            assert split["lengths"].tolist() == lengths[rows].tolist()
            assert [f.decode() if isinstance(f, bytes) else f for f in split["frame_ids"]] == [
                frames[row].decode() for row in rows]
            assert np.array_equal(split["data"], data[rows])
        assert sorted(ts for split in splits.values() for ts in split["ts"]) == sorted(as_dict["ts"])
        assert np.array_equal(splits["train"]["ts"], [0.0, 1.0, 2.0])  # by time: earliest rows first

    windows = list(ml.iter_ml_windows(as_dict, size=2))
    assert [window["lengths"].tolist() for window in windows] == [[1], [1, 4], [4, 2], [2, 3], [3, 4]]
    assert [list(window["frame_ids"]) for window in windows][1] == ["cam", "lidar"]
    batch = ml.collate_samples(windows)
    assert batch["lengths"].shape == (5, 2) and batch["lengths"][2].tolist() == [4, 2]
    assert batch["data_lengths"].tolist() == [1, 2, 2, 2, 2]
