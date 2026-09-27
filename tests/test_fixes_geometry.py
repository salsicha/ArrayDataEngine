"""Regression tests for geometry / point-cloud fixes (registration gating,
bounded nearest-neighbor search, NaN handling, lens distortion, frames)."""

import tracemalloc

import numpy as np
import pytest

from arraydataengine.ops import geometry as g
from arraydataengine.ops import point_cloud as pc


@pytest.fixture
def no_scipy(monkeypatch):
    """Force the pure-NumPy nearest-neighbor fallback."""

    monkeypatch.setattr(pc, "_scipy_ckdtree", lambda: None)


def _has_scipy():
    return pc._scipy_ckdtree() is not None


def _plane_and_blob(seed=0, n=400):
    rng = np.random.default_rng(seed)
    plane = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n), np.zeros(n)])
    blob = rng.normal(size=(n, 3)) + [50.0, 50.0, 50.0]
    return plane, blob


def _corner_scene(n=300, seed=0):
    rng = np.random.default_rng(seed)
    target = np.concatenate([
        np.column_stack([rng.uniform(0, 5, n), rng.uniform(0, 5, n), np.zeros(n)]),
        np.column_stack([np.zeros(n), rng.uniform(0, 5, n), rng.uniform(0, 5, n)]),
        np.column_stack([rng.uniform(0, 5, n), np.zeros(n), rng.uniform(0, 5, n)]),
    ])
    normals = np.repeat(np.eye(3)[[2, 0, 1]], n, axis=0)
    return target, normals


def _small_rotation(rotvec):
    angle = np.linalg.norm(rotvec)
    axis = np.asarray(rotvec) / angle
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


# --- 1. registration fitness / gating / reported metrics -------------------


def test_verify_loop_closures_default_gate_rejects_unrelated_scans():
    plane, blob = _plane_and_blob()
    clouds = [plane, blob, plane, blob]
    trajectory = np.zeros((4, 7))
    trajectory[:, 6] = 1.0
    out = pc.verify_loop_closures(clouds, trajectory, radius=1.0, min_separation=1, min_fitness=0.9, return_all=True)
    # Candidates are (1->0), (2->0), (3->0); only the plane/plane pair is real.
    assert out["source_index"].tolist() == [1, 2, 3]
    assert out["accepted"].tolist() == [False, True, False]
    assert out["fitness"][0] == 0.0 and out["fitness"][2] == 0.0

    out = pc.verify_loop_closures(
        clouds, trajectory, radius=1.0, min_separation=1, min_fitness=0.9, return_all=True,
        method="multi_scale", voxel_sizes=(1.0, 0.0),
    )
    assert out["accepted"].tolist() == [False, True, False]


def test_default_loop_closure_gate():
    grid = np.stack(np.meshgrid(np.arange(20.0), np.arange(20.0), [0.0]), -1).reshape(-1, 3) * 0.5
    assert pc._default_loop_closure_gate(grid, voxel_size=0.2) == pytest.approx(0.4)
    assert pc._default_loop_closure_gate(grid, voxel_size=None) == pytest.approx(3 * 0.5)


def test_multi_scale_icp_final_metrics_are_gated_and_iterations_are_actual():
    plane, blob = _plane_and_blob()
    result = pc.multi_scale_icp(plane, blob, voxel_sizes=(1.0, 0.5), max_correspondence_distances=(2.0, 1.0))
    assert result["fitness"] == 0.0
    assert np.isinf(result["inlier_rmse"])
    assert result["iterations"] == sum(level["result"]["iterations"] for level in result["levels"])
    assert result["iterations"] < 40

    # Default per-level gates (2x voxel) also gate the final metrics.
    result = pc.multi_scale_icp(plane, blob, voxel_sizes=(1.0, 0.5))
    assert result["fitness"] == 0.0

    aligned = pc.multi_scale_icp(plane, plane + [0.05, 0.0, 0.0], voxel_sizes=(1.0, 0.5), max_iterations=10)
    assert aligned["fitness"] == 1.0
    assert aligned["iterations"] == sum(level["result"]["iterations"] for level in aligned["levels"])


def test_point_to_plane_reports_point_to_point_inlier_rmse():
    # Two different samplings of the same plane: the point-to-plane residual
    # is zero, the point-to-point distance between samples is not.
    rng = np.random.default_rng(3)
    target = np.column_stack([rng.uniform(0, 5, 400), rng.uniform(0, 5, 400), np.zeros(400)])
    source = np.column_stack([rng.uniform(1, 4, 200), rng.uniform(1, 4, 200), np.zeros(200)])
    normals = np.tile([0.0, 0.0, 1.0], (400, 1))
    result = pc.point_to_plane_icp(
        source, target, target_normals=normals, max_iterations=3, max_correspondence_distance=0.5,
        min_correspondences=3, return_correspondences=True,
    )
    corr = result["correspondences"]
    moved = pc._transform_points_xyz(source, result["transform"])
    expected = np.sqrt(np.mean(np.sum((moved[corr["source_indices"]] - target[corr["target_indices"]]) ** 2, axis=1)))
    assert expected > 0.01
    assert result["inlier_rmse"] == pytest.approx(expected, rel=1e-12)


def test_icp_does_one_neighbor_search_per_iteration(monkeypatch):
    calls = []
    original = pc._nearest_correspondences

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pc, "_nearest_correspondences", counting)
    rng = np.random.default_rng(1)
    target = rng.uniform(0, 5, (300, 3))
    result = pc.point_to_point_icp(target + [0.05, 0.0, 0.0], target, max_correspondence_distance=0.5)
    assert result["converged"]
    assert np.allclose(result["transform"][:3, 3], [-0.05, 0.0, 0.0])
    assert len(calls) == result["iterations"] + 1


# --- 2. bounded, backend-independent nearest-neighbor search ---------------


def _reference_knn(points, queries, k):
    distances = np.linalg.norm(points[None, :, :3] - queries[:, None, :3], axis=2)
    order = np.argsort(distances, axis=1)[:, :k]
    return np.take_along_axis(distances, order, axis=1), order


@pytest.mark.parametrize("fallback", [False, True])
def test_knn_search_matches_reference(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(pc, "_scipy_ckdtree", lambda: None)
    elif not _has_scipy():
        pytest.skip("scipy not installed")
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, (700, 3))
    queries = rng.uniform(-1, 11, (300, 3))
    for k in (1, 5, 1000):
        distances, indices = pc.knn_search(points, queries, k=k)
        ref_d, ref_i = _reference_knn(points, queries, min(k, 700))
        assert np.array_equal(indices, ref_i)
        assert np.allclose(distances, ref_d, rtol=1e-14, atol=1e-12)


def test_knn_fallback_is_chunked_and_memory_bounded(no_scipy):
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 20, (3000, 3))
    tracemalloc.start()
    normals = pc.estimate_normals(points, k=8)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert np.isfinite(normals).all()
    # The old (Q, N, 3) broadcast needed > 200 MB here.
    assert peak < 100e6


def test_knn_search_skips_non_finite_points_and_queries(no_scipy):
    points = np.array([[0.0, 0, 0], [np.nan, 0, 0], [1.0, 0, 0]])
    queries = np.array([[0.1, 0, 0], [np.nan, 0, 0]])
    distances, indices = pc.knn_search(points, queries, k=3)
    assert indices.tolist() == [[0, 2, -1], [-1, -1, -1]]
    assert np.allclose(distances[0, :2], [0.1, 0.9]) and np.isinf(distances[0, 2]) and np.isinf(distances[1]).all()


@pytest.mark.skipif(not _has_scipy(), reason="scipy not installed")
def test_radius_and_hybrid_search_identical_across_backends(monkeypatch):
    rng = np.random.default_rng(2)
    grid = rng.integers(0, 5, (400, 3)).astype(float) * 0.5  # many exact ties
    grid[7] = np.nan
    queries = np.vstack([rng.integers(0, 10, (150, 3)).astype(float) * 0.25, [[np.nan, 0, 0]]])
    results = []
    for use_tree in (True, False):
        if not use_tree:
            monkeypatch.setattr(pc, "_scipy_ckdtree", lambda: None)
        radius = pc.radius_search(grid, queries, 0.8)
        hybrid = pc.hybrid_search(grid, queries, 0.8, 3)
        results.append((radius, hybrid))
    (r_tree, h_tree), (r_grid, h_grid) = results
    assert all(np.array_equal(a, b) for a, b in zip(r_tree, r_grid))
    assert all(np.all(np.diff(a) > 0) and 7 not in a for a in r_tree)
    for a, b in zip(h_tree, h_grid):
        assert np.array_equal(a, b)
    # Ties are broken by lowest index, as documented by the brute-force path.
    d, i, c = h_tree
    for row in range(len(queries) - 1):
        cand = r_grid[row]
        dist = np.linalg.norm(grid[cand] - queries[row], axis=1)
        expected = cand[np.lexsort((cand, dist))][:3]
        assert i[row, : c[row]].tolist() == expected.tolist()


# --- 3. point-to-plane linearization far from the origin -------------------


@pytest.mark.parametrize("offset", [1.0e3, 5.0e5])
def test_point_to_plane_icp_converges_far_from_origin(offset):
    target, normals = _corner_scene(200)
    target = target + offset
    centroid = target.mean(axis=0)
    rotation = _small_rotation([0.03, -0.02, 0.05])
    source = (target - centroid - [0.1, -0.05, 0.08]) @ rotation + centroid
    result = pc.point_to_plane_icp(
        source, target, target_normals=normals, max_iterations=30, max_correspondence_distance=0.5,
    )
    assert result["converged"]
    assert np.abs(pc._transform_points_xyz(source, result["transform"]) - target).max() < 1e-6


# --- 4. non-finite rows -----------------------------------------------------


@pytest.fixture
def cloud_with_nan():
    rng = np.random.default_rng(0)
    cloud = rng.uniform(0, 5, (300, 3))
    with_nan = np.vstack([cloud[:7], [[np.nan, np.nan, np.nan]], cloud[7:]])
    return cloud, with_nan


@pytest.mark.parametrize("gate", [None, 0.5])
@pytest.mark.parametrize("fallback", [False, True])
def test_icp_ignores_non_finite_rows(monkeypatch, cloud_with_nan, gate, fallback):
    if fallback:
        monkeypatch.setattr(pc, "_scipy_ckdtree", lambda: None)
    cloud, with_nan = cloud_with_nan
    result = pc.point_to_point_icp(with_nan + [0.05, 0, 0], with_nan, max_correspondence_distance=gate)
    assert np.allclose(result["transform"][:3, 3], [-0.05, 0, 0], atol=1e-9)
    assert result["fitness"] == 1.0
    plane_result = pc.point_to_plane_icp(with_nan + [0, 0, 0.02], with_nan, max_correspondence_distance=gate)
    assert np.isfinite(plane_result["transform"]).all()


def test_statistical_outlier_filter_with_nan(cloud_with_nan):
    cloud, with_nan = cloud_with_nan
    filtered, mask = pc.statistical_outlier_filter(with_nan, k=5, return_mask=True)
    _, reference_mask = pc.statistical_outlier_filter(cloud, k=5, return_mask=True)
    assert not mask[7]
    assert np.array_equal(np.delete(mask, 7), reference_mask)
    assert np.isfinite(filtered).all() and filtered.shape[0] > 250


def test_voxel_and_farthest_point_downsample_skip_nan(cloud_with_nan):
    cloud, with_nan = cloud_with_nan
    assert np.array_equal(pc.voxel_downsample(with_nan, 0.5), pc.voxel_downsample(cloud, 0.5))
    sampled, indices = pc.farthest_point_downsample(with_nan, 10, start_index=7, return_indices=True)
    assert 7 not in indices and indices[0] == 0 and np.isfinite(sampled).all()
    _, reference = pc.farthest_point_downsample(cloud, 10, start_index=0, return_indices=True)
    assert np.array_equal(np.where(indices > 7, indices - 1, indices), reference)
    assert 7 not in pc.farthest_point_downsample(with_nan, 400, return_indices=True)[1]


def test_per_point_outputs_stay_aligned_with_nan(cloud_with_nan):
    cloud, with_nan = cloud_with_nan
    normals = pc.estimate_normals(with_nan, k=8, orient_toward=[0, 0, 10])
    assert np.isnan(normals[7]).all()
    assert np.allclose(np.delete(normals, 7, axis=0), pc.estimate_normals(cloud, k=8, orient_toward=[0, 0, 10]))
    covariances, indices = pc.local_covariances(with_nan, k=6, return_indices=True)
    assert np.isnan(covariances[7]).all() and (indices[7] == -1).all() and 7 not in indices
    descriptors = pc.curvature_descriptors(with_nan, k=6)
    assert np.isnan(descriptors["eigenentropy"][7]) and np.isfinite(np.delete(descriptors["curvature"], 7)).all()
    stats = pc.nearest_neighbor_distance_stats(with_nan, k=2)
    assert np.isnan(stats["distances"][7]).all()
    assert stats["global_mean"] == pytest.approx(pc.nearest_neighbor_distance_stats(cloud, k=2)["global_mean"])
    labels = pc.connected_components(with_nan, radius=1.0)
    assert labels[7] == -1
    plane, mask = pc.segment_plane(np.vstack([with_nan[:, :2].T, np.zeros(301)]).T, 0.01, seed=1)
    assert not mask[7] and mask.sum() == 300 and abs(plane[2]) == pytest.approx(1.0)


# --- 5./6. lens distortion ---------------------------------------------------


K = g.camera_matrix(500, 480, 320, 240)


@pytest.mark.parametrize("distortion", [
    np.array([-0.28, 0.07, 0.0, 0.0, 0.0]),
    np.array([-0.40, 0.15, 0.001, -0.0015, -0.02]),
    np.array([-0.28, 0.07, 0.001, -0.0015, 0.01, 0.02, -0.01, 0.005]),
])
def test_undistort_matches_opencv_converged(distortion):
    cv2 = pytest.importorskip("cv2")
    uv = np.stack(np.meshgrid(np.linspace(0, 639, 9), np.linspace(0, 479, 7)), -1).reshape(-1, 2)
    reference = cv2.undistortPoints(
        uv[:, None, :], K, distortion, P=K, criteria=(cv2.TERM_CRITERIA_COUNT, 300, 0),
    )[:, 0]
    assert np.abs(g.undistort_pixels(uv, K, distortion) - reference).max() < 1e-6
    assert np.abs(g.distort_pixels(g.undistort_pixels(uv, K, distortion), K, distortion) - uv).max() < 1e-6


def test_backproject_pixels_accepts_iterations_and_is_converged_by_default():
    distortion = np.array([-0.40, 0.15, 0.0, 0.0, -0.02])
    corner = np.array([[0.0, 0.0]])
    converged = g.undistort_normalized_points(g.pixels_to_normalized_points(corner, K), distortion, iterations=200, tolerance=None)
    points = g.backproject_pixels(corner, 10.0, K, distortion)
    assert np.allclose(points[0, :2], converged[0] * 10.0, atol=1e-9)
    coarse = g.backproject_pixels(corner, 10.0, K, distortion, iterations=2, tolerance=None)
    assert np.abs(coarse[0, :2] - converged[0] * 10.0).max() > 1e-3


def test_undistort_outside_model_domain_is_nan():
    # k1 = -1: the radial factor hits zero at r = 1.
    assert np.isnan(g.undistort_normalized_points(np.array([[0.9, 0.0]]), np.array([-1.0, 0, 0, 0]))).all()


def test_projection_marks_folded_points_invalid():
    Kc = g.camera_matrix(500, 500, 320, 240)
    distortion = np.array([-0.3, 0.0, 0.0, 0.0, 0.0])
    angles = np.deg2rad([30, 45, 55, 61, 70])
    points = np.column_stack([np.tan(angles), np.zeros(5), np.ones(5)])
    pixels, valid = g.project_camera_points(points, camera_matrix=Kc, distortion=distortion)
    # f(r) = r (1 - 0.3 r^2) peaks at r^2 = 1/0.9 (46.5 degrees).
    assert valid.tolist() == [True, True, False, False, False]
    assert np.isnan(pixels[2:]).all()
    # 61 degrees used to fold back to column ~341 and be reported valid.
    _, in_image = g.project_camera_points(points, camera_matrix=Kc, distortion=distortion, image_shape=(480, 640))
    assert in_image.tolist() == [True, False, False, False, False]
    assert g._radial_distortion_limit(g._distortion_coefficients(distortion)) == pytest.approx(1 / 0.9)
    assert g._radial_distortion_limit(g._distortion_coefficients(np.array([0.1, 0.01, 0, 0]))) is None


def test_projection_helpers_accept_distortion():
    distortion = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-0.8, 0.8, 200), rng.uniform(2, 4, 200)])
    pixels, valid = g.project_points_to_image(points, camera_matrix=K, distortion=distortion, image_shape=(480, 640))
    image = np.broadcast_to(np.arange(640.0), (480, 640))
    colors, mask = g.colorize_points(points, image, camera_matrix=K, distortion=distortion, return_mask=True)
    assert np.array_equal(mask, valid) and np.allclose(colors[mask, 3], pixels[mask, 0])
    samples = g.sample_image_at_points(image, points, camera_matrix=K, distortion=distortion)
    assert np.allclose(samples[valid], pixels[valid, 0])
    depth, index = g.points_to_depth_image(points, (480, 640), camera_matrix=K, distortion=distortion, return_indices=True)
    hit = index >= 0
    rows, cols = np.nonzero(hit)
    assert np.array_equal(np.rint(pixels[index[hit]]).astype(int), np.column_stack([cols, rows]))

    flat = np.full((48, 64), 3.0)
    Ks = g.camera_matrix(50, 50, 32, 24)
    xyz = g.rgbd_to_points(flat, np.zeros((48, 64)), camera_matrix=Ks, distortion=distortion)[:, :3]
    reprojected, ok = g.project_points_to_image(xyz, camera_matrix=Ks, distortion=distortion)
    grid = np.stack(np.meshgrid(np.arange(64.0), np.arange(48.0)), -1)[flat > 0]
    assert ok.all() and np.abs(reprojected - grid).max() < 1e-6


# --- 7. odometry twist frame -------------------------------------------------


def test_transform_odometry_twist_frame():
    odom = np.zeros((8, 4))
    odom[2] = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    odom[4] = [1.0, 0.0, 0.0, 0.0]
    odom[5] = [1.0, 4.0, 9.0, 0.0]
    odom[6] = [0.0, 0.0, 0.5, 0.0]
    odom[7] = [2.0, 3.0, 5.0, 0.0]
    transform = np.eye(4)
    transform[:3, :3] = _small_rotation([0.0, 0.0, np.pi / 2])
    out = g.transform_odometry(odom, transform)
    assert np.array_equal(out[4:], odom[4:])
    assert np.allclose(out[2], [0, 0, 1, 0]) or np.allclose(out[2], [0, 0, -1, 0])
    parent = g.transform_odometry(odom, transform, twist_frame="parent")
    assert np.allclose(parent[4, :3], [0, 1, 0]) and np.allclose(parent[5, :3], [4, 1, 9])
    with pytest.raises(ValueError):
        g.transform_odometry(odom, transform, twist_frame="world")


# --- 8./9./10. camera matrix scaling, CameraModel equality, voxel dtype ------


def test_scale_camera_matrix_half_pixel_centers():
    Ks = g.camera_matrix(500, 400, 200, 150)
    assert np.allclose(g.scale_camera_matrix(Ks, 0.5), g.camera_matrix(250, 200, 100, 75))
    half = g.scale_camera_matrix(Ks, 0.5, 0.25, half_pixel_centers=True)
    assert np.allclose(half, g.camera_matrix(250, 100, 99.75, 37.125))


def test_scale_camera_matrix_half_pixel_matches_cv2_resize():
    cv2 = pytest.importorskip("cv2")
    yy, xx = np.mgrid[0:480, 0:640]
    image = np.exp(-((xx - 200.0) ** 2 + (yy - 150.0) ** 2) / (2 * 6.0 ** 2))
    small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    y, x = np.mgrid[0:small.shape[0], 0:small.shape[1]]
    centre = [(small * x).sum() / small.sum(), (small * y).sum() / small.sum()]
    scaled = g.scale_camera_matrix(g.camera_matrix(500, 500, 200, 150), 0.25, half_pixel_centers=True)
    assert np.allclose(scaled[:2, 2], centre, atol=1e-3)


def test_camera_model_equality_and_hash():
    a = g.camera_model(fx=1, fy=1, cx=0, cy=0, distortion=[0.1, 0, 0, 0])
    b = g.camera_model(fx=1, fy=1, cx=-0.0, cy=0, distortion=np.array([0.1, 0, 0, 0, 0, 0, 0, 0]))
    c = g.camera_model(fx=1, fy=1, cx=0, cy=0)
    assert a == b and hash(a) == hash(b)
    assert a != c and c != g.camera_model(fx=1, fy=1, cx=0, cy=0, image_shape=(4, 4))
    assert len({a, b, c}) == 2
    assert (a == "camera") is False


def test_voxel_downsample_promotes_integer_input():
    out = pc.voxel_downsample(np.array([[0, 0, 0], [1, 1, 1]]), 10.0)
    assert out.dtype == np.float64 and np.allclose(out, [[0.5, 0.5, 0.5]])
    assert pc.voxel_downsample(np.zeros((0, 3), dtype=np.int32), 1.0).dtype == np.float64
    assert pc.voxel_downsample(np.ones((3, 3), dtype=np.float32), 1.0).dtype == np.float32


# --- improvements ------------------------------------------------------------


def test_local_covariances_match_per_point_loop():
    rng = np.random.default_rng(4)
    points = rng.uniform(0, 3, (250, 3))
    covariances, indices = pc.local_covariances(points, k=7, return_indices=True)
    for row in (0, 17, 249):
        local = points[indices[row]]
        centered = local - local.mean(axis=0)
        assert np.array_equal(covariances[row], centered.T @ centered / 6)
    normals = pc.estimate_normals(points, k=7, orient_toward=[0, 0, 10])
    for row in (0, 17, 249):
        normal = np.linalg.eigh(covariances[row])[1][:, 0]
        if np.dot(normal, [0, 0, 10] - points[row]) < 0:
            normal = -normal
        assert np.allclose(normals[row], normal)


def test_segment_plane_refines_and_handles_tiny_scale():
    rng = np.random.default_rng(5)
    points = np.column_stack([rng.uniform(-5, 5, 600), rng.uniform(-5, 5, 600), rng.normal(0, 0.01, 600)])
    points[:100] += rng.normal(0, 2, (100, 3))
    rough, rough_mask = pc.segment_plane(points, 0.03, iterations=20, seed=3, refine=False)
    plane, mask = pc.segment_plane(points, 0.03, iterations=20, seed=3)
    assert mask.sum() >= rough_mask.sum()
    assert abs(plane[2]) > abs(rough[2])
    assert np.array_equal(mask, np.abs(points @ plane[:3] + plane[3]) <= 0.03)
    # Relative degeneracy test: nanometre-scale clouds are not "collinear".
    tiny_plane, tiny_mask = pc.segment_plane(points[100:] * 1e-9, 3e-11, iterations=20, seed=3)
    assert tiny_mask.sum() > 400 and abs(tiny_plane[2]) > 0.99
