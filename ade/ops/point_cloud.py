from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np

from .geometry import _as_points, _as_transform_matrix, apply_transform, crop_bounds, points_to_depth_image
from .nav import quaternion_to_rotation_matrix


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    arr = _as_points(points)
    if arr.size == 0:
        return arr.copy()

    voxels = np.floor(arr[:, :3] / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxels, axis=0, return_inverse=True)
    downsampled = np.zeros((inverse.max() + 1, arr.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse)
    for dim in range(arr.shape[1]):
        downsampled[:, dim] = np.bincount(inverse, weights=arr[:, dim]) / counts
    return downsampled.astype(arr.dtype, copy=False)


def uniform_downsample(
    points: np.ndarray,
    every_k: int,
    start_index: int = 0,
    return_indices: bool = False,
):
    """Select every `every_k` point, preserving input order."""

    arr = _as_points(points)
    every_k = int(every_k)
    start_index = int(start_index)
    if every_k < 1:
        raise ValueError("every_k must be at least 1")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    indices = np.arange(start_index, arr.shape[0], every_k, dtype=np.int64)
    sampled = arr[indices].copy()
    return (sampled, indices) if return_indices else sampled


def random_downsample(
    points: np.ndarray,
    count: int | None = None,
    ratio: float | None = None,
    seed: int | None = None,
    replace: bool = False,
    return_indices: bool = False,
):
    """Randomly sample points by absolute count or ratio."""

    arr = _as_points(points)
    sample_count = _sampling_count(arr.shape[0], count=count, ratio=ratio, replace=replace)
    rng = np.random.default_rng(seed)
    indices = rng.choice(arr.shape[0], size=sample_count, replace=replace).astype(np.int64, copy=False)
    if not replace:
        indices.sort()
    sampled = arr[indices].copy()
    return (sampled, indices) if return_indices else sampled


def farthest_point_downsample(
    points: np.ndarray,
    count: int,
    start_index: int | None = 0,
    seed: int | None = None,
    return_indices: bool = False,
):
    """Sample points with greedy farthest-point sampling over XYZ coordinates."""

    arr = _as_points(points)
    n_points = arr.shape[0]
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0 or n_points == 0:
        indices = np.empty((0,), dtype=np.int64)
        sampled = arr[:0].copy()
        return (sampled, indices) if return_indices else sampled
    if count >= n_points:
        indices = np.arange(n_points, dtype=np.int64)
        sampled = arr.copy()
        return (sampled, indices) if return_indices else sampled

    if start_index is None:
        rng = np.random.default_rng(seed)
        current = int(rng.integers(0, n_points))
    else:
        current = int(start_index)
        if current < 0 or current >= n_points:
            raise ValueError("start_index must refer to an existing point")

    xyz = arr[:, :3].astype(np.float64, copy=False)
    indices = np.empty(count, dtype=np.int64)
    min_distances = np.full(n_points, np.inf, dtype=np.float64)
    selected = np.zeros(n_points, dtype=bool)

    for sample_index in range(count):
        indices[sample_index] = current
        selected[current] = True
        diff = xyz - xyz[current]
        distances = np.einsum("ij,ij->i", diff, diff)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected] = -np.inf
        if sample_index + 1 < count:
            current = int(np.argmax(min_distances))

    sampled = arr[indices].copy()
    return (sampled, indices) if return_indices else sampled


def knn_search(points: np.ndarray, queries: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    arr = _as_points(points)
    query = np.asarray(queries, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[1] < 3:
        raise ValueError("queries must have shape (Q, 3+) or (3,)")
    if k < 1:
        raise ValueError("k must be at least 1")
    if arr.shape[0] == 0:
        return np.empty((query.shape[0], 0)), np.empty((query.shape[0], 0), dtype=np.int64)

    k = min(k, arr.shape[0])
    distances = np.linalg.norm(arr[None, :, :3] - query[:, None, :3], axis=2)
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    row = np.arange(query.shape[0])[:, None]
    order = np.argsort(distances[row, indices], axis=1)
    indices = indices[row, order]
    return distances[row, indices], indices


def _sampling_count(n_points: int, count: int | None, ratio: float | None, replace: bool) -> int:
    if (count is None) == (ratio is None):
        raise ValueError("provide exactly one of count or ratio")
    if count is not None:
        sample_count = int(count)
        if sample_count < 0:
            raise ValueError("count must be non-negative")
    else:
        ratio = float(ratio)
        if ratio < 0.0:
            raise ValueError("ratio must be non-negative")
        if ratio > 1.0 and not replace:
            raise ValueError("ratio cannot exceed 1.0 when replace=False")
        sample_count = int(np.ceil(n_points * ratio))

    if not replace and sample_count > n_points:
        raise ValueError("count cannot exceed the number of points when replace=False")
    return sample_count


def _radius_neighbors(points: np.ndarray, queries: np.ndarray, radius: float) -> list[np.ndarray]:
    arr = _as_points(points).astype(np.float64, copy=False)
    query = np.asarray(queries, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[1] < 3:
        raise ValueError("queries must have shape (Q, 3+) or (3,)")
    if arr.shape[0] == 0:
        return [np.empty((0,), dtype=np.int64) for _ in range(query.shape[0])]

    if radius == 0:
        return [
            np.flatnonzero(np.all(arr[:, :3] == item[:3], axis=1)).astype(np.int64, copy=False)
            for item in query
        ]

    cell_size = float(radius)
    radius_squared = cell_size * cell_size
    point_cells = np.floor(arr[:, :3] / cell_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for point_index, cell in enumerate(point_cells):
        buckets.setdefault(tuple(int(v) for v in cell), []).append(point_index)

    offsets = tuple(product((-1, 0, 1), repeat=3))
    neighborhoods = []
    for item in query:
        cell = np.floor(item[:3] / cell_size).astype(np.int64)
        candidate_indices = []
        for offset in offsets:
            key = tuple(int(cell[dim] + offset[dim]) for dim in range(3))
            candidate_indices.extend(buckets.get(key, ()))
        if not candidate_indices:
            neighborhoods.append(np.empty((0,), dtype=np.int64))
            continue

        candidates = np.asarray(sorted(set(candidate_indices)), dtype=np.int64)
        diff = arr[candidates, :3] - item[:3]
        distances = np.einsum("ij,ij->i", diff, diff)
        neighborhoods.append(candidates[distances <= radius_squared])
    return neighborhoods


def radius_search(points: np.ndarray, queries: np.ndarray, radius: float) -> list[np.ndarray]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return _radius_neighbors(points, queries, radius)


def hybrid_search(
    points: np.ndarray,
    queries: np.ndarray,
    radius: float,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find up to `max_neighbors` nearest points within `radius` for each query."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    max_neighbors = int(max_neighbors)
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1")

    arr = _as_points(points).astype(np.float64, copy=False)
    query = np.asarray(queries, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[1] < 3:
        raise ValueError("queries must have shape (Q, 3+) or (3,)")

    neighborhoods = _radius_neighbors(arr, query, radius)
    distances = np.full((query.shape[0], max_neighbors), np.inf, dtype=np.float64)
    indices = np.full((query.shape[0], max_neighbors), -1, dtype=np.int64)
    counts = np.zeros((query.shape[0],), dtype=np.int64)

    for row, candidates in enumerate(neighborhoods):
        if candidates.size == 0:
            continue
        candidate_distances = np.linalg.norm(arr[candidates, :3] - query[row, :3], axis=1)
        order = np.argsort(candidate_distances, kind="stable")[:max_neighbors]
        selected = candidates[order]
        count = selected.size
        counts[row] = count
        indices[row, :count] = selected
        distances[row, :count] = candidate_distances[order]

    return distances, indices, counts


def calibrate_point_cloud_metric_scale(
    relative_points: np.ndarray,
    accurate_points: np.ndarray,
    correspondence: str = "index",
    fit_offset: bool = True,
    max_correspondence_distance: float | None = None,
    return_adjusted: bool = False,
):
    """Estimate an isotropic metric-scale calibration for a relative point cloud.

    The fitted model is `accurate_xyz ~= scale * relative_xyz + offset`.
    `correspondence="index"` pairs rows directly. `correspondence="nearest"`
    pairs each relative point with its nearest accurate point and can be gated
    with `max_correspondence_distance`.
    """

    relative = np.asarray(_as_points(relative_points), dtype=np.float64)
    accurate = np.asarray(_as_points(accurate_points), dtype=np.float64)
    source_xyz, target_xyz = _metric_point_correspondences(
        relative,
        accurate,
        correspondence=correspondence,
        max_correspondence_distance=max_correspondence_distance,
    )
    scale, offset = _fit_isotropic_metric_scale(source_xyz, target_xyz, fit_offset=fit_offset)
    adjusted = apply_point_cloud_metric_scale(relative_points, {"scale": scale, "offset": offset})
    rmse = _metric_rmse(source_xyz * scale + offset, target_xyz)
    calibration = {
        "kind": "point_cloud_metric_scale",
        "scale": float(scale),
        "offset": offset,
        "fit_offset": bool(fit_offset),
        "correspondence": correspondence,
        "correspondence_count": int(source_xyz.shape[0]),
        "rmse": float(rmse),
    }
    return (calibration, adjusted) if return_adjusted else calibration



def valid_point_cloud_points(points: np.ndarray, finite: bool = True, drop_zero_xyz: bool = True) -> np.ndarray:
    """Return point-cloud rows that contain valid XYZ coordinates.

    PointCloud2 readers may pad scans to a fixed row count with all-zero rows;
    `drop_zero_xyz=True` removes that padding while keeping real nonzero points.
    """

    arr = _as_points(points)
    mask = np.ones(arr.shape[0], dtype=bool)
    if finite:
        mask &= np.isfinite(arr[:, :3]).all(axis=1)
    if drop_zero_xyz:
        mask &= np.any(arr[:, :3] != 0, axis=1)
    return arr[mask].copy()

def apply_point_cloud_metric_scale(points: np.ndarray, calibration: Mapping | float, offset=None) -> np.ndarray:
    """Apply a metric-scale calibration to point-cloud XYZ columns."""

    arr = _as_points(points)
    scale, translation = _metric_scale_offset(calibration, offset=offset, width=3)
    result = arr.astype(np.result_type(arr.dtype, np.float64), copy=True)
    result[:, :3] = np.asarray(arr[:, :3], dtype=np.float64) * scale + translation
    return result


def calibrate_depth_metric_scale(
    relative_depth: np.ndarray,
    accurate_points: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    camera_matrix: np.ndarray | None = None,
    transform: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    fit_offset: bool = True,
    min_depth: float = 0.0,
    return_adjusted: bool = False,
):
    """Calibrate a relative depth image against an accurate camera-frame point cloud.

    Accurate points are rasterized into the depth image plane, then the model
    `metric_depth ~= scale * relative_depth + offset` is fitted over pixels
    valid in both arrays.
    """

    relative = np.asarray(relative_depth, dtype=np.float64)
    if relative.ndim != 2:
        raise ValueError("relative_depth must have shape (H, W)")
    accurate_depth = points_to_depth_image(
        accurate_points,
        image_shape=relative.shape,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_matrix=camera_matrix,
        transform=transform,
        fill_value=0.0,
    )
    valid = np.isfinite(relative) & np.isfinite(accurate_depth) & (relative > min_depth) & (accurate_depth > 0.0)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != relative.shape:
            raise ValueError("mask must match relative_depth shape")
        valid &= keep
    if not np.any(valid):
        raise ValueError("no valid overlapping depth samples for calibration")

    scale, offset = _fit_scalar_metric_scale(relative[valid], accurate_depth[valid], fit_offset=fit_offset)
    adjusted = apply_depth_metric_scale(relative_depth, {"scale": scale, "offset": offset}, min_depth=min_depth)
    residual = adjusted[valid] - accurate_depth[valid]
    calibration = {
        "kind": "depth_metric_scale",
        "scale": float(scale),
        "offset": float(offset),
        "fit_offset": bool(fit_offset),
        "correspondence_count": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }
    return (calibration, adjusted) if return_adjusted else calibration


def apply_depth_metric_scale(
    depth: np.ndarray,
    calibration: Mapping | float,
    offset: float | None = None,
    min_depth: float = 0.0,
    preserve_invalid: bool = True,
) -> np.ndarray:
    """Apply a metric-scale calibration to a relative depth image or sequence."""

    arr = np.asarray(depth, dtype=np.float64)
    scale, bias = _metric_scale_offset(calibration, offset=offset, width=1)
    adjusted = arr * scale + float(bias[0])
    if not preserve_invalid:
        return adjusted
    valid = np.isfinite(arr) & (arr > min_depth)
    return np.where(valid, adjusted, arr)


def iter_loop_closure_candidates(
    trajectory,
    radius: float,
    min_separation: int = 30,
    max_candidates_per_pose: int = 1,
):
    """Yield pose-index loop closure candidates using a streaming spatial index.

    Each yielded record uses the current/source pose as `source_index` and an
    earlier nearby pose as `target_index`. Candidate generation only keeps a
    lightweight spatial index of prior pose positions, so callers can process
    long trajectories without materializing all pairwise distances.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative")
    min_separation = int(min_separation)
    max_candidates_per_pose = int(max_candidates_per_pose)
    if min_separation < 1:
        raise ValueError("min_separation must be at least 1")
    if max_candidates_per_pose < 1:
        raise ValueError("max_candidates_per_pose must be at least 1")

    positions = _trajectory_positions(trajectory)
    if positions.shape[0] <= min_separation:
        return

    if radius == 0:
        buckets: dict[tuple[float, float, float], list[int]] = {}
        for index, position in enumerate(positions):
            cutoff = index - min_separation
            if cutoff >= 0:
                key = tuple(float(value) for value in positions[cutoff])
                buckets.setdefault(key, []).append(cutoff)
            key = tuple(float(value) for value in position)
            candidates = [(0.0, target_index) for target_index in buckets.get(key, ()) if target_index <= cutoff]
            for distance, target_index in candidates[:max_candidates_per_pose]:
                yield {
                    "source_index": int(index),
                    "target_index": int(target_index),
                    "pose_distance": float(distance),
                }
        return

    cell_size = float(radius)
    radius_squared = cell_size * cell_size
    buckets: dict[tuple[int, int, int], list[int]] = {}
    offsets = tuple(product((-1, 0, 1), repeat=3))

    for index, position in enumerate(positions):
        cutoff = index - min_separation
        if cutoff >= 0:
            key = _loop_closure_cell(positions[cutoff], cell_size)
            buckets.setdefault(key, []).append(cutoff)

        if not buckets:
            continue

        cell = _loop_closure_cell(position, cell_size)
        candidates = []
        for offset in offsets:
            key = tuple(int(cell[dim] + offset[dim]) for dim in range(3))
            for target_index in buckets.get(key, ()):
                diff = position - positions[target_index]
                distance_squared = float(np.dot(diff, diff))
                if distance_squared <= radius_squared:
                    candidates.append((distance_squared, int(target_index)))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[0], item[1]))
        for distance_squared, target_index in candidates[:max_candidates_per_pose]:
            yield {
                "source_index": int(index),
                "target_index": int(target_index),
                "pose_distance": float(np.sqrt(distance_squared)),
            }


def find_loop_closure_candidates(
    trajectory,
    radius: float,
    min_separation: int = 30,
    max_candidates_per_pose: int = 1,
) -> dict[str, np.ndarray]:
    """Collect loop closure candidates from `iter_loop_closure_candidates`."""

    records = list(
        iter_loop_closure_candidates(
            trajectory,
            radius=radius,
            min_separation=min_separation,
            max_candidates_per_pose=max_candidates_per_pose,
        )
    )
    return _loop_closure_records_to_arrays(records)


def verify_loop_closures(
    point_clouds,
    trajectory,
    candidates=None,
    radius: float | None = None,
    min_separation: int = 30,
    max_candidates_per_pose: int = 1,
    method: str = "point_to_point",
    voxel_size: float | None = None,
    max_correspondence_distance: float | None = None,
    min_fitness: float = 0.3,
    max_inlier_rmse: float | None = None,
    return_all: bool = False,
    **icp_kwargs,
) -> dict[str, np.ndarray]:
    """Verify loop closure candidates for aligned point cloud and pose streams.

    `point_clouds` can be a sequence of point arrays or a topic mapping with a
    `data` field. `trajectory` can be a common trajectory mapping with `position`
    and `orientation`, or an array shaped `(N, 7+)` containing XYZ + XYZW poses.
    The returned transform maps each source/current point cloud into the target
    loop-closure point cloud frame.
    """

    clouds = _point_cloud_sequence(point_clouds)
    poses = _trajectory_pose_array(trajectory)
    if len(clouds) != poses.shape[0]:
        raise ValueError("point_clouds and trajectory must contain the same number of samples")

    pose_matrices = _pose_matrices(poses)
    if candidates is None:
        if radius is None:
            raise ValueError("provide candidates or a candidate search radius")
        candidate_iter = iter_loop_closure_candidates(
            trajectory,
            radius=radius,
            min_separation=min_separation,
            max_candidates_per_pose=max_candidates_per_pose,
        )
    else:
        candidate_iter = _iter_candidate_records(candidates)

    records = []
    for candidate in candidate_iter:
        source_index = int(candidate["source_index"])
        target_index = int(candidate["target_index"])
        if source_index < 0 or source_index >= len(clouds) or target_index < 0 or target_index >= len(clouds):
            raise ValueError("loop closure candidate indices must refer to point_clouds")

        source = _loop_closure_cloud(clouds[source_index], voxel_size)
        target = _loop_closure_cloud(clouds[target_index], voxel_size)
        seed = np.linalg.inv(pose_matrices[target_index]) @ pose_matrices[source_index]
        registration_kwargs = dict(icp_kwargs)
        if method == "multi_scale":
            registration_kwargs.setdefault("max_correspondence_distances", max_correspondence_distance)
        else:
            registration_kwargs.setdefault("max_correspondence_distance", max_correspondence_distance)
        result = odometry_seeded_icp(
            source,
            target,
            odometry_transform=seed,
            method=method,
            **registration_kwargs,
        )
        accepted = bool(result["fitness"] >= min_fitness)
        if max_inlier_rmse is not None:
            accepted = accepted and bool(result["inlier_rmse"] <= max_inlier_rmse)
        if accepted or return_all:
            records.append({
                "source_index": source_index,
                "target_index": target_index,
                "pose_distance": float(candidate.get("pose_distance", np.nan)),
                "accepted": accepted,
                "fitness": float(result["fitness"]),
                "inlier_rmse": float(result["inlier_rmse"]),
                "correspondence_count": int(result["correspondence_count"]),
                "transform": result["transform"],
                "odometry_seed": result["odometry_seed"],
            })

    return _loop_closure_records_to_arrays(records, include_verification=True)


def connected_components(
    points: np.ndarray,
    radius: float,
    min_component_size: int = 1,
    return_counts: bool = False,
):
    """Label radius-connected point components."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    min_component_size = int(min_component_size)
    if min_component_size < 1:
        raise ValueError("min_component_size must be at least 1")

    arr = _as_points(points)
    labels = np.full(arr.shape[0], -1, dtype=np.int64)
    if arr.shape[0] == 0:
        counts = np.empty((0,), dtype=np.int64)
        return (labels, counts) if return_counts else labels

    neighborhoods = _radius_neighbors(arr, arr[:, :3], radius)
    raw_components: list[list[int]] = []
    visited = np.zeros(arr.shape[0], dtype=bool)
    for point_index in range(arr.shape[0]):
        if visited[point_index]:
            continue

        component = []
        queue = deque([point_index])
        visited[point_index] = True
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in neighborhoods[current]:
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        raw_components.append(component)

    component_id = 0
    counts = []
    for component in raw_components:
        if len(component) < min_component_size:
            continue
        labels[np.asarray(component, dtype=np.int64)] = component_id
        counts.append(len(component))
        component_id += 1

    counts_array = np.asarray(counts, dtype=np.int64)
    return (labels, counts_array) if return_counts else labels


def local_covariances(points: np.ndarray, k: int = 8, return_indices: bool = False):
    """Estimate per-point local XYZ covariance matrices from KNN neighborhoods."""

    arr = _as_points(points).astype(np.float64, copy=False)
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if arr.shape[0] == 0:
        covariances = np.empty((0, 3, 3), dtype=np.float64)
        indices = np.empty((0, 0), dtype=np.int64)
        return (covariances, indices) if return_indices else covariances

    neighbor_count = min(k, arr.shape[0])
    _, indices = knn_search(arr, arr[:, :3], k=neighbor_count)
    covariances = np.empty((arr.shape[0], 3, 3), dtype=np.float64)
    for point_index, neighbors in enumerate(indices):
        local = arr[neighbors, :3]
        centered = local - local.mean(axis=0)
        covariances[point_index] = centered.T @ centered / max(local.shape[0] - 1, 1)
    return (covariances, indices) if return_indices else covariances


def curvature_descriptors(points: np.ndarray, k: int = 8) -> dict[str, np.ndarray]:
    """Compute eigenvalue-based local shape descriptors for each point."""

    covariances = local_covariances(points, k=k)
    eigenvalues = np.linalg.eigvalsh(covariances)
    eigenvalues = np.clip(eigenvalues[:, ::-1], 0.0, None)
    if eigenvalues.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "eigenvalues": eigenvalues,
            "linearity": empty,
            "planarity": empty,
            "scattering": empty,
            "anisotropy": empty,
            "omnivariance": empty,
            "eigenentropy": empty,
            "curvature": empty,
            "surface_variation": empty,
        }

    l1, l2, l3 = eigenvalues.T
    eps = np.finfo(np.float64).eps
    largest = np.maximum(l1, eps)
    total = np.maximum(eigenvalues.sum(axis=1), eps)
    probabilities = eigenvalues / total[:, None]
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    entropy_terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    curvature = l3 / total

    return {
        "eigenvalues": eigenvalues,
        "linearity": (l1 - l2) / largest,
        "planarity": (l2 - l3) / largest,
        "scattering": l3 / largest,
        "anisotropy": (l1 - l3) / largest,
        "omnivariance": np.cbrt(np.prod(eigenvalues, axis=1)),
        "eigenentropy": -entropy_terms.sum(axis=1),
        "curvature": curvature,
        "surface_variation": curvature,
    }


def nearest_neighbor_distances(points: np.ndarray, k: int = 1) -> np.ndarray:
    """Return distances to each point's nearest neighbors, excluding the point itself."""

    arr = _as_points(points)
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if arr.shape[0] <= 1:
        return np.empty((arr.shape[0], 0), dtype=np.float64)

    neighbor_count = min(k + 1, arr.shape[0])
    distances, _ = knn_search(arr, arr[:, :3], k=neighbor_count)
    return distances[:, 1:]


def nearest_neighbor_distance_stats(points: np.ndarray, k: int = 1) -> dict[str, np.ndarray | float]:
    """Compute per-point and global nearest-neighbor distance statistics."""

    distances = nearest_neighbor_distances(points, k=k)
    if distances.shape[1] == 0:
        per_point = np.full((distances.shape[0],), np.nan, dtype=np.float64)
        return {
            "distances": distances,
            "per_point_mean": per_point.copy(),
            "per_point_std": per_point.copy(),
            "per_point_min": per_point.copy(),
            "per_point_max": per_point.copy(),
            "global_mean": np.nan,
            "global_std": np.nan,
            "global_min": np.nan,
            "global_max": np.nan,
        }

    return {
        "distances": distances,
        "per_point_mean": distances.mean(axis=1),
        "per_point_std": distances.std(axis=1),
        "per_point_min": distances.min(axis=1),
        "per_point_max": distances.max(axis=1),
        "global_mean": float(distances.mean()),
        "global_std": float(distances.std()),
        "global_min": float(distances.min()),
        "global_max": float(distances.max()),
    }


def estimate_normals(points: np.ndarray, k: int = 8, orient_toward: np.ndarray | None = None) -> np.ndarray:
    arr = _as_points(points).astype(np.float64, copy=False)
    if arr.shape[0] < 3:
        raise ValueError("at least three points are required to estimate normals")
    k = max(3, min(k, arr.shape[0]))
    covariances = local_covariances(arr, k=k)

    normals = np.zeros((arr.shape[0], 3), dtype=np.float64)
    for i, covariance in enumerate(covariances):
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        if orient_toward is not None and np.dot(normal, np.asarray(orient_toward) - arr[i, :3]) < 0:
            normal = -normal
        normals[i] = normal / max(np.linalg.norm(normal), np.finfo(float).eps)
    return normals


def statistical_outlier_filter(points: np.ndarray, k: int = 8, std_ratio: float = 2.0, return_mask: bool = False):
    arr = _as_points(points)
    if arr.shape[0] == 0:
        mask = np.array([], dtype=bool)
        return (arr.copy(), mask) if return_mask else arr.copy()
    neighbor_count = max(2, min(k + 1, arr.shape[0]))
    distances, _ = knn_search(arr, arr[:, :3], k=neighbor_count)
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    mask = mean_distances <= threshold
    filtered = arr[mask].copy()
    return (filtered, mask) if return_mask else filtered


def radius_outlier_filter(points: np.ndarray, radius: float, min_neighbors: int, return_mask: bool = False):
    arr = _as_points(points)
    neighborhoods = radius_search(arr, arr[:, :3], radius)
    mask = np.asarray([neighbors.size - 1 >= min_neighbors for neighbors in neighborhoods], dtype=bool)
    filtered = arr[mask].copy()
    return (filtered, mask) if return_mask else filtered


def cluster_dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")

    arr = _as_points(points)
    neighborhoods = radius_search(arr, arr[:, :3], eps)
    labels = np.full(arr.shape[0], -1, dtype=np.int64)
    visited = np.zeros(arr.shape[0], dtype=bool)
    cluster_id = 0

    for point_index in range(arr.shape[0]):
        if visited[point_index]:
            continue
        visited[point_index] = True
        neighbors = neighborhoods[point_index]
        if neighbors.size < min_points:
            continue

        labels[point_index] = cluster_id
        queue = deque(int(i) for i in neighbors if i != point_index)
        while queue:
            neighbor = queue.popleft()
            if not visited[neighbor]:
                visited[neighbor] = True
                neighbor_neighbors = neighborhoods[neighbor]
                if neighbor_neighbors.size >= min_points:
                    queue.extend(int(i) for i in neighbor_neighbors if labels[i] < 0)
            if labels[neighbor] < 0:
                labels[neighbor] = cluster_id
        cluster_id += 1

    return labels


def _plane_from_points(points: np.ndarray) -> np.ndarray | None:
    p0, p1, p2 = points
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm <= np.finfo(float).eps:
        return None
    normal = normal / norm
    return np.r_[normal, -np.dot(normal, p0)]


def _segment_plane_ransac(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int,
    seed: int,
    normal: np.ndarray | None = None,
    max_angle_degrees: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    arr = _as_points(points).astype(np.float64, copy=False)
    if arr.shape[0] < 3:
        raise ValueError("at least three points are required to segment a plane")

    normal_filter = None
    if normal is not None:
        normal_filter = np.asarray(normal, dtype=np.float64)
        if normal_filter.shape != (3,):
            raise ValueError("normal must have shape (3,)")
        norm = np.linalg.norm(normal_filter)
        if norm == 0:
            raise ValueError("normal must be non-zero")
        normal_filter = normal_filter / norm
    if max_angle_degrees is not None:
        if max_angle_degrees < 0:
            raise ValueError("max_angle_degrees must be non-negative")
        min_alignment = np.cos(np.deg2rad(max_angle_degrees))
    else:
        min_alignment = None

    rng = np.random.default_rng(seed)
    best_plane = None
    best_mask = np.zeros(arr.shape[0], dtype=bool)
    for _ in range(max(iterations, 1)):
        sample = arr[rng.choice(arr.shape[0], size=3, replace=False), :3]
        plane = _plane_from_points(sample)
        if plane is None:
            continue
        if normal_filter is not None and min_alignment is not None:
            if abs(float(np.dot(plane[:3], normal_filter))) < min_alignment:
                continue
        distances = np.abs(arr[:, :3] @ plane[:3] + plane[3])
        mask = distances <= distance_threshold
        if mask.sum() > best_mask.sum():
            best_plane = plane
            best_mask = mask

    if best_plane is None:
        raise ValueError("could not find a non-degenerate plane")
    return best_plane, best_mask


def segment_plane(points: np.ndarray, distance_threshold: float, iterations: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return _segment_plane_ransac(points, distance_threshold, iterations, seed)


def segment_ground(
    points: np.ndarray,
    distance_threshold: float,
    up_axis=(0.0, 0.0, 1.0),
    max_slope_degrees: float = 20.0,
    iterations: int = 100,
    seed: int = 0,
    return_plane: bool = False,
):
    """Split points into ground and non-ground sets using an up-aligned RANSAC plane."""

    arr = _as_points(points)
    plane, ground_mask = _segment_plane_ransac(
        arr,
        distance_threshold=distance_threshold,
        iterations=iterations,
        seed=seed,
        normal=np.asarray(up_axis, dtype=np.float64),
        max_angle_degrees=max_slope_degrees,
    )
    up = np.asarray(up_axis, dtype=np.float64)
    up = up / np.linalg.norm(up)
    if np.dot(plane[:3], up) < 0:
        plane = -plane

    ground = arr[ground_mask].copy()
    non_ground = arr[~ground_mask].copy()
    if return_plane:
        return ground, non_ground, ground_mask, plane
    return ground, non_ground, ground_mask


def point_to_point_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_transform: np.ndarray | None = None,
    max_iterations: int = 20,
    max_correspondence_distance: float | None = None,
    relative_rmse_tolerance: float = 1.0e-6,
    relative_fitness_tolerance: float = 1.0e-6,
    min_correspondences: int = 3,
    return_correspondences: bool = False,
) -> dict:
    """Register `source` to `target` with point-to-point ICP."""

    source_arr, target_arr, transform = _registration_inputs(source, target, initial_transform)
    previous_rmse = np.inf
    previous_fitness = 0.0
    converged = False
    correspondences = _empty_correspondences()
    iteration = 0

    for iteration in range(1, max(int(max_iterations), 0) + 1):
        transformed = _transform_points_xyz(source_arr[:, :3], transform)
        correspondences = _nearest_correspondences(
            transformed,
            target_arr,
            max_correspondence_distance=max_correspondence_distance,
        )
        if correspondences["source_indices"].size < min_correspondences:
            break

        source_matches = transformed[correspondences["source_indices"]]
        target_matches = target_arr[correspondences["target_indices"], :3]
        delta = _best_fit_transform(source_matches, target_matches)
        transform = delta @ transform

        metrics = _registration_metrics(
            source_arr,
            target_arr,
            transform,
            max_correspondence_distance=max_correspondence_distance,
        )
        rmse = metrics["inlier_rmse"]
        fitness = metrics["fitness"]
        if (
            abs(previous_rmse - rmse) <= relative_rmse_tolerance
            and abs(previous_fitness - fitness) <= relative_fitness_tolerance
        ):
            correspondences = metrics["correspondences"]
            converged = True
            break
        previous_rmse = rmse
        previous_fitness = fitness
        correspondences = metrics["correspondences"]

    return _registration_result(
        transform,
        source_arr,
        target_arr,
        correspondences,
        iteration,
        converged,
        "point_to_point",
        return_correspondences=return_correspondences,
    )


def point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray | None = None,
    initial_transform: np.ndarray | None = None,
    max_iterations: int = 20,
    max_correspondence_distance: float | None = None,
    relative_rmse_tolerance: float = 1.0e-6,
    min_correspondences: int = 6,
    normal_k: int = 8,
    return_correspondences: bool = False,
) -> dict:
    """Register `source` to `target` with linearized point-to-plane ICP."""

    source_arr, target_arr, transform = _registration_inputs(source, target, initial_transform)
    normals = _registration_normals(target_arr, target_normals, normal_k)
    previous_rmse = np.inf
    converged = False
    correspondences = _empty_correspondences()
    iteration = 0

    for iteration in range(1, max(int(max_iterations), 0) + 1):
        transformed = _transform_points_xyz(source_arr[:, :3], transform)
        correspondences = _nearest_correspondences(
            transformed,
            target_arr,
            max_correspondence_distance=max_correspondence_distance,
        )
        if correspondences["source_indices"].size < min_correspondences:
            break

        source_matches = transformed[correspondences["source_indices"]]
        target_matches = target_arr[correspondences["target_indices"], :3]
        normal_matches = normals[correspondences["target_indices"]]
        delta = _point_to_plane_delta(source_matches, target_matches, normal_matches)
        transform = delta @ transform

        metrics = _registration_metrics(
            source_arr,
            target_arr,
            transform,
            target_normals=normals,
            max_correspondence_distance=max_correspondence_distance,
        )
        rmse = metrics["inlier_rmse"]
        if abs(previous_rmse - rmse) <= relative_rmse_tolerance:
            correspondences = metrics["correspondences"]
            converged = True
            break
        previous_rmse = rmse
        correspondences = metrics["correspondences"]

    return _registration_result(
        transform,
        source_arr,
        target_arr,
        correspondences,
        iteration,
        converged,
        "point_to_plane",
        target_normals=normals,
        return_correspondences=return_correspondences,
    )


def multi_scale_icp(
    source: np.ndarray,
    target: np.ndarray,
    voxel_sizes=(1.0, 0.5, 0.25),
    method: str = "point_to_point",
    initial_transform: np.ndarray | None = None,
    max_iterations: int | tuple[int, ...] | list[int] = 20,
    max_correspondence_distances: float | tuple[float, ...] | list[float] | None = None,
    **kwargs,
) -> dict:
    """Run ICP from coarse to fine voxel scales."""

    source_arr, target_arr, transform = _registration_inputs(source, target, initial_transform)
    scales = tuple(float(size) for size in voxel_sizes)
    if not scales:
        scales = (0.0,)
    if any(size < 0.0 for size in scales):
        raise ValueError("voxel_sizes must be non-negative")

    iterations = _scale_parameter(max_iterations, len(scales), "max_iterations")
    distances = _scale_parameter(max_correspondence_distances, len(scales), "max_correspondence_distances")
    levels = []
    for level, voxel_size in enumerate(scales):
        level_source = source_arr if voxel_size == 0.0 else voxel_downsample(source_arr, voxel_size)
        level_target = target_arr if voxel_size == 0.0 else voxel_downsample(target_arr, voxel_size)
        max_distance = distances[level]
        if max_distance is None and voxel_size > 0.0:
            max_distance = voxel_size * 2.0

        icp_kwargs = dict(kwargs)
        icp_kwargs.pop("return_correspondences", None)
        if method == "point_to_point":
            result = point_to_point_icp(
                level_source,
                level_target,
                initial_transform=transform,
                max_iterations=int(iterations[level]),
                max_correspondence_distance=max_distance,
                return_correspondences=False,
                **icp_kwargs,
            )
        elif method == "point_to_plane":
            result = point_to_plane_icp(
                level_source,
                level_target,
                initial_transform=transform,
                max_iterations=int(iterations[level]),
                max_correspondence_distance=max_distance,
                return_correspondences=False,
                **icp_kwargs,
            )
        else:
            raise ValueError("method must be 'point_to_point' or 'point_to_plane'")

        transform = result["transform"]
        levels.append({
            "voxel_size": voxel_size,
            "result": result,
        })

    final_metrics = _registration_metrics(source_arr, target_arr, transform)
    final = _registration_result(
        transform,
        source_arr,
        target_arr,
        final_metrics["correspondences"],
        int(sum(int(value) for value in iterations)),
        bool(levels and levels[-1]["result"]["converged"]),
        f"multi_scale_{method}",
    )
    final["levels"] = levels
    return final


def odometry_seeded_icp(
    source: np.ndarray,
    target: np.ndarray,
    odometry_transform: np.ndarray | None = None,
    source_pose: np.ndarray | None = None,
    target_pose: np.ndarray | None = None,
    method: str = "point_to_point",
    **kwargs,
) -> dict:
    """Run ICP with an odometry-derived initial source-to-target transform."""

    if odometry_transform is not None:
        seed = _as_transform_matrix(odometry_transform)
    elif source_pose is not None and target_pose is not None:
        seed = np.linalg.inv(_as_transform_matrix(target_pose)) @ _as_transform_matrix(source_pose)
    else:
        raise ValueError("provide odometry_transform or both source_pose and target_pose")

    if method == "point_to_point":
        result = point_to_point_icp(source, target, initial_transform=seed, **kwargs)
    elif method == "point_to_plane":
        result = point_to_plane_icp(source, target, initial_transform=seed, **kwargs)
    elif method == "multi_scale":
        result = multi_scale_icp(source, target, initial_transform=seed, **kwargs)
    else:
        raise ValueError("method must be 'point_to_point', 'point_to_plane', or 'multi_scale'")

    result["odometry_seed"] = seed
    return result


def to_open3d_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    color_columns: tuple[int, int, int] | None = None,
    normal_columns: tuple[int, int, int] | None = None,
    normalize_colors: bool = True,
):
    """Convert an ADE/NumPy point array to an Open3D `PointCloud`."""

    o3d = _import_open3d()
    arr = _as_points(points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(arr[:, :3], dtype=np.float64))

    color_values = _optional_columns(arr, colors, color_columns, "colors")
    if color_values is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(
            _normalize_open3d_colors(color_values, normalize=normalize_colors)
        )

    normal_values = _optional_columns(arr, normals, normal_columns, "normals")
    if normal_values is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(_normalize_vector3_array(normal_values, "normals"))

    return point_cloud


def from_open3d_point_cloud(
    point_cloud,
    include_colors: bool = True,
    include_normals: bool = True,
    as_dict: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Convert an Open3D `PointCloud` to NumPy arrays."""

    _import_open3d()
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Open3D point cloud points must have shape (N, 3)")

    colors = _open3d_optional_array(point_cloud, "colors", points.shape[0]) if include_colors else None
    normals = _open3d_optional_array(point_cloud, "normals", points.shape[0]) if include_normals else None

    if as_dict:
        result = {"points": points.copy()}
        if colors is not None:
            result["colors"] = colors.copy()
        if normals is not None:
            result["normals"] = normals.copy()
        return result

    arrays = [points]
    if colors is not None:
        arrays.append(colors)
    if normals is not None:
        arrays.append(normals)
    return np.column_stack(arrays)


def _registration_inputs(source: np.ndarray, target: np.ndarray, initial_transform: np.ndarray | None):
    source_arr = _as_points(source)
    target_arr = _as_points(target)
    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        raise ValueError("source and target must contain at least one point")
    transform = np.eye(4, dtype=np.float64) if initial_transform is None else _as_transform_matrix(initial_transform)
    return source_arr, target_arr, transform.copy()


def _transform_points_xyz(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = _as_transform_matrix(transform)
    arr = np.asarray(points, dtype=np.float64)
    return arr[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]


def _nearest_correspondences(
    transformed_source_xyz: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float | None,
) -> dict[str, np.ndarray]:
    if max_correspondence_distance is not None:
        distances, target_indices, counts = hybrid_search(
            target,
            transformed_source_xyz,
            radius=float(max_correspondence_distance),
            max_neighbors=1,
        )
        mask = counts > 0
        source_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        return {
            "source_indices": source_indices,
            "target_indices": target_indices[mask, 0],
            "distances": distances[mask, 0],
        }

    distances, target_indices = knn_search(target, transformed_source_xyz, k=1)
    return {
        "source_indices": np.arange(transformed_source_xyz.shape[0], dtype=np.int64),
        "target_indices": target_indices[:, 0],
        "distances": distances[:, 0],
    }


def _best_fit_transform(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    source_centroid = source_xyz.mean(axis=0)
    target_centroid = target_xyz.mean(axis=0)
    source_centered = source_xyz - source_centroid
    target_centered = target_xyz - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _registration_normals(target: np.ndarray, target_normals: np.ndarray | None, normal_k: int) -> np.ndarray:
    if target_normals is None:
        return estimate_normals(target, k=normal_k)

    normals = np.asarray(target_normals, dtype=np.float64)
    if normals.shape != (target.shape[0], 3):
        raise ValueError("target_normals must have shape (N, 3)")
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("target_normals cannot contain zero-length normals")
    return normals / norm


def _point_to_plane_delta(source_xyz: np.ndarray, target_xyz: np.ndarray, normals: np.ndarray) -> np.ndarray:
    cross_terms = np.cross(source_xyz, normals)
    a = np.column_stack((cross_terms, normals))
    b = -np.einsum("ij,ij->i", normals, source_xyz - target_xyz)
    twist, *_ = np.linalg.lstsq(a, b, rcond=None)
    return _se3_from_twist(twist)


def _se3_from_twist(twist: np.ndarray) -> np.ndarray:
    omega = np.asarray(twist[:3], dtype=np.float64)
    translation = np.asarray(twist[3:6], dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    if theta <= np.finfo(np.float64).eps:
        rotation = np.eye(3, dtype=np.float64)
    else:
        axis = omega / theta
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _registration_metrics(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    target_normals: np.ndarray | None = None,
    max_correspondence_distance: float | None = None,
) -> dict:
    transformed = _transform_points_xyz(source[:, :3], transform)
    correspondences = _nearest_correspondences(
        transformed,
        target,
        max_correspondence_distance=max_correspondence_distance,
    )
    return _metrics_from_correspondences(source, target, transform, correspondences, target_normals)


def _metrics_from_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    correspondences: dict[str, np.ndarray],
    target_normals: np.ndarray | None = None,
) -> dict:
    if correspondences["source_indices"].size == 0:
        return {
            "fitness": 0.0,
            "inlier_rmse": np.inf,
            "correspondences": correspondences,
        }

    transformed = _transform_points_xyz(source[:, :3], transform)
    source_matches = transformed[correspondences["source_indices"]]
    target_matches = target[correspondences["target_indices"], :3]
    if target_normals is None:
        residuals = np.linalg.norm(source_matches - target_matches, axis=1)
    else:
        normals = target_normals[correspondences["target_indices"]]
        residuals = np.abs(np.einsum("ij,ij->i", normals, source_matches - target_matches))

    return {
        "fitness": float(correspondences["source_indices"].size / source.shape[0]),
        "inlier_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "correspondences": correspondences,
    }


def _registration_result(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    correspondences: dict[str, np.ndarray],
    iterations: int,
    converged: bool,
    method: str,
    target_normals: np.ndarray | None = None,
    return_correspondences: bool = False,
) -> dict:
    metrics = _metrics_from_correspondences(source, target, transform, correspondences, target_normals)
    result = {
        "transform": transform,
        "fitness": metrics["fitness"],
        "inlier_rmse": metrics["inlier_rmse"],
        "correspondence_count": int(metrics["correspondences"]["source_indices"].size),
        "iterations": int(iterations),
        "converged": bool(converged),
        "method": method,
    }
    if return_correspondences:
        result["correspondences"] = metrics["correspondences"]
    return result


def _empty_correspondences() -> dict[str, np.ndarray]:
    return {
        "source_indices": np.empty((0,), dtype=np.int64),
        "target_indices": np.empty((0,), dtype=np.int64),
        "distances": np.empty((0,), dtype=np.float64),
    }


def _scale_parameter(value, count: int, name: str):
    if isinstance(value, (tuple, list)):
        if len(value) != count:
            raise ValueError(f"{name} length must match voxel_sizes length")
        return tuple(value)
    return tuple(value for _ in range(count))


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D point cloud adapters require the optional `open3d` dependency. "
            "Install it directly or use the `visualization` extra."
        ) from exc
    return o3d


def _optional_columns(
    points: np.ndarray,
    values: np.ndarray | None,
    columns: tuple[int, int, int] | None,
    name: str,
) -> np.ndarray | None:
    if values is not None and columns is not None:
        raise ValueError(f"provide either {name} or {name[:-1]}_columns, not both")
    if values is not None:
        return _normalize_vector3_array(values, name)
    if columns is None:
        return None
    if len(columns) != 3:
        raise ValueError(f"{name[:-1]}_columns must contain three column indices")
    if any(column < 0 or column >= points.shape[1] for column in columns):
        raise ValueError(f"{name[:-1]}_columns must refer to valid point-array columns")
    return np.asarray(points[:, columns], dtype=np.float64)


def _normalize_vector3_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return arr


def _normalize_open3d_colors(colors: np.ndarray, normalize: bool) -> np.ndarray:
    arr = _normalize_vector3_array(colors, "colors")
    if normalize and arr.size and (np.issubdtype(np.asarray(colors).dtype, np.integer) or np.nanmax(arr) > 1.0):
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _open3d_optional_array(point_cloud, attribute: str, count: int) -> np.ndarray | None:
    values = np.asarray(getattr(point_cloud, attribute), dtype=np.float64)
    if values.size == 0:
        return None
    if values.shape != (count, 3):
        raise ValueError(f"Open3D point cloud {attribute} must have shape (N, 3)")
    return values


def _metric_point_correspondences(
    relative: np.ndarray,
    accurate: np.ndarray,
    correspondence: str,
    max_correspondence_distance: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = correspondence.lower()
    rel = relative[:, :3]
    acc = accurate[:, :3]
    rel_valid = np.isfinite(rel).all(axis=1)
    acc_valid = np.isfinite(acc).all(axis=1)

    if mode == "index":
        if rel.shape[0] != acc.shape[0]:
            raise ValueError("index correspondence requires point clouds with the same row count")
        keep = rel_valid & acc_valid
        source = rel[keep]
        target = acc[keep]
    elif mode == "nearest":
        source_candidates = rel[rel_valid]
        target_candidates = acc[acc_valid]
        if source_candidates.shape[0] == 0 or target_candidates.shape[0] == 0:
            raise ValueError("point clouds must contain finite XYZ points")
        distances, indices = knn_search(target_candidates, source_candidates, k=1)
        keep = np.ones(source_candidates.shape[0], dtype=bool)
        if max_correspondence_distance is not None:
            if max_correspondence_distance < 0:
                raise ValueError("max_correspondence_distance must be non-negative")
            keep &= distances[:, 0] <= max_correspondence_distance
        source = source_candidates[keep]
        target = target_candidates[indices[keep, 0]]
    else:
        raise ValueError("correspondence must be 'index' or 'nearest'")

    if source.shape[0] < 2:
        raise ValueError("at least two finite correspondences are required for metric calibration")
    return source, target


def _fit_isotropic_metric_scale(source_xyz: np.ndarray, target_xyz: np.ndarray, fit_offset: bool) -> tuple[float, np.ndarray]:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both have shape (N, 3)")

    if fit_offset:
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        centered_source = source - source_mean
        centered_target = target - target_mean
        denominator = float(np.einsum("ij,ij->", centered_source, centered_source))
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("relative point correspondences do not span enough geometry to estimate scale")
        scale = float(np.einsum("ij,ij->", centered_source, centered_target) / denominator)
        offset = target_mean - scale * source_mean
    else:
        denominator = float(np.einsum("ij,ij->", source, source))
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("relative point correspondences do not span enough geometry to estimate scale")
        scale = float(np.einsum("ij,ij->", source, target) / denominator)
        offset = np.zeros(3, dtype=np.float64)

    if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float64).eps:
        raise ValueError("estimated metric scale is invalid")
    return scale, offset.astype(np.float64, copy=False)


def _fit_scalar_metric_scale(source: np.ndarray, target: np.ndarray, fit_offset: bool) -> tuple[float, float]:
    src = np.asarray(source, dtype=np.float64).reshape(-1)
    dst = np.asarray(target, dtype=np.float64).reshape(-1)
    keep = np.isfinite(src) & np.isfinite(dst)
    src = src[keep]
    dst = dst[keep]
    if src.size < 2:
        raise ValueError("at least two finite scalar correspondences are required for metric calibration")

    if fit_offset:
        source_mean = float(src.mean())
        target_mean = float(dst.mean())
        centered_source = src - source_mean
        centered_target = dst - target_mean
        denominator = float(np.dot(centered_source, centered_source))
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("relative depth values do not span enough range to estimate scale")
        scale = float(np.dot(centered_source, centered_target) / denominator)
        offset = target_mean - scale * source_mean
    else:
        denominator = float(np.dot(src, src))
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("relative depth values do not span enough range to estimate scale")
        scale = float(np.dot(src, dst) / denominator)
        offset = 0.0

    if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float64).eps:
        raise ValueError("estimated metric scale is invalid")
    return scale, float(offset)


def _metric_scale_offset(calibration: Mapping | float, offset, width: int) -> tuple[float, np.ndarray]:
    if isinstance(calibration, Mapping):
        scale = float(calibration["scale"])
        raw_offset = calibration.get("offset", 0.0 if offset is None else offset)
    else:
        scale = float(calibration)
        raw_offset = 0.0 if offset is None else offset
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")

    values = np.asarray(raw_offset, dtype=np.float64)
    if values.ndim == 0:
        values = np.full((width,), float(values), dtype=np.float64)
    if values.shape != (width,):
        raise ValueError(f"offset must be scalar or have shape ({width},)")
    if not np.isfinite(values).all():
        raise ValueError("offset must be finite")
    return scale, values


def _metric_rmse(source: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(source, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", residual, residual))))


def _trajectory_positions(trajectory) -> np.ndarray:
    if isinstance(trajectory, Mapping):
        if "position" in trajectory:
            positions = np.asarray(trajectory["position"], dtype=np.float64)
        elif "pose" in trajectory:
            positions = np.asarray(trajectory["pose"], dtype=np.float64)[..., :3]
        elif "data" in trajectory:
            data = np.asarray(trajectory["data"], dtype=np.float64)
            positions = data[..., :3]
        else:
            raise ValueError("trajectory mappings must contain 'position', 'pose', or 'data'")
    else:
        arr = np.asarray(trajectory, dtype=np.float64)
        positions = arr[..., :3]

    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError("trajectory positions must have shape (N, 3+)")
    if not np.isfinite(positions[:, :3]).all():
        raise ValueError("trajectory positions must be finite")
    return positions[:, :3]


def _trajectory_pose_array(trajectory) -> np.ndarray:
    if isinstance(trajectory, Mapping):
        if "pose" in trajectory:
            poses = np.asarray(trajectory["pose"], dtype=np.float64)
        elif "position" in trajectory and "orientation" in trajectory:
            poses = np.column_stack((
                np.asarray(trajectory["position"], dtype=np.float64),
                np.asarray(trajectory["orientation"], dtype=np.float64),
            ))
        elif "data" in trajectory:
            poses = np.asarray(trajectory["data"], dtype=np.float64)
        else:
            raise ValueError("trajectory mappings must contain pose data")
    else:
        poses = np.asarray(trajectory, dtype=np.float64)

    if poses.ndim != 2 or poses.shape[1] < 7:
        raise ValueError("trajectory poses must have shape (N, 7+) with XYZ + XYZW quaternion")
    if not np.isfinite(poses[:, :7]).all():
        raise ValueError("trajectory poses must be finite")
    return poses[:, :7]


def _pose_matrices(poses: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], arr.shape[0], axis=0)
    matrices[:, :3, :3] = quaternion_to_rotation_matrix(arr[:, 3:7])
    matrices[:, :3, 3] = arr[:, :3]
    return matrices


def _point_cloud_sequence(point_clouds):
    if isinstance(point_clouds, Mapping):
        if "data" not in point_clouds:
            raise ValueError("point cloud mappings must contain a 'data' field")
        values = point_clouds["data"]
    else:
        values = point_clouds

    if isinstance(values, np.ndarray) and values.ndim >= 3:
        return [values[index] for index in range(values.shape[0])]
    if isinstance(values, np.ndarray) and values.dtype == object and values.ndim == 1:
        return list(values)
    if isinstance(values, Sequence):
        return list(values)
    raise ValueError("point_clouds must be a sequence, object array, stacked array, or mapping with 'data'")


def _loop_closure_cell(position: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    cell = np.floor(np.asarray(position, dtype=np.float64)[:3] / cell_size).astype(np.int64)
    return tuple(int(value) for value in cell)


def _loop_closure_cloud(points, voxel_size: float | None) -> np.ndarray:
    cloud = np.asarray(_as_points(points), dtype=np.float64)
    if voxel_size is None:
        return cloud
    return voxel_downsample(cloud, voxel_size)


def _iter_candidate_records(candidates):
    if isinstance(candidates, Mapping):
        sources = np.asarray(candidates["source_index"], dtype=np.int64)
        targets = np.asarray(candidates["target_index"], dtype=np.int64)
        distances = np.asarray(candidates.get("pose_distance", np.full(sources.shape, np.nan)), dtype=np.float64)
        if sources.shape != targets.shape or sources.shape != distances.shape:
            raise ValueError("candidate arrays must have matching shapes")
        for source_index, target_index, distance in zip(sources, targets, distances, strict=True):
            yield {
                "source_index": int(source_index),
                "target_index": int(target_index),
                "pose_distance": float(distance),
            }
        return

    for candidate in candidates:
        if isinstance(candidate, Mapping):
            yield candidate
            continue
        if len(candidate) == 2:
            source_index, target_index = candidate
            distance = np.nan
        elif len(candidate) == 3:
            source_index, target_index, distance = candidate
        else:
            raise ValueError("candidate records must have 2 or 3 values")
        yield {
            "source_index": int(source_index),
            "target_index": int(target_index),
            "pose_distance": float(distance),
        }


def _loop_closure_records_to_arrays(records: list[dict], include_verification: bool = False) -> dict[str, np.ndarray]:
    count = len(records)
    result = {
        "source_index": np.asarray([record["source_index"] for record in records], dtype=np.int64),
        "target_index": np.asarray([record["target_index"] for record in records], dtype=np.int64),
        "pose_distance": np.asarray([record["pose_distance"] for record in records], dtype=np.float64),
    }
    if include_verification:
        result.update({
            "accepted": np.asarray([record["accepted"] for record in records], dtype=bool),
            "fitness": np.asarray([record["fitness"] for record in records], dtype=np.float64),
            "inlier_rmse": np.asarray([record["inlier_rmse"] for record in records], dtype=np.float64),
            "correspondence_count": np.asarray(
                [record["correspondence_count"] for record in records],
                dtype=np.int64,
            ),
            "transform": np.asarray(
                [record["transform"] for record in records],
                dtype=np.float64,
            ).reshape((count, 4, 4)),
            "odometry_seed": np.asarray(
                [record["odometry_seed"] for record in records],
                dtype=np.float64,
            ).reshape((count, 4, 4)),
        })
    return result


__all__ = [
    "apply_depth_metric_scale",
    "apply_point_cloud_metric_scale",
    "apply_transform",
    "calibrate_depth_metric_scale",
    "calibrate_point_cloud_metric_scale",
    "cluster_dbscan",
    "connected_components",
    "crop_bounds",
    "curvature_descriptors",
    "estimate_normals",
    "farthest_point_downsample",
    "find_loop_closure_candidates",
    "from_open3d_point_cloud",
    "hybrid_search",
    "iter_loop_closure_candidates",
    "knn_search",
    "local_covariances",
    "nearest_neighbor_distance_stats",
    "nearest_neighbor_distances",
    "multi_scale_icp",
    "odometry_seeded_icp",
    "point_to_plane_icp",
    "point_to_point_icp",
    "radius_outlier_filter",
    "radius_search",
    "random_downsample",
    "segment_ground",
    "segment_plane",
    "statistical_outlier_filter",
    "to_open3d_point_cloud",
    "uniform_downsample",
    "valid_point_cloud_points",
    "verify_loop_closures",
    "voxel_downsample",
]
