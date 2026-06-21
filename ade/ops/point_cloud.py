from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import _as_points, apply_transform, crop_bounds


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


def radius_search(points: np.ndarray, queries: np.ndarray, radius: float) -> list[np.ndarray]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    arr = _as_points(points)
    query = np.asarray(queries, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances = np.linalg.norm(arr[None, :, :3] - query[:, None, :3], axis=2)
    return [np.flatnonzero(row <= radius) for row in distances]


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


def segment_plane(points: np.ndarray, distance_threshold: float, iterations: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    arr = _as_points(points).astype(np.float64, copy=False)
    if arr.shape[0] < 3:
        raise ValueError("at least three points are required to segment a plane")

    rng = np.random.default_rng(seed)
    best_plane = None
    best_mask = np.zeros(arr.shape[0], dtype=bool)
    for _ in range(max(iterations, 1)):
        sample = arr[rng.choice(arr.shape[0], size=3, replace=False), :3]
        plane = _plane_from_points(sample)
        if plane is None:
            continue
        distances = np.abs(arr[:, :3] @ plane[:3] + plane[3])
        mask = distances <= distance_threshold
        if mask.sum() > best_mask.sum():
            best_plane = plane
            best_mask = mask

    if best_plane is None:
        raise ValueError("could not find a non-degenerate plane")
    return best_plane, best_mask


__all__ = [
    "apply_transform",
    "cluster_dbscan",
    "crop_bounds",
    "curvature_descriptors",
    "estimate_normals",
    "farthest_point_downsample",
    "knn_search",
    "local_covariances",
    "nearest_neighbor_distance_stats",
    "nearest_neighbor_distances",
    "radius_outlier_filter",
    "radius_search",
    "random_downsample",
    "segment_plane",
    "statistical_outlier_filter",
    "uniform_downsample",
    "voxel_downsample",
]
