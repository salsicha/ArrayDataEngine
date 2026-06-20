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


def radius_search(points: np.ndarray, queries: np.ndarray, radius: float) -> list[np.ndarray]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    arr = _as_points(points)
    query = np.asarray(queries, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances = np.linalg.norm(arr[None, :, :3] - query[:, None, :3], axis=2)
    return [np.flatnonzero(row <= radius) for row in distances]


def estimate_normals(points: np.ndarray, k: int = 8, orient_toward: np.ndarray | None = None) -> np.ndarray:
    arr = _as_points(points).astype(np.float64, copy=False)
    if arr.shape[0] < 3:
        raise ValueError("at least three points are required to estimate normals")
    k = max(3, min(k, arr.shape[0]))
    _, indices = knn_search(arr, arr[:, :3], k=k)

    normals = np.zeros((arr.shape[0], 3), dtype=np.float64)
    for i, neighbors in enumerate(indices):
        local = arr[neighbors, :3]
        centered = local - local.mean(axis=0)
        covariance = centered.T @ centered / max(local.shape[0] - 1, 1)
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
    "estimate_normals",
    "knn_search",
    "radius_outlier_filter",
    "radius_search",
    "segment_plane",
    "statistical_outlier_filter",
    "voxel_downsample",
]
