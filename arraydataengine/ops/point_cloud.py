from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import chain, product

import numpy as np

from .geometry import _as_points, _as_transform_matrix, apply_transform, crop_bounds, points_to_depth_image
from .nav import quaternion_to_rotation_matrix


# Voxel coordinates within +/-2^20 (about 1e6 voxels per axis) can be packed
# into a single int64 grouping key; anything beyond falls back to row-wise
# unique.
_VOXEL_PACK_LIMIT = 1 << 20

# Without SciPy, k-NN falls back to brute force over blocks of queries; the
# block size keeps each (block, N) distance matrix around 2M float64 values.
_KNN_MAX_QUERY_CHUNK = 1024
_KNN_CHUNK_ELEMENTS = 1 << 21
# Queries per cKDTree.query_ball_point call (bounds the Python-list output).
_BALL_QUERY_CHUNK = 4096
# Tree searches use a slightly inflated radius; candidates are then filtered
# with the exact squared-distance test so results match the brute-force path.
_RADIUS_SEARCH_SLACK = 1.0e-9
# Points per block when gathering (block, k, 3) neighborhoods.
_NEIGHBORHOOD_CHUNK = 1 << 15
# verify_loop_closures without a gating distance or voxel size gates at this
# multiple of the target cloud's median nearest-neighbor spacing.
_LOOP_CLOSURE_SPACING_FACTOR = 3.0
_SPACING_SAMPLE_COUNT = 2048


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Average points that fall into the same voxel.

    Rows with non-finite XYZ are dropped. Integer inputs are promoted to
    float64 so voxel means are not truncated; floating inputs keep their dtype.
    """

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    arr = _as_points(points)
    output_dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.dtype(np.float64)
    finite = np.isfinite(arr[:, :3]).all(axis=1)
    if not finite.all():
        arr = arr[finite]
    if arr.size == 0:
        return arr.astype(output_dtype, copy=True)

    voxels = np.floor(arr[:, :3] / voxel_size).astype(np.int64)
    if ((voxels > -_VOXEL_PACK_LIMIT) & (voxels < _VOXEL_PACK_LIMIT)).all():
        # Packing the three coordinates into one int64 turns the slow
        # row-wise unique into a 1-D integer unique. The shift keeps every
        # component non-negative, so key order == lexicographic voxel order
        # and the output matches the unique(axis=0) path exactly.
        shifted = voxels + _VOXEL_PACK_LIMIT
        keys = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
        _, inverse = np.unique(keys, return_inverse=True)
    else:
        _, inverse = np.unique(voxels, axis=0, return_inverse=True)

    downsampled = np.zeros((inverse.max() + 1, arr.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse)
    for dim in range(arr.shape[1]):
        downsampled[:, dim] = np.bincount(inverse, weights=arr[:, dim]) / counts
    return downsampled.astype(output_dtype, copy=False)


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
    """Sample points with greedy farthest-point sampling over XYZ coordinates.

    Rows with non-finite XYZ are never selected. If `start_index` refers to
    such a row, sampling starts from the first finite point instead.
    """

    arr = _as_points(points)
    n_points = arr.shape[0]
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    xyz = arr[:, :3].astype(np.float64, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    n_finite = finite_indices.size
    if count == 0 or n_finite == 0:
        indices = np.empty((0,), dtype=np.int64)
        sampled = arr[:0].copy()
        return (sampled, indices) if return_indices else sampled
    if count >= n_finite:
        indices = finite_indices.astype(np.int64, copy=False)
        sampled = arr[indices].copy()
        return (sampled, indices) if return_indices else sampled

    if start_index is None:
        rng = np.random.default_rng(seed)
        current = int(finite_indices[int(rng.integers(0, n_finite))])
    else:
        current = int(start_index)
        if current < 0 or current >= n_points:
            raise ValueError("start_index must refer to an existing point")
        if not finite[current]:
            current = int(finite_indices[0])

    indices = np.empty(count, dtype=np.int64)
    min_distances = np.full(n_points, np.inf, dtype=np.float64)
    blocked = ~finite

    for sample_index in range(count):
        indices[sample_index] = current
        blocked[current] = True
        diff = xyz - xyz[current]
        distances = np.einsum("ij,ij->i", diff, diff)
        min_distances = np.minimum(min_distances, distances)
        min_distances[blocked] = -np.inf
        if sample_index + 1 < count:
            current = int(np.argmax(min_distances))

    sampled = arr[indices].copy()
    return (sampled, indices) if return_indices else sampled


def knn_search(points: np.ndarray, queries: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and indices of the `k` nearest points for each query.

    Results have shape `(Q, min(k, N))`, sorted by increasing distance. Uses
    `scipy.spatial.cKDTree` when SciPy is installed and memory-bounded
    brute force otherwise. Rows with non-finite XYZ are never returned as
    neighbors; slots that cannot be filled (non-finite queries, or `k` larger
    than the number of finite points) hold distance `inf` and index `-1`.
    """

    arr = _as_points(points)
    query = _query_array(queries)
    if k < 1:
        raise ValueError("k must be at least 1")
    if arr.shape[0] == 0:
        return np.empty((query.shape[0], 0)), np.empty((query.shape[0], 0), dtype=np.int64)
    return _PointIndex(arr).knn(query, int(k))


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
    return _PointIndex(points).radius_neighbors(_query_array(queries), float(radius))


def radius_search(points: np.ndarray, queries: np.ndarray, radius: float) -> list[np.ndarray]:
    """Return sorted indices of the points within `radius` of each query."""

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

    arr = _as_points(points)
    query = _query_array(queries)
    return _PointIndex(arr).hybrid(query, float(radius), max_neighbors)


@lru_cache(maxsize=None)
def _scipy_ckdtree():
    """Return `scipy.spatial.cKDTree`, or None when SciPy is not installed."""

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    return cKDTree


def _query_array(queries: np.ndarray) -> np.ndarray:
    query = np.asarray(queries, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[1] < 3:
        raise ValueError("queries must have shape (Q, 3+) or (3,)")
    return np.ascontiguousarray(query[:, :3])


class _PointIndex:
    """Nearest-neighbor index over the finite XYZ rows of a point array.

    Backed by `scipy.spatial.cKDTree` when SciPy is importable; otherwise
    k-NN queries use chunked brute force and radius queries a voxel hash grid.
    Rows with non-finite XYZ are excluded, and every returned index refers to
    the original array. Building the index once and querying it repeatedly
    (e.g. across ICP iterations) avoids rebuilding the tree.
    """

    def __init__(self, points: np.ndarray):
        xyz = np.asarray(_as_points(points)[:, :3], dtype=np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        self.size = int(xyz.shape[0])
        self.points_xyz = xyz
        if finite.all():
            self._finite_indices = None
            self.xyz = xyz
        else:
            self._finite_indices = np.flatnonzero(finite).astype(np.int64, copy=False)
            self.xyz = xyz[self._finite_indices]
        self._tree = None
        self._tree_built = False
        self._buckets: dict[float, dict[tuple[int, int, int], np.ndarray]] = {}

    @property
    def finite_count(self) -> int:
        return int(self.xyz.shape[0])

    def _original(self, indices: np.ndarray) -> np.ndarray:
        return indices if self._finite_indices is None else self._finite_indices[indices]

    def tree(self):
        if not self._tree_built:
            self._tree_built = True
            ckdtree = _scipy_ckdtree()
            if ckdtree is not None and self.finite_count:
                self._tree = ckdtree(np.ascontiguousarray(self.xyz))
        return self._tree

    def knn(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        width = min(k, self.size)
        valid_k = min(width, self.finite_count)
        distances = np.full((query.shape[0], width), np.inf, dtype=np.float64)
        indices = np.full((query.shape[0], width), -1, dtype=np.int64)
        query_rows = np.isfinite(query).all(axis=1)
        all_rows = bool(query_rows.all())
        queries = query if all_rows else query[query_rows]
        if valid_k == 0 or queries.shape[0] == 0:
            return distances, indices

        tree = self.tree()
        if tree is not None:
            found_distances, found_indices = tree.query(queries, k=valid_k)
            found_distances = np.asarray(found_distances, dtype=np.float64).reshape(-1, valid_k)
            found_indices = np.asarray(found_indices, dtype=np.int64).reshape(-1, valid_k)
        else:
            found_distances, found_indices = _knn_bruteforce(self.xyz, queries, valid_k)
        found_indices = self._original(found_indices)
        if all_rows and valid_k == width:
            return found_distances, found_indices
        distances[query_rows, :valid_k] = found_distances
        indices[query_rows, :valid_k] = found_indices
        return distances, indices

    def radius_neighbors(self, query: np.ndarray, radius: float) -> list[np.ndarray]:
        if radius == 0.0 or self.tree() is None:
            return self._grid_radius_neighbors(query, radius)
        rows, cols = self._tree_radius_pairs(query, radius, np.flatnonzero(np.isfinite(query).all(axis=1)))
        counts = np.bincount(rows, minlength=query.shape[0])
        return np.split(cols, np.cumsum(counts)[:-1])

    def radius_pairs(self, query: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Return `(query_rows, point_indices)` within `radius`, sorted by row then index."""

        if radius == 0.0 or self.tree() is None:
            neighborhoods = self._grid_radius_neighbors(query, radius)
            counts = np.fromiter(map(len, neighborhoods), dtype=np.int64, count=len(neighborhoods))
            rows = np.repeat(np.arange(len(neighborhoods), dtype=np.int64), counts)
            cols = np.concatenate(neighborhoods) if neighborhoods else np.empty((0,), dtype=np.int64)
            return rows, cols.astype(np.int64, copy=False)
        return self._tree_radius_pairs(query, radius, np.flatnonzero(np.isfinite(query).all(axis=1)))

    def hybrid(self, query: np.ndarray, radius: float, max_neighbors: int):
        n_query = query.shape[0]
        distances = np.full((n_query, max_neighbors), np.inf, dtype=np.float64)
        indices = np.full((n_query, max_neighbors), -1, dtype=np.int64)
        counts = np.zeros((n_query,), dtype=np.int64)
        if n_query == 0 or self.finite_count == 0:
            return distances, indices, counts

        tree = self.tree()
        if radius == 0.0 or tree is None:
            rows, cols = self.radius_pairs(query, radius)
            self._fill_nearest(query, rows, cols, max_neighbors, distances, indices, counts)
            return distances, indices, counts

        # Ask the tree for one neighbor more than needed: if the extra one is
        # clearly farther than the last kept one, the kept set is exact.
        # Rows with a (near) tie at that boundary are resolved with a complete
        # ball query so tie-breaking matches the brute-force path (lowest index).
        query_rows = np.flatnonzero(np.isfinite(query).all(axis=1))
        if query_rows.size == 0:
            return distances, indices, counts
        request = min(max_neighbors + 1, self.finite_count)
        search_radius = radius * (1.0 + _RADIUS_SEARCH_SLACK)
        _, found = tree.query(query[query_rows], k=request, distance_upper_bound=search_radius)
        found = np.asarray(found, dtype=np.int64).reshape(-1, request)
        present = found < self.finite_count

        row_slot, slot = np.nonzero(present)
        rows = query_rows[row_slot]
        cols = found[row_slot, slot]
        diff = self.xyz[cols] - query[rows]
        inside = np.einsum("ij,ij->i", diff, diff) <= radius * radius
        rows, cols = rows[inside], self._original(cols[inside])

        if request > max_neighbors:
            full_rows = query_rows[present.all(axis=1)]
            if full_rows.size:
                full = found[present.all(axis=1)]
                exact = np.linalg.norm(self.xyz[full] - query[full_rows][:, None, :], axis=2)
                exact.sort(axis=1)
                tied = exact[:, max_neighbors] <= exact[:, max_neighbors - 1] * (1.0 + 1.0e-12)
                if tied.any():
                    # Every point that can rank in the top `max_neighbors`
                    # lies within the extra neighbor's distance.
                    tie_rows = full_rows[tied]
                    keep = ~np.isin(rows, tie_rows)
                    tie_pairs = self._tree_radius_pairs(
                        query,
                        radius,
                        tie_rows,
                        ball_radius=np.minimum(exact[tied, max_neighbors], radius) * (1.0 + _RADIUS_SEARCH_SLACK),
                    )
                    rows = np.concatenate((rows[keep], tie_pairs[0]))
                    cols = np.concatenate((cols[keep], tie_pairs[1]))

        self._fill_nearest(query, rows, cols, max_neighbors, distances, indices, counts)
        return distances, indices, counts

    def _fill_nearest(self, query, rows, cols, max_neighbors, distances, indices, counts) -> None:
        if rows.size == 0:
            return
        candidate_distances = np.linalg.norm(self.points_xyz[cols] - query[rows], axis=1)
        order = np.lexsort((cols, candidate_distances, rows))
        rows = rows[order]
        cols = cols[order]
        candidate_distances = candidate_distances[order]
        rank = np.arange(rows.size) - np.searchsorted(rows, rows, side="left")
        keep = rank < max_neighbors
        rows, rank = rows[keep], rank[keep]
        distances[rows, rank] = candidate_distances[keep]
        indices[rows, rank] = cols[keep]
        counts += np.bincount(rows, minlength=counts.size)

    def _tree_radius_pairs(self, query: np.ndarray, radius: float, query_rows: np.ndarray, ball_radius=None):
        tree = self.tree()
        radius_squared = radius * radius
        if ball_radius is None:
            ball_radius = np.full(query_rows.size, radius * (1.0 + _RADIUS_SEARCH_SLACK))
        row_parts = []
        col_parts = []
        for start in range(0, query_rows.size, _BALL_QUERY_CHUNK):
            chunk_rows = query_rows[start:start + _BALL_QUERY_CHUNK]
            neighborhoods = tree.query_ball_point(
                query[chunk_rows],
                ball_radius[start:start + _BALL_QUERY_CHUNK],
                return_sorted=False,
            )
            lengths = np.fromiter(map(len, neighborhoods), dtype=np.int64, count=len(neighborhoods))
            cols = np.fromiter(chain.from_iterable(neighborhoods), dtype=np.int64, count=int(lengths.sum()))
            rows = np.repeat(chunk_rows, lengths)
            diff = self.xyz[cols] - query[rows]
            inside = np.einsum("ij,ij->i", diff, diff) <= radius_squared
            row_parts.append(rows[inside])
            col_parts.append(cols[inside])
        if not row_parts:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        order = np.lexsort((cols, rows))
        return rows[order], self._original(cols[order])

    def _grid_radius_neighbors(self, query: np.ndarray, radius: float) -> list[np.ndarray]:
        empty = np.empty((0,), dtype=np.int64)
        if self.finite_count == 0:
            return [empty for _ in range(query.shape[0])]
        query_finite = np.isfinite(query).all(axis=1)

        if radius == 0.0:
            return [
                self._original(np.flatnonzero(np.all(self.xyz == item, axis=1)).astype(np.int64, copy=False))
                if finite else empty
                for item, finite in zip(query, query_finite)
            ]

        cell_size = float(radius)
        radius_squared = cell_size * cell_size
        buckets = self._grid_buckets(cell_size)
        offsets = tuple(product((-1, 0, 1), repeat=3))
        neighborhoods = []
        for item, finite in zip(query, query_finite):
            if not finite:
                neighborhoods.append(empty)
                continue
            cell = np.floor(item / cell_size).astype(np.int64)
            candidate_parts = []
            for offset in offsets:
                key = (int(cell[0] + offset[0]), int(cell[1] + offset[1]), int(cell[2] + offset[2]))
                bucket = buckets.get(key)
                if bucket is not None:
                    candidate_parts.append(bucket)
            if not candidate_parts:
                neighborhoods.append(empty)
                continue

            # Each point lives in exactly one cell, so the parts are disjoint.
            candidates = np.sort(np.concatenate(candidate_parts))
            diff = self.xyz[candidates] - item
            distances = np.einsum("ij,ij->i", diff, diff)
            neighborhoods.append(self._original(candidates[distances <= radius_squared]))
        return neighborhoods

    def _grid_buckets(self, cell_size: float) -> dict[tuple[int, int, int], np.ndarray]:
        buckets = self._buckets.get(cell_size)
        if buckets is None:
            cells = np.floor(self.xyz / cell_size).astype(np.int64)
            unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            order = np.argsort(inverse, kind="stable")
            splits = np.split(order.astype(np.int64, copy=False), np.cumsum(np.bincount(inverse))[:-1])
            buckets = {tuple(int(value) for value in cell): members for cell, members in zip(unique_cells, splits)}
            self._buckets[cell_size] = buckets
        return buckets


def _knn_bruteforce(points_xyz: np.ndarray, query_xyz: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN by brute force over query blocks (bounded memory)."""

    n_points = points_xyz.shape[0]
    n_query = query_xyz.shape[0]
    distances = np.empty((n_query, k), dtype=np.float64)
    indices = np.empty((n_query, k), dtype=np.int64)
    chunk = int(max(1, min(_KNN_MAX_QUERY_CHUNK, _KNN_CHUNK_ELEMENTS // max(n_points, 1))))
    px = points_xyz[:, 0][None, :]
    py = points_xyz[:, 1][None, :]
    pz = points_xyz[:, 2][None, :]
    for start in range(0, n_query, chunk):
        block = query_xyz[start:start + chunk]
        squared = px - block[:, 0:1]
        np.multiply(squared, squared, out=squared)
        axis_diff = py - block[:, 1:2]
        np.multiply(axis_diff, axis_diff, out=axis_diff)
        squared += axis_diff
        np.subtract(pz, block[:, 2:3], out=axis_diff)
        np.multiply(axis_diff, axis_diff, out=axis_diff)
        squared += axis_diff
        if k == 1:
            nearest = np.argmin(squared, axis=1)[:, None]
        elif k < n_points:
            nearest = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
        else:
            nearest = np.broadcast_to(np.arange(n_points), squared.shape).copy()
        selected = np.take_along_axis(squared, nearest, axis=1)
        order = np.argsort(selected, axis=1)
        stop = start + block.shape[0]
        indices[start:stop] = np.take_along_axis(nearest, order, axis=1)
        distances[start:stop] = np.sqrt(np.take_along_axis(selected, order, axis=1))
    return distances, indices


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


def calibrate_depth_anything_point_cloud(
    points: np.ndarray,
    calibration=None,
    scale: float | None = None,
    offset: float | None = None,
    min_depth: float | None = 0.1,
    max_depth: float | None = 8.0,
) -> np.ndarray:
    """Convert relative Depth Anything camera-frame points into metric camera-frame points.

    MapEverything publishes `/mapping/pointcloud/depth_anything` as camera-ray
    coordinates scaled by raw relative depth. The paired calibration reconstructs
    metric depth as `metric_depth = scale * relative_depth + offset`; X/Y must be
    recomputed by scaling the camera ray by the metric depth, not by applying a
    simple XYZ affine transform.
    """

    arr = _as_points(points).astype(np.float64, copy=False)
    scale_value, offset_value = _depth_anything_scale_offset(calibration, scale, offset)
    relative_depth = -arr[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        metric_depth = scale_value * relative_depth + offset_value
        ratio = metric_depth / relative_depth

    mask = np.isfinite(ratio) & np.isfinite(metric_depth) & (relative_depth > 0.0)
    if min_depth is not None:
        mask &= metric_depth >= float(min_depth)
    if max_depth is not None:
        mask &= metric_depth <= float(max_depth)

    result = arr[mask].copy()
    result[:, :2] *= ratio[mask, None]
    result[:, 2] = -metric_depth[mask]
    return result


def _depth_anything_scale_offset(calibration, scale, offset) -> tuple[float, float]:
    if calibration is not None:
        if isinstance(calibration, Mapping):
            scale = calibration.get("scale", scale)
            offset = calibration.get("offset", offset)
        else:
            values = np.asarray(calibration, dtype=np.float64).reshape(-1)
            if values.size >= 2:
                scale = values[0]
                offset = values[1]
    if scale is None or offset is None:
        raise ValueError("provide calibration or both scale and offset")
    return float(scale), float(offset)

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

    Registration always uses a gating distance so that `fitness` (the fraction
    of source points within it of the target after alignment) can reject
    unrelated scans. When `max_correspondence_distance` is None it defaults to
    `2 * voxel_size` if `voxel_size` is given, otherwise to 3x the median
    nearest-neighbor spacing of the target cloud. For `method="multi_scale"`
    levels with a positive voxel size gate at twice that size (see
    `multi_scale_icp`) and voxel-size-0 levels use the same default. Pass an
    explicit distance that covers the expected odometry drift for reliable
    results.
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
    default_gates: dict[int, float] = {}
    for candidate in candidate_iter:
        source_index = int(candidate["source_index"])
        target_index = int(candidate["target_index"])
        if source_index < 0 or source_index >= len(clouds) or target_index < 0 or target_index >= len(clouds):
            raise ValueError("loop closure candidate indices must refer to point_clouds")

        source = _loop_closure_cloud(clouds[source_index], voxel_size)
        target = _loop_closure_cloud(clouds[target_index], voxel_size)
        seed = np.linalg.inv(pose_matrices[target_index]) @ pose_matrices[source_index]

        def default_gate() -> float:
            if target_index not in default_gates:
                default_gates[target_index] = _default_loop_closure_gate(target, voxel_size)
            return default_gates[target_index]

        registration_kwargs = dict(icp_kwargs)
        if method == "multi_scale":
            distances = max_correspondence_distance
            if distances is None and "max_correspondence_distances" not in registration_kwargs:
                sizes = tuple(float(size) for size in registration_kwargs.get("voxel_sizes", (1.0, 0.5, 0.25)))
                sizes = sizes or (0.0,)
                if any(size == 0.0 for size in sizes):
                    distances = tuple(2.0 * size if size > 0.0 else default_gate() for size in sizes)
            registration_kwargs.setdefault("max_correspondence_distances", distances)
        else:
            gate = max_correspondence_distance
            registration_kwargs["max_correspondence_distance"] = default_gate() if gate is None else gate
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
    # Rows with non-finite XYZ belong to no component (label -1).
    visited = ~np.isfinite(np.asarray(arr[:, :3], dtype=np.float64)).all(axis=1)
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
    """Estimate per-point local XYZ covariance matrices from KNN neighborhoods.

    Neighborhoods are drawn from finite points only. Rows with non-finite XYZ
    get NaN covariances (and `-1` neighbor indices), keeping outputs aligned
    with the input rows.
    """

    arr = _as_points(points).astype(np.float64, copy=False)
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if arr.shape[0] == 0:
        covariances = np.empty((0, 3, 3), dtype=np.float64)
        indices = np.empty((0, 0), dtype=np.int64)
        return (covariances, indices) if return_indices else covariances

    xyz = arr[:, :3]
    finite_rows = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    neighbor_count = min(k, finite_rows.size)
    covariances = np.full((arr.shape[0], 3, 3), np.nan, dtype=np.float64)
    indices = np.full((arr.shape[0], neighbor_count), -1, dtype=np.int64)
    if finite_rows.size:
        _, neighbors = _PointIndex(xyz).knn(np.ascontiguousarray(xyz[finite_rows]), neighbor_count)
        indices[finite_rows] = neighbors
        covariances[finite_rows] = _neighborhood_covariances(xyz, neighbors)
    return (covariances, indices) if return_indices else covariances


def _neighborhood_covariances(xyz: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    covariances = np.empty((neighbors.shape[0], 3, 3), dtype=np.float64)
    denominator = max(neighbors.shape[1] - 1, 1)
    for start in range(0, neighbors.shape[0], _NEIGHBORHOOD_CHUNK):
        local = xyz[neighbors[start:start + _NEIGHBORHOOD_CHUNK]]
        centered = local - local.mean(axis=1, keepdims=True)
        covariances[start:start + _NEIGHBORHOOD_CHUNK] = (
            np.matmul(centered.transpose(0, 2, 1), centered) / denominator
        )
    return covariances


def curvature_descriptors(points: np.ndarray, k: int = 8) -> dict[str, np.ndarray]:
    """Compute eigenvalue-based local shape descriptors for each point.

    Rows with non-finite XYZ get NaN descriptors.
    """

    covariances = local_covariances(points, k=k)
    names = (
        "linearity",
        "planarity",
        "scattering",
        "anisotropy",
        "omnivariance",
        "eigenentropy",
        "curvature",
        "surface_variation",
    )
    valid = np.isfinite(covariances).all(axis=(1, 2))
    if not valid.all():
        result = {"eigenvalues": np.full((covariances.shape[0], 3), np.nan, dtype=np.float64)}
        result.update({name: np.full((covariances.shape[0],), np.nan, dtype=np.float64) for name in names})
        if valid.any():
            for name, values in _curvature_from_covariances(covariances[valid]).items():
                result[name][valid] = values
        return result
    return _curvature_from_covariances(covariances)


def _curvature_from_covariances(covariances: np.ndarray) -> dict[str, np.ndarray]:
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
    """Return distances to each point's nearest neighbors, excluding the point itself.

    Only finite points are considered; rows with non-finite XYZ are NaN.
    """

    arr = _as_points(points)
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if arr.shape[0] <= 1:
        return np.empty((arr.shape[0], 0), dtype=np.float64)

    xyz = np.asarray(arr[:, :3], dtype=np.float64)
    finite_rows = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    neighbor_count = min(k + 1, finite_rows.size)
    if finite_rows.size == arr.shape[0]:
        distances, _ = knn_search(xyz, xyz, k=neighbor_count)
        return distances[:, 1:]
    distances = np.full((arr.shape[0], max(neighbor_count - 1, 0)), np.nan, dtype=np.float64)
    if neighbor_count > 1:
        found, _ = _PointIndex(xyz).knn(np.ascontiguousarray(xyz[finite_rows]), neighbor_count)
        distances[finite_rows] = found[:, 1:]
    return distances


def nearest_neighbor_distance_stats(points: np.ndarray, k: int = 1) -> dict[str, np.ndarray | float]:
    """Compute per-point and global nearest-neighbor distance statistics.

    Global statistics ignore rows with non-finite XYZ.
    """

    distances = nearest_neighbor_distances(points, k=k)
    finite_distances = distances[np.isfinite(distances).all(axis=1)]
    if distances.shape[1] == 0 or finite_distances.shape[0] == 0:
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
        "global_mean": float(finite_distances.mean()),
        "global_std": float(finite_distances.std()),
        "global_min": float(finite_distances.min()),
        "global_max": float(finite_distances.max()),
    }


def estimate_normals(points: np.ndarray, k: int = 8, orient_toward: np.ndarray | None = None) -> np.ndarray:
    """Estimate unit normals from the smallest-eigenvalue direction of local covariances.

    With `orient_toward`, normals are flipped to face that point. Rows with
    non-finite XYZ get NaN normals and are not used as neighbors.
    """

    arr = _as_points(points).astype(np.float64, copy=False)
    if arr.shape[0] < 3:
        raise ValueError("at least three points are required to estimate normals")
    finite = np.isfinite(arr[:, :3]).all(axis=1)
    finite_count = int(finite.sum())
    if finite_count < 3:
        raise ValueError("at least three finite points are required to estimate normals")
    k = max(3, min(k, finite_count))
    covariances = local_covariances(arr, k=k)

    normals = np.full((arr.shape[0], 3), np.nan, dtype=np.float64)
    rows = np.flatnonzero(finite)
    _, vectors = np.linalg.eigh(covariances[rows])
    normal = vectors[:, :, 0]
    if orient_toward is not None:
        toward = np.asarray(orient_toward) - arr[rows, :3]
        flip = np.einsum("ij,ij->i", normal, toward) < 0
        normal[flip] = -normal[flip]
    norms = np.linalg.norm(normal, axis=1)
    normals[rows] = normal / np.maximum(norms, np.finfo(float).eps)[:, None]
    return normals


def statistical_outlier_filter(points: np.ndarray, k: int = 8, std_ratio: float = 2.0, return_mask: bool = False):
    """Remove points whose mean neighbor distance is unusually large.

    Rows with non-finite XYZ are always removed (mask `False`) and do not
    take part in the neighbor statistics.
    """

    arr = _as_points(points)
    if arr.shape[0] == 0:
        mask = np.array([], dtype=bool)
        return (arr.copy(), mask) if return_mask else arr.copy()
    xyz = np.asarray(arr[:, :3], dtype=np.float64)
    finite_rows = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    mask = np.zeros(arr.shape[0], dtype=bool)
    if finite_rows.size == 1:
        # A single point has no neighbors to judge it by; keep it.
        mask[finite_rows] = True
    elif finite_rows.size > 1:
        neighbor_count = max(2, min(k + 1, finite_rows.size))
        distances, _ = _PointIndex(xyz).knn(np.ascontiguousarray(xyz[finite_rows]), neighbor_count)
        mean_distances = distances[:, 1:].mean(axis=1)
        threshold = mean_distances.mean() + std_ratio * mean_distances.std()
        mask[finite_rows] = mean_distances <= threshold
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


# Samples whose edge vectors are this close to parallel (|a x b| relative to
# |a| |b|, i.e. sin of the angle between them) are treated as collinear.
_PLANE_DEGENERACY_TOLERANCE = 1.0e-12


def _plane_from_points(points: np.ndarray) -> np.ndarray | None:
    p0, p1, p2 = points
    first = p1 - p0
    second = p2 - p0
    normal = np.cross(first, second)
    norm = np.linalg.norm(normal)
    scale = np.linalg.norm(first) * np.linalg.norm(second)
    if not np.isfinite(norm) or norm <= _PLANE_DEGENERACY_TOLERANCE * scale:
        return None
    normal = normal / norm
    return np.r_[normal, -np.dot(normal, p0)]


def _refine_plane(xyz: np.ndarray, inliers: np.ndarray, plane: np.ndarray) -> np.ndarray | None:
    """Least-squares plane through `inliers`, oriented like `plane`."""

    inlier_xyz = xyz[inliers]
    if inlier_xyz.shape[0] < 3:
        return None
    centroid = inlier_xyz.mean(axis=0)
    centered = inlier_xyz - centroid
    _, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    if np.dot(normal, plane[:3]) < 0:
        normal = -normal
    return np.r_[normal, -np.dot(normal, centroid)]


def _segment_plane_ransac(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int,
    seed: int,
    normal: np.ndarray | None = None,
    max_angle_degrees: float | None = None,
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    arr = _as_points(points).astype(np.float64, copy=False)
    if arr.shape[0] < 3:
        raise ValueError("at least three points are required to segment a plane")
    xyz = arr[:, :3]
    finite_rows = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    if finite_rows.size < 3:
        raise ValueError("at least three finite points are required to segment a plane")
    sample_xyz = xyz if finite_rows.size == arr.shape[0] else xyz[finite_rows]

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

    def aligned(candidate: np.ndarray) -> bool:
        if normal_filter is None or min_alignment is None:
            return True
        return abs(float(np.dot(candidate[:3], normal_filter))) >= min_alignment

    rng = np.random.default_rng(seed)
    best_plane = None
    best_mask = np.zeros(arr.shape[0], dtype=bool)
    for _ in range(max(iterations, 1)):
        sample = sample_xyz[rng.choice(sample_xyz.shape[0], size=3, replace=False)]
        plane = _plane_from_points(sample)
        if plane is None or not aligned(plane):
            continue
        distances = np.abs(xyz @ plane[:3] + plane[3])
        mask = distances <= distance_threshold
        if mask.sum() > best_mask.sum():
            best_plane = plane
            best_mask = mask

    if best_plane is None:
        raise ValueError("could not find a non-degenerate plane")
    if refine:
        # Replace the 3-point hypothesis with a least-squares fit to its
        # inliers; keep it only if it still satisfies the normal constraint
        # and does not lose inliers. The mask is always the set of points
        # within `distance_threshold` of the returned plane.
        refined = _refine_plane(xyz, best_mask, best_plane)
        if refined is not None and aligned(refined):
            refined_mask = np.abs(xyz @ refined[:3] + refined[3]) <= distance_threshold
            if refined_mask.sum() >= best_mask.sum():
                best_plane = refined
                best_mask = refined_mask
    return best_plane, best_mask


def segment_plane(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int = 100,
    seed: int = 0,
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane `[a, b, c, d]` (unit normal) with RANSAC.

    Returns the plane and the mask of points within `distance_threshold` of
    it. Rows with non-finite XYZ are never sampled or counted as inliers.
    With `refine=True` the best hypothesis is re-fitted by least squares to
    its inliers (kept only if it does not lose inliers).
    """

    return _segment_plane_ransac(points, distance_threshold, iterations, seed, refine=refine)


def segment_ground(
    points: np.ndarray,
    distance_threshold: float,
    up_axis=(0.0, 0.0, 1.0),
    max_slope_degrees: float = 20.0,
    iterations: int = 100,
    seed: int = 0,
    return_plane: bool = False,
    refine: bool = True,
):
    """Split points into ground and non-ground sets using an up-aligned RANSAC plane.

    See `segment_plane` for `refine`.
    """

    arr = _as_points(points)
    plane, ground_mask = _segment_plane_ransac(
        arr,
        distance_threshold=distance_threshold,
        iterations=iterations,
        seed=seed,
        normal=np.asarray(up_axis, dtype=np.float64),
        max_angle_degrees=max_slope_degrees,
        refine=refine,
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
    """Register `source` to `target` with point-to-point ICP.

    `fitness` is the fraction of finite source points with a correspondence
    and `inlier_rmse` the RMS point-to-point distance of those
    correspondences. Without `max_correspondence_distance` every finite
    source point is matched to its nearest target point, so `fitness` is 1.0
    by construction; pass a gating distance whenever fitness must separate
    good from bad alignments. Rows with non-finite XYZ are ignored.
    """

    source_arr, target_arr, transform = _registration_inputs(source, target, initial_transform)
    target_index = _PointIndex(target_arr)
    source_count = _finite_row_count(source_arr)
    previous_rmse = np.inf
    previous_fitness = 0.0
    converged = False
    correspondences = _empty_correspondences()
    iteration = 0
    max_iterations = max(int(max_iterations), 0)
    transformed = _transform_points_xyz(source_arr[:, :3], transform)
    if max_iterations:
        correspondences = _nearest_correspondences(transformed, target_index, max_correspondence_distance)

    for iteration in range(1, max_iterations + 1):
        if correspondences["source_indices"].size < min_correspondences:
            break

        source_matches = transformed[correspondences["source_indices"]]
        target_matches = target_arr[correspondences["target_indices"], :3]
        delta = _best_fit_transform(source_matches, target_matches)
        transform = delta @ transform

        # These correspondences also seed the next iteration, so each
        # iteration performs a single nearest-neighbor search.
        transformed = _transform_points_xyz(source_arr[:, :3], transform)
        correspondences = _nearest_correspondences(transformed, target_index, max_correspondence_distance)
        metrics = _metrics_from_correspondences(transformed, target_arr, correspondences, source_count)
        rmse = metrics["inlier_rmse"]
        fitness = metrics["fitness"]
        if (
            abs(previous_rmse - rmse) <= relative_rmse_tolerance
            and abs(previous_fitness - fitness) <= relative_fitness_tolerance
        ):
            converged = True
            break
        previous_rmse = rmse
        previous_fitness = fitness

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
    """Register `source` to `target` with linearized point-to-plane ICP.

    Convergence is judged on the point-to-plane residual, but the reported
    `inlier_rmse` is the point-to-point distance of the correspondences, as
    for the other registration methods (see `point_to_point_icp` for
    `fitness` and gating). Each step linearizes the rotation about the
    matched source centroid, so clouds far from the origin (e.g. UTM
    coordinates) converge like clouds near it.
    """

    source_arr, target_arr, transform = _registration_inputs(source, target, initial_transform)
    normals = _registration_normals(target_arr, target_normals, normal_k)
    target_index = _PointIndex(target_arr)
    previous_rmse = np.inf
    converged = False
    correspondences = _empty_correspondences()
    iteration = 0
    max_iterations = max(int(max_iterations), 0)
    transformed = _transform_points_xyz(source_arr[:, :3], transform)
    if max_iterations:
        correspondences = _nearest_correspondences(transformed, target_index, max_correspondence_distance)

    for iteration in range(1, max_iterations + 1):
        if correspondences["source_indices"].size < min_correspondences:
            break

        source_matches = transformed[correspondences["source_indices"]]
        target_matches = target_arr[correspondences["target_indices"], :3]
        normal_matches = normals[correspondences["target_indices"]]
        delta = _point_to_plane_delta(source_matches, target_matches, normal_matches)
        transform = delta @ transform

        transformed = _transform_points_xyz(source_arr[:, :3], transform)
        correspondences = _nearest_correspondences(transformed, target_index, max_correspondence_distance)
        rmse = _point_to_plane_rmse(transformed, target_arr, normals, correspondences)
        if abs(previous_rmse - rmse) <= relative_rmse_tolerance:
            converged = True
            break
        previous_rmse = rmse

    return _registration_result(
        transform,
        source_arr,
        target_arr,
        correspondences,
        iteration,
        converged,
        "point_to_plane",
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
    """Run ICP from coarse to fine voxel scales.

    Each level without an explicit correspondence distance gates at twice its
    voxel size (a level with voxel size 0 and no distance is ungated). The
    final `fitness` / `inlier_rmse` are evaluated on the full-resolution
    clouds with the finest level's gating distance, and `iterations` is the
    total number of iterations actually run across levels.
    """

    source_arr, target_arr, transform = _registration_inputs(source, target, initial_transform)
    scales = tuple(float(size) for size in voxel_sizes)
    if not scales:
        scales = (0.0,)
    if any(size < 0.0 for size in scales):
        raise ValueError("voxel_sizes must be non-negative")

    iterations = _scale_parameter(max_iterations, len(scales), "max_iterations")
    distances = _scale_parameter(max_correspondence_distances, len(scales), "max_correspondence_distances")
    levels = []
    total_iterations = 0
    max_distance = None
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
        total_iterations += int(result["iterations"])
        levels.append({
            "voxel_size": voxel_size,
            "result": result,
        })

    final_correspondences = _nearest_correspondences(
        _transform_points_xyz(source_arr[:, :3], transform),
        _PointIndex(target_arr),
        max_distance,
    )
    final = _registration_result(
        transform,
        source_arr,
        target_arr,
        final_correspondences,
        total_iterations,
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
    if _finite_row_count(source_arr) == 0 or _finite_row_count(target_arr) == 0:
        raise ValueError("source and target must contain at least one finite point")
    transform = np.eye(4, dtype=np.float64) if initial_transform is None else _as_transform_matrix(initial_transform)
    return source_arr, target_arr, transform.copy()


def _finite_row_count(points: np.ndarray) -> int:
    return int(np.isfinite(np.asarray(points[:, :3], dtype=np.float64)).all(axis=1).sum())


def _transform_points_xyz(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = _as_transform_matrix(transform)
    arr = np.asarray(points, dtype=np.float64)
    return arr[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]


def _nearest_correspondences(
    transformed_source_xyz: np.ndarray,
    target,
    max_correspondence_distance: float | None,
) -> dict[str, np.ndarray]:
    """Match each finite source point to its nearest finite target point.

    `target` is a point array or a prebuilt `_PointIndex`. With a gating
    distance, matches farther than it are dropped.
    """

    index = target if isinstance(target, _PointIndex) else _PointIndex(target)
    query = np.ascontiguousarray(transformed_source_xyz[:, :3], dtype=np.float64)
    if max_correspondence_distance is not None:
        radius = float(max_correspondence_distance)
        if radius < 0:
            raise ValueError("max_correspondence_distance must be non-negative")
        distances, target_indices, counts = index.hybrid(query, radius, 1)
        mask = counts > 0
        source_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        return {
            "source_indices": source_indices,
            "target_indices": target_indices[mask, 0],
            "distances": distances[mask, 0],
        }

    finite = np.isfinite(query).all(axis=1)
    source_indices = np.flatnonzero(finite).astype(np.int64, copy=False)
    if source_indices.size == 0 or index.finite_count == 0:
        return _empty_correspondences()
    distances, target_indices = index.knn(query if finite.all() else query[source_indices], 1)
    return {
        "source_indices": source_indices,
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
    # Linearize the rotation about the source centroid instead of the world
    # origin; otherwise the small-angle error scales with the clouds' distance
    # from the origin and ICP diverges for georeferenced (e.g. UTM) data.
    centroid = source_xyz.mean(axis=0)
    source_centered = source_xyz - centroid
    cross_terms = np.cross(source_centered, normals)
    a = np.column_stack((cross_terms, normals))
    b = -np.einsum("ij,ij->i", normals, source_xyz - target_xyz)
    twist, *_ = np.linalg.lstsq(a, b, rcond=None)
    delta = _se3_from_twist(twist)
    delta[:3, 3] += centroid - delta[:3, :3] @ centroid
    return delta


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


def _point_to_plane_rmse(
    transformed_source_xyz: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
    correspondences: dict[str, np.ndarray],
) -> float:
    if correspondences["source_indices"].size == 0:
        return np.inf
    source_matches = transformed_source_xyz[correspondences["source_indices"]]
    target_matches = target[correspondences["target_indices"], :3]
    normals = target_normals[correspondences["target_indices"]]
    residuals = np.abs(np.einsum("ij,ij->i", normals, source_matches - target_matches))
    return float(np.sqrt(np.mean(residuals ** 2)))


def _metrics_from_correspondences(
    transformed_source_xyz: np.ndarray,
    target: np.ndarray,
    correspondences: dict[str, np.ndarray],
    source_count: int,
) -> dict:
    """Fitness (inliers / finite source points) and point-to-point inlier RMSE."""

    if correspondences["source_indices"].size == 0:
        return {
            "fitness": 0.0,
            "inlier_rmse": np.inf,
            "correspondences": correspondences,
        }

    source_matches = transformed_source_xyz[correspondences["source_indices"]]
    target_matches = target[correspondences["target_indices"], :3]
    residuals = np.linalg.norm(source_matches - target_matches, axis=1)
    return {
        "fitness": float(correspondences["source_indices"].size / max(source_count, 1)),
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
    return_correspondences: bool = False,
) -> dict:
    metrics = _metrics_from_correspondences(
        _transform_points_xyz(source[:, :3], transform),
        target,
        correspondences,
        _finite_row_count(source),
    )
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


def _default_loop_closure_gate(target: np.ndarray, voxel_size: float | None) -> float:
    """Gating distance used by `verify_loop_closures` when none is given."""

    if voxel_size is not None and voxel_size > 0:
        return 2.0 * float(voxel_size)
    return _LOOP_CLOSURE_SPACING_FACTOR * _median_point_spacing(target)


def _median_point_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbor distance (sampled on up to ~2k query points)."""

    xyz = np.asarray(points[:, :3], dtype=np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.shape[0] < 2:
        return 0.0
    stride = max(1, xyz.shape[0] // _SPACING_SAMPLE_COUNT)
    distances, _ = _PointIndex(xyz).knn(np.ascontiguousarray(xyz[::stride]), 2)
    spacing = distances[:, 1]
    spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    return float(np.median(spacing)) if spacing.size else 0.0


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
    "calibrate_depth_anything_point_cloud",
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
