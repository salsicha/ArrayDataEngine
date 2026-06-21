from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .nav import enu_to_navsat, navsat_to_enu


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+) with XYZ in the first three columns")
    return arr


def _as_transform_matrix(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (3, 3):
        normalized = np.eye(4, dtype=np.float64)
        normalized[:3, :3] = matrix
        return normalized
    if matrix.shape == (3, 4):
        normalized = np.eye(4, dtype=np.float64)
        normalized[:3, :] = matrix
        return normalized
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError("transform must have shape (3, 3), (3, 4), or (4, 4)")


@dataclass(frozen=True)
class _FrameEdge:
    source: str
    target: str
    transforms: np.ndarray
    timestamps: np.ndarray | None = None
    inverse: bool = False

    @property
    def is_static(self) -> bool:
        return self.timestamps is None


class FrameGraph:
    """Graph of static and time-varying SE(3) transforms between named frames."""

    def __init__(self):
        self._edges: dict[str, list[_FrameEdge]] = {}

    @property
    def frames(self) -> tuple[str, ...]:
        return tuple(sorted(self._edges))

    def add_transform(
        self,
        source_frame: str,
        target_frame: str,
        transform: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> "FrameGraph":
        """Add a transform that maps coordinates from `source_frame` to `target_frame`."""

        if source_frame == target_frame:
            raise ValueError("source_frame and target_frame must differ")

        transforms, ts = _normalize_frame_transform(transform, timestamps)
        forward = _FrameEdge(source_frame, target_frame, transforms, ts)
        inverse = _FrameEdge(target_frame, source_frame, transforms, ts, inverse=True)

        self._add_edge(forward)
        self._add_edge(inverse)
        return self

    def add_static_transform(self, source_frame: str, target_frame: str, transform: np.ndarray) -> "FrameGraph":
        return self.add_transform(source_frame, target_frame, transform)

    def add_time_varying_transform(
        self,
        source_frame: str,
        target_frame: str,
        timestamps: np.ndarray,
        transforms: np.ndarray,
    ) -> "FrameGraph":
        return self.add_transform(source_frame, target_frame, transforms, timestamps=timestamps)

    def has_frame(self, frame: str) -> bool:
        return frame in self._edges

    def lookup_transform(self, source_frame: str, target_frame: str, timestamp: float | None = None) -> np.ndarray:
        """Return the composed transform from `source_frame` to `target_frame`."""

        if source_frame == target_frame:
            return np.eye(4, dtype=np.float64)
        path = self._find_path(source_frame, target_frame)
        if path is None:
            raise KeyError(f"no transform path from {source_frame!r} to {target_frame!r}")

        composed = np.eye(4, dtype=np.float64)
        for edge in path:
            composed = _edge_transform_at(edge, timestamp) @ composed
        return composed

    def transform_points(
        self,
        points: np.ndarray,
        source_frame: str,
        target_frame: str,
        timestamp: float | None = None,
    ) -> np.ndarray:
        return apply_transform(points, self.lookup_transform(source_frame, target_frame, timestamp=timestamp))

    def transform_vectors(
        self,
        vectors: np.ndarray,
        source_frame: str,
        target_frame: str,
        timestamp: float | None = None,
    ) -> np.ndarray:
        return transform_vectors(vectors, self.lookup_transform(source_frame, target_frame, timestamp=timestamp))

    def transform_poses(
        self,
        poses: np.ndarray,
        source_frame: str,
        target_frame: str,
        timestamp: float | None = None,
    ) -> np.ndarray:
        return transform_poses(poses, self.lookup_transform(source_frame, target_frame, timestamp=timestamp))

    def _add_edge(self, edge: _FrameEdge) -> None:
        self._edges.setdefault(edge.source, [])
        self._edges.setdefault(edge.target, [])
        self._edges[edge.source] = [
            candidate for candidate in self._edges[edge.source] if candidate.target != edge.target
        ]
        self._edges[edge.source].append(edge)

    def _find_path(self, source_frame: str, target_frame: str) -> list[_FrameEdge] | None:
        if source_frame not in self._edges:
            raise KeyError(f"unknown source frame: {source_frame!r}")
        if target_frame not in self._edges:
            raise KeyError(f"unknown target frame: {target_frame!r}")

        queue = deque([(source_frame, [])])
        visited = {source_frame}
        while queue:
            frame, path = queue.popleft()
            for edge in self._edges.get(frame, ()):
                if edge.target in visited:
                    continue
                next_path = [*path, edge]
                if edge.target == target_frame:
                    return next_path
                visited.add(edge.target)
                queue.append((edge.target, next_path))
        return None


def _transform_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    coords = np.asarray(xyz, dtype=np.float64)
    if coords.shape[-1] < 3:
        raise ValueError("coordinates must have XYZ in the last dimension")
    matrix = _as_transform_matrix(transform)
    transformed = coords[..., :3] @ matrix[:3, :3].T + matrix[:3, 3]
    if coords.shape[-1] == 3:
        return transformed

    result = coords.copy()
    result[..., :3] = transformed
    return result


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 3x3, 3x4, or 4x4 transform to point XYZ columns."""

    arr = _as_points(points)
    xyz = _transform_xyz(arr[:, :3], transform)

    result = arr.copy()
    result[:, :3] = xyz
    return result


def transform_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Rotate XYZ vectors by the rotation part of a transform."""

    arr = np.asarray(vectors, dtype=np.float64)
    if arr.shape[-1] < 3:
        raise ValueError("vectors must have XYZ in the last dimension")
    rotation = _as_transform_matrix(transform)[:3, :3]
    rotated = arr[..., :3] @ rotation.T
    if arr.shape[-1] == 3:
        return rotated

    result = arr.copy()
    result[..., :3] = rotated
    return result


def transform_poses(poses: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to pose arrays shaped (..., 7+) as XYZ + XYZW quaternion."""

    arr = np.asarray(poses, dtype=np.float64)
    if arr.shape[-1] < 7:
        raise ValueError("poses must have shape (..., 7+) with XYZ and XYZW quaternion")

    matrix = _as_transform_matrix(transform)
    result = arr.copy()
    result[..., :3] = _transform_xyz(arr[..., :3], matrix)
    result[..., 3:7] = _transform_quaternions(arr[..., 3:7], matrix)
    return result


def transform_odometry(odometry: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to ADE odometry arrays shaped (..., 8, 4)."""

    arr = np.asarray(odometry, dtype=np.float64)
    if arr.shape[-2:] != (8, 4):
        raise ValueError("odometry must have shape (..., 8, 4)")

    matrix = _as_transform_matrix(transform)
    rotation = matrix[:3, :3]
    result = arr.copy()
    result[..., 0, :3] = _transform_xyz(arr[..., 0, :3], matrix)
    result[..., 2, :4] = _transform_quaternions(arr[..., 2, :4], matrix)
    result[..., 4, :3] = transform_vectors(arr[..., 4, :3], matrix)
    result[..., 6, :3] = transform_vectors(arr[..., 6, :3], matrix)

    for row in (1, 3, 5, 7):
        result[..., row, :3] = _rotate_diagonal_covariances(arr[..., row, :3], rotation)
    return result


def transform_navsat(
    navsat: np.ndarray,
    transform: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
) -> np.ndarray:
    """Transform NavSat `[lat, lon, alt]` samples by converting through local ENU."""

    arr = np.asarray(navsat, dtype=np.float64)
    if arr.shape[-1] < 3:
        raise ValueError("navsat must have latitude, longitude, and altitude in the last dimension")

    enu = navsat_to_enu(arr[..., 0], arr[..., 1], arr[..., 2], ref_lat, ref_lon, ref_alt)
    transformed_enu = _transform_xyz(enu, transform)
    transformed_llh = enu_to_navsat(transformed_enu, ref_lat, ref_lon, ref_alt)

    result = arr.copy()
    result[..., :3] = transformed_llh
    return result


def dem_grid_to_points(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Convert a DEM elevation grid to a `(rows, cols, 3)` XYZ point surface."""

    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("elevation must be a two-dimensional DEM grid")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    if x is None:
        x_coords = origin[0] + np.arange(z.shape[1], dtype=np.float64) * resolution
    else:
        x_coords = np.asarray(x, dtype=np.float64)
    if y is None:
        y_coords = origin[1] + np.arange(z.shape[0], dtype=np.float64) * resolution
    else:
        y_coords = np.asarray(y, dtype=np.float64)

    if x_coords.ndim == 1 and y_coords.ndim == 1:
        if x_coords.size != z.shape[1] or y_coords.size != z.shape[0]:
            raise ValueError("x and y coordinate vectors must match elevation columns and rows")
        xx, yy = np.meshgrid(x_coords, y_coords)
    else:
        xx = np.asarray(x_coords, dtype=np.float64)
        yy = np.asarray(y_coords, dtype=np.float64)
        if xx.shape != z.shape or yy.shape != z.shape:
            raise ValueError("x and y coordinate grids must match elevation shape")

    return np.stack((xx, yy, z), axis=-1)


def transform_dem_grid(
    elevation: np.ndarray,
    transform: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Apply an SE(3) transform to a DEM grid and return transformed XYZ grid points."""

    return _transform_xyz(
        dem_grid_to_points(elevation, x=x, y=y, resolution=resolution, origin=origin),
        transform,
    )


def crop_bounds(points: np.ndarray, min_bound=None, max_bound=None, return_mask: bool = False):
    """Select points inside axis-aligned XYZ bounds."""

    arr = _as_points(points)
    mask = np.ones(arr.shape[0], dtype=bool)
    if min_bound is not None:
        mask &= np.all(arr[:, :3] >= np.asarray(min_bound, dtype=np.float64), axis=1)
    if max_bound is not None:
        mask &= np.all(arr[:, :3] <= np.asarray(max_bound, dtype=np.float64), axis=1)
    cropped = arr[mask].copy()
    return (cropped, mask) if return_mask else cropped


def _transform_quaternions(quaternions: np.ndarray, transform: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions(np.asarray(quaternions, dtype=np.float64))
    rotation_q = _rotation_matrix_to_quaternion(_as_transform_matrix(transform)[:3, :3])
    return _normalize_quaternions(_quaternion_multiply(rotation_q, q))


def _normalize_frame_transform(transform: np.ndarray, timestamps: np.ndarray | None):
    matrix = np.asarray(transform, dtype=np.float64)
    if timestamps is None:
        return _as_transform_matrix(matrix), None

    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if ts.size == 0:
        raise ValueError("timestamps cannot be empty")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if matrix.ndim != 3 or matrix.shape[0] != ts.size:
        raise ValueError("time-varying transforms must have shape (N, 3, 3), (N, 3, 4), or (N, 4, 4)")
    return np.stack([_as_transform_matrix(item) for item in matrix]), ts


def _edge_transform_at(edge: _FrameEdge, timestamp: float | None) -> np.ndarray:
    if edge.is_static:
        transform = edge.transforms
    else:
        if timestamp is None:
            raise ValueError(
                f"timestamp is required for time-varying transform {edge.source!r}->{edge.target!r}"
            )
        transform = _interpolate_transform(edge.transforms, edge.timestamps, float(timestamp))
    if edge.inverse:
        return np.linalg.inv(transform)
    return transform


def _interpolate_transform(transforms: np.ndarray, timestamps: np.ndarray, timestamp: float) -> np.ndarray:
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise ValueError("timestamp is outside the time-varying transform range")

    right = int(np.searchsorted(timestamps, timestamp, side="left"))
    if right < timestamps.size and timestamps[right] == timestamp:
        return transforms[right]
    if right == 0:
        return transforms[0]
    if right >= timestamps.size:
        return transforms[-1]

    left = right - 1
    fraction = (timestamp - timestamps[left]) / (timestamps[right] - timestamps[left])
    translation = (1.0 - fraction) * transforms[left, :3, 3] + fraction * transforms[right, :3, 3]
    q0 = _rotation_matrix_to_quaternion(transforms[left, :3, :3])
    q1 = _rotation_matrix_to_quaternion(transforms[right, :3, :3])
    quaternion = _slerp_quaternion(q0, q1, fraction)

    interpolated = np.eye(4, dtype=np.float64)
    interpolated[:3, :3] = _quaternion_to_rotation_matrix(quaternion)
    interpolated[:3, 3] = translation
    return interpolated


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero-length quaternion cannot be normalized")
    return q / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), axis=-1)


def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalize_quaternions(q0)
    q1 = _normalize_quaternions(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return _normalize_quaternions(q0 + fraction * (q1 - q0))

    theta_0 = np.arccos(dot)
    theta = theta_0 * fraction
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    return np.cos(theta) * q0 + sin_theta * (q1 - q0 * dot) / sin_theta_0


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternions(quaternion)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return _normalize_quaternions(np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ]))

    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = [
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        ]
    elif axis == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = [
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        ]
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = [
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    return _normalize_quaternions(np.asarray(quat, dtype=np.float64))


def _rotate_diagonal_covariances(covariances: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", np.asarray(rotation, dtype=np.float64) ** 2, covariances)
