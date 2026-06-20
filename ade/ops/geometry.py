from __future__ import annotations

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
