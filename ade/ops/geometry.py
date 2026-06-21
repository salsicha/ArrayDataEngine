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


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Return a 3x3 pinhole camera intrinsic matrix."""

    if fx == 0 or fy == 0:
        raise ValueError("fx and fy must be non-zero")
    return np.array([
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def project_points_to_image(
    points: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    image_shape: tuple[int, int] | None = None,
    transform: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    return_depth: bool = False,
):
    """Project XYZ points into image pixels as `(col, row)` coordinates."""

    intrinsics = _camera_intrinsics(fx, fy, cx, cy, camera_matrix)
    arr = _as_points(points)
    xyz = (
        _transform_xyz(arr[:, :3], transform)
        if transform is not None
        else np.asarray(arr[:, :3], dtype=np.float64)
    )
    depth = xyz[:, 2]

    valid = np.isfinite(xyz).all(axis=1) & (depth > 0.0)
    pixels = np.full((arr.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        pixels[valid, 0] = intrinsics[0] * xyz[valid, 0] / depth[valid] + intrinsics[2]
        pixels[valid, 1] = intrinsics[1] * xyz[valid, 1] / depth[valid] + intrinsics[3]

    if image_shape is not None:
        height, width = _image_height_width(image_shape)
        valid &= (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= height - 1)
        )

    if return_depth:
        return pixels, depth.copy(), valid
    return pixels, valid


def sample_image_at_pixels(
    image: np.ndarray,
    pixels: np.ndarray,
    bilinear: bool = True,
    fill_value=np.nan,
    return_mask: bool = False,
):
    """Sample an image at floating-point `(col, row)` pixel coordinates."""

    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    coords = np.asarray(pixels, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2) with `(col, row)` coordinates")

    height, width = arr.shape[:2]
    valid = (
        np.isfinite(coords).all(axis=1)
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= width - 1)
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= height - 1)
    )

    dtype = np.result_type(
        arr.dtype,
        np.asarray(fill_value).dtype,
        np.float64 if bilinear else arr.dtype,
    )
    samples = np.full((coords.shape[0], *arr.shape[2:]), fill_value, dtype=dtype)
    if not np.any(valid):
        return (samples, valid) if return_mask else samples

    cols = coords[valid, 0]
    rows = coords[valid, 1]
    if not bilinear:
        samples[valid] = arr[np.rint(rows).astype(int), np.rint(cols).astype(int)]
        return (samples, valid) if return_mask else samples

    c0 = np.floor(cols).astype(int)
    r0 = np.floor(rows).astype(int)
    c1 = np.clip(c0 + 1, 0, width - 1)
    r1 = np.clip(r0 + 1, 0, height - 1)
    wc = cols - c0
    wr = rows - r0

    w00 = (1.0 - wr) * (1.0 - wc)
    w10 = wr * (1.0 - wc)
    w01 = (1.0 - wr) * wc
    w11 = wr * wc
    if arr.ndim == 3:
        w00 = w00[:, None]
        w10 = w10[:, None]
        w01 = w01[:, None]
        w11 = w11[:, None]

    samples[valid] = (
        arr[r0, c0] * w00
        + arr[r1, c0] * w10
        + arr[r0, c1] * w01
        + arr[r1, c1] * w11
    )
    return (samples, valid) if return_mask else samples


def sample_image_at_points(
    image: np.ndarray,
    points: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    transform: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    bilinear: bool = True,
    fill_value=np.nan,
    return_mask: bool = False,
):
    """Project points into an image and sample pixel values at their projected locations."""

    pixels, projected = project_points_to_image(
        points,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_shape=np.asarray(image).shape[:2],
        transform=transform,
        camera_matrix=camera_matrix,
    )
    samples, sampled = sample_image_at_pixels(
        image,
        pixels,
        bilinear=bilinear,
        fill_value=fill_value,
        return_mask=True,
    )
    valid = projected & sampled
    return (samples, valid) if return_mask else samples


def colorize_points(
    points: np.ndarray,
    image: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    transform: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    bilinear: bool = True,
    fill_value=np.nan,
    return_mask: bool = False,
):
    """Append sampled image channels to a point cloud projected into the camera frame."""

    arr = _as_points(points)
    colors, mask = sample_image_at_points(
        image,
        arr,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        transform=transform,
        camera_matrix=camera_matrix,
        bilinear=bilinear,
        fill_value=fill_value,
        return_mask=True,
    )
    if colors.ndim == 1:
        colors = colors[:, None]
    result = np.concatenate((
        arr.astype(np.result_type(arr.dtype, colors.dtype), copy=False),
        colors,
    ), axis=1)
    return (result, mask) if return_mask else result


def points_to_depth_image(
    points: np.ndarray,
    image_shape: tuple[int, int],
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    transform: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    fill_value: float = 0.0,
    return_indices: bool = False,
):
    """Rasterize a point cloud into a depth image using nearest-depth z-buffering."""

    height, width = _image_height_width(image_shape)
    pixels, depth_values, valid = project_points_to_image(
        points,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_shape=None,
        transform=transform,
        camera_matrix=camera_matrix,
        return_depth=True,
    )

    valid_idx = np.flatnonzero(valid)
    cols = np.rint(pixels[valid_idx, 0]).astype(int)
    rows = np.rint(pixels[valid_idx, 1]).astype(int)
    in_bounds = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    valid_idx = valid_idx[in_bounds]
    cols = cols[in_bounds]
    rows = rows[in_bounds]

    depth = np.full((height, width), np.inf, dtype=np.float64)
    if valid_idx.size:
        np.minimum.at(depth, (rows, cols), depth_values[valid_idx])

    if return_indices:
        indices = np.full((height, width), -1, dtype=np.int64)
        if valid_idx.size:
            order = np.argsort(depth_values[valid_idx])[::-1]
            ordered_idx = valid_idx[order]
            ordered_cols = cols[order]
            ordered_rows = rows[order]
            for point_index, row, col in zip(ordered_idx, ordered_rows, ordered_cols):
                indices[row, col] = point_index

    depth[~np.isfinite(depth)] = fill_value
    return (depth, indices) if return_indices else depth


def rgbd_to_points(
    depth: np.ndarray,
    image: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject a depth image to XYZ and append aligned image channels."""

    if scale == 0:
        raise ValueError("scale must be non-zero")
    intrinsics = _camera_intrinsics(fx, fy, cx, cy, camera_matrix)
    depth_arr = np.asarray(depth, dtype=np.float64)
    image_arr = np.asarray(image)
    if depth_arr.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if image_arr.shape[:2] != depth_arr.shape:
        raise ValueError("image and depth must have matching height and width")

    z = depth_arr / scale
    valid = np.isfinite(z) & (z > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    rows, cols = np.nonzero(valid)
    z_valid = z[rows, cols]
    x = (cols - intrinsics[2]) * z_valid / intrinsics[0]
    y = (rows - intrinsics[3]) * z_valid / intrinsics[1]
    colors = image_arr[rows, cols]
    if colors.ndim == 1:
        colors = colors[:, None]
    return np.column_stack((x, y, z_valid, colors))


def project_dem_to_image(
    elevation: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    image_shape: tuple[int, int] | None = None,
    transform: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    return_depth: bool = False,
):
    """Project DEM grid points into image pixels, preserving the DEM grid shape."""

    points = dem_grid_to_points(elevation, x=x, y=y, resolution=resolution, origin=origin)
    pixels, depth, valid = project_points_to_image(
        points.reshape(-1, 3),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_shape=image_shape,
        transform=transform,
        camera_matrix=camera_matrix,
        return_depth=True,
    )

    grid_shape = points.shape[:2]
    pixels = pixels.reshape((*grid_shape, 2))
    valid = valid.reshape(grid_shape)
    if return_depth:
        return pixels, depth.reshape(grid_shape), valid
    return pixels, valid


def select_mask(values: np.ndarray, mask: np.ndarray, axis: int = 0, copy: bool = True) -> np.ndarray:
    """Select rows or leading grid cells from an array with a boolean mask."""

    arr = np.asarray(values)
    keep = np.asarray(mask, dtype=bool)
    if arr.ndim == 0:
        raise ValueError("values must have at least one dimension")
    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("axis is out of bounds for values")

    if axis != 0 and keep.ndim == 1 and keep.shape[0] == arr.shape[axis]:
        selected = np.compress(keep, arr, axis=axis)
    elif keep.shape == arr.shape[: keep.ndim]:
        selected = arr[keep]
    elif keep.ndim == 1 and arr.ndim > 0 and keep.shape[0] == arr.shape[axis]:
        selected = np.compress(keep, arr, axis=axis)
    else:
        raise ValueError("mask must match leading dimensions or the selected axis length")
    return selected.copy() if copy else selected


def bounds_mask(
    points: np.ndarray,
    min_bound=None,
    max_bound=None,
    columns: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a mask for rows inside axis-aligned bounds."""

    coords, min_array, max_array = _axis_bounds_inputs(points, min_bound, max_bound, columns)
    return np.logical_and(coords >= min_array, coords <= max_array).all(axis=1)


def crop_bounds(
    points: np.ndarray,
    min_bound=None,
    max_bound=None,
    columns: tuple[int, ...] | None = None,
    return_mask: bool = False,
):
    """Select rows inside axis-aligned coordinate bounds."""

    arr = np.asarray(points)
    mask = bounds_mask(arr, min_bound=min_bound, max_bound=max_bound, columns=columns)
    cropped = select_mask(arr, mask)
    return (cropped, mask) if return_mask else cropped


def oriented_bounds_mask(
    points: np.ndarray,
    center,
    extent,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Return a mask for XYZ points inside an oriented bounding box."""

    arr = _as_points(points)
    center_array = _vector3(center, "center")
    extent_array = _vector3(extent, "extent")
    if np.any(extent_array < 0.0):
        raise ValueError("extent values must be non-negative")

    if rotation is None:
        rotation_matrix = np.eye(3, dtype=np.float64)
    else:
        rotation_matrix = np.asarray(rotation, dtype=np.float64)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")

    local = (np.asarray(arr[:, :3], dtype=np.float64) - center_array) @ rotation_matrix
    half_extent = extent_array / 2.0
    return np.less_equal(np.abs(local), half_extent).all(axis=1)


def crop_oriented_bounds(
    points: np.ndarray,
    center,
    extent,
    rotation: np.ndarray | None = None,
    return_mask: bool = False,
):
    """Select points inside an oriented bounding box."""

    arr = _as_points(points)
    mask = oriented_bounds_mask(arr, center=center, extent=extent, rotation=rotation)
    cropped = select_mask(arr, mask)
    return (cropped, mask) if return_mask else cropped


def geographic_bounds_mask(
    coordinates: np.ndarray,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    columns: tuple[int, int] = (0, 1),
    wrap_longitude: bool = True,
) -> np.ndarray:
    """Return a mask for latitude/longitude coordinates inside a geographic box."""

    if min_lat > max_lat:
        raise ValueError("min_lat must be less than or equal to max_lat")
    if len(columns) != 2:
        raise ValueError("columns must contain latitude and longitude column indices")

    coords = _last_axis_columns(coordinates, columns)
    lat = coords[..., 0]
    lon = coords[..., 1]
    lat_mask = np.isfinite(lat) & (lat >= min_lat) & (lat <= max_lat)
    if min_lon <= max_lon:
        lon_mask = (lon >= min_lon) & (lon <= max_lon)
    elif wrap_longitude:
        lon_mask = (lon >= min_lon) | (lon <= max_lon)
    else:
        raise ValueError("min_lon must be less than or equal to max_lon unless wrap_longitude=True")
    return lat_mask & np.isfinite(lon) & lon_mask


def crop_geographic_bounds(
    coordinates: np.ndarray,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    columns: tuple[int, int] = (0, 1),
    wrap_longitude: bool = True,
    return_mask: bool = False,
):
    """Select latitude/longitude rows or grid cells inside a geographic bounding box."""

    arr = np.asarray(coordinates)
    mask = geographic_bounds_mask(
        arr,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        columns=columns,
        wrap_longitude=wrap_longitude,
    )
    cropped = select_mask(arr, mask)
    return (cropped, mask) if return_mask else cropped


def _transform_quaternions(quaternions: np.ndarray, transform: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions(np.asarray(quaternions, dtype=np.float64))
    rotation_q = _rotation_matrix_to_quaternion(_as_transform_matrix(transform)[:3, :3])
    return _normalize_quaternions(_quaternion_multiply(rotation_q, q))


def _camera_intrinsics(
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    matrix: np.ndarray | None,
) -> np.ndarray:
    if matrix is not None:
        intrinsics = np.asarray(matrix, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
    elif fx is None or fy is None or cx is None or cy is None:
        raise ValueError("provide fx, fy, cx, and cy or camera_matrix")

    if fx == 0 or fy == 0:
        raise ValueError("fx and fy must be non-zero")
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def _image_height_width(image_shape: tuple[int, int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width")
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    return height, width


def _axis_bounds_inputs(points: np.ndarray, min_bound, max_bound, columns: tuple[int, ...] | None):
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("points must have shape (N, D)")

    if columns is None:
        if min_bound is not None:
            dim = int(np.asarray(min_bound).size)
        elif max_bound is not None:
            dim = int(np.asarray(max_bound).size)
        else:
            dim = min(3, arr.shape[1])
        columns = tuple(range(dim))
    else:
        columns = tuple(int(column) for column in columns)
        dim = len(columns)

    if dim == 0:
        raise ValueError("bounds must include at least one dimension")
    if any(column < 0 or column >= arr.shape[1] for column in columns):
        raise ValueError("columns must be valid point-array column indices")

    min_array = _optional_bound(min_bound, dim, -np.inf, "min_bound")
    max_array = _optional_bound(max_bound, dim, np.inf, "max_bound")
    if np.any(min_array > max_array):
        raise ValueError("min_bound must be less than or equal to max_bound")

    return arr[:, columns], min_array, max_array


def _optional_bound(bound, dim: int, fill_value: float, name: str) -> np.ndarray:
    if bound is None:
        return np.full(dim, fill_value, dtype=np.float64)
    array = np.asarray(bound, dtype=np.float64)
    if array.ndim == 0 and dim == 1:
        array = array.reshape(1)
    if array.ndim != 1 or array.size != dim:
        raise ValueError(f"{name} must contain {dim} values")
    return array


def _vector3(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return array


def _last_axis_columns(values: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("coordinates must have at least one dimension")
    if any(column < 0 for column in columns):
        raise ValueError("columns must be non-negative")
    if max(columns) >= arr.shape[-1]:
        raise ValueError("columns must be valid coordinate-array column indices")
    return np.take(arr, columns, axis=-1)


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
