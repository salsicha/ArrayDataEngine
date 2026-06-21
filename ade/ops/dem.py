from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .nav import navsat_to_enu


def crop_raster(raster: np.ndarray, row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
    return np.asarray(raster)[row_start:row_stop, col_start:col_stop].copy()


def mosaic_tiles(tiles: dict[tuple[int, int], np.ndarray] | list[list[np.ndarray]]) -> np.ndarray:
    if isinstance(tiles, dict):
        rows = sorted({key[0] for key in tiles})
        cols = sorted({key[1] for key in tiles})
        return np.vstack([
            np.hstack([np.asarray(tiles[(row, col)]) for col in cols])
            for row in rows
        ])

    return np.vstack([np.hstack([np.asarray(tile) for tile in row]) for row in tiles])


def mosaic_dem_tiles(tiles, fill_value=np.nan, return_index: bool = False):
    """Mosaic SRTM-style DEM tiles using tile names or DEMSource messages.

    Accepts mappings of `{name: raster}`, iterables of `(name, raster)` pairs,
    or DEMSource-style messages containing `name` and `data`. Tile names must
    look like `N37W122` or `S02E003`. Missing cells in a sparse tile grid are
    filled with `fill_value`.
    """

    named_tiles = _normalize_dem_tiles(tiles)
    if not named_tiles:
        raise ValueError("tiles must contain at least one DEM tile")

    keyed_tiles: dict[tuple[int, int], np.ndarray] = {}
    tile_shape: tuple[int, int] | None = None
    for name, raster in named_tiles:
        lat, lon = _parse_dem_tile_name(name)
        arr = np.asarray(raster)
        if arr.ndim != 2:
            raise ValueError("DEM tile rasters must be two-dimensional")
        if tile_shape is None:
            tile_shape = arr.shape
        elif arr.shape != tile_shape:
            raise ValueError("all DEM tiles must have the same shape")
        key = (lat, lon)
        if key in keyed_tiles:
            raise ValueError(f"duplicate DEM tile coordinate {key}")
        keyed_tiles[key] = arr

    assert tile_shape is not None
    latitudes = np.array(sorted({lat for lat, _ in keyed_tiles}, reverse=True), dtype=np.int64)
    longitudes = np.array(sorted({lon for _, lon in keyed_tiles}), dtype=np.int64)
    dtype = np.result_type(*(tile.dtype for tile in keyed_tiles.values()), np.asarray(fill_value).dtype)
    tile_rows, tile_cols = tile_shape
    mosaic = np.full(
        (latitudes.size * tile_rows, longitudes.size * tile_cols),
        fill_value,
        dtype=dtype,
    )

    for row_index, lat in enumerate(latitudes):
        row_start = row_index * tile_rows
        row_stop = row_start + tile_rows
        for col_index, lon in enumerate(longitudes):
            tile = keyed_tiles.get((int(lat), int(lon)))
            if tile is None:
                continue
            col_start = col_index * tile_cols
            col_stop = col_start + tile_cols
            mosaic[row_start:row_stop, col_start:col_stop] = tile

    if return_index:
        return mosaic, latitudes, longitudes
    return mosaic


def resample_raster(raster: np.ndarray, shape: tuple[int, int], method: str = "bilinear") -> np.ndarray:
    """Resample a DEM/raster grid to `(rows, cols)` using nearest or bilinear sampling."""

    arr = _elevation_grid(raster)
    rows, cols = _output_shape(shape)
    row_coords = np.linspace(0.0, arr.shape[0] - 1, rows)
    col_coords = np.linspace(0.0, arr.shape[1] - 1, cols)
    sample_rows, sample_cols = np.meshgrid(row_coords, col_coords, indexing="ij")
    return sample_grid(arr, sample_rows, sample_cols, bilinear=_sampling_method(method))


def reproject_raster(
    raster: np.ndarray,
    src_bounds: tuple[float, float, float, float],
    dst_bounds: tuple[float, float, float, float] | None = None,
    shape: tuple[int, int] | None = None,
    transform=None,
    method: str = "bilinear",
    fill_value=np.nan,
) -> np.ndarray:
    """Sample a raster into a new coordinate grid.

    Bounds are `(min_x, min_y, max_x, max_y)`. `transform`, when provided, maps
    destination `x, y` coordinate arrays back into source coordinates. It can be
    either a callable returning `(x, y)` or a 3x3 homogeneous matrix.
    """

    arr = _elevation_grid(raster)
    src = _bounds(src_bounds, "src_bounds")
    dst = src if dst_bounds is None else _bounds(dst_bounds, "dst_bounds")
    rows, cols = arr.shape if shape is None else _output_shape(shape)
    x_coords = np.linspace(dst[0], dst[2], cols)
    y_coords = np.linspace(dst[1], dst[3], rows)
    dst_x, dst_y = np.meshgrid(x_coords, y_coords)
    src_x, src_y = _apply_coordinate_transform(dst_x, dst_y, transform)

    src_cols = (src_x - src[0]) / (src[2] - src[0]) * (arr.shape[1] - 1)
    src_rows = (src_y - src[1]) / (src[3] - src[1]) * (arr.shape[0] - 1)
    sampled = sample_grid(arr, src_rows, src_cols, bilinear=_sampling_method(method))
    inside = (
        (src_cols >= 0.0)
        & (src_cols <= arr.shape[1] - 1)
        & (src_rows >= 0.0)
        & (src_rows <= arr.shape[0] - 1)
    )
    if inside.all():
        return sampled
    result = np.full(sampled.shape, fill_value, dtype=np.result_type(sampled.dtype, np.asarray(fill_value).dtype))
    result[inside] = sampled[inside]
    return result


def write_dem_cache(
    cache_dir,
    name: str,
    raster: np.ndarray,
    metadata: dict | None = None,
    compressed: bool = True,
) -> Path:
    """Write a DEM tile and optional JSON metadata to a local `.npz` cache file."""

    path = _cache_path(cache_dir, name, suffix=".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, data=np.asarray(raster), metadata=np.asarray(json.dumps(metadata or {})))
    return path


def read_dem_cache(cache_dir, name: str, return_metadata: bool = False):
    """Read a DEM tile written by `write_dem_cache`."""

    path = _cache_path(cache_dir, name, suffix=".npz")
    with np.load(path, allow_pickle=False) as archive:
        data = archive["data"].copy()
        metadata = json.loads(str(archive["metadata"].item())) if "metadata" in archive else {}
    return (data, metadata) if return_metadata else data


def slope_aspect(elevation: np.ndarray, resolution: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    dz_dx, dz_dy = terrain_gradients(elevation, resolution=resolution)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    return slope, aspect


def hillshade(elevation: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0, resolution: float = 1.0) -> np.ndarray:
    slope, aspect = slope_aspect(elevation, resolution)
    azimuth_rad = np.deg2rad(360.0 - azimuth + 90.0)
    altitude_rad = np.deg2rad(altitude)
    shaded = (
        np.sin(altitude_rad) * np.cos(slope)
        + np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )
    return np.clip(shaded, 0.0, 1.0)


def sample_grid(raster: np.ndarray, rows: np.ndarray, cols: np.ndarray, bilinear: bool = True) -> np.ndarray:
    arr = np.asarray(raster)
    if arr.ndim < 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("raster must have at least one row and one column")
    row = np.asarray(rows, dtype=np.float64)
    col = np.asarray(cols, dtype=np.float64)

    if not bilinear:
        rr = np.clip(np.rint(row).astype(int), 0, arr.shape[0] - 1)
        cc = np.clip(np.rint(col).astype(int), 0, arr.shape[1] - 1)
        return arr[rr, cc]

    row = np.clip(row, 0.0, arr.shape[0] - 1)
    col = np.clip(col, 0.0, arr.shape[1] - 1)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.clip(r0 + 1, 0, arr.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, arr.shape[1] - 1)
    wr = row - r0
    wc = col - c0
    return (
        arr[r0, c0] * (1 - wr) * (1 - wc)
        + arr[r1, c0] * wr * (1 - wc)
        + arr[r0, c1] * (1 - wr) * wc
        + arr[r1, c1] * wr * wc
    )


def terrain_gradients(elevation: np.ndarray, resolution: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return `(dz_dx, dz_dy)` gradients for a DEM grid."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    arr = _elevation_grid(elevation)
    dz_dy, dz_dx = np.gradient(arr, resolution, resolution)
    return dz_dx, dz_dy


def terrain_normals(elevation: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """Estimate per-cell terrain normals as `(rows, cols, 3)` XYZ vectors."""

    dz_dx, dz_dy = terrain_gradients(elevation, resolution=resolution)
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)


def roughness_map(elevation: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Compute local elevation roughness as an edge-padded standard deviation map."""

    arr = _elevation_grid(elevation)
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_size == 1:
        return np.zeros_like(arr, dtype=np.float64)

    before = window_size // 2
    after = window_size - 1 - before
    padded = np.pad(arr, ((before, after), (before, after)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
    valid = np.isfinite(windows)
    counts = valid.sum(axis=(-2, -1))
    sums = np.where(valid, windows, 0.0).sum(axis=(-2, -1))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    squared = np.where(valid, (windows - means[..., None, None]) ** 2, 0.0).sum(axis=(-2, -1))
    variance = np.divide(squared, counts, out=np.full_like(squared, np.nan), where=counts > 0)
    return np.sqrt(variance)


def traversability_map(
    elevation: np.ndarray,
    resolution: float = 1.0,
    max_slope_degrees: float = 30.0,
    max_roughness: float | None = None,
    roughness_window: int = 3,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Score terrain traversability from 0 to 1 using slope and optional roughness."""

    if max_slope_degrees <= 0:
        raise ValueError("max_slope_degrees must be positive")
    slope, _ = slope_aspect(elevation, resolution=resolution)
    max_slope = np.deg2rad(max_slope_degrees)
    slope_score = 1.0 - np.clip(slope / max_slope, 0.0, 1.0)
    score = slope_score

    if max_roughness is not None:
        if max_roughness <= 0:
            raise ValueError("max_roughness must be positive")
        roughness = roughness_map(elevation, window_size=roughness_window)
        roughness_score = 1.0 - np.clip(roughness / max_roughness, 0.0, 1.0)
        score = np.minimum(score, roughness_score)

    score = np.where(np.isfinite(_elevation_grid(elevation)), score, 0.0)
    if return_mask:
        return score, score > 0.0
    return score


def sample_elevation(
    elevation: np.ndarray,
    x,
    y,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    bilinear: bool = True,
) -> np.ndarray:
    """Sample a DEM at local XY coordinates using the DEM grid convention."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    cols = (np.asarray(x, dtype=np.float64) - origin[0]) / resolution
    rows = (np.asarray(y, dtype=np.float64) - origin[1]) / resolution
    return sample_grid(elevation, rows, cols, bilinear=bilinear)


def sample_elevation_at_navsat(
    elevation: np.ndarray,
    navsat: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    bilinear: bool = True,
) -> np.ndarray:
    """Sample DEM elevations at WGS84 latitude/longitude/altitude points."""

    samples = np.asarray(navsat, dtype=np.float64)
    if samples.ndim == 0 or samples.shape[-1] < 3:
        raise ValueError("navsat must have latitude, longitude, and altitude in the last dimension")
    enu = navsat_to_enu(samples[..., 0], samples[..., 1], samples[..., 2], ref_lat, ref_lon, ref_alt)
    return sample_elevation(
        elevation,
        enu[..., 0],
        enu[..., 1],
        resolution=resolution,
        origin=origin,
        bilinear=bilinear,
    )


def terrain_patch(
    elevation: np.ndarray,
    center,
    size: int | tuple[int, int],
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    fill_value=np.nan,
    return_origin: bool = False,
):
    """Extract a fixed-size DEM patch centered on local XY coordinates."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    arr = _elevation_grid(elevation)
    rows, cols = _patch_size(size)
    xy = np.asarray(center, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError("center must have shape (2,)")
    center_col = int(np.rint((xy[0] - origin[0]) / resolution))
    center_row = int(np.rint((xy[1] - origin[1]) / resolution))
    row_start = center_row - rows // 2
    col_start = center_col - cols // 2

    patch = np.full(
        (rows, cols),
        fill_value,
        dtype=np.result_type(arr.dtype, np.asarray(fill_value).dtype),
    )

    source_row_start = max(row_start, 0)
    source_col_start = max(col_start, 0)
    source_row_stop = min(row_start + rows, arr.shape[0])
    source_col_stop = min(col_start + cols, arr.shape[1])
    if source_row_start < source_row_stop and source_col_start < source_col_stop:
        dest_row_start = source_row_start - row_start
        dest_col_start = source_col_start - col_start
        patch[
            dest_row_start:dest_row_start + (source_row_stop - source_row_start),
            dest_col_start:dest_col_start + (source_col_stop - source_col_start),
        ] = arr[source_row_start:source_row_stop, source_col_start:source_col_stop]

    patch_origin = (
        origin[0] + col_start * resolution,
        origin[1] + row_start * resolution,
    )
    return (patch, patch_origin) if return_origin else patch


def terrain_patch_at_navsat(
    elevation: np.ndarray,
    navsat,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    size: int | tuple[int, int] = 3,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    fill_value=np.nan,
    return_origin: bool = False,
):
    """Extract a fixed-size DEM patch centered on a WGS84 NavSat point."""

    sample = np.asarray(navsat, dtype=np.float64)
    if sample.ndim == 0 or sample.shape[-1] < 3:
        raise ValueError("navsat must have latitude, longitude, and altitude in the last dimension")
    enu = navsat_to_enu(sample[..., 0], sample[..., 1], sample[..., 2], ref_lat, ref_lon, ref_alt)
    center = np.asarray(enu)[..., :2]
    if center.ndim != 1:
        raise ValueError("terrain_patch_at_navsat expects a single NavSat sample")
    return terrain_patch(
        elevation,
        center,
        size=size,
        resolution=resolution,
        origin=origin,
        fill_value=fill_value,
        return_origin=return_origin,
    )


def dem_to_point_cloud(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    include_nan: bool = False,
) -> np.ndarray:
    """Convert a DEM grid to an unorganized XYZ point cloud."""

    points = _grid_points(elevation, x=x, y=y, resolution=resolution, origin=origin).reshape((-1, 3))
    if include_nan:
        return points
    return points[np.isfinite(points).all(axis=1)]


def dem_to_mesh(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    include_nan: bool = False,
) -> dict[str, np.ndarray]:
    """Convert a DEM grid to triangle mesh vertices and faces."""

    grid = _grid_points(elevation, x=x, y=y, resolution=resolution, origin=origin)
    rows, cols = grid.shape[:2]
    vertices = grid.reshape((-1, 3))
    faces = []
    finite = np.isfinite(grid).all(axis=-1)
    for row in range(rows - 1):
        for col in range(cols - 1):
            quad = np.array([
                row * cols + col,
                row * cols + col + 1,
                (row + 1) * cols + col,
                (row + 1) * cols + col + 1,
            ], dtype=np.int64)
            if include_nan or finite[row:row + 2, col:col + 2].all():
                faces.append([quad[0], quad[2], quad[1]])
                faces.append([quad[1], quad[2], quad[3]])
    faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    if include_nan:
        return {"vertices": vertices, "faces": faces}
    if faces.size == 0:
        return {
            "vertices": np.empty((0, 3), dtype=vertices.dtype),
            "faces": faces,
        }
    used = np.unique(faces.ravel())
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return {
        "vertices": vertices[used],
        "faces": remap[faces],
    }


def _elevation_grid(elevation: np.ndarray) -> np.ndarray:
    arr = np.asarray(elevation, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("elevation must be a two-dimensional DEM grid")
    return arr


def _coordinate_grid(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    z = _elevation_grid(elevation)
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
        return np.meshgrid(x_coords, y_coords)

    if x_coords.shape != z.shape or y_coords.shape != z.shape:
        raise ValueError("x and y coordinate grids must match elevation shape")
    return x_coords, y_coords


def _grid_points(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    z = _elevation_grid(elevation)
    xx, yy = _coordinate_grid(z, x=x, y=y, resolution=resolution, origin=origin)
    return np.stack((xx, yy, z), axis=-1)


def _normalize_dem_tiles(tiles) -> list[tuple[object, np.ndarray]]:
    if isinstance(tiles, dict):
        return list(tiles.items())

    normalized = []
    for tile in tiles:
        if isinstance(tile, dict):
            if "name" not in tile or "data" not in tile:
                raise ValueError("DEM tile messages must contain 'name' and 'data'")
            normalized.append((tile["name"], tile["data"]))
            continue

        try:
            name, raster = tile
        except (TypeError, ValueError) as exc:
            raise ValueError("DEM tiles must be messages or (name, raster) pairs") from exc
        normalized.append((name, raster))
    return normalized


def _parse_dem_tile_name(name) -> tuple[int, int]:
    if isinstance(name, tuple) and len(name) == 2:
        return int(name[0]), int(name[1])

    tile_name = Path(str(name)).name.upper()
    match = re.match(r"^([NS])(\d+)([EW])(\d+)", tile_name)
    if match is None:
        raise ValueError("DEM tile names must look like N37W122 or S02E003")

    lat_hemi, lat_value, lon_hemi, lon_value = match.groups()
    lat = int(lat_value)
    lon = int(lon_value)
    if lat_hemi == "S":
        lat = -lat
    if lon_hemi == "W":
        lon = -lon
    return lat, lon


def _patch_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, tuple):
        if len(size) != 2:
            raise ValueError("size tuple must contain (rows, cols)")
        rows, cols = (int(size[0]), int(size[1]))
    else:
        rows = cols = int(size)
    if rows < 1 or cols < 1:
        raise ValueError("size dimensions must be at least 1")
    return rows, cols


def _output_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must contain (rows, cols)")
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("shape dimensions must be at least 1")
    return rows, cols


def _sampling_method(method: str) -> bool:
    normalized = method.lower().replace("_", "-")
    if normalized == "bilinear":
        return True
    if normalized in {"nearest", "nearest-neighbor"}:
        return False
    raise ValueError("method must be 'bilinear' or 'nearest'")


def _bounds(bounds, name: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bounds)
    if len(values) != 4:
        raise ValueError(f"{name} must contain (min_x, min_y, max_x, max_y)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    if values[0] == values[2] or values[1] == values[3]:
        raise ValueError(f"{name} must span non-zero width and height")
    return values


def _apply_coordinate_transform(x: np.ndarray, y: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    if transform is None:
        return x, y
    if callable(transform):
        src_x, src_y = transform(x, y)
        return np.asarray(src_x, dtype=np.float64), np.asarray(src_y, dtype=np.float64)

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("transform must be a callable or a 3x3 homogeneous matrix")
    homogeneous = np.stack((x, y, np.ones_like(x)), axis=0).reshape((3, -1))
    mapped = matrix @ homogeneous
    scale = np.where(mapped[2] == 0.0, 1.0, mapped[2])
    src_x = (mapped[0] / scale).reshape(x.shape)
    src_y = (mapped[1] / scale).reshape(y.shape)
    return src_x, src_y


def _cache_path(cache_dir, name: str, suffix: str) -> Path:
    safe_name = str(name).replace("/", "_").replace("\\", "_")
    if not safe_name:
        raise ValueError("cache tile name cannot be empty")
    return Path(cache_dir) / f"{safe_name}{suffix}"
