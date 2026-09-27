from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .core import TopicView, topic_parts
from .nav import navsat_to_enu

# Sample counts of standard SRTM HGT tiles (3 and 1 arc-second). These tiles
# repeat the edge row/column of their neighbours.
_SRTM_TILE_SIDES = (1201, 3601)


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


def mosaic_dem_tiles(
    tiles,
    fill_value=np.nan,
    return_index: bool = False,
    overlap: int | None = None,
    nodata=-32768,
):
    """Mosaic SRTM-style DEM tiles using tile names or DEMSource messages.

    Accepts mappings of `{name: raster}`, iterables of `(name, raster)` pairs,
    DEMSource-style messages containing `name` and `data`, or a buffered DEM
    topic (structured topic array or topic dict whose ids are tile names).
    Tile names must look like `N37W122` or `S02E003`. Tiles are placed
    north-up (row 0 = north), like the HGT rasters DEMSource decodes.

    `overlap` is the number of edge rows/columns adjacent tiles share. SRTM
    HGT tiles repeat their neighbour's edge (a 1-degree SRTM1 tile is
    3601 x 3601 samples, not 3600 x 3600), so their seams must be merged
    rather than duplicated. The default (`None`) uses `overlap=1` for the
    standard SRTM tile sizes (1201 and 3601 samples per side) and `0`
    otherwise; pass an explicit value for other layouts. Samples equal to
    `nodata` (SRTM voids are -32768) and NaNs are treated as missing, like
    absent tiles, and become `fill_value`; pass `nodata=None` to keep them.
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
    tile_rows, tile_cols = tile_shape
    if overlap is None:
        overlap = 1 if tile_rows == tile_cols and tile_rows in _SRTM_TILE_SIDES else 0
    overlap = int(overlap)
    if overlap < 0 or overlap >= min(tile_rows, tile_cols):
        raise ValueError("overlap must be non-negative and smaller than the tile size")
    # Span the full coordinate range so fully-missing rows/columns become
    # fill_value bands instead of silently collapsing the grid.
    present_lats = {lat for lat, _ in keyed_tiles}
    present_lons = {lon for _, lon in keyed_tiles}
    latitudes = np.arange(max(present_lats), min(present_lats) - 1, -1, dtype=np.int64)
    longitudes = np.arange(min(present_lons), max(present_lons) + 1, dtype=np.int64)
    grid_cells = int(latitudes.size) * int(longitudes.size)
    if grid_cells > max(64, 8 * len(keyed_tiles)):
        raise ValueError(
            f"tile grid spans {latitudes.size} x {longitudes.size} cells for only "
            f"{len(keyed_tiles)} tiles; filling the gaps would allocate an enormous "
            "raster. Mosaic contiguous tile subsets instead."
        )
    dtype = np.result_type(*(tile.dtype for tile in keyed_tiles.values()), np.asarray(fill_value).dtype)
    row_step = tile_rows - overlap
    col_step = tile_cols - overlap
    mosaic = np.full(
        (latitudes.size * row_step + overlap, longitudes.size * col_step + overlap),
        fill_value,
        dtype=dtype,
    )

    for row_index, lat in enumerate(latitudes):
        row_start = row_index * row_step
        row_stop = row_start + tile_rows
        for col_index, lon in enumerate(longitudes):
            tile = keyed_tiles.get((int(lat), int(lon)))
            if tile is None:
                continue
            col_start = col_index * col_step
            col_stop = col_start + tile_cols
            # Only write real samples, so voids stay `fill_value` and a void
            # on a shared seam never hides the neighbour's valid sample.
            present = _dem_samples_present(tile, nodata)
            target = mosaic[row_start:row_stop, col_start:col_stop]
            if present is None:
                target[...] = tile
            else:
                target[present] = tile[present]

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
    north_up: bool = False,
) -> np.ndarray:
    """Sample a raster into a new coordinate grid.

    Bounds are `(min_x, min_y, max_x, max_y)`. `transform`, when provided, maps
    destination `x, y` coordinate arrays back into source coordinates. It can be
    either a callable returning `(x, y)` or a 3x3 homogeneous matrix.

    Rows follow the module's native convention (row 0 = `min_y`, rows grow
    toward +y/north) unless `north_up=True`, which treats both the source and
    the output as GIS-style rasters with row 0 at `max_y` (e.g. HGT tiles and
    `mosaic_dem_tiles` output).
    """

    arr = _elevation_grid(raster)
    src = _bounds(src_bounds, "src_bounds")
    dst = src if dst_bounds is None else _bounds(dst_bounds, "dst_bounds")
    rows, cols = arr.shape if shape is None else _output_shape(shape)
    x_coords = np.linspace(dst[0], dst[2], cols)
    y_coords = np.linspace(dst[1], dst[3], rows)
    if north_up:
        y_coords = y_coords[::-1]
    dst_x, dst_y = np.meshgrid(x_coords, y_coords)
    src_x, src_y = _apply_coordinate_transform(dst_x, dst_y, transform)

    src_cols = (src_x - src[0]) / (src[2] - src[0]) * (arr.shape[1] - 1)
    src_rows = (src_y - src[1]) / (src[3] - src[1]) * (arr.shape[0] - 1)
    if north_up:
        src_rows = (arr.shape[0] - 1) - src_rows
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


def slope_aspect(
    elevation: np.ndarray,
    resolution: float | tuple[float, float] = 1.0,
    north_up: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Slope and compass aspect of a DEM grid, in radians.

    Aspect is the compass bearing of the downslope direction, clockwise from
    north in `[0, 2*pi)` (0 = north, pi/2 = east, pi = south, 3*pi/2 = west);
    flat cells have no aspect and are NaN. `resolution` is a scalar cell size
    or `(dx, dy)`.

    The module's native grid convention is row = +y = north (matching
    `terrain_normals` and `sample_elevation_at_navsat`). Pass `north_up=True`
    for GIS-style rasters where row 0 is the northernmost row — including the
    output of `mosaic_dem_tiles`."""

    slope, aspect = _slope_and_signed_aspect(elevation, resolution, north_up)
    # Wrap the (-pi, pi] bearing into [0, 2*pi).
    aspect = np.where(aspect < 0.0, aspect + 2.0 * np.pi, aspect)
    aspect = np.where(aspect >= 2.0 * np.pi, 0.0, aspect)
    aspect = np.where(slope == 0.0, np.nan, aspect)
    return slope, aspect


def _slope_and_signed_aspect(elevation, resolution, north_up: bool) -> tuple[np.ndarray, np.ndarray]:
    dz_dx, dz_dy = terrain_gradients(elevation, resolution=resolution, north_up=north_up)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    # Compass bearing of the downslope direction (-dz_dx, -dz_dy) with
    # 0 = north and pi/2 = east, in (-pi, pi].
    return slope, np.arctan2(-dz_dx, -dz_dy)


def hillshade(
    elevation: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    resolution: float | tuple[float, float] = 1.0,
    north_up: bool = False,
) -> np.ndarray:
    """Hillshade a DEM grid; see `slope_aspect` for the `north_up` convention."""

    slope, aspect = _slope_and_signed_aspect(elevation, resolution, north_up)
    # aspect is already a compass bearing, so the sun azimuth can be compared
    # directly; cos() of the relative angle is convention-independent. Flat
    # cells (NaN aspect) contribute no directional term.
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)
    directional = np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    directional = np.where(slope == 0.0, 0.0, directional)
    shaded = np.sin(altitude_rad) * np.cos(slope) + directional
    return np.clip(shaded, 0.0, 1.0)


def sample_grid(
    raster: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    bilinear: bool = True,
    fill_value=np.nan,
) -> np.ndarray:
    """Sample a raster at fractional `(row, col)` positions.

    Positions that are non-finite or lie more than half a cell outside the
    grid return `fill_value` (NaN by default); positions within that half-cell
    border are clamped to the edge. The output dtype is
    `np.result_type(raster_dtype_or_float, fill_value)`, so nearest sampling of
    an integer raster returns floats unless an integer `fill_value` is given.
    Bilinear taps with zero weight are ignored, so a sample exactly on a valid
    node is not poisoned by a NaN neighbour.
    """

    arr = np.asarray(raster)
    if arr.ndim < 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("raster must have at least one row and one column")
    row, col = np.broadcast_arrays(np.asarray(rows, dtype=np.float64), np.asarray(cols, dtype=np.float64))
    height, width = arr.shape[:2]
    with np.errstate(invalid="ignore"):
        valid = (
            np.isfinite(row)
            & np.isfinite(col)
            & (row >= -0.5)
            & (row <= height - 0.5)
            & (col >= -0.5)
            & (col <= width - 0.5)
        )
    row = np.where(valid, np.clip(row, 0.0, height - 1), 0.0)
    col = np.where(valid, np.clip(col, 0.0, width - 1), 0.0)
    channel_shape = (1,) * (arr.ndim - 2)

    if not bilinear:
        sampled = arr[np.rint(row).astype(np.intp), np.rint(col).astype(np.intp)]
    else:
        r0 = np.floor(row).astype(np.intp)
        c0 = np.floor(col).astype(np.intp)
        r1 = np.minimum(r0 + 1, height - 1)
        c1 = np.minimum(c0 + 1, width - 1)
        wr = row - r0
        wc = col - c0
        sampled = None
        for rr, cc, weight in (
            (r0, c0, (1 - wr) * (1 - wc)),
            (r1, c0, wr * (1 - wc)),
            (r0, c1, (1 - wr) * wc),
            (r1, c1, wr * wc),
        ):
            weight = weight.reshape(weight.shape + channel_shape)
            term = np.where(weight > 0.0, arr[rr, cc] * weight, 0.0)
            sampled = term if sampled is None else sampled + term

    result = sampled.astype(np.result_type(sampled.dtype, fill_value), copy=False)
    if not valid.all():
        result = result.copy()
        result[~valid] = fill_value
    return result


def terrain_gradients(
    elevation: np.ndarray,
    resolution: float | tuple[float, float] = 1.0,
    north_up: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(dz_dx, dz_dy)` gradients for a DEM grid.

    `resolution` is a scalar cell size or `(dx, dy)`. With the native layout
    rows grow toward +y; pass `north_up=True` for rasters whose row 0 is the
    northernmost row so `dz_dy` still points north.
    """

    dx, dy = _resolution_xy(resolution)
    arr = _elevation_grid(elevation)
    dz_dy, dz_dx = np.gradient(arr, dy, dx)
    if north_up:
        dz_dy = -dz_dy
    return dz_dx, dz_dy


def terrain_normals(
    elevation: np.ndarray,
    resolution: float | tuple[float, float] = 1.0,
    north_up: bool = False,
) -> np.ndarray:
    """Estimate per-cell terrain normals as `(rows, cols, 3)` XYZ vectors.

    Normals point up (+z) with +y = north; see `terrain_gradients` for
    `resolution` and `north_up`.
    """

    dz_dx, dz_dy = terrain_gradients(elevation, resolution=resolution, north_up=north_up)
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)


def roughness_map(elevation: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Compute local elevation roughness as an edge-padded standard deviation map.

    NaN cells are ignored inside each window; windows without finite cells
    are NaN. Memory use is a few rasters, independent of `window_size`.
    """

    arr = _elevation_grid(elevation)
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_size == 1:
        return np.zeros_like(arr, dtype=np.float64)

    before = window_size // 2
    after = window_size - 1 - before
    padded = np.pad(arr, ((before, after), (before, after)), mode="edge")
    valid = np.isfinite(padded)
    invalid = ~valid
    filled = np.where(valid, padded, 0.0)
    height, width = arr.shape
    offsets = [(row, col) for row in range(window_size) for col in range(window_size)]

    counts = np.zeros(arr.shape, dtype=np.float64)
    sums = np.zeros(arr.shape, dtype=np.float64)
    for row, col in offsets:
        counts += valid[row:row + height, col:col + width]
        sums += filled[row:row + height, col:col + width]
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    # Two-pass variance over shifted views: exact like the windowed version,
    # without materializing an (H, W, k, k) array.
    squared = np.zeros(arr.shape, dtype=np.float64)
    deviation = np.empty(arr.shape, dtype=np.float64)
    for row, col in offsets:
        np.subtract(filled[row:row + height, col:col + width], means, out=deviation)
        np.multiply(deviation, deviation, out=deviation)
        np.copyto(deviation, 0.0, where=invalid[row:row + height, col:col + width])
        squared += deviation
    variance = np.divide(squared, counts, out=np.full_like(squared, np.nan), where=counts > 0)
    return np.sqrt(variance)


def traversability_map(
    elevation: np.ndarray,
    resolution: float | tuple[float, float] = 1.0,
    max_slope_degrees: float = 30.0,
    max_roughness: float | None = None,
    roughness_window: int = 3,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Score terrain traversability from 0 to 1 using slope and optional roughness.

    Cells whose slope or roughness cannot be evaluated (NaN elevation, or
    cells whose gradient touches a NaN neighbour) score 0 (untraversable),
    so the result always stays within `[0, 1]`. Slope and roughness are
    orientation-independent, so no `north_up` flag is needed.
    """

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
    score = np.nan_to_num(score, nan=0.0)
    if return_mask:
        return score, score > 0.0
    return score


def sample_elevation(
    elevation: np.ndarray,
    x,
    y,
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    bilinear: bool = True,
    north_up: bool = False,
) -> np.ndarray:
    """Sample a DEM at local XY coordinates using the DEM grid convention.

    `origin` is the XY position of the south-west (min x, min y) cell and
    `resolution` a scalar cell size or `(dx, dy)`. By default rows grow toward
    +y (north); pass `north_up=True` for rasters whose row 0 is the
    northernmost row (HGT tiles, `mosaic_dem_tiles` output). Points that are
    non-finite or more than half a cell outside the grid are NaN.
    """

    dx, dy = _resolution_xy(resolution)
    arr = np.asarray(elevation)
    cols = (np.asarray(x, dtype=np.float64) - origin[0]) / dx
    rows = (np.asarray(y, dtype=np.float64) - origin[1]) / dy
    if north_up:
        if arr.ndim < 2:
            raise ValueError("raster must have at least one row and one column")
        rows = (arr.shape[0] - 1) - rows
    return sample_grid(arr, rows, cols, bilinear=bilinear)


def sample_elevation_at_navsat(
    elevation: np.ndarray,
    navsat: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    bilinear: bool = True,
    north_up: bool = False,
) -> np.ndarray:
    """Sample DEM elevations at WGS84 latitude/longitude/altitude points.

    Points are converted to ENU relative to the reference; `origin` is the ENU
    XY of the DEM's south-west cell. See `sample_elevation` for `north_up`.
    NaN fixes (GPS dropouts) sample as NaN.
    """

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
        north_up=north_up,
    )


def terrain_patch(
    elevation: np.ndarray,
    center,
    size: int | tuple[int, int],
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    fill_value=np.nan,
    return_origin: bool = False,
    north_up: bool = False,
):
    """Extract a fixed-size DEM patch centered on local XY coordinates.

    The patch keeps the input's row layout. `origin` (and the returned patch
    origin) is the XY of the south-west cell; see `sample_elevation` for
    `north_up`.
    """

    dx, dy = _resolution_xy(resolution)
    arr = _elevation_grid(elevation)
    if north_up:
        arr = arr[::-1]
    rows, cols = _patch_size(size)
    xy = np.asarray(center, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError("center must have shape (2,)")
    center_col = int(np.rint((xy[0] - origin[0]) / dx))
    center_row = int(np.rint((xy[1] - origin[1]) / dy))
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

    if north_up:
        patch = np.ascontiguousarray(patch[::-1])
    patch_origin = (
        origin[0] + col_start * dx,
        origin[1] + row_start * dy,
    )
    return (patch, patch_origin) if return_origin else patch


def terrain_patch_at_navsat(
    elevation: np.ndarray,
    navsat,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    size: int | tuple[int, int] = 3,
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    fill_value=np.nan,
    return_origin: bool = False,
    north_up: bool = False,
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
        north_up=north_up,
    )


def dem_to_point_cloud(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    include_nan: bool = False,
    north_up: bool = False,
) -> np.ndarray:
    """Convert a DEM grid to an unorganized XYZ point cloud.

    Without explicit `x`/`y`, coordinates come from `origin` (south-west cell)
    and `resolution` (scalar or `(dx, dy)`); `north_up=True` assigns the
    largest y to row 0. Points are emitted in the raster's row-major order.
    """

    points = _grid_points(
        elevation,
        x=x,
        y=y,
        resolution=resolution,
        origin=origin,
        north_up=north_up,
    ).reshape((-1, 3))
    if include_nan:
        return points
    return points[np.isfinite(points).all(axis=1)]


def dem_to_mesh(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    include_nan: bool = False,
    north_up: bool = False,
) -> dict[str, np.ndarray]:
    """Convert a DEM grid to triangle mesh vertices and faces.

    Each grid cell becomes two triangles wound counter-clockwise when seen
    from above, so face normals point +z like `terrain_normals` for either
    row convention (see `dem_to_point_cloud` for coordinates and `north_up`).
    """

    grid = _grid_points(elevation, x=x, y=y, resolution=resolution, origin=origin, north_up=north_up)
    rows, cols = grid.shape[:2]
    vertices = grid.reshape((-1, 3))
    if rows < 2 or cols < 2:
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        index = np.arange(rows * cols, dtype=np.int64).reshape((rows, cols))
        q0 = index[:-1, :-1].ravel()
        q1 = index[:-1, 1:].ravel()
        q2 = index[1:, :-1].ravel()
        q3 = index[1:, 1:].ravel()
        if not include_nan:
            finite = np.isfinite(grid).all(axis=-1)
            keep = (finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]).ravel()
            q0, q1, q2, q3 = q0[keep], q1[keep], q2[keep], q3[keep]
        faces = np.stack((
            np.stack((q0, q1, q2), axis=-1),
            np.stack((q1, q3, q2), axis=-1),
        ), axis=1).reshape((-1, 3))
        faces = _wind_counter_clockwise(vertices, faces)
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


def _wind_counter_clockwise(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip triangles whose XY winding is clockwise so normals point +z."""

    a = vertices[faces[:, 0], :2]
    b = vertices[faces[:, 1], :2]
    c = vertices[faces[:, 2], :2]
    cross_z = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    clockwise = cross_z < 0.0
    if np.any(clockwise):
        faces = faces.copy()
        faces[clockwise, 1], faces[clockwise, 2] = faces[clockwise, 2], faces[clockwise, 1].copy()
    return faces


def _elevation_grid(elevation: np.ndarray) -> np.ndarray:
    arr = np.asarray(elevation, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("elevation must be a two-dimensional DEM grid")
    return arr


def _resolution_xy(resolution) -> tuple[float, float]:
    values = np.asarray(resolution, dtype=np.float64)
    if values.ndim == 0:
        dx = dy = float(values)
    elif values.shape == (2,):
        dx, dy = float(values[0]), float(values[1])
    else:
        raise ValueError("resolution must be a scalar or (dx, dy)")
    if not (dx > 0 and dy > 0):
        raise ValueError("resolution must be positive")
    return dx, dy


def _coordinate_grid(
    elevation: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    north_up: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    z = _elevation_grid(elevation)
    dx, dy = _resolution_xy(resolution)

    if x is None:
        x_coords = origin[0] + np.arange(z.shape[1], dtype=np.float64) * dx
    else:
        x_coords = np.asarray(x, dtype=np.float64)
    if y is None:
        y_coords = origin[1] + np.arange(z.shape[0], dtype=np.float64) * dy
        if north_up:
            y_coords = y_coords[::-1]
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
    resolution: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    north_up: bool = False,
) -> np.ndarray:
    z = _elevation_grid(elevation)
    xx, yy = _coordinate_grid(z, x=x, y=y, resolution=resolution, origin=origin, north_up=north_up)
    return np.stack((xx, yy, z), axis=-1)


def _dem_samples_present(tile: np.ndarray, nodata) -> np.ndarray | None:
    present = None
    if nodata is not None:
        present = tile != nodata
    if np.issubdtype(tile.dtype, np.inexact):
        finite = ~np.isnan(tile)
        present = finite if present is None else present & finite
    if present is not None and present.all():
        return None
    return present


def _normalize_dem_tiles(tiles) -> list[tuple[object, np.ndarray]]:
    if isinstance(tiles, TopicView) or (
        isinstance(tiles, np.ndarray) and tiles.dtype.names is not None and "data" in tiles.dtype.names
    ) or (isinstance(tiles, dict) and "ts" in tiles and "data" in tiles):
        # A buffered DEM topic: tile names travel as message ids.
        ids, _, data = topic_parts(tiles)
        if ids is None:
            raise ValueError("buffered DEM topics must carry tile names as message ids")
        return [
            (name.decode() if isinstance(name, bytes) else str(name), raster)
            for name, raster in zip(np.asarray(ids).tolist(), data)
        ]
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
