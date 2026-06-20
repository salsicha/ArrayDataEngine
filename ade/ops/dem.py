from __future__ import annotations

import numpy as np


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


def slope_aspect(elevation: np.ndarray, resolution: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    dz_dy, dz_dx = np.gradient(np.asarray(elevation, dtype=np.float64), resolution, resolution)
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
    row = np.asarray(rows, dtype=np.float64)
    col = np.asarray(cols, dtype=np.float64)

    if not bilinear:
        rr = np.clip(np.rint(row).astype(int), 0, arr.shape[0] - 1)
        cc = np.clip(np.rint(col).astype(int), 0, arr.shape[1] - 1)
        return arr[rr, cc]

    r0 = np.clip(np.floor(row).astype(int), 0, arr.shape[0] - 1)
    c0 = np.clip(np.floor(col).astype(int), 0, arr.shape[1] - 1)
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
