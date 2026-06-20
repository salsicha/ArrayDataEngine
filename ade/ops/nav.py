from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6378137.0


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero-length quaternion cannot be normalized")
    return q / norm


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        return normalize_quaternion(q0 + fraction * (q1 - q0))

    theta_0 = np.arccos(dot)
    theta = theta_0 * fraction
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    return np.cos(theta) * q0 + sin_theta * (q1 - q0 * dot) / sin_theta_0


def interpolate_timeseries(timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    targets = np.asarray(target_timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if ts.size == 0:
        raise ValueError("timestamps cannot be empty")
    if vals.shape[0] != ts.size:
        raise ValueError("values must have the same first dimension as timestamps")

    flat = vals.reshape((vals.shape[0], -1))
    interpolated = np.column_stack([np.interp(targets, ts, flat[:, dim]) for dim in range(flat.shape[1])])
    return interpolated.reshape((targets.size,) + vals.shape[1:])


def navsat_to_enu(lat, lon, alt, ref_lat: float, ref_lon: float, ref_alt: float = 0.0) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)
    ref_lat_rad = np.deg2rad(ref_lat)
    east = np.deg2rad(lon - ref_lon) * EARTH_RADIUS_M * np.cos(ref_lat_rad)
    north = np.deg2rad(lat - ref_lat) * EARTH_RADIUS_M
    up = alt - ref_alt
    return np.stack((east, north, up), axis=-1)


def enu_to_navsat(enu: np.ndarray, ref_lat: float, ref_lon: float, ref_alt: float = 0.0) -> np.ndarray:
    arr = np.asarray(enu, dtype=np.float64)
    ref_lat_rad = np.deg2rad(ref_lat)
    lat = ref_lat + np.rad2deg(arr[..., 1] / EARTH_RADIUS_M)
    lon = ref_lon + np.rad2deg(arr[..., 0] / (EARTH_RADIUS_M * np.cos(ref_lat_rad)))
    alt = ref_alt + arr[..., 2]
    return np.stack((lat, lon, alt), axis=-1)


def trajectory_speed(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if ts.size < 2:
        return np.zeros(ts.shape, dtype=np.float64)
    dt = np.gradient(ts)
    if np.any(dt == 0):
        raise ValueError("timestamps must not contain duplicate values")
    velocity = np.gradient(pos, axis=0) / dt.reshape((-1,) + (1,) * (pos.ndim - 1))
    return np.linalg.norm(velocity, axis=-1)
