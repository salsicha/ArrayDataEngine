from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def imu_to_trajectory(imu_data, timestamps: np.ndarray | None = None, position=None, linear_velocity=None) -> dict:
    """Convert ADE IMU arrays shaped `(N, 6, 4)` into a common trajectory dict."""

    values, ts, metadata = _topic_data_and_timestamps(imu_data, timestamps)
    imu = _as_sensor_stream(values, (6, 4), "imu")
    position_array = _optional_vector(position, ts.size, default=np.nan)
    linear_velocity_array = _optional_vector(linear_velocity, ts.size, default=np.nan)
    return _trajectory_dict(
        ts=ts,
        position=position_array,
        orientation=imu[:, 0, :4],
        linear_velocity=linear_velocity_array,
        angular_velocity=imu[:, 2, :3],
        linear_acceleration=imu[:, 4, :3],
        position_covariance=_filled_vectors(ts.size, np.nan),
        orientation_covariance=imu[:, 1, :3],
        linear_velocity_covariance=_filled_vectors(ts.size, np.nan),
        angular_velocity_covariance=imu[:, 3, :3],
        linear_acceleration_covariance=imu[:, 5, :3],
        source="imu",
        metadata=metadata,
    )


def odometry_to_trajectory(odometry_data, timestamps: np.ndarray | None = None) -> dict:
    """Convert ADE odometry arrays shaped `(N, 8, 4)` into a common trajectory dict."""

    values, ts, metadata = _topic_data_and_timestamps(odometry_data, timestamps)
    odom = _as_sensor_stream(values, (8, 4), "odometry")
    return _trajectory_dict(
        ts=ts,
        position=odom[:, 0, :3],
        orientation=odom[:, 2, :4],
        linear_velocity=odom[:, 4, :3],
        angular_velocity=odom[:, 6, :3],
        position_covariance=odom[:, 1, :3],
        orientation_covariance=odom[:, 3, :3],
        linear_velocity_covariance=odom[:, 5, :3],
        angular_velocity_covariance=odom[:, 7, :3],
        linear_acceleration=_filled_vectors(ts.size, np.nan),
        linear_acceleration_covariance=_filled_vectors(ts.size, np.nan),
        source="odometry",
        metadata=metadata,
    )


def navsat_to_trajectory(
    navsat_data,
    timestamps: np.ndarray | None = None,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
    ref_alt: float | None = None,
    compute_velocity: bool = True,
) -> dict:
    """Convert NavSat `[lat, lon, alt]` samples into local ENU trajectory arrays."""

    values, ts, metadata = _topic_data_and_timestamps(navsat_data, timestamps)
    navsat = _as_navsat_stream(values)
    if ref_lat is None:
        ref_lat = float(navsat[0, 0])
    if ref_lon is None:
        ref_lon = float(navsat[0, 1])
    if ref_alt is None:
        ref_alt = float(navsat[0, 2])

    position = navsat_to_enu(navsat[:, 0], navsat[:, 1], navsat[:, 2], ref_lat, ref_lon, ref_alt)
    linear_velocity = _velocity_from_positions(ts, position) if compute_velocity else _filled_vectors(ts.size, np.nan)
    trajectory = _trajectory_dict(
        ts=ts,
        position=position,
        orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (ts.size, 1)),
        linear_velocity=linear_velocity,
        angular_velocity=_filled_vectors(ts.size, np.nan),
        linear_acceleration=_filled_vectors(ts.size, np.nan),
        position_covariance=_filled_vectors(ts.size, np.nan),
        orientation_covariance=_filled_vectors(ts.size, np.nan),
        linear_velocity_covariance=_filled_vectors(ts.size, np.nan),
        angular_velocity_covariance=_filled_vectors(ts.size, np.nan),
        linear_acceleration_covariance=_filled_vectors(ts.size, np.nan),
        source="navsat",
        metadata=metadata,
    )
    trajectory["navsat"] = navsat.copy()
    trajectory["reference"] = {
        "lat": float(ref_lat),
        "lon": float(ref_lon),
        "alt": float(ref_alt),
    }
    return trajectory


def sensor_to_trajectory(sensor_data, kind: str, timestamps: np.ndarray | None = None, **kwargs) -> dict:
    """Convert a named ADE sensor stream into the common trajectory representation."""

    normalized = kind.lower().replace("-", "_")
    if normalized in {"imu", "imudata"}:
        return imu_to_trajectory(sensor_data, timestamps=timestamps, **kwargs)
    if normalized in {"odom", "odometry"}:
        return odometry_to_trajectory(sensor_data, timestamps=timestamps)
    if normalized in {"navsat", "navsatfix", "gps"}:
        return navsat_to_trajectory(sensor_data, timestamps=timestamps, **kwargs)
    raise ValueError("kind must be 'imu', 'odometry', or 'navsat'")


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


def _topic_data_and_timestamps(sensor_data, timestamps: np.ndarray | None = None):
    metadata: dict[str, Any] = {}
    if isinstance(sensor_data, Mapping):
        values = sensor_data["data"]
        if timestamps is None:
            timestamps = sensor_data.get("ts", sensor_data.get("timestamp"))
        for key in ("topic", "source_uri", "frame_id"):
            if key in sensor_data:
                metadata[key] = sensor_data[key]
    else:
        values = sensor_data

    arr = np.asarray(values, dtype=np.float64)
    if timestamps is None:
        count = 1 if arr.ndim in {1, 2} else arr.shape[0]
        ts = np.arange(count, dtype=np.float64)
    else:
        ts = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))
    return arr, ts, metadata


def _as_sensor_stream(values: np.ndarray, message_shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == message_shape:
        arr = arr.reshape((1,) + message_shape)
    if arr.ndim != 3 or arr.shape[1:] != message_shape:
        raise ValueError(f"{name} data must have shape {message_shape} or (N, {message_shape[0]}, {message_shape[1]})")
    return arr


def _as_navsat_stream(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] < 3:
            raise ValueError("navsat data must contain latitude, longitude, and altitude")
        arr = arr.reshape((1, arr.shape[0]))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("navsat data must have shape (3+) or (N, 3+)")
    return arr[:, :3]


def _trajectory_dict(
    ts: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    linear_acceleration: np.ndarray,
    position_covariance: np.ndarray,
    orientation_covariance: np.ndarray,
    linear_velocity_covariance: np.ndarray,
    angular_velocity_covariance: np.ndarray,
    linear_acceleration_covariance: np.ndarray,
    source: str,
    metadata: Mapping[str, Any],
) -> dict:
    count = _validated_stream_count(ts, position, orientation)
    normalized_orientation = normalize_quaternion(orientation)
    pose = np.concatenate((position, normalized_orientation), axis=1)
    trajectory = np.concatenate((pose, linear_velocity, angular_velocity), axis=1)
    result = {
        "ts": ts.copy(),
        "position": position.copy(),
        "orientation": normalized_orientation,
        "pose": pose,
        "linear_velocity": linear_velocity.copy(),
        "angular_velocity": angular_velocity.copy(),
        "linear_acceleration": linear_acceleration.copy(),
        "trajectory": trajectory,
        "position_covariance": position_covariance.copy(),
        "orientation_covariance": orientation_covariance.copy(),
        "linear_velocity_covariance": linear_velocity_covariance.copy(),
        "angular_velocity_covariance": angular_velocity_covariance.copy(),
        "linear_acceleration_covariance": linear_acceleration_covariance.copy(),
        "source": source,
    }
    for key, value in metadata.items():
        result[key] = value
    if count == 0:
        result["pose"] = np.empty((0, 7), dtype=np.float64)
        result["trajectory"] = np.empty((0, 13), dtype=np.float64)
    return result


def _validated_stream_count(ts: np.ndarray, *arrays: np.ndarray) -> int:
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    for array in arrays:
        if array.shape[0] != ts.size:
            raise ValueError("stream arrays must have the same leading dimension as timestamps")
    return int(ts.size)


def _optional_vector(value, count: int, default: float) -> np.ndarray:
    if value is None:
        return _filled_vectors(count, default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("optional vector must have shape (3,) or (N, 3)")
        arr = np.tile(arr, (count, 1))
    if arr.shape != (count, 3):
        raise ValueError("optional vector must have shape (3,) or (N, 3)")
    return arr


def _filled_vectors(count: int, value: float) -> np.ndarray:
    return np.full((count, 3), value, dtype=np.float64)


def _velocity_from_positions(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if timestamps.size < 2:
        return np.zeros((timestamps.size, 3), dtype=np.float64)
    if np.any(np.diff(timestamps) == 0):
        return _filled_vectors(timestamps.size, np.nan)
    dt = np.gradient(timestamps)
    return np.gradient(positions, axis=0) / dt.reshape((-1, 1))
