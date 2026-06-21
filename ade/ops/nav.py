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


def euler_to_quaternion(roll, pitch=None, yaw=None, degrees: bool = False) -> np.ndarray:
    """Convert roll/pitch/yaw angles to XYZW quaternions."""

    if pitch is None and yaw is None:
        angles = np.asarray(roll, dtype=np.float64)
        if angles.shape[-1:] != (3,):
            raise ValueError("Euler angles must have shape (..., 3)")
        roll_values = angles[..., 0]
        pitch_values = angles[..., 1]
        yaw_values = angles[..., 2]
    elif pitch is not None and yaw is not None:
        roll_values, pitch_values, yaw_values = np.broadcast_arrays(
            np.asarray(roll, dtype=np.float64),
            np.asarray(pitch, dtype=np.float64),
            np.asarray(yaw, dtype=np.float64),
        )
    else:
        raise ValueError("provide either an (..., 3) Euler array or roll, pitch, and yaw")

    if degrees:
        roll_values = np.deg2rad(roll_values)
        pitch_values = np.deg2rad(pitch_values)
        yaw_values = np.deg2rad(yaw_values)

    half_roll = roll_values * 0.5
    half_pitch = pitch_values * 0.5
    half_yaw = yaw_values * 0.5
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)

    quaternion = np.stack((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ), axis=-1)
    return normalize_quaternion(quaternion)


def quaternion_to_euler(quaternion: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Convert XYZW quaternions to roll/pitch/yaw angles."""

    q = normalize_quaternion(quaternion)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    angles = np.stack((roll, pitch, yaw), axis=-1)
    return np.rad2deg(angles) if degrees else angles


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to rotation matrices."""

    q = normalize_quaternion(quaternion)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def rotate_vectors_by_quaternion(vectors: np.ndarray, quaternion: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Rotate XYZ vectors by XYZW quaternions."""

    vec = np.asarray(vectors, dtype=np.float64)
    if vec.shape[-1:] != (3,):
        raise ValueError("vectors must have shape (..., 3)")
    matrix = quaternion_to_rotation_matrix(quaternion)
    if inverse:
        matrix = np.swapaxes(matrix, -1, -2)
    return np.einsum("...ij,...j->...i", matrix, vec)


def compensate_gravity(
    linear_acceleration: np.ndarray,
    orientation: np.ndarray,
    gravity=(0.0, 0.0, -9.80665),
    orientation_maps_body_to_world: bool = True,
) -> np.ndarray:
    """Remove gravity from body-frame accelerometer samples."""

    acceleration = np.asarray(linear_acceleration, dtype=np.float64)
    if acceleration.shape[-1:] != (3,):
        raise ValueError("linear_acceleration must have shape (..., 3)")
    gravity_world = np.asarray(gravity, dtype=np.float64)
    if gravity_world.shape != (3,):
        raise ValueError("gravity must have shape (3,)")
    gravity_vectors = np.broadcast_to(gravity_world, acceleration.shape)
    gravity_body = rotate_vectors_by_quaternion(
        gravity_vectors,
        orientation,
        inverse=orientation_maps_body_to_world,
    )
    return acceleration + gravity_body


def estimate_bias(values: np.ndarray, expected=0.0, mask: np.ndarray | None = None, axis=0) -> np.ndarray:
    """Estimate additive sensor bias as the mean residual from expected values."""

    arr = np.asarray(values, dtype=np.float64)
    residual = arr - np.asarray(expected, dtype=np.float64)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        while keep.ndim < residual.ndim:
            keep = keep[..., None]
        residual = np.where(keep, residual, np.nan)
    return np.nanmean(residual, axis=axis)


def correct_bias(
    values: np.ndarray,
    bias: np.ndarray | None = None,
    expected=0.0,
    mask: np.ndarray | None = None,
    axis=0,
    return_bias: bool = False,
):
    """Subtract an additive bias from sensor values."""

    arr = np.asarray(values, dtype=np.float64)
    estimated = estimate_bias(arr, expected=expected, mask=mask, axis=axis) if bias is None else np.asarray(
        bias,
        dtype=np.float64,
    )
    corrected = arr - estimated
    return (corrected, estimated) if return_bias else corrected


def compensate_imu_gravity(
    imu_data,
    gravity=(0.0, 0.0, -9.80665),
    orientation_maps_body_to_world: bool = True,
):
    """Remove gravity from ADE IMU arrays shaped `(N, 6, 4)`."""

    imu, original_shape, result = _mutable_imu_data(imu_data)
    imu[:, 4, :3] = compensate_gravity(
        imu[:, 4, :3],
        imu[:, 0, :4],
        gravity=gravity,
        orientation_maps_body_to_world=orientation_maps_body_to_world,
    )
    return _restore_imu_data(result, imu, original_shape)


def correct_imu_bias(
    imu_data,
    angular_velocity_bias: np.ndarray | None = None,
    linear_acceleration_bias: np.ndarray | None = None,
    angular_velocity_expected=0.0,
    linear_acceleration_expected=0.0,
    sample_slice=None,
    return_bias: bool = False,
):
    """Bias-correct ADE IMU angular velocity and linear acceleration rows."""

    imu, original_shape, result = _mutable_imu_data(imu_data)
    calibration = imu if sample_slice is None else imu[sample_slice]
    if calibration.shape[0] == 0:
        raise ValueError("sample_slice selects no calibration samples")

    angular_bias = (
        estimate_bias(calibration[:, 2, :3], expected=angular_velocity_expected, axis=0)
        if angular_velocity_bias is None
        else np.asarray(angular_velocity_bias, dtype=np.float64)
    )
    linear_bias = (
        estimate_bias(calibration[:, 4, :3], expected=linear_acceleration_expected, axis=0)
        if linear_acceleration_bias is None
        else np.asarray(linear_acceleration_bias, dtype=np.float64)
    )
    imu[:, 2, :3] -= angular_bias
    imu[:, 4, :3] -= linear_bias
    restored = _restore_imu_data(result, imu, original_shape)
    biases = {"angular_velocity": angular_bias, "linear_acceleration": linear_bias}
    return (restored, biases) if return_bias else restored


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


def interpolate_quaternions(
    timestamps: np.ndarray,
    orientations: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate XYZW quaternions at target timestamps using SLERP."""

    ts, targets = _interpolation_timestamps(timestamps, target_timestamps)
    quaternions = np.asarray(orientations, dtype=np.float64)
    if quaternions.shape != (ts.size, 4):
        raise ValueError("orientations must have shape (N, 4)")
    quaternions = normalize_quaternion(quaternions)
    if ts.size == 1:
        return np.repeat(quaternions, targets.size, axis=0)

    result = np.empty((targets.size, 4), dtype=np.float64)
    for index, target in enumerate(targets):
        if target <= ts[0]:
            result[index] = quaternions[0]
        elif target >= ts[-1]:
            result[index] = quaternions[-1]
        else:
            upper = int(np.searchsorted(ts, target, side="right"))
            lower = upper - 1
            fraction = float((target - ts[lower]) / (ts[upper] - ts[lower]))
            result[index] = slerp(quaternions[lower], quaternions[upper], fraction)
    return normalize_quaternion(result)


def interpolate_trajectory(trajectory: Mapping[str, Any], target_timestamps: np.ndarray) -> dict:
    """Interpolate a common trajectory dict onto target timestamps."""

    ts, targets = _interpolation_timestamps(trajectory["ts"], target_timestamps)
    position = interpolate_timeseries(ts, _trajectory_field(trajectory, "position", ts.size, 3), targets)
    orientation = interpolate_quaternions(ts, _trajectory_field(trajectory, "orientation", ts.size, 4), targets)
    linear_velocity = interpolate_timeseries(ts, _trajectory_field(trajectory, "linear_velocity", ts.size, 3), targets)
    angular_velocity = interpolate_timeseries(ts, _trajectory_field(trajectory, "angular_velocity", ts.size, 3), targets)
    linear_acceleration = interpolate_timeseries(
        ts,
        _trajectory_field(trajectory, "linear_acceleration", ts.size, 3),
        targets,
    )
    result = _trajectory_dict(
        ts=targets,
        position=position,
        orientation=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        position_covariance=interpolate_timeseries(
            ts,
            _trajectory_field(trajectory, "position_covariance", ts.size, 3),
            targets,
        ),
        orientation_covariance=interpolate_timeseries(
            ts,
            _trajectory_field(trajectory, "orientation_covariance", ts.size, 3),
            targets,
        ),
        linear_velocity_covariance=interpolate_timeseries(
            ts,
            _trajectory_field(trajectory, "linear_velocity_covariance", ts.size, 3),
            targets,
        ),
        angular_velocity_covariance=interpolate_timeseries(
            ts,
            _trajectory_field(trajectory, "angular_velocity_covariance", ts.size, 3),
            targets,
        ),
        linear_acceleration_covariance=interpolate_timeseries(
            ts,
            _trajectory_field(trajectory, "linear_acceleration_covariance", ts.size, 3),
            targets,
        ),
        source=str(trajectory.get("source", "trajectory")),
        metadata=_trajectory_metadata(trajectory),
    )

    if "reference" in trajectory:
        reference = dict(trajectory["reference"])
        result["reference"] = reference
        if "navsat" in trajectory:
            result["navsat"] = enu_to_navsat(
                result["position"],
                reference["lat"],
                reference["lon"],
                reference.get("alt", 0.0),
            )
    return result


def resample_trajectory(
    trajectory: Mapping[str, Any],
    period: float | None = None,
    target_timestamps: np.ndarray | None = None,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    """Resample a common trajectory dict to target timestamps or a fixed period."""

    targets = _resample_targets(trajectory["ts"], period=period, target_timestamps=target_timestamps, start=start, end=end)
    return interpolate_trajectory(trajectory, targets)


def resample_imu(
    imu_data,
    period: float | None = None,
    target_timestamps: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    position=None,
    linear_velocity=None,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    """Convert and resample an IMU stream as a common trajectory dict."""

    trajectory = imu_to_trajectory(
        imu_data,
        timestamps=timestamps,
        position=position,
        linear_velocity=linear_velocity,
    )
    return resample_trajectory(trajectory, period=period, target_timestamps=target_timestamps, start=start, end=end)


def resample_odometry(
    odometry_data,
    period: float | None = None,
    target_timestamps: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    """Convert and resample an odometry stream as a common trajectory dict."""

    trajectory = odometry_to_trajectory(odometry_data, timestamps=timestamps)
    return resample_trajectory(trajectory, period=period, target_timestamps=target_timestamps, start=start, end=end)


def resample_navsat(
    navsat_data,
    period: float | None = None,
    target_timestamps: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
    ref_alt: float | None = None,
    compute_velocity: bool = True,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    """Convert and resample a NavSat stream as a local ENU trajectory dict."""

    trajectory = navsat_to_trajectory(
        navsat_data,
        timestamps=timestamps,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        ref_alt=ref_alt,
        compute_velocity=compute_velocity,
    )
    return resample_trajectory(trajectory, period=period, target_timestamps=target_timestamps, start=start, end=end)


def smooth_timeseries(values: np.ndarray, window_size: int = 3, axis: int = 0) -> np.ndarray:
    """Smooth numeric samples with an edge-padded moving average."""

    arr = np.asarray(values, dtype=np.float64)
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_size == 1:
        return arr.copy()
    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("axis is out of bounds")

    moved = np.moveaxis(arr, axis, 0)
    before = window_size // 2
    after = window_size - 1 - before
    padded = np.pad(moved, [(before, after), *([(0, 0)] * (moved.ndim - 1))], mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size, axis=0)
    valid = np.isfinite(windows)
    counts = valid.sum(axis=-1)
    sums = np.where(valid, windows, 0.0).sum(axis=-1)
    smoothed = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return np.moveaxis(smoothed, 0, axis)


def smooth_trajectory(
    trajectory: Mapping[str, Any],
    window_size: int = 3,
    fields: tuple[str, ...] | None = None,
    smooth_orientation: bool = True,
) -> dict:
    """Smooth common trajectory fields with moving averages."""

    ts = np.asarray(trajectory["ts"], dtype=np.float64)
    if fields is None:
        fields = (
            "position",
            "linear_velocity",
            "angular_velocity",
            "linear_acceleration",
            "position_covariance",
            "orientation_covariance",
            "linear_velocity_covariance",
            "angular_velocity_covariance",
            "linear_acceleration_covariance",
        )
    values = {
        key: _trajectory_field(trajectory, key, ts.size, 3)
        for key in (
            "position",
            "linear_velocity",
            "angular_velocity",
            "linear_acceleration",
            "position_covariance",
            "orientation_covariance",
            "linear_velocity_covariance",
            "angular_velocity_covariance",
            "linear_acceleration_covariance",
        )
    }
    for field in fields:
        if field not in values:
            raise ValueError(f"unsupported trajectory field for smoothing: {field}")
        values[field] = smooth_timeseries(values[field], window_size=window_size, axis=0)

    orientation = _trajectory_field(trajectory, "orientation", ts.size, 4)
    if smooth_orientation:
        orientation = _smooth_quaternions(orientation, window_size=window_size)

    result = _trajectory_dict(
        ts=ts,
        position=values["position"],
        orientation=orientation,
        linear_velocity=values["linear_velocity"],
        angular_velocity=values["angular_velocity"],
        linear_acceleration=values["linear_acceleration"],
        position_covariance=values["position_covariance"],
        orientation_covariance=values["orientation_covariance"],
        linear_velocity_covariance=values["linear_velocity_covariance"],
        angular_velocity_covariance=values["angular_velocity_covariance"],
        linear_acceleration_covariance=values["linear_acceleration_covariance"],
        source=str(trajectory.get("source", "trajectory")),
        metadata=_trajectory_metadata(trajectory),
    )
    return _preserve_navsat_reference(result, trajectory)


def differentiate_timeseries(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Differentiate numeric samples with respect to timestamps."""

    ts = _strict_timestamps(timestamps)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] != ts.size:
        raise ValueError("values must have the same first dimension as timestamps")
    if ts.size < 2:
        return np.zeros_like(arr, dtype=np.float64)
    return np.gradient(arr, ts, axis=0)


def angular_velocity_from_quaternions(
    timestamps: np.ndarray,
    orientations: np.ndarray,
    angular_velocity_frame: str = "body",
) -> np.ndarray:
    """Estimate angular velocity vectors from XYZW orientation samples."""

    ts = _strict_timestamps(timestamps)
    q = normalize_quaternion(orientations)
    if q.shape != (ts.size, 4):
        raise ValueError("orientations must have shape (N, 4)")
    if ts.size < 2:
        return np.zeros((ts.size, 3), dtype=np.float64)

    frame = _angular_velocity_frame(angular_velocity_frame)
    interval_omega = np.empty((ts.size - 1, 3), dtype=np.float64)
    for index in range(ts.size - 1):
        dt = ts[index + 1] - ts[index]
        if frame == "body":
            delta = _quaternion_multiply(_quaternion_conjugate(q[index]), q[index + 1])
        else:
            delta = _quaternion_multiply(q[index + 1], _quaternion_conjugate(q[index]))
        if delta[3] < 0.0:
            delta = -delta
        interval_omega[index] = _rotation_vector_from_quaternion(delta) / dt

    omega = np.empty((ts.size, 3), dtype=np.float64)
    omega[0] = interval_omega[0]
    omega[-1] = interval_omega[-1]
    if ts.size > 2:
        omega[1:-1] = 0.5 * (interval_omega[:-1] + interval_omega[1:])
    return omega


def differentiate_trajectory(trajectory: Mapping[str, Any], angular_velocity_frame: str = "body") -> dict:
    """Differentiate pose fields to estimate velocity and acceleration."""

    ts = _strict_timestamps(trajectory["ts"])
    position = _trajectory_field(trajectory, "position", ts.size, 3)
    orientation = _trajectory_field(trajectory, "orientation", ts.size, 4)
    linear_velocity = differentiate_timeseries(ts, position)
    angular_velocity = angular_velocity_from_quaternions(
        ts,
        orientation,
        angular_velocity_frame=angular_velocity_frame,
    )
    linear_acceleration = differentiate_timeseries(ts, linear_velocity)
    result = _trajectory_dict(
        ts=ts,
        position=position,
        orientation=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        position_covariance=_trajectory_field(trajectory, "position_covariance", ts.size, 3),
        orientation_covariance=_trajectory_field(trajectory, "orientation_covariance", ts.size, 3),
        linear_velocity_covariance=_trajectory_field(trajectory, "linear_velocity_covariance", ts.size, 3),
        angular_velocity_covariance=_trajectory_field(trajectory, "angular_velocity_covariance", ts.size, 3),
        linear_acceleration_covariance=_trajectory_field(trajectory, "linear_acceleration_covariance", ts.size, 3),
        source=str(trajectory.get("source", "trajectory")),
        metadata=_trajectory_metadata(trajectory),
    )
    return _preserve_navsat_reference(result, trajectory)


def integrate_timeseries(timestamps: np.ndarray, rates: np.ndarray, initial=0.0) -> np.ndarray:
    """Integrate derivative samples with trapezoidal integration."""

    ts = _strict_timestamps(timestamps)
    rate = np.asarray(rates, dtype=np.float64)
    if rate.shape[0] != ts.size:
        raise ValueError("rates must have the same first dimension as timestamps")
    initial_array = np.broadcast_to(np.asarray(initial, dtype=np.float64), rate.shape[1:])
    result = np.empty_like(rate, dtype=np.float64)
    result[0] = initial_array
    for index in range(1, ts.size):
        dt = ts[index] - ts[index - 1]
        result[index] = result[index - 1] + 0.5 * (rate[index - 1] + rate[index]) * dt
    return result


def integrate_orientations(
    timestamps: np.ndarray,
    angular_velocity: np.ndarray,
    initial_orientation=(0.0, 0.0, 0.0, 1.0),
    angular_velocity_frame: str = "body",
) -> np.ndarray:
    """Integrate angular velocity samples into XYZW orientations."""

    ts = _strict_timestamps(timestamps)
    omega = np.asarray(angular_velocity, dtype=np.float64)
    if omega.shape != (ts.size, 3):
        raise ValueError("angular_velocity must have shape (N, 3)")
    frame = _angular_velocity_frame(angular_velocity_frame)
    orientations = np.empty((ts.size, 4), dtype=np.float64)
    orientations[0] = normalize_quaternion(np.asarray(initial_orientation, dtype=np.float64))
    for index in range(1, ts.size):
        dt = ts[index] - ts[index - 1]
        average_omega = 0.5 * (omega[index - 1] + omega[index])
        delta = _quaternion_from_rotation_vector(average_omega * dt)
        if frame == "body":
            orientations[index] = _quaternion_multiply(orientations[index - 1], delta)
        else:
            orientations[index] = _quaternion_multiply(delta, orientations[index - 1])
        orientations[index] = normalize_quaternion(orientations[index])
    return orientations


def integrate_trajectory(
    trajectory: Mapping[str, Any],
    initial_position=None,
    initial_orientation=None,
    body_frame_velocity: bool = False,
    angular_velocity_frame: str = "body",
) -> dict:
    """Integrate trajectory velocity fields into position and orientation."""

    return dead_reckon_trajectory(
        trajectory,
        initial_position=initial_position,
        initial_orientation=initial_orientation,
        body_frame_velocity=body_frame_velocity,
        angular_velocity_frame=angular_velocity_frame,
    )


def dead_reckon_trajectory(
    trajectory: Mapping[str, Any],
    initial_position=None,
    initial_orientation=None,
    body_frame_velocity: bool = False,
    angular_velocity_frame: str = "body",
) -> dict:
    """Dead-reckon a trajectory from linear and angular velocity samples."""

    ts = _strict_timestamps(trajectory["ts"])
    velocity = _trajectory_field(trajectory, "linear_velocity", ts.size, 3)
    angular_velocity = _trajectory_field(trajectory, "angular_velocity", ts.size, 3)
    source_position = _trajectory_field(trajectory, "position", ts.size, 3)
    source_orientation = _trajectory_field(trajectory, "orientation", ts.size, 4)
    if initial_position is None:
        initial_position = source_position[0] if np.isfinite(source_position[0]).all() else np.zeros(3)
    if initial_orientation is None:
        initial_orientation = source_orientation[0]

    orientation = integrate_orientations(
        ts,
        angular_velocity,
        initial_orientation=initial_orientation,
        angular_velocity_frame=angular_velocity_frame,
    )
    world_velocity = (
        rotate_vectors_by_quaternion(velocity, orientation)
        if body_frame_velocity
        else velocity
    )
    position = integrate_timeseries(ts, world_velocity, initial=initial_position)
    linear_acceleration = differentiate_timeseries(ts, world_velocity)
    result = _trajectory_dict(
        ts=ts,
        position=position,
        orientation=orientation,
        linear_velocity=world_velocity,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        position_covariance=_trajectory_field(trajectory, "position_covariance", ts.size, 3),
        orientation_covariance=_trajectory_field(trajectory, "orientation_covariance", ts.size, 3),
        linear_velocity_covariance=_trajectory_field(trajectory, "linear_velocity_covariance", ts.size, 3),
        angular_velocity_covariance=_trajectory_field(trajectory, "angular_velocity_covariance", ts.size, 3),
        linear_acceleration_covariance=_trajectory_field(trajectory, "linear_acceleration_covariance", ts.size, 3),
        source=str(trajectory.get("source", "trajectory")),
        metadata=_trajectory_metadata(trajectory),
    )
    return _preserve_navsat_reference(result, trajectory)


def propagate_trajectory_covariance(
    trajectory: Mapping[str, Any],
    process_noise: Mapping[str, Any] | float | None = None,
    fill_missing: bool = True,
) -> dict:
    """Accumulate diagonal process noise into trajectory covariance fields."""

    ts = _strict_timestamps(trajectory["ts"])
    elapsed = np.concatenate(([0.0], np.cumsum(np.diff(ts))))
    result = _copy_trajectory_mapping(trajectory)
    for state_key, covariance_key in _trajectory_covariance_keys().items():
        covariance = _trajectory_field(trajectory, covariance_key, ts.size, 3)
        noise = _process_noise_vector(process_noise, state_key, covariance_key)
        propagated = covariance.copy()
        if fill_missing and np.any(noise != 0.0):
            propagated = np.where(np.isfinite(propagated), propagated, 0.0)
        result[covariance_key] = propagated + elapsed[:, None] * noise[None, :]
    return result


def trajectory_quality_mask(
    trajectory: Mapping[str, Any],
    required_fields: tuple[str, ...] | str = ("position", "orientation"),
    covariance_limits: Mapping[str, Any] | None = None,
    status_key: str = "status",
    valid_statuses=None,
) -> np.ndarray:
    """Build a boolean mask from finite fields, covariance thresholds, and statuses."""

    count = _trajectory_count(trajectory)
    mask = np.ones(count, dtype=bool)
    for field in _as_field_tuple(required_fields):
        if field not in trajectory:
            mask &= False
            continue
        values = np.asarray(trajectory[field])
        if values.shape[0] != count:
            raise ValueError(f"trajectory field {field!r} must match timestamp count")
        finite = np.isfinite(values.reshape((count, -1))).all(axis=1)
        if field == "orientation" and values.shape[-1:] == (4,):
            finite &= np.linalg.norm(values, axis=-1) > 0.0
        mask &= finite

    if covariance_limits is not None:
        for field, limit in covariance_limits.items():
            covariance_key = field if str(field).endswith("_covariance") else f"{field}_covariance"
            if covariance_key not in trajectory:
                mask &= False
                continue
            covariance = _trajectory_field(trajectory, covariance_key, count, 3)
            limit_vector = _covariance_vector(limit, "covariance limit")
            covariance_ok = np.isfinite(covariance).all(axis=1) & np.all(covariance <= limit_vector, axis=1)
            mask &= covariance_ok

    if valid_statuses is not None:
        if status_key not in trajectory:
            mask &= False
        else:
            status = _status_array(trajectory[status_key], count)
            if isinstance(valid_statuses, (set, list, tuple)):
                valid_values = np.asarray(list(valid_statuses))
            else:
                valid_values = np.atleast_1d(np.asarray(valid_statuses))
            mask &= np.isin(status, valid_values)
    return mask


def add_trajectory_quality_mask(
    trajectory: Mapping[str, Any],
    mask_key: str = "quality_mask",
    **kwargs,
) -> dict:
    """Return a trajectory copy with a computed quality mask field."""

    result = _copy_trajectory_mapping(trajectory)
    result[mask_key] = trajectory_quality_mask(trajectory, **kwargs)
    return result


def mask_trajectory(
    trajectory: Mapping[str, Any],
    mask: np.ndarray,
    drop: bool = False,
    invalid_value=np.nan,
    mask_key: str = "quality_mask",
) -> dict:
    """Apply a quality mask to trajectory rows by dropping or invalidating samples."""

    count = _trajectory_count(trajectory)
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != (count,):
        raise ValueError("mask must have shape (N,)")

    result = {}
    for key, value in trajectory.items():
        row_values = _row_aligned_array(value, count)
        if row_values is not None:
            if drop:
                result[key] = row_values[keep].copy()
            elif key == "ts":
                result[key] = row_values.copy()
            else:
                result[key] = _invalidate_rows(row_values, keep, invalid_value)
        else:
            result[key] = _copy_trajectory_value(value)
    result[mask_key] = keep[keep].copy() if drop else keep.copy()
    return result


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

    status = _mapping_value(navsat_data, "status")
    position_covariance = _mapping_value(navsat_data, "position_covariance", "covariance")
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
    position_covariance_array = _optional_covariance(position_covariance, ts.size, default=np.nan)
    trajectory = _trajectory_dict(
        ts=ts,
        position=position,
        orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (ts.size, 1)),
        linear_velocity=linear_velocity,
        angular_velocity=_filled_vectors(ts.size, np.nan),
        linear_acceleration=_filled_vectors(ts.size, np.nan),
        position_covariance=position_covariance_array,
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
    if status is not None:
        trajectory["status"] = _status_array(status, ts.size).copy()
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


def enu_to_ned(enu: np.ndarray) -> np.ndarray:
    """Convert local ENU coordinates to NED coordinates."""

    arr = _local_xyz(enu, "enu")
    return np.stack((arr[..., 1], arr[..., 0], -arr[..., 2]), axis=-1)


def ned_to_enu(ned: np.ndarray) -> np.ndarray:
    """Convert local NED coordinates to ENU coordinates."""

    arr = _local_xyz(ned, "ned")
    return np.stack((arr[..., 1], arr[..., 0], -arr[..., 2]), axis=-1)


def navsat_to_ned(lat, lon, alt, ref_lat: float, ref_lon: float, ref_alt: float = 0.0) -> np.ndarray:
    """Convert WGS84 latitude/longitude/altitude samples to local NED coordinates."""

    return enu_to_ned(navsat_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt))


def enu_to_navsat(enu: np.ndarray, ref_lat: float, ref_lon: float, ref_alt: float = 0.0) -> np.ndarray:
    arr = np.asarray(enu, dtype=np.float64)
    ref_lat_rad = np.deg2rad(ref_lat)
    lat = ref_lat + np.rad2deg(arr[..., 1] / EARTH_RADIUS_M)
    lon = ref_lon + np.rad2deg(arr[..., 0] / (EARTH_RADIUS_M * np.cos(ref_lat_rad)))
    alt = ref_alt + arr[..., 2]
    return np.stack((lat, lon, alt), axis=-1)


def ned_to_navsat(ned: np.ndarray, ref_lat: float, ref_lon: float, ref_alt: float = 0.0) -> np.ndarray:
    """Convert local NED coordinates back to WGS84 latitude/longitude/altitude."""

    return enu_to_navsat(ned_to_enu(ned), ref_lat, ref_lon, ref_alt)


def navsat_to_local(
    navsat_data,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
    ref_alt: float | None = None,
    frame: str = "enu",
    return_reference: bool = False,
):
    """Convert WGS84 NavSat samples to local ENU or NED coordinates."""

    values, _, _ = _topic_data_and_timestamps(navsat_data, timestamps=None)
    navsat = _as_navsat_stream(values)
    if ref_lat is None:
        ref_lat = float(navsat[0, 0])
    if ref_lon is None:
        ref_lon = float(navsat[0, 1])
    if ref_alt is None:
        ref_alt = float(navsat[0, 2])

    normalized_frame = _local_frame(frame)
    enu = navsat_to_enu(navsat[:, 0], navsat[:, 1], navsat[:, 2], ref_lat, ref_lon, ref_alt)
    local = enu if normalized_frame == "enu" else enu_to_ned(enu)
    if return_reference:
        return local, {"lat": float(ref_lat), "lon": float(ref_lon), "alt": float(ref_alt), "frame": normalized_frame}
    return local


def local_to_navsat(
    local_coordinates: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    frame: str = "enu",
) -> np.ndarray:
    """Convert local ENU or NED coordinates back to WGS84 NavSat samples."""

    normalized_frame = _local_frame(frame)
    local = _local_xyz(local_coordinates, "local_coordinates")
    enu = local if normalized_frame == "enu" else ned_to_enu(local)
    return enu_to_navsat(enu, ref_lat, ref_lon, ref_alt)


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


def _interpolation_timestamps(
    timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(timestamps, dtype=np.float64)
    targets = np.asarray(target_timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if targets.ndim != 1:
        raise ValueError("target_timestamps must be one-dimensional")
    if ts.size == 0:
        raise ValueError("timestamps cannot be empty")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return ts, targets


def _resample_targets(
    timestamps: np.ndarray,
    period: float | None,
    target_timestamps: np.ndarray | None,
    start: float | None,
    end: float | None,
) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    if target_timestamps is not None:
        return np.asarray(target_timestamps, dtype=np.float64)
    if period is None:
        raise ValueError("period or target_timestamps is required")
    period = float(period)
    if period <= 0.0:
        raise ValueError("period must be positive")
    if ts.ndim != 1 or ts.size == 0:
        raise ValueError("trajectory timestamps must be a non-empty one-dimensional array")
    start_value = float(ts[0] if start is None else start)
    end_value = float(ts[-1] if end is None else end)
    if start_value > end_value:
        raise ValueError("start must be less than or equal to end")
    count = int(np.floor((end_value - start_value) / period + 1e-12)) + 1
    return start_value + np.arange(count, dtype=np.float64) * period


def _trajectory_field(trajectory: Mapping[str, Any], key: str, count: int, width: int) -> np.ndarray:
    if key not in trajectory:
        return np.full((count, width), np.nan, dtype=np.float64)
    arr = np.asarray(trajectory[key], dtype=np.float64)
    if arr.shape[0] != count:
        raise ValueError(f"trajectory field {key!r} must match timestamp count")
    return arr


def _trajectory_metadata(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    return {key: trajectory[key] for key in ("topic", "source_uri", "frame_id") if key in trajectory}


def _trajectory_covariance_keys() -> dict[str, str]:
    return {
        "position": "position_covariance",
        "orientation": "orientation_covariance",
        "linear_velocity": "linear_velocity_covariance",
        "angular_velocity": "angular_velocity_covariance",
        "linear_acceleration": "linear_acceleration_covariance",
    }


def _trajectory_count(trajectory: Mapping[str, Any]) -> int:
    ts = np.asarray(trajectory["ts"])
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    return int(ts.size)


def _copy_trajectory_mapping(trajectory: Mapping[str, Any]) -> dict:
    return {key: _copy_trajectory_value(value) for key, value in trajectory.items()}


def _copy_trajectory_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    return value


def _row_aligned_array(value, count: int):
    if not isinstance(value, (np.ndarray, list, tuple)):
        return None
    arr = np.asarray(value)
    return arr if arr.shape[:1] == (count,) else None


def _process_noise_vector(process_noise, state_key: str, covariance_key: str) -> np.ndarray:
    if process_noise is None:
        value = 0.0
    elif isinstance(process_noise, Mapping):
        value = process_noise.get(covariance_key, process_noise.get(state_key, 0.0))
    else:
        value = process_noise
    return _covariance_vector(value, "process_noise")


def _covariance_vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    if arr.shape == (3,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape (3,)")


def _as_field_tuple(fields: tuple[str, ...] | str | None) -> tuple[str, ...]:
    if fields is None:
        return ()
    if isinstance(fields, str):
        return (fields,)
    return tuple(fields)


def _status_array(status, count: int) -> np.ndarray:
    arr = np.asarray(status)
    if arr.ndim == 0:
        return np.full(count, arr.item(), dtype=arr.dtype)
    if arr.shape[0] != count:
        raise ValueError("status must have the same first dimension as timestamps")
    if arr.ndim == 1:
        return arr
    return arr.reshape((count, -1))[:, 0]


def _invalidate_rows(values: np.ndarray, keep: np.ndarray, invalid_value):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        try:
            dtype = np.result_type(arr.dtype, np.asarray(invalid_value).dtype)
            invalid_is_finite = bool(np.isfinite(invalid_value))
        except TypeError:
            dtype = object
            invalid_is_finite = False
        if np.issubdtype(dtype, np.integer) and not invalid_is_finite:
            dtype = np.float64
        result = arr.astype(dtype, copy=True)
    else:
        result = arr.astype(object, copy=True)
    result[~keep] = invalid_value
    return result


def _mapping_value(value, *keys: str):
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        if key in value:
            return value[key]
    return None


def _optional_covariance(value, count: int, default: float) -> np.ndarray:
    if value is None:
        return _filled_vectors(count, default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape == (3,):
            return np.tile(arr, (count, 1))
        if arr.shape == (9,):
            return np.tile(arr[[0, 4, 8]], (count, 1))
    if arr.ndim == 2:
        if arr.shape == (count, 3):
            return arr
        if arr.shape == (count, 9):
            return arr[:, [0, 4, 8]]
        if arr.shape == (3, 3):
            return np.tile(np.diag(arr), (count, 1))
    if arr.ndim == 3 and arr.shape == (count, 3, 3):
        return np.diagonal(arr, axis1=1, axis2=2)
    raise ValueError("covariance must have shape (3,), (9,), (N, 3), (N, 9), (3, 3), or (N, 3, 3)")


def _strict_timestamps(timestamps: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if ts.size == 0:
        raise ValueError("timestamps cannot be empty")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return ts


def _smooth_quaternions(orientations: np.ndarray, window_size: int) -> np.ndarray:
    quaternions = normalize_quaternion(orientations)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("orientations must have shape (N, 4)")
    if window_size <= 1 or quaternions.shape[0] <= 1:
        return quaternions.copy()

    aligned = quaternions.copy()
    for index in range(1, aligned.shape[0]):
        if np.dot(aligned[index - 1], aligned[index]) < 0.0:
            aligned[index] = -aligned[index]

    smoothed = smooth_timeseries(aligned, window_size=window_size, axis=0)
    for index in range(smoothed.shape[0]):
        if np.dot(smoothed[index], aligned[index]) < 0.0:
            smoothed[index] = -smoothed[index]
    return normalize_quaternion(smoothed)


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    arr = np.asarray(quaternion, dtype=np.float64)
    result = arr.copy()
    result[..., :3] *= -1.0
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), axis=-1)


def _rotation_vector_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    if q[3] < 0.0:
        q = -q
    vector = q[:3]
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(norm, np.clip(q[3], -1.0, 1.0))
    return vector * (angle / norm)


def _quaternion_from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("rotation_vector must have shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = vector / angle
    half_angle = 0.5 * angle
    return normalize_quaternion(np.array([
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
        np.cos(half_angle),
    ], dtype=np.float64))


def _angular_velocity_frame(frame: str) -> str:
    normalized = str(frame).lower()
    if normalized not in {"body", "world"}:
        raise ValueError("angular_velocity_frame must be 'body' or 'world'")
    return normalized


def _preserve_navsat_reference(result: dict, trajectory: Mapping[str, Any]) -> dict:
    if "reference" in trajectory:
        reference = dict(trajectory["reference"])
        result["reference"] = reference
        if "navsat" in trajectory:
            result["navsat"] = enu_to_navsat(
                result["position"],
                reference["lat"],
                reference["lon"],
                reference.get("alt", 0.0),
            )
    if np.asarray(result["ts"]).shape == np.asarray(trajectory["ts"]).shape:
        for key in ("status", "quality_mask", "valid_mask", "position_covariance_type"):
            if key in trajectory:
                result[key] = _copy_trajectory_value(trajectory[key])
    return result


def _local_frame(frame: str) -> str:
    normalized = str(frame).lower()
    if normalized not in {"enu", "ned"}:
        raise ValueError("frame must be 'enu' or 'ned'")
    return normalized


def _local_xyz(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3)")
    return arr


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


def _mutable_imu_data(imu_data):
    if isinstance(imu_data, Mapping):
        values = np.asarray(imu_data["data"], dtype=np.float64)
        result = dict(imu_data)
    else:
        values = np.asarray(imu_data, dtype=np.float64)
        result = None
    original_shape = values.shape
    imu = _as_sensor_stream(values, (6, 4), "imu").copy()
    return imu, original_shape, result


def _restore_imu_data(result: dict[str, Any] | None, imu: np.ndarray, original_shape: tuple[int, ...]):
    data = imu[0] if original_shape == (6, 4) else imu
    if result is None:
        return data
    result["data"] = data
    return result


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
