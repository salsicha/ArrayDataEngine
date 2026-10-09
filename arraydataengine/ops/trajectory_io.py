from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .core import TopicView, topic_view
from .nav import quaternion_to_rotation_matrix


def _pose_topic_arrays(topic) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(ts, position, orientation)` of a pose-like topic: PoseStamped rows
    ``[x, y, z, qx, qy, qz, qw]`` (extra columns ignored), bare ``[x, y,
    z]`` positions, or odometry rows shaped ``(8, 4)``."""

    view = topic_view(topic, copy=False)
    ts = np.atleast_1d(np.asarray(view.timestamps, dtype=np.float64))
    data = np.asarray(view.data)
    if data.dtype == object or not np.issubdtype(data.dtype, np.number):
        raise ValueError("pose topics need numeric data shaped (N, 7) [x y z qx qy qz qw], (N, 3) or (N, 8, 4)")
    data = data.astype(np.float64, copy=False)
    if data.ndim == 3 and data.shape[1:] == (8, 4):
        # ADE odometry layout: row 0 position, row 2 orientation quaternion.
        return ts, data[:, 0, :3], data[:, 2, :4]
    if data.shape[0] == 0:
        return ts, np.empty((0, 3)), np.empty((0, 4))
    flat = data.reshape((data.shape[0], -1))
    if data.ndim == 2 and flat.shape[1] >= 7:
        # PoseStamped rows: reuse the nav conversion (all-zero quaternions
        # count as missing orientations).
        from .nav import pose_to_trajectory

        trajectory = pose_to_trajectory(view)
        return trajectory["ts"], trajectory["position"], trajectory["orientation"]
    if flat.shape[1] >= 7:
        return ts, flat[:, :3], flat[:, 3:7]
    if flat.shape[1] == 3:
        return ts, flat, np.full((ts.size, 4), np.nan)
    raise ValueError(
        f"pose topics need rows of [x y z qx qy qz qw] (or [x y z]); got rows of {flat.shape[1]} values"
    )


def _trajectory_arrays(trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(trajectory, Mapping) and "position" in trajectory:
        ts = np.atleast_1d(np.asarray(trajectory["ts"], dtype=np.float64))
        position = np.asarray(trajectory["position"], dtype=np.float64).reshape((-1, 3))
        orientation = np.asarray(
            trajectory.get("orientation", np.full((ts.size, 4), np.nan)), dtype=np.float64
        ).reshape((-1, 4))
    elif isinstance(trajectory, (Mapping, TopicView, np.ndarray)):
        # A buffered pose/odometry topic (dict, TopicView or structured array).
        ts, position, orientation = _pose_topic_arrays(trajectory)
    else:
        raise TypeError("trajectory must be a trajectory dict (ts, position[, orientation]) or a pose topic")
    if not (ts.shape[0] == position.shape[0] == orientation.shape[0]):
        raise ValueError("trajectory ts, position, and orientation lengths differ")

    # Trajectories without orientation (e.g. navsat) carry NaN (or all-zero)
    # quaternions; export those rows as identity.
    orientation = orientation.copy()
    norms = np.linalg.norm(orientation, axis=1, keepdims=True)
    invalid = ~np.isfinite(orientation).all(axis=1) | (norms[:, 0] == 0)
    orientation[invalid] = (0.0, 0.0, 0.0, 1.0)
    norms[invalid] = 1.0
    return ts, position, orientation / norms


def to_tum_trajectory(trajectory) -> np.ndarray:
    """Convert a trajectory to TUM rows `[ts x y z qx qy qz qw]`.

    Accepts trajectory dicts (``ts``, ``position``, optional
    ``orientation``) and pose topics: dicts, TopicViews or structured
    arrays whose data rows are ``[x, y, z, qx, qy, qz, qw]`` (PoseStamped),
    ``[x, y, z]``, or ADE odometry ``(8, 4)`` blocks. Missing orientations
    are written as identity."""

    ts, position, orientation = _trajectory_arrays(trajectory)
    return np.column_stack((ts, position, orientation))


def write_tum_trajectory(path, trajectory) -> Path:
    """Write a trajectory dict or pose topic (see `to_tum_trajectory`) in
    TUM format (compatible with `evo`, rgbd tools)."""

    rows = to_tum_trajectory(trajectory)
    path = Path(path)
    header = "timestamp tx ty tz qx qy qz qw"
    np.savetxt(path, rows, fmt="%.9f", header=header)
    return path


def read_tum_trajectory(path) -> dict:
    """Read a TUM-format file back into a trajectory dict."""

    rows = np.loadtxt(Path(path), comments="#", ndmin=2)
    if rows.size == 0:
        rows = np.empty((0, 8), dtype=np.float64)
    if rows.shape[1] != 8:
        raise ValueError("TUM trajectories need 8 columns: ts x y z qx qy qz qw")

    ts = rows[:, 0]
    position = rows[:, 1:4]
    orientation = rows[:, 4:8]
    pose = np.concatenate((position, orientation), axis=1)
    return {
        "ts": ts,
        "position": position,
        "orientation": orientation,
        "pose": pose,
        "source": "tum",
    }


def write_kitti_trajectory(path, trajectory) -> Path:
    """Write a trajectory in KITTI odometry format (12 floats per line,
    the row-major 3x4 `[R | t]` of each pose)."""

    _, position, orientation = _trajectory_arrays(trajectory)
    rotations = quaternion_to_rotation_matrix(orientation)
    matrices = np.concatenate((rotations, position[:, :, None]), axis=2)
    path = Path(path)
    np.savetxt(path, matrices.reshape((-1, 12)), fmt="%.9f")
    return path
