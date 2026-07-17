from __future__ import annotations

from pathlib import Path

import numpy as np

from .nav import quaternion_to_rotation_matrix


def _trajectory_arrays(trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.atleast_1d(np.asarray(trajectory["ts"], dtype=np.float64))
    position = np.asarray(trajectory["position"], dtype=np.float64).reshape((-1, 3))
    orientation = np.asarray(
        trajectory.get("orientation", np.full((ts.size, 4), np.nan)), dtype=np.float64
    ).reshape((-1, 4))
    if not (ts.shape[0] == position.shape[0] == orientation.shape[0]):
        raise ValueError("trajectory ts, position, and orientation lengths differ")

    # Trajectories without orientation (e.g. navsat) carry NaN quaternions;
    # export those rows as identity.
    invalid = ~np.isfinite(orientation).all(axis=1)
    orientation = orientation.copy()
    orientation[invalid] = (0.0, 0.0, 0.0, 1.0)
    norms = np.linalg.norm(orientation, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ts, position, orientation / norms


def to_tum_trajectory(trajectory) -> np.ndarray:
    """Convert a trajectory dict to TUM rows `[ts x y z qx qy qz qw]`."""

    ts, position, orientation = _trajectory_arrays(trajectory)
    return np.column_stack((ts, position, orientation))


def write_tum_trajectory(path, trajectory) -> Path:
    """Write a trajectory in TUM format (compatible with `evo`, rgbd tools)."""

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
