from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor


# Default per-message row capacity buffers use when storing variable-size
# point clouds (sources themselves do not pad or limit clouds by default).
DEFAULT_MAX_POINTS = 30000


def finite_points(points: np.ndarray, max_points: int | None = None) -> tuple[np.ndarray, int]:
    """Apply the shared point-cloud policy to an (N, 3) XYZ array.

    Rows with any non-finite coordinate are dropped; the remaining points are
    returned unpadded as ``(count, 3)`` in the input dtype together with
    ``count``. Sources report ``count`` as the message's ``point_count``.

    Raises ValueError when ``max_points`` is set and more than ``max_points``
    finite points remain; sources skip such messages with a warning instead
    of truncating them.
    """

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N, 3), got {points.shape}")
    finite_rows = np.isfinite(points).all(axis=1)
    finite = points if finite_rows.all() else points[finite_rows]
    count = int(finite.shape[0])
    if max_points is not None and count > max_points:
        raise ValueError(
            f"PointCloud2 has {count} points, which exceeds max_points={max_points}"
        )
    return np.ascontiguousarray(finite), count


def fixed_size_points(points: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> tuple[np.ndarray, int]:
    """Drop non-finite rows and zero-pad to a fixed ``(max_points, 3)`` array.

    Sources no longer pad (see `finite_points`); this helper remains for
    callers that need fixed-shape storage. Returns ``(padded, point_count)``
    and raises ValueError when more than ``max_points`` finite points remain.
    """

    finite, count = finite_points(points, max_points)
    padded = np.zeros((max_points, 3), dtype=finite.dtype)
    padded[:count] = finite
    return padded, count


class PointCloudSensor(BaseSensor):
    """Convert sensor_msgs/PointCloud2 messages to unpadded ``(N, 3)`` XYZ arrays.

    Non-finite rows are dropped and the field dtype is kept (float64 stays
    float64). With `max_points` set, larger clouds raise ValueError.
    """

    DEFAULT_MAX_POINTS = DEFAULT_MAX_POINTS

    def __init__(self, rawdata, msgtype, max_points: int | None = None, deserializer=None):
        """Constructor

        """
        super().__init__(rawdata, msgtype, deserializer=deserializer)

        # None = no limit; an int rejects (never truncates) larger clouds.
        self.max_points = max_points
        # Number of valid (finite) points in the last converted message.
        self.point_count: int | None = None

    def numpyify(self):
        from ..sources.cdr import pointcloud_xyz

        msg = self.deserialize()
        self._capture_header_metadata(msg)
        pc_2_np = pointcloud_xyz(
            bytes(msg.data),
            [dict(name=field.name, offset=field.offset, datatype=field.datatype,
                  count=field.count) for field in msg.fields],
            msg.height, msg.width, msg.point_step, msg.row_step, bool(msg.is_bigendian),
        )
        npified, self.point_count = finite_points(pc_2_np, self.max_points)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return npified, msg.__class__.__name__, ts
