from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor


DEFAULT_MAX_POINTS = 30000


def fixed_size_points(points: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> tuple[np.ndarray, int]:
    """Apply the shared point-cloud policy to an (N, 3) XYZ array.

    Rows with any non-finite coordinate are dropped and the remaining points
    are zero-padded to a fixed ``(max_points, 3)`` array of the input dtype,
    so every message on a topic has the same shape (ring buffers and Arrow
    stores require it). Returns ``(padded, point_count)``.

    Raises ValueError when more than ``max_points`` finite points remain;
    sources skip such messages with a warning instead of truncating them.
    """

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N, 3), got {points.shape}")
    finite = points[np.isfinite(points).all(axis=1)]
    count = int(finite.shape[0])
    if count > max_points:
        raise ValueError(
            f"PointCloud2 has {count} points, which exceeds max_points={max_points}"
        )
    padded = np.zeros((max_points, 3), dtype=points.dtype)
    padded[:count] = finite
    return padded, count


class PointCloudSensor(BaseSensor):
    """Point Cloud Sensor Class
    Attributes:
    Args:
    Returns:
    """

    DEFAULT_MAX_POINTS = DEFAULT_MAX_POINTS

    def __init__(self, rawdata, msgtype, max_points: int | None = None, deserializer=None):
        """Constructor

        """
        super().__init__(rawdata, msgtype, deserializer=deserializer)

        # PointCloud2 message has variable length due to sensor dropping some points
        # The max number of points in a scan for the vlp-16 should be 30000
        self.max_points = self.DEFAULT_MAX_POINTS if max_points is None else max_points
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
        npified, self.point_count = fixed_size_points(pc_2_np, self.max_points)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return npified, msg.__class__.__name__, ts
