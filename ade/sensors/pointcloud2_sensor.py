from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class PointCloudSensor(BaseSensor):
    """Point Cloud Sensor Class
    Attributes:
    Args:
    Returns:
    """

    DEFAULT_MAX_POINTS = 30000

    def __init__(self, rawdata, msgtype, max_points: int | None = None):
        """Constructor

        """
        super().__init__(rawdata, msgtype)

        # PointCloud2 message has variable length due to sensor dropping some points
        # The max number of points in a scan for the vlp-16 should be 30000
        self.max_points = self.DEFAULT_MAX_POINTS if max_points is None else max_points

    def numpyify(self):
        import ros2_numpy as rnp

        msg = self.deserialize()
        pc_2_np = rnp.point_cloud2.point_cloud2_to_array(msg)["xyz"]
        if pc_2_np.shape[0] > self.max_points:
            raise ValueError(
                f"PointCloud2 has {pc_2_np.shape[0]} points, which exceeds max_points={self.max_points}"
            )

        npified = np.zeros((self.max_points, 3), dtype=pc_2_np.dtype)
        npified[:pc_2_np.shape[0]] = pc_2_np
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return npified, msg.__class__.__name__, ts
