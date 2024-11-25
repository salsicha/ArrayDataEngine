from __future__ import annotations

import numpy as np
import ros2_numpy as rnp

from .base_sensor import BaseSensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from rosbags.serde import deserialize_cdr


class PointCloudSensor(BaseSensor):
    """Point Cloud Sensor Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, rawdata, msgtype):
        """Constructor

        """
        super().__init__(rawdata, msgtype)

        # PointCloud2 message has variable length due to sensor dropping some points
        # The max number of points in a scan for the vlp-16 should be 30000
        self.max_vel = 30000


    def deserialize(self):
        msg = deserialize_cdr(self.rawdata, self.msgtype)
        return msg


    def numpyify(self):
        msg = self.deserialize()
        pc_2_np = rnp.point_cloud2.point_cloud2_to_array(msg)["xyz"]
        npified = np.zeros((self.max_vel, 3), dtype=pc_2_np.dtype)
        npified[:pc_2_np.shape[0]] = pc_2_np
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return npified, msg.__class__.__name__, ts