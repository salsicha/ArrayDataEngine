from __future__ import annotations

import numpy as np
import ros2_numpy as rnp

from .base_sensor import BaseSensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from rosbags.serde import deserialize_cdr


class IMUSensor(BaseSensor):
    """Point Cloud Sensor Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, rawdata, msgtype):
        """Constructor

        """
        super().__init__(rawdata, msgtype)


    def numpyify(self) -> tuple:
        """
        6 x 4 matrix:
        orient 
        x, y, z, w
        orient cov
        x, y, z, 0
        ang vel
        x, y, z, 0
        ang vel cov
        x, y, z, 0
        lin acc
        x, y, z, 0
        lin acc cov
        x, y, z, 0
        """

        msg = self.deserialize()

        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9

        orient = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        orient_cov = [msg.orientation_covariance[0], msg.orientation_covariance[3], msg.orientation_covariance[6], 0.0]
        ang_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, 0.0]
        ang_vel_cov = [msg.angular_velocity_covariance[0], msg.angular_velocity_covariance[3], msg.angular_velocity_covariance[6], 0.0]
        lin_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z, 0.0]
        lin_acc_cov = [msg.linear_acceleration_covariance[0], msg.linear_acceleration_covariance[3], msg.linear_acceleration_covariance[6], 0.0]

        npified = np.array([orient, orient_cov, ang_vel, ang_vel_cov, lin_acc, lin_acc_cov])

        return npified, msg.__class__.__name__, ts
