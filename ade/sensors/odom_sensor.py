from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class OdomSensor(BaseSensor):
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
        8 x 4 matrix:
        """

        msg = self.deserialize()

        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9

        pose_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, 0.0]
        poes_pos_cov = [msg.pose.covariance[0], msg.pose.covariance[6], msg.pose.covariance[12], 0.0]
        pose_orient = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pose_orient_cov = [msg.pose.covariance[18], msg.pose.covariance[24], msg.pose.covariance[30], 0.0]
        twist_lin = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z, 0.0]
        twist_lin_cov = [msg.twist.covariance[0], msg.twist.covariance[6], msg.twist.covariance[12], 0.0]
        twist_ang = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z, 0.0]
        twist_ang_cov = [msg.twist.covariance[18], msg.twist.covariance[24], msg.twist.covariance[30], 0.0]

        npified = np.array([pose_pos, poes_pos_cov, pose_orient, pose_orient_cov, twist_lin, twist_lin_cov, twist_ang, twist_ang_cov])

        return npified, msg.__class__.__name__, ts
