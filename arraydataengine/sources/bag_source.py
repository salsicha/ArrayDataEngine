from __future__ import annotations

import os

from .ros_source import RosSource
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor
from ..sensors.imu_sensor import IMUSensor
from ..sensors.odom_sensor import OdomSensor
from ..sensors.nav_sensor import NavSensor
from ..sensors.pose_sensor import PoseSensor


class BagSource(RosSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """

    SENSOR_TYPES = {
        "pointcloud2": PointCloudSensor,
        "image": ImageSensor,
        "imu": IMUSensor,
        "odometry": OdomSensor,
        "navsatfix": NavSensor,
        "pose": PoseSensor,
        "posestamped": PoseSensor,
    }


    def __init__(self, data_path: str):
        """Constructor

        """
        super().__init__(data_path)

        self.data_path = data_path


    def data_exists(self) -> bool:
        return os.path.isfile(self.data_path)
