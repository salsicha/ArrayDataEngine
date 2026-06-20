from __future__ import annotations

import os
from .ros_source import RosSource
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor
from ..sensors.imu_sensor import IMUSensor
from ..sensors.odom_sensor import OdomSensor
from ..sensors.nav_sensor import NavSensor


class DB3Source(RosSource):
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
    }


    def __init__(self, data_path: str):
        """Constructor

        """
        super().__init__(data_path)

        if os.path.isdir(data_path):
            self.data_path = data_path
        else:
            self.data_path = os.path.dirname(data_path) or "."


    def data_exists(self) -> bool:
        if not os.path.isdir(self.data_path):
            return False

        if os.path.exists(os.path.join(self.data_path, "metadata.yaml")):
            return True

        return any(name.lower().endswith(".db3") for name in os.listdir(self.data_path))
