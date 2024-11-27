from __future__ import annotations

from rosbags.highlevel import AnyReader

from pathlib import Path

from ..sensors.base_sensor import BaseSensor
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor



class BaseSource:
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path, debug=False):
        """Constructor

        """
        self.data_path = data_path
        self._debug = debug


    def get_topics(self):
        topics = []
        with AnyReader([Path(self.data_path)]) as reader:
            for conn in reader.connections:
                topics.append(conn.topic)
        return topics


    def get_duration(self):
        any_reader = AnyReader([Path(self.data_path)])
        bag_duration = (any_reader.end_time - any_reader.start_time) * 1e-9
        return bag_duration


    def get_data_path(self):
        return self.data_path
    
