from __future__ import annotations

import os

from .ros_source import DEFAULT_SENSOR_TYPES, RosSource


class BagSource(RosSource):
    """ROS1 ``.bag`` recordings read through rosbags.

    See `RosSource` for the message format shared by every ROS read path.
    """

    SENSOR_TYPES = dict(DEFAULT_SENSOR_TYPES)


    def __init__(self, data_path: str, max_points: int | None = None, topics=None):
        """Constructor

        `max_points` (default None: no limit) skips point clouds with more
        finite points, with a warning. `topics` restricts reading to those
        topic names.
        """
        data_path = os.fspath(data_path)
        super().__init__(data_path, max_points=max_points, topics=topics)

        self.data_path = data_path


    def data_exists(self) -> bool:
        return os.path.isfile(self.data_path)
