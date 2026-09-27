from __future__ import annotations

import os

from .ros_source import DEFAULT_SENSOR_TYPES, RosSource


class BagSource(RosSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """

    SENSOR_TYPES = dict(DEFAULT_SENSOR_TYPES)


    def __init__(self, data_path: str, max_points: int | None = None):
        """Constructor

        `max_points` is the fixed row count PointCloud2 messages are padded to
        (default 30000); larger clouds are skipped with a warning.
        """
        super().__init__(data_path, max_points=max_points)

        self.data_path = data_path


    def data_exists(self) -> bool:
        return os.path.isfile(self.data_path)
