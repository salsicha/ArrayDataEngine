from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class NavSensor(BaseSensor):
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
        msg = self.deserialize()

        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9

        # TODO: status and covariances

        npified = np.array([msg.latitude, msg.longitude, msg.altitude])

        return npified, msg.__class__.__name__, ts
    