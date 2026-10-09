from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor


def navsat_extra(status, position_covariance) -> dict:
    """Message keys shared by every NavSatFix read path.

    `status` is ``NavSatStatus.status`` (-1 = no fix) and
    `position_covariance` the row-major 3x3 ENU position covariance (m^2).
    Missing values are left out.
    """

    extra = {}
    if status is not None:
        extra["status"] = int(status)
    if position_covariance is not None:
        covariance = np.asarray(position_covariance, dtype=np.float64)
        if covariance.size == 9:
            extra["position_covariance"] = covariance.reshape(3, 3)
    return extra


class NavSensor(BaseSensor):
    """Convert sensor_msgs/NavSatFix messages to ``[lat, lon, alt]`` arrays.

    The fix status and 3x3 position covariance are reported in `extra` as
    ``status`` and ``position_covariance``.
    """

    def __init__(self, rawdata, msgtype, deserializer=None):
        super().__init__(rawdata, msgtype, deserializer=deserializer)
        self.extra: dict | None = None

    def numpyify(self) -> tuple:
        msg = self.deserialize()
        self._capture_header_metadata(msg)

        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9

        npified = np.array([msg.latitude, msg.longitude, msg.altitude], dtype=np.float64)
        status = getattr(getattr(msg, "status", None), "status", None)
        self.extra = navsat_extra(
            None if status is None else int(status), getattr(msg, "position_covariance", None)
        )

        return npified, msg.__class__.__name__, ts
