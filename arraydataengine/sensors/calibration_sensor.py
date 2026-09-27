from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


class DepthAnythingCalibrationSensor(BaseSensor):
    """Convert mapeverything_msgs/DepthAnythingCalibration messages.

    Used for payloads the dedicated CDR decoder cannot read (e.g. ROS1 bags);
    produces the same array and `extra` fields as `decode_depth_anything_calibration`.
    """

    def __init__(self, rawdata, msgtype, deserializer=None):
        super().__init__(rawdata, msgtype, deserializer=deserializer)
        self.extra: dict | None = None

    def numpyify(self) -> tuple:
        from ..sources.cdr import DEPTH_ANYTHING_CALIBRATION_FIELDS, depth_anything_calibration_message

        msg = self.deserialize()
        self._capture_header_metadata(msg)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        fields = {
            name: _plain(getattr(msg, name))
            for name in DEPTH_ANYTHING_CALIBRATION_FIELDS
            if hasattr(msg, name)
        }
        decoded = depth_anything_calibration_message(fields, ts, self.frame_id)
        self.extra = decoded.extra
        return decoded.data, decoded.name, decoded.timestamp
