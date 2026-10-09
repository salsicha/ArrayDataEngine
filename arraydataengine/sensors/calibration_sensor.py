from __future__ import annotations

import dataclasses

import numpy as np

from .base_sensor import BaseSensor


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _message_fields(msg) -> list[str]:
    """Field names of a deserialized message in definition order."""

    if dataclasses.is_dataclass(msg):
        names = [field.name for field in dataclasses.fields(msg)]
    else:
        slots = getattr(type(msg), "__slots__", None)
        names = list(slots) if isinstance(slots, (list, tuple)) else []
    # rosbags also stores constants (upper case) and __msgtype__ as fields.
    return [name for name in names if not name.startswith("_") and not name.isupper()]


class DepthAnythingCalibrationSensor(BaseSensor):
    """Convert mapeverything_msgs/DepthAnythingCalibration messages.

    Used for payloads the dedicated CDR decoder cannot read (ROS1 bags) and for
    rosbag2 recordings that embed the message definition, so newer schema
    versions decode by field name. Produces the same array and `extra` fields
    as `decode_depth_anything_calibration`, plus any additional scalar or
    string body fields a newer definition adds.
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
        extra = dict(decoded.extra or {})
        for name in _message_fields(msg):
            if name == "header" or name in fields:
                continue
            value = _plain(getattr(msg, name, None))
            if isinstance(value, (bool, int, float, str)):
                extra.setdefault(name, value)
        self.extra = extra
        return decoded.data, decoded.name, decoded.timestamp
