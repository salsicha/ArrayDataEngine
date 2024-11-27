from __future__ import annotations

import numpy as np
import ros2_numpy as rnp

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from rosbags.serde import deserialize_cdr


class BaseSensor:
    """Base Sensor Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, rawdata: bytes, msgtype: str):
        """Constructor

        """

        self.rawdata = rawdata
        self.msgtype = msgtype

        # ROSbags to native ROS class converter 
        self.NATIVE_CLASSES: dict[str, Any] = {}


    def deserialize(self):
        msg = deserialize_cdr(self.rawdata, self.msgtype)
        return msg


    def numpyify(self) -> tuple:
        msg = self.deserialize()
        msg = self.rosbags_to_native(msg)
        npified = rnp.numpify(msg)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return npified, msg.__class__.__name__, ts


    def rosbags_to_native(self, msg: Any) -> Any:  # noqa: ANN401
        """Convert rosbags message to native message.

        Args:
            msg: Rosbags message.

        Returns:
            Native message.

        """

        msgtype: str = msg.__msgtype__
        if msgtype not in self.NATIVE_CLASSES:
            pkg, name = msgtype.rsplit('/', 1)
            self.NATIVE_CLASSES[msgtype] = getattr(importlib.import_module(pkg.replace('/', '.')), name)

        fields = {}
        for name, field in msg.__dataclass_fields__.items():
            if 'ClassVar' in field.type:
                continue
            value = getattr(msg, name)
            if '__msg__' in field.type:
                value = self.rosbags_to_native(value)
            elif isinstance(value, list):
                value = [self.rosbags_to_native(x) for x in value]
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            fields[name] = value

        return self.NATIVE_CLASSES[msgtype](**fields)
    
