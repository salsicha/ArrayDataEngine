from __future__ import annotations

import numpy as np

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

try:
    from rosbags.serde import deserialize_cdr as _deserialize_cdr
except ImportError:
    _deserialize_cdr = None


@lru_cache(maxsize=1)
def _get_typestore():
    from rosbags.typesys import Stores, get_typestore

    store = Stores.ROS2_JAZZY if hasattr(Stores, "ROS2_JAZZY") else Stores.LATEST
    return get_typestore(store)


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
        self.frame_id: str | None = None

        # ROSbags to native ROS class converter 
        self.NATIVE_CLASSES: dict[str, Any] = {}


    def deserialize(self):
        if _deserialize_cdr is not None:
            msg = _deserialize_cdr(self.rawdata, self.msgtype)
        else:
            msg = _get_typestore().deserialize_cdr(self.rawdata, self.msgtype)
        return msg


    def numpyify(self) -> tuple:
        import ros2_numpy as rnp

        msg = self.deserialize()
        self._capture_header_metadata(msg)
        msg = self.rosbags_to_native(msg)
        npified = rnp.numpify(msg)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return npified, msg.__class__.__name__, ts

    def _capture_header_metadata(self, msg: Any) -> None:  # noqa: ANN401
        header = getattr(msg, "header", None)
        self.frame_id = self._decode_optional_text(getattr(header, "frame_id", None))

    def _decode_optional_text(self, value: Any) -> str | None:  # noqa: ANN401
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                return None
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, str):
            return value
        return None


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
    
