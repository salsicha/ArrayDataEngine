from __future__ import annotations

import numpy as np

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


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


    def __init__(self, rawdata: bytes, msgtype: str, deserializer=None):
        """Constructor

        `deserializer` is a callable `(rawdata, msgtype) -> message` supplied
        by the reader that produced the raw payload (e.g. AnyReader.deserialize),
        which knows whether the payload is ROS1- or CDR-serialized. Without it,
        deserialization falls back to CDR, which is only correct for ROS2 data.
        """

        self.rawdata = rawdata
        self.msgtype = msgtype
        self.frame_id: str | None = None
        self._deserializer = deserializer


    def deserialize(self):
        if self._deserializer is not None:
            return self._deserializer(self.rawdata, self.msgtype)
        return _get_typestore().deserialize_cdr(self.rawdata, self.msgtype)


    def numpyify(self) -> tuple:
        """Return `(array, type_name, timestamp)`; implemented by each sensor."""

        raise NotImplementedError(f"{type(self).__name__} does not implement numpyify()")

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
