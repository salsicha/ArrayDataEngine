from __future__ import annotations

import numpy as np
from .base_sensor import BaseSensor

_ENCODINGS = {
    "mono16": (np.uint16, 1),
    "16uc1": (np.uint16, 1),
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgb": (np.uint8, 3),
    "bgr": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "8uc1": (np.uint8, 1),
}


class ImageSensor(BaseSensor):
    """Image Sensor Class
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

        dtype_channels = _ENCODINGS.get(msg.encoding.lower())
        if dtype_channels is None:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            return data, msg.__class__.__name__, ts

        base_dtype, channels = dtype_channels
        dtype = np.dtype(base_dtype)
        if dtype.itemsize > 1:
            is_bigendian = getattr(msg, "is_bigendian", False)
            if not isinstance(is_bigendian, bool):
                is_bigendian = False
            dtype = dtype.newbyteorder(">" if is_bigendian else "<")

        bytes_per_pixel = dtype.itemsize * channels
        image_row_bytes = msg.width * bytes_per_pixel
        row_bytes = getattr(msg, "step", image_row_bytes)
        if not isinstance(row_bytes, (int, np.integer)):
            row_bytes = image_row_bytes
        if row_bytes < image_row_bytes:
            raise ValueError(
                f"Image step {row_bytes} is too small for {msg.width} pixels with encoding {msg.encoding}"
            )

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        expected_bytes = msg.height * row_bytes
        if raw.size < expected_bytes:
            raise ValueError(f"Image data has {raw.size} bytes, expected at least {expected_bytes}")

        rows = raw[:expected_bytes].reshape((msg.height, row_bytes))[:, :image_row_bytes]
        data = np.ascontiguousarray(rows).view(dtype)
        if dtype != np.dtype(base_dtype):
            data = data.astype(base_dtype)

        if channels == 1:
            npified = data.reshape((msg.height, msg.width))
        else:
            npified = data.reshape((msg.height, msg.width, channels))

        return npified, msg.__class__.__name__, ts
