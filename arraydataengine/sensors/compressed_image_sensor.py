from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base_sensor import BaseSensor


@lru_cache(maxsize=1)
def compressed_image_available() -> bool:
    """True when OpenCV is importable, which CompressedImage decoding needs."""

    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def decode_compressed_image(data) -> np.ndarray:
    """Decode a JPEG/PNG/... payload with OpenCV (BGR channel order, like ImgSource)."""

    import cv2

    if isinstance(data, np.ndarray):
        buffer = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    else:
        buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED) if buffer.size else None
    if image is None:
        raise ValueError("Unable to decode compressed image payload")
    return image


class CompressedImageSensor(BaseSensor):
    """Decode sensor_msgs/CompressedImage payloads into image arrays."""

    @staticmethod
    def is_available() -> bool:
        return compressed_image_available()

    def numpyify(self) -> tuple:
        msg = self.deserialize()
        self._capture_header_metadata(msg)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        return decode_compressed_image(msg.data), msg.__class__.__name__, ts
