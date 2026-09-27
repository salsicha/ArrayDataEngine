from __future__ import annotations

import re

import numpy as np
from .base_sensor import BaseSensor

# Named sensor_msgs/image_encodings: (base dtype, channels).
_ENCODINGS = {
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgb": (np.uint8, 3),
    "bgr": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "bgr16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgra16": (np.uint16, 4),
    # Packed 4:2:2 YUV: two bytes per pixel, returned as (H, W, 2).
    "yuv422": (np.uint8, 2),
    "uyvy": (np.uint8, 2),
    "yuyv": (np.uint8, 2),
    "yuv422_yuy2": (np.uint8, 2),
}
# OpenCV-style encodings such as 8UC3, 16SC1, 32FC1, 64FC1.
_CV_ENCODING = re.compile(r"(8|16|32|64)([usf])c(\d+)")
# Raw Bayer mosaics are single-channel images.
_BAYER_ENCODING = re.compile(r"bayer_(?:rggb|bggr|gbrg|grbg)(8|16)")
_CV_KINDS = {"u": "u", "s": "i", "f": "f"}


def image_encoding_layout(encoding: str) -> tuple[np.dtype, int]:
    """Return `(dtype, channels)` for a sensor_msgs/Image encoding.

    Raises ValueError for encodings without a known pixel layout.
    """

    key = str(encoding).strip().lower()
    if key in _ENCODINGS:
        base_dtype, channels = _ENCODINGS[key]
        return np.dtype(base_dtype), channels

    match = _BAYER_ENCODING.fullmatch(key)
    if match:
        return np.dtype(np.uint16 if match.group(1) == "16" else np.uint8), 1

    match = _CV_ENCODING.fullmatch(key)
    if match:
        bits, kind, channels = int(match.group(1)), match.group(2), int(match.group(3))
        if channels >= 1 and not (kind == "f" and bits == 8):
            return np.dtype(f"{_CV_KINDS[kind]}{bits // 8}"), channels

    raise ValueError(f"Unsupported image encoding {encoding!r}")


class ImageSensor(BaseSensor):
    """Image Sensor Class
    Attributes:
    Args:
    Returns:
    """


    def numpyify(self) -> tuple:
        msg = self.deserialize()
        self._capture_header_metadata(msg)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9

        base_dtype, channels = image_encoding_layout(msg.encoding)
        dtype = base_dtype
        if dtype.itemsize > 1:
            # sensor_msgs/Image.is_bigendian is uint8; rosbags yields int, not bool
            is_bigendian = bool(getattr(msg, "is_bigendian", False) or False)
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

        # Drop per-row padding (step > width * pixel size) before viewing the
        # pixel bytes with the encoding's dtype.
        rows = raw[:expected_bytes].reshape((msg.height, row_bytes))[:, :image_row_bytes]
        data = np.ascontiguousarray(rows).view(dtype)
        if dtype != base_dtype:
            data = data.astype(base_dtype)

        if channels == 1:
            npified = data.reshape((msg.height, msg.width))
        else:
            npified = data.reshape((msg.height, msg.width, channels))

        return npified, msg.__class__.__name__, ts
