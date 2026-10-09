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
# Colour encodings returned in RGB(A) order: source encoding -> returned encoding.
_TO_RGB = {
    "bgr8": "rgb8",
    "bgr": "rgb8",
    "rgb": "rgb8",
    "bgr16": "rgb16",
    "bgra8": "rgba8",
    "bgra16": "rgba16",
}
_NAMED_ARRAY_ENCODINGS = {
    (np.dtype(np.uint8), 1): "mono8",
    (np.dtype(np.uint8), 3): "rgb8",
    (np.dtype(np.uint8), 4): "rgba8",
    (np.dtype(np.uint16), 1): "mono16",
    (np.dtype(np.uint16), 3): "rgb16",
    (np.dtype(np.uint16), 4): "rgba16",
}
_CV_DEPTHS = {"u": "U", "i": "S", "f": "F"}


def canonical_encoding(encoding: str) -> str:
    """Normalized encoding name: OpenCV-style upper case (``32FC1``), others lower case."""

    key = str(encoding).strip().lower()
    if _CV_ENCODING.fullmatch(key):
        return key.upper()
    return key


def array_encoding(image, color_order: str = "rgb") -> str:
    """sensor_msgs-style encoding describing an image array as returned.

    uint8/uint16 arrays with 1, 3, or 4 channels are ``mono8``/``rgb8``/
    ``rgba8`` (``mono16``/``rgb16``/``rgba16``; ``bgr*`` when `color_order`
    is "bgr"); anything else uses the OpenCV form, e.g. ``32FC1``.
    """

    array = np.asarray(image)
    channels = 1 if array.ndim == 2 else int(array.shape[-1])
    named = _NAMED_ARRAY_ENCODINGS.get((array.dtype.newbyteorder("="), channels))
    if named is not None:
        if color_order == "bgr" and channels in (3, 4):
            named = named.replace("rgb", "bgr")
        return named
    kind = _CV_DEPTHS.get(array.dtype.kind)
    if kind is None:
        raise ValueError(f"No image encoding for dtype {array.dtype}")
    return f"{array.dtype.itemsize * 8}{kind}C{channels}"


_CV_COLOR_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """New array with the colour channels of a BGR or BGRA image swapped (alpha stays last).

    Arrays that are not 3- or 4-channel images are returned unchanged.
    """

    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return image
    if image.dtype in _CV_COLOR_DTYPES:
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            # ~20x faster than a NumPy channel gather on large frames.
            code = cv2.COLOR_BGR2RGB if image.shape[-1] == 3 else cv2.COLOR_BGRA2RGBA
            return cv2.cvtColor(np.ascontiguousarray(image), code)
    order = [2, 1, 0] if image.shape[-1] == 3 else [2, 1, 0, 3]
    return np.ascontiguousarray(image[..., order])


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
    """Convert sensor_msgs/Image messages to arrays.

    Colour images are returned in RGB(A) channel order (``bgr8``/``bgra8``/
    ``bgr16``/``bgra16`` are converted); `extra["encoding"]` names the
    returned layout, e.g. ``rgb8``, ``mono16``, or ``32FC1``.
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

        encoding = canonical_encoding(msg.encoding)
        if encoding in _TO_RGB:
            if encoding.startswith("bgr"):
                npified = bgr_to_rgb(npified)
            encoding = _TO_RGB[encoding]
        self.extra = {"encoding": encoding}
        return npified, msg.__class__.__name__, ts
