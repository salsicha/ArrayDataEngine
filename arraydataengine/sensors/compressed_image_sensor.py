from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np

from .base_sensor import BaseSensor
from .image_sensor import array_encoding, bgr_to_rgb, canonical_encoding

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# compressed_depth_image_transport ConfigHeader: int32 format + float32 depthParam[2].
_DEPTH_HEADER = struct.Struct("<iff")


@lru_cache(maxsize=1)
def compressed_image_available() -> bool:
    """True when OpenCV is importable, which CompressedImage decoding needs."""

    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _as_uint8_buffer(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    return np.frombuffer(bytes(data), dtype=np.uint8)


def _imdecode(buffer: np.ndarray) -> np.ndarray:
    import cv2

    if not buffer.size:
        raise ValueError("Compressed image payload is empty")
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        # Corrupt or oversized payloads (e.g. dimensions above OpenCV's pixel
        # limit) raise cv2.error; report them like any other bad payload.
        raise ValueError(f"Unable to decode compressed image payload: {exc}") from exc
    if image is None:
        raise ValueError("Unable to decode compressed image payload")
    return image


def is_compressed_depth(fmt) -> bool:
    return isinstance(fmt, str) and "compresseddepth" in fmt.lower()


def decode_compressed_depth(data, fmt: str) -> tuple[np.ndarray, str]:
    """Decode a compressed_depth_image_transport payload.

    The payload is a 12-byte config header (format id and two float32 depth
    quantization parameters) followed by a PNG. ``16UC1`` images are returned
    as uint16; ``32FC1`` images are de-quantized to float32 metres with NaN
    for invalid (zero) pixels.
    """

    buffer = _as_uint8_buffer(data)
    lowered = fmt.lower()
    if "rvl" in lowered:
        raise ValueError(f"Unsupported compressedDepth codec in format {fmt!r} (only PNG is supported)")
    raw = buffer.tobytes()
    params = None
    if len(raw) >= _DEPTH_HEADER.size and not raw.startswith(_PNG_MAGIC):
        _, quant_a, quant_b = _DEPTH_HEADER.unpack_from(raw)
        params = (quant_a, quant_b)
        buffer = buffer[_DEPTH_HEADER.size:]
    image = _imdecode(buffer)
    if image.ndim != 2 or image.dtype != np.uint16:
        raise ValueError(
            f"compressedDepth payload decoded to {image.dtype} with shape {image.shape}, expected 16-bit single channel"
        )

    encoding = canonical_encoding(fmt.split(";", 1)[0]) if ";" in fmt else "16UC1"
    if encoding == "32FC1":
        if params is None:
            raise ValueError("32FC1 compressedDepth payload has no quantization header")
        quant_a, quant_b = params
        depth = np.full(image.shape, np.nan, dtype=np.float32)
        valid = image != 0
        depth[valid] = np.float32(quant_a) / (image[valid].astype(np.float32) - np.float32(quant_b))
        return depth, "32FC1"
    return image, encoding if encoding == "mono16" else "16UC1"


def decode_compressed_image_with_encoding(data, fmt: str | None = None) -> tuple[np.ndarray, str]:
    """Decode a CompressedImage payload; returns ``(image, encoding)``.

    Colour images come back in RGB(A) order (OpenCV decodes to BGR); the
    encoding names the returned layout (``rgb8``, ``mono8``, ``mono16``, ...).
    `fmt` is the message's ``format`` field; ``compressedDepth`` payloads are
    decoded with `decode_compressed_depth`.
    """

    if is_compressed_depth(fmt):
        return decode_compressed_depth(data, fmt)
    image = bgr_to_rgb(_imdecode(_as_uint8_buffer(data)))
    return image, array_encoding(image)


def decode_compressed_image(data, fmt: str | None = None) -> np.ndarray:
    """Decode a JPEG/PNG/... payload with OpenCV (colour images in RGB order)."""

    return decode_compressed_image_with_encoding(data, fmt)[0]


class CompressedImageSensor(BaseSensor):
    """Decode sensor_msgs/CompressedImage payloads into image arrays.

    Colour images are returned in RGB(A) order and `extra["encoding"]`
    describes the returned array.
    """

    def __init__(self, rawdata, msgtype, deserializer=None):
        super().__init__(rawdata, msgtype, deserializer=deserializer)
        self.extra: dict | None = None

    @staticmethod
    def is_available() -> bool:
        return compressed_image_available()

    def numpyify(self) -> tuple:
        msg = self.deserialize()
        self._capture_header_metadata(msg)
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9
        fmt = self._decode_optional_text(getattr(msg, "format", None))
        image, encoding = decode_compressed_image_with_encoding(msg.data, fmt)
        self.extra = {"encoding": encoding}
        return image, msg.__class__.__name__, ts
