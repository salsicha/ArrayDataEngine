from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

POSE_STAMPED_TYPE = "geometry_msgs/msg/PoseStamped"
NAVSATFIX_TYPE = "sensor_msgs/msg/NavSatFix"
POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
DEPTH_ANYTHING_CALIBRATION_TYPE = "mapeverything_msgs/msg/DepthAnythingCalibration"

# Types with a dedicated (rosbags-free) CDR decoder in this module.
CDR_DECODED_TYPES = frozenset(
    {POSE_STAMPED_TYPE, POINTCLOUD2_TYPE, DEPTH_ANYTHING_CALIBRATION_TYPE}
)
# Types decode_rosbridge_json_message understands; CompressedImage also needs cv2.
JSON_DECODED_TYPES = frozenset(
    {POSE_STAMPED_TYPE, NAVSATFIX_TYPE, POINTCLOUD2_TYPE, DEPTH_ANYTHING_CALIBRATION_TYPE}
)

# DepthAnythingCalibration body fields in message order. The body's own
# frame_id is reported as "calibration_frame_id" so it can never overwrite the
# header frame_id of the yielded message.
DEPTH_ANYTHING_CALIBRATION_FIELDS = (
    "schema_version", "source", "relative_pointcloud_topic", "overlay_mesh_source",
    "frame_id", "relative_depth_width", "relative_depth_height", "image_width",
    "image_height", "scale", "offset", "equation", "relative_depth_units",
    "metric_depth_units", "calibration_source", "metadata_json",
)
_CALIBRATION_EXTRA_KEYS = {"frame_id": "calibration_frame_id"}

# Representation identifiers from the DDS-XTypes encapsulation header.
_ENCAPSULATIONS = {
    0x0000: "CDR_BE",
    0x0001: "CDR_LE",
    0x0002: "PL_CDR_BE",
    0x0003: "PL_CDR_LE",
    0x0006: "CDR2_BE",
    0x0007: "CDR2_LE",
    0x0008: "D_CDR2_BE",
    0x0009: "D_CDR2_LE",
    0x000A: "PL_CDR2_BE",
    0x000B: "PL_CDR2_LE",
}


@dataclass(frozen=True)
class DecodedMessage:
    data: np.ndarray
    name: str
    timestamp: float
    frame_id: str | None = None
    extra: dict | None = None


class CDRReader:
    def __init__(self, data: bytes):
        if len(data) < 4:
            raise ValueError("CDR payload is too short")
        kind = (data[0] << 8) | data[1]
        if kind not in (0x0000, 0x0001):
            # Parameter-list (PL_CDR) and XCDR2 payloads use different
            # alignment and framing rules; decoding them as plain CDR would
            # silently return garbage.
            raise ValueError(
                f"Unsupported CDR encapsulation 0x{kind:04x} "
                f"({_ENCAPSULATIONS.get(kind, 'unknown')}); only plain XCDR1 "
                "(CDR_BE/CDR_LE) payloads are supported"
            )
        self.data = memoryview(data)
        self.offset = 4
        self.endian = "<" if kind == 0x0001 else ">"

    def align(self, size: int) -> None:
        remainder = (self.offset - 4) % size
        if remainder:
            self.offset += size - remainder

    def read(self, fmt: str, size: int):
        self.align(size)
        value = struct.unpack_from(self.endian + fmt, self.data, self.offset)[0]
        self.offset += size
        return value

    def read_bool(self) -> bool:
        return bool(self.read("?", 1))

    def read_uint8(self) -> int:
        return int(self.read("B", 1))

    def read_int32(self) -> int:
        return int(self.read("i", 4))

    def read_uint32(self) -> int:
        return int(self.read("I", 4))

    def read_float64(self) -> float:
        return float(self.read("d", 8))

    def read_string(self) -> str:
        # CDR has no padding after string/sequence payloads; alignment is
        # driven by the next field's own type.
        length = self.read_uint32()
        raw = bytes(self.data[self.offset:self.offset + length])
        self.offset += length
        return raw.rstrip(b"\x00").decode(errors="replace")

    def read_bytes(self) -> bytes:
        length = self.read_uint32()
        raw = bytes(self.data[self.offset:self.offset + length])
        self.offset += length
        return raw


def cdr_type_supported(msgtype: str) -> bool:
    """True when `msgtype` has a dedicated CDR decoder in this module."""

    return msgtype in CDR_DECODED_TYPES


def json_type_supported(msgtype: str) -> bool:
    """True when rosbridge JSON payloads of `msgtype` decode to arrays."""

    if msgtype == COMPRESSED_IMAGE_TYPE:
        from ..sensors.compressed_image_sensor import compressed_image_available

        return compressed_image_available()
    return msgtype in JSON_DECODED_TYPES


def decode_supported_cdr_message(rawdata: bytes, msgtype: str) -> DecodedMessage | None:
    # Bags recorded through rosbridge/foxglove can carry JSON envelopes
    # instead of CDR; route those to the JSON decoder.
    if _looks_like_json(rawdata):
        return decode_rosbridge_json_message(rawdata, msgtype)
    if msgtype == POSE_STAMPED_TYPE:
        return decode_pose_stamped(rawdata)
    if msgtype == POINTCLOUD2_TYPE:
        return decode_pointcloud2(rawdata)
    if msgtype == DEPTH_ANYTHING_CALIBRATION_TYPE:
        return decode_depth_anything_calibration(rawdata)
    return None


_JSON_LEADING_BYTES = frozenset(b"{ \t\r\n")


def _looks_like_json(rawdata: bytes) -> bool:
    # CDR payloads always start with the 0x00 high byte of the encapsulation
    # id, so a JSON object (optionally preceded by whitespace) is unambiguous.
    if not rawdata or rawdata[0] not in _JSON_LEADING_BYTES:
        return False
    stripped = rawdata.strip()
    return stripped.startswith(b"{") and stripped.endswith(b"}")


looks_like_json = _looks_like_json


def _json_timestamp(header) -> float:
    stamp = header.get("stamp") if isinstance(header, dict) else None
    if not isinstance(stamp, dict):
        return 0.0
    # ROS 2 rosbridge uses sec/nanosec; ROS 1 rosbridge uses secs/nsecs.
    sec = stamp.get("sec", stamp.get("secs", 0))
    nanosec = stamp.get("nanosec", stamp.get("nsec", stamp.get("nsecs", 0)))
    return float(sec or 0) + float(nanosec or 0) * 1e-9


def _json_bytes(value) -> bytes:
    # rosbridge encodes uint8[] fields as base64 strings.
    if isinstance(value, str):
        import base64

        return base64.b64decode(value)
    return bytes(value or b"")


def decode_rosbridge_json_message(rawdata: bytes, msgtype: str) -> DecodedMessage | None:
    """Decode a rosbridge-style JSON envelope ({"op": "publish", "msg": ...})."""

    try:
        envelope = json.loads(rawdata.decode(errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(envelope, dict):
        return None
    msg = envelope.get("msg", envelope)
    if not isinstance(msg, dict):
        return None

    header = msg.get("header")
    if not isinstance(header, dict):
        header = {}
    timestamp = _json_timestamp(header)
    frame_id = header.get("frame_id")

    if msgtype == POSE_STAMPED_TYPE:
        pose = msg.get("pose", {})
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})
        values = np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
                float(orientation.get("w", 1.0)),
            ],
            dtype=np.float64,
        )
        return DecodedMessage(values, "PoseStamped", timestamp, frame_id)

    if msgtype == NAVSATFIX_TYPE:
        values = np.array(
            [
                float(msg.get("latitude", 0.0)),
                float(msg.get("longitude", 0.0)),
                float(msg.get("altitude", 0.0)),
            ],
            dtype=np.float64,
        )
        return DecodedMessage(values, "NavSatFix", timestamp, frame_id)

    if msgtype == POINTCLOUD2_TYPE:
        fields = [dict(field) for field in msg.get("fields", [])]
        points = pointcloud_xyz(
            _json_bytes(msg.get("data", "")),
            fields,
            int(msg.get("height", 0)),
            int(msg.get("width", 0)),
            int(msg.get("point_step", 0)),
            int(msg.get("row_step", 0)),
            bool(msg.get("is_bigendian", False)),
        )
        return DecodedMessage(points, "PointCloud2", timestamp, frame_id)

    if msgtype == COMPRESSED_IMAGE_TYPE:
        from ..sensors.compressed_image_sensor import compressed_image_available, decode_compressed_image

        if not compressed_image_available():
            return None
        image = decode_compressed_image(_json_bytes(msg.get("data", "")))
        return DecodedMessage(image, "CompressedImage", timestamp, frame_id)

    if msgtype == DEPTH_ANYTHING_CALIBRATION_TYPE:
        return depth_anything_calibration_message(msg, timestamp, frame_id)

    return None


def depth_anything_calibration_message(
    fields: dict, timestamp: float, header_frame_id: str | None
) -> DecodedMessage:
    """Build the shared DepthAnythingCalibration output from body `fields`.

    `data` holds [scale, offset, relative_depth_width, relative_depth_height,
    image_width, image_height]; every body field present is returned in
    `extra` (the body frame_id as ``calibration_frame_id``).
    """

    values = np.array(
        [
            float(fields.get(key, 0.0))
            for key in (
                "scale",
                "offset",
                "relative_depth_width",
                "relative_depth_height",
                "image_width",
                "image_height",
            )
        ],
        dtype=np.float64,
    )
    extra = {
        _CALIBRATION_EXTRA_KEYS.get(key, key): fields[key]
        for key in DEPTH_ANYTHING_CALIBRATION_FIELDS
        if key in fields
    }
    return DecodedMessage(values, "DepthAnythingCalibration", timestamp, header_frame_id, extra)


def decode_header(reader: CDRReader) -> tuple[float, str]:
    sec = reader.read_int32()
    nanosec = reader.read_uint32()
    frame_id = reader.read_string()
    return sec + nanosec * 1e-9, frame_id


def decode_pose_stamped(rawdata: bytes) -> DecodedMessage:
    reader = CDRReader(rawdata)
    timestamp, frame_id = decode_header(reader)
    pose = np.array(
        [
            reader.read_float64(),
            reader.read_float64(),
            reader.read_float64(),
            reader.read_float64(),
            reader.read_float64(),
            reader.read_float64(),
            reader.read_float64(),
        ],
        dtype=np.float64,
    )
    return DecodedMessage(pose, "PoseStamped", timestamp, frame_id)


def decode_depth_anything_calibration(rawdata: bytes) -> DecodedMessage:
    reader = CDRReader(rawdata)
    timestamp, header_frame_id = decode_header(reader)
    fields = {
        "schema_version": reader.read_uint32(),
        "source": reader.read_string(),
        "relative_pointcloud_topic": reader.read_string(),
        "overlay_mesh_source": reader.read_string(),
        "frame_id": reader.read_string(),
        "relative_depth_width": reader.read_uint32(),
        "relative_depth_height": reader.read_uint32(),
        "image_width": reader.read_uint32(),
        "image_height": reader.read_uint32(),
        "scale": reader.read_float64(),
        "offset": reader.read_float64(),
        "equation": reader.read_string(),
        "relative_depth_units": reader.read_string(),
        "metric_depth_units": reader.read_string(),
        "calibration_source": reader.read_string(),
        "metadata_json": reader.read_string(),
    }
    return depth_anything_calibration_message(fields, timestamp, header_frame_id)


def decode_pointcloud2(rawdata: bytes) -> DecodedMessage:
    reader = CDRReader(rawdata)
    timestamp, frame_id = decode_header(reader)
    height = reader.read_uint32()
    width = reader.read_uint32()
    fields = [decode_point_field(reader) for _ in range(reader.read_uint32())]
    is_bigendian = reader.read_bool()
    point_step = reader.read_uint32()
    row_step = reader.read_uint32()
    point_bytes = reader.read_bytes()
    reader.read_bool()  # is_dense

    points = pointcloud_xyz(
        point_bytes,
        fields,
        int(height),
        int(width),
        int(point_step),
        int(row_step),
        bool(is_bigendian),
    )
    return DecodedMessage(points, "PointCloud2", timestamp, frame_id)


def decode_point_field(reader: CDRReader) -> dict[str, int | str]:
    name = reader.read_string()
    offset = reader.read_uint32()
    datatype = reader.read_uint8()
    count = reader.read_uint32()
    return {"name": name, "offset": offset, "datatype": datatype, "count": count}


def pointcloud_xyz(
    point_bytes: bytes,
    fields: list[dict[str, int | str]],
    height: int,
    width: int,
    point_step: int,
    row_step: int,
    is_bigendian: bool,
) -> np.ndarray:
    """Extract an (N, 3) XYZ array from a PointCloud2 payload.

    The output keeps the precision of the XYZ fields: float64 fields (e.g. UTM
    coordinates) stay float64, float32 fields stay float32, and small integer
    fields widen to float32.
    """

    by_name = {str(field["name"]): field for field in fields}
    missing = [name for name in ("x", "y", "z") if name not in by_name]
    if missing:
        raise ValueError(f"PointCloud2 is missing XYZ field(s): {missing}")
    if height < 0 or width < 0 or point_step < 0 or row_step < 0:
        raise ValueError("PointCloud2 height, width, point_step, and row_step must be non-negative")

    byteorder = ">" if is_bigendian else "<"
    layout = []
    for name in ("x", "y", "z"):
        field = by_name[name]
        layout.append((int(field["offset"]), point_field_dtype(int(field["datatype"]), byteorder)))
    output_dtype = np.result_type(np.float32, *(dtype.newbyteorder("=") for _, dtype in layout))

    point_count = height * width
    output = np.empty((point_count, 3), dtype=output_dtype)
    if point_count == 0:
        return output

    if point_step <= 0:
        raise ValueError(f"PointCloud2 point_step must be positive for {point_count} points, got {point_step}")
    for name, (offset, dtype) in zip(("x", "y", "z"), layout):
        if offset < 0 or offset + dtype.itemsize > point_step:
            raise ValueError(
                f"PointCloud2 field {name!r} at offset {offset} ({dtype.itemsize} bytes) "
                f"does not fit in point_step {point_step}"
            )
    field_end = max(offset + dtype.itemsize for offset, dtype in layout)

    # Organized clouds may pad each row; honor row_step when it is larger
    # than the packed row size AND the buffer is actually big enough for the
    # padded layout — some writers lie about row_step but serialize densely.
    padded_rows = (
        height > 1
        and row_step > width * point_step
        and len(point_bytes) >= (height - 1) * row_step + width * point_step
    )
    if padded_rows:
        required = (height - 1) * row_step + (width - 1) * point_step + field_end
    else:
        required = (point_count - 1) * point_step + field_end
    if len(point_bytes) < required:
        raise ValueError(
            f"PointCloud2 data has {len(point_bytes)} bytes, expected at least {required} "
            f"for {height}x{width} points with point_step {point_step}"
        )

    for column, (offset, dtype) in enumerate(layout):
        if padded_rows:
            values = np.ndarray(
                shape=(height, width),
                dtype=dtype,
                buffer=point_bytes,
                offset=offset,
                strides=(row_step, point_step),
            ).reshape(-1)
        else:
            values = np.ndarray(
                shape=(point_count,),
                dtype=dtype,
                buffer=point_bytes,
                offset=offset,
                strides=(point_step,),
            )
        output[:, column] = values
    return output


def point_field_dtype(datatype: int, byteorder: str) -> np.dtype:
    mapping = {
        1: "i1",
        2: "u1",
        3: "i2",
        4: "u2",
        5: "i4",
        6: "u4",
        7: "f4",
        8: "f8",
    }
    if datatype not in mapping:
        raise ValueError(f"Unsupported PointField datatype: {datatype}")
    code = mapping[datatype]
    if code.endswith("1"):
        return np.dtype(code)
    return np.dtype(byteorder + code)
