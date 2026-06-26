from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


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
        self.data = memoryview(data)
        self.offset = 4
        self.endian = "<" if data[1] == 1 else ">"

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
        length = self.read_uint32()
        raw = bytes(self.data[self.offset:self.offset + length])
        self.offset += length
        self.align(4)
        return raw.rstrip(b"\x00").decode(errors="replace")

    def read_bytes(self) -> bytes:
        length = self.read_uint32()
        raw = bytes(self.data[self.offset:self.offset + length])
        self.offset += length
        self.align(4)
        return raw


def decode_supported_cdr_message(rawdata: bytes, msgtype: str) -> DecodedMessage | None:
    if msgtype == "geometry_msgs/msg/PoseStamped":
        return decode_pose_stamped(rawdata)
    if msgtype == "sensor_msgs/msg/PointCloud2":
        return decode_pointcloud2(rawdata)
    if msgtype == "mapeverything_msgs/msg/DepthAnythingCalibration":
        return decode_depth_anything_calibration(rawdata)
    return None


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
    schema_version = reader.read_uint32()
    source = reader.read_string()
    relative_pointcloud_topic = reader.read_string()
    overlay_mesh_source = reader.read_string()
    frame_id = reader.read_string()
    relative_depth_width = reader.read_uint32()
    relative_depth_height = reader.read_uint32()
    image_width = reader.read_uint32()
    image_height = reader.read_uint32()
    scale = reader.read_float64()
    offset = reader.read_float64()
    equation = reader.read_string()
    relative_depth_units = reader.read_string()
    metric_depth_units = reader.read_string()
    calibration_source = reader.read_string()
    metadata_json = reader.read_string()
    values = np.array(
        [
            scale,
            offset,
            relative_depth_width,
            relative_depth_height,
            image_width,
            image_height,
        ],
        dtype=np.float64,
    )
    extra = {
        "schema_version": schema_version,
        "source": source,
        "relative_pointcloud_topic": relative_pointcloud_topic,
        "overlay_mesh_source": overlay_mesh_source,
        "frame_id": frame_id,
        "relative_depth_width": relative_depth_width,
        "relative_depth_height": relative_depth_height,
        "image_width": image_width,
        "image_height": image_height,
        "scale": scale,
        "offset": offset,
        "equation": equation,
        "relative_depth_units": relative_depth_units,
        "metric_depth_units": metric_depth_units,
        "calibration_source": calibration_source,
        "metadata_json": metadata_json,
    }
    return DecodedMessage(values, "DepthAnythingCalibration", timestamp, header_frame_id, extra)


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

    point_count = int(height) * int(width)
    points = pointcloud_xyz(point_bytes, fields, point_count, int(point_step), bool(is_bigendian))
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
    point_count: int,
    point_step: int,
    is_bigendian: bool,
) -> np.ndarray:
    by_name = {str(field["name"]): field for field in fields}
    missing = [name for name in ("x", "y", "z") if name not in by_name]
    if missing:
        raise ValueError(f"PointCloud2 is missing XYZ field(s): {missing}")

    byteorder = ">" if is_bigendian else "<"
    output = np.empty((point_count, 3), dtype=np.float32)
    for column, name in enumerate(("x", "y", "z")):
        field = by_name[name]
        dtype = point_field_dtype(int(field["datatype"]), byteorder)
        values = np.ndarray(
            shape=(point_count,),
            dtype=dtype,
            buffer=point_bytes,
            offset=int(field["offset"]),
            strides=(point_step,),
        )
        output[:, column] = values.astype(np.float32, copy=False)
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
