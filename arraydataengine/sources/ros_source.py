from __future__ import annotations

import importlib.util
import logging
import struct

from .base_source import BaseSource, accepts_parameter
from ..sensors.base_sensor import BaseSensor
from ..sensors.calibration_sensor import DepthAnythingCalibrationSensor
from ..sensors.compressed_image_sensor import CompressedImageSensor
from ..sensors.image_sensor import ImageSensor
from ..sensors.imu_sensor import IMUSensor
from ..sensors.nav_sensor import NavSensor
from ..sensors.odom_sensor import OdomSensor
from ..sensors.pointcloud2_sensor import DEFAULT_MAX_POINTS, PointCloudSensor, fixed_size_points
from ..sensors.pose_sensor import PoseSensor

_logger = logging.getLogger(__name__)

# Sensor converters keyed by lower-case short message type name.
DEFAULT_SENSOR_TYPES = {
    "pointcloud2": PointCloudSensor,
    "image": ImageSensor,
    "compressedimage": CompressedImageSensor,
    "imu": IMUSensor,
    "odometry": OdomSensor,
    "navsatfix": NavSensor,
    "pose": PoseSensor,
    "posestamped": PoseSensor,
    "depthanythingcalibration": DepthAnythingCalibrationSensor,
}

# Exceptions a single malformed or unexpected payload can raise while being
# decoded (rosbags raises struct.error/AssertionError/KeyError). Such a message
# is skipped with a warning instead of ending the whole stream.
DECODE_ERRORS = (
    struct.error,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    AssertionError,
    OverflowError,
)

# Only the first few skipped messages per topic are logged as warnings.
_MAX_SKIP_WARNINGS = 5


def short_type_name(name) -> str:
    """'sensor_msgs/msg/Image' or rosbags' 'sensor_msgs__msg__Image' -> 'Image'."""

    return str(name).rsplit("/", 1)[-1].rsplit("__", 1)[-1]


def rosbags_available() -> bool:
    try:
        return importlib.util.find_spec("rosbags") is not None
    except (ImportError, ValueError):
        return False


def serialization_format(connection) -> str | None:
    """rosbag2 serialization format of a reader connection; None for ROS1."""

    fmt = getattr(getattr(connection, "ext", None), "serialization_format", None)
    return fmt if isinstance(fmt, str) else None


def is_json_format(fmt: str | None) -> bool:
    return fmt is not None and "json" in fmt.lower()


def validate_max_points(max_points) -> int:
    if max_points is None:
        return DEFAULT_MAX_POINTS
    value = int(max_points)
    if value < 1 or value != max_points:
        raise ValueError(f"max_points must be a positive integer, got {max_points!r}")
    return value


class SkippedMessages:
    """Rate-limited logging of messages skipped because they failed to decode."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def record(self, topic, msgtype, exc) -> None:
        count = self.counts.get(topic, 0) + 1
        self.counts[topic] = count
        if count <= _MAX_SKIP_WARNINGS:
            _logger.warning("Skipping undecodable %s message on %s: %s", msgtype, topic, exc)
            if count == _MAX_SKIP_WARNINGS:
                _logger.warning("Further skipped messages on %s are logged at DEBUG level", topic)
        else:
            _logger.debug("Skipping undecodable %s message on %s: %s", msgtype, topic, exc)

    def summarize(self) -> None:
        for topic, count in self.counts.items():
            if count > _MAX_SKIP_WARNINGS:
                _logger.warning("Skipped %d undecodable messages on %s", count, topic)


def iter_reader_messages(reader, connections):
    """reader.messages() restricted to `connections` when the reader supports it."""

    if accepts_parameter(reader.messages, "connections"):
        return reader.messages(connections=connections)
    return reader.messages()


class RosSource(BaseSource):
    """Shared message conversion for ROS bag-like sources.

    Every read path (rosbags reader or the built-in SQLite reader) goes through
    `_decode_message`, so one recording yields identical messages regardless of
    how it is opened:

    - `name` is the short message type name (e.g. "PoseStamped");
    - a zero header stamp (or a header-less type) falls back to the log time;
    - PointCloud2 data drops non-finite XYZ rows and is zero-padded to
      `(max_points, 3)` keeping the field dtype; the number of valid points is
      reported as `point_count`, and clouds with more than `max_points` finite
      points are skipped with a warning;
    - messages that fail to decode are skipped with a warning.
    """

    SENSOR_TYPES = DEFAULT_SENSOR_TYPES

    def __init__(self, data_path, debug=False, max_points: int | None = None):
        super().__init__(data_path, debug=debug)
        self.max_points = validate_max_points(max_points)

    def get_topic_types(self) -> dict:
        """Message type of every recorded topic, including ones get_topics omits
        because no decoder exists for them."""

        return dict(self._metadata().get("types", {}))

    def _sensor_class(self, msgtype):
        return self.SENSOR_TYPES.get(short_type_name(msgtype).lower())

    def _type_supported(
        self,
        msgtype: str,
        fmt: str | None = None,
        *,
        rosbag2: bool = False,
        has_deserializer: bool = True,
    ) -> bool:
        """Whether messages of `msgtype` decode to arrays on this read path."""

        from .cdr import cdr_type_supported, json_type_supported

        if is_json_format(fmt):
            return json_type_supported(msgtype)
        if rosbag2 and cdr_type_supported(msgtype):
            return True
        sensor_cls = self._sensor_class(msgtype)
        if sensor_cls is None:
            return False
        if (
            not has_deserializer
            and isinstance(sensor_cls, type)
            and issubclass(sensor_cls, BaseSensor)
            and not rosbags_available()
        ):
            # BaseSensor falls back to the rosbags typestore for CDR payloads.
            return False
        is_available = getattr(sensor_cls, "is_available", None)
        return bool(is_available()) if callable(is_available) else True

    def _connection_supported(self, connection) -> bool:
        msgtype = getattr(connection, "msgtype", None)
        if not isinstance(msgtype, str):
            return False
        fmt = serialization_format(connection)
        return self._type_supported(msgtype, fmt, rosbag2=fmt is not None)

    def _build_sensor(self, sensor_cls, rawdata, msgtype, deserializer):
        if deserializer is None:
            sensor = sensor_cls(rawdata, msgtype)
        else:
            sensor = sensor_cls(rawdata, msgtype, deserializer=deserializer)
        if hasattr(sensor, "max_points"):
            sensor.max_points = self.max_points
        return sensor

    def _decode_message(self, topic, msgtype, rawdata, log_time_ns, *, rosbag2, deserializer=None):
        """Decode one payload into a message dict.

        Returns None for types without a decoder; raises one of DECODE_ERRORS
        for malformed payloads. `rosbag2` marks CDR/JSON payloads (anything
        but ROS1), which can use the dedicated decoders in `cdr`.
        """

        from .cdr import decode_rosbridge_json_message, decode_supported_cdr_message, looks_like_json

        if not isinstance(rawdata, bytes):
            rawdata = bytes(rawdata)

        decoded = None
        if rosbag2:
            if looks_like_json(rawdata):
                decoded = decode_rosbridge_json_message(rawdata, msgtype)
                if decoded is None:
                    return None
            else:
                decoded = decode_supported_cdr_message(rawdata, msgtype)

        point_count = None
        if decoded is not None:
            data, name, timestamp = decoded.data, decoded.name, decoded.timestamp
            frame_id, extra = decoded.frame_id, decoded.extra
            if name == "PointCloud2":
                data, point_count = fixed_size_points(data, self.max_points)
        else:
            if not self._type_supported(msgtype, has_deserializer=deserializer is not None):
                if self._debug:
                    _logger.debug("Message type not supported: %s", msgtype)
                return None
            sensor = self._build_sensor(self._sensor_class(msgtype), rawdata, msgtype, deserializer)
            data, name, timestamp = sensor.numpyify()
            frame_id = getattr(sensor, "frame_id", None)
            extra = getattr(sensor, "extra", None)
            point_count = getattr(sensor, "point_count", None)

        if timestamp == 0:
            # Zero header stamps and header-less types use the bag log time.
            timestamp = int(log_time_ns) * 1e-9
        message = {
            "data": data,
            "timestamp": timestamp,
            "topic": str(topic),
            "name": short_type_name(name),
        }
        if frame_id is not None:
            message["frame_id"] = frame_id
        if isinstance(point_count, int):
            message["point_count"] = point_count
        if extra:
            for key, value in extra.items():
                # Message-body fields never overwrite the core keys.
                message.setdefault(key, value)
        return message

    def _reader_messages(self, reader_context):
        """Yield decoded messages from an (unopened) rosbags reader."""

        skipped = SkippedMessages()
        try:
            with reader_context as reader:
                # AnyReader.deserialize knows whether payloads are ROS1- or
                # CDR-serialized; without it ROS1 .bag payloads would be
                # mis-parsed as CDR.
                deserializer = getattr(reader, "deserialize", None)
                connections = [
                    connection
                    for connection in reader.connections
                    if self._connection_supported(connection)
                ]
                if not connections:
                    return
                for connection, timestamp, rawdata in iter_reader_messages(reader, connections):
                    fmt = serialization_format(connection)
                    try:
                        message = self._decode_message(
                            connection.topic,
                            connection.msgtype,
                            rawdata,
                            timestamp,
                            rosbag2=fmt is not None,
                            deserializer=deserializer,
                        )
                    except DECODE_ERRORS as exc:
                        skipped.record(connection.topic, connection.msgtype, exc)
                        continue
                    if message is not None:
                        yield message
        finally:
            skipped.summarize()

    def messages(self):
        """Yield NumPy-oriented messages from a ROS source."""

        yield from self._reader_messages(self.reader())
