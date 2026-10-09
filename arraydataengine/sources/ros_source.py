from __future__ import annotations

import importlib.util
import logging
import struct
from dataclasses import dataclass

from .base_source import BaseSource, accepts_parameter
from .cdr import (
    CUSTOM_CDR_TYPES,
    cdr_decoder,
    decode_rosbridge_json_message,
    json_type_supported,
    looks_like_json,
)
from ..sensors.base_sensor import BaseSensor
from ..sensors.calibration_sensor import DepthAnythingCalibrationSensor
from ..sensors.compressed_image_sensor import CompressedImageSensor
from ..sensors.image_sensor import ImageSensor
from ..sensors.imu_sensor import IMUSensor
from ..sensors.nav_sensor import NavSensor
from ..sensors.odom_sensor import OdomSensor
from ..sensors.pointcloud2_sensor import PointCloudSensor, finite_points
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


def serialization_format(connection, overrides: dict | None = None) -> str | None:
    """rosbag2 serialization format of a reader connection; None for ROS1.

    `overrides` maps topic names to the format recorded in the storage file
    itself, which wins over the (sometimes wrong) metadata.yaml value.
    """

    if overrides:
        fmt = overrides.get(getattr(connection, "topic", None))
        if isinstance(fmt, str):
            return fmt
    fmt = getattr(getattr(connection, "ext", None), "serialization_format", None)
    return fmt if isinstance(fmt, str) else None


def has_embedded_definition(connection) -> bool:
    """Whether a rosbags connection carries a message definition from the bag."""

    msgdef = getattr(connection, "msgdef", None)
    data = getattr(msgdef, "data", None)
    return isinstance(data, str) and bool(data.strip())


def is_json_format(fmt: str | None) -> bool:
    return fmt is not None and "json" in fmt.lower()


def validate_max_points(max_points) -> int | None:
    """None (no limit) or a positive integer row limit for point clouds."""

    if max_points is None:
        return None
    if isinstance(max_points, bool):
        raise ValueError(f"max_points must be a positive integer, got {max_points!r}")
    try:
        value = int(max_points)
    except (TypeError, ValueError):
        raise ValueError(f"max_points must be a positive integer, got {max_points!r}") from None
    if value < 1 or value != max_points:
        raise ValueError(f"max_points must be a positive integer, got {max_points!r}")
    return value


def normalize_topics(topics) -> tuple[str, ...] | None:
    """None (all topics) or a tuple of topic names; a single string is one topic."""

    if topics is None:
        return None
    if isinstance(topics, str):
        topics = [topics]
    try:
        names = tuple(dict.fromkeys(str(topic) for topic in topics))
    except TypeError:
        raise TypeError(f"topics must be a topic name or an iterable of names, got {topics!r}") from None
    return names


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


def merge_default_types(reader) -> None:
    """Let connections without an embedded definition use the default typestore.

    rosbags only falls back to `default_typestore` when *no* connection of a
    bag embeds a definition; when some do, standard types recorded without
    one would fail on every message.
    """

    typestore = getattr(reader, "typestore", None)
    default = getattr(reader, "default_typestore", None)
    fielddefs = getattr(typestore, "fielddefs", None)
    default_defs = getattr(default, "fielddefs", None)
    if not isinstance(fielddefs, dict) or not isinstance(default_defs, dict):
        return
    if not any(
        isinstance(getattr(connection, "msgtype", None), str) and connection.msgtype not in fielddefs
        for connection in getattr(reader, "connections", [])
    ):
        return
    missing = {name: value for name, value in default_defs.items() if name not in fielddefs}
    if not missing:
        return
    try:
        typestore.register(missing)
    except Exception as exc:  # a conflicting definition must not make the bag unreadable
        _logger.debug("Could not merge default message types: %s", exc)


def _message_count(connection) -> int:
    count = getattr(connection, "msgcount", 0)
    return int(count) if isinstance(count, int) and count > 0 else 0


class EnteredReader:
    """Context wrapper around a reader context that has already been entered."""

    def __init__(self, context, reader):
        self._context = context
        self._reader = reader

    def __enter__(self):
        return self._reader

    def __exit__(self, exc_type, exc, tb):
        return self._context.__exit__(exc_type, exc, tb)


@dataclass(frozen=True)
class DecodePlan:
    """How messages of one type are decoded on one read path (cached per type)."""

    supported: bool
    cdr: object = None  # dedicated CDR decoder (rosbag2 payloads)
    sensor: object = None  # sensor converter class


_UNSUPPORTED = DecodePlan(False)


class RosSource(BaseSource):
    """Shared message conversion for ROS bag-like sources.

    Every read path (rosbags reader or the built-in SQLite reader) goes through
    `_decode_message`, so one recording yields identical messages regardless of
    how it is opened:

    - `name` is the short message type name (e.g. "PoseStamped");
    - a zero header stamp (or a header-less type) falls back to the log time;
    - PointCloud2 data is the unpadded ``(N, 3)`` array of finite XYZ rows in
      the field dtype, with ``point_count`` = N; when `max_points` is set,
      larger clouds are skipped with a warning (never truncated);
    - colour images are RGB(A) and carry an ``encoding`` key;
    - NavSatFix messages carry ``status`` and ``position_covariance``;
    - messages that fail to decode are skipped with a warning.

    `topics` restricts reading (and get_topics/get_count) to those topics.
    """

    SENSOR_TYPES = DEFAULT_SENSOR_TYPES
    # DataSources leaves `topics` filtering to sources that set this.
    supports_topic_filter = True

    def __init__(self, data_path, debug=False, max_points: int | None = None, topics=None):
        super().__init__(data_path, debug=debug)
        self.max_points = validate_max_points(max_points)
        self.topics = normalize_topics(topics)
        self._plans: dict[tuple, DecodePlan] = {}

    def get_topic_types(self) -> dict:
        """Message type of every recorded topic, including ones get_topics omits
        because no decoder exists for them (or `topics` excludes them)."""

        return dict(self._metadata().get("types", {}))

    def _metadata(self):
        metadata = super()._metadata()
        self._warn_missing_topics(metadata)
        return metadata

    def _warn_missing_topics(self, metadata) -> None:
        if self.topics is None or getattr(self, "_warned_missing", False):
            return
        self._warned_missing = True
        missing = [topic for topic in self.topics if topic not in metadata.get("counts", {})]
        if missing:
            _logger.warning(
                "Requested topic(s) %s not found or not decodable in %s", missing, self.data_path
            )

    def _topic_wanted(self, topic) -> bool:
        return self.topics is None or topic in self.topics

    def _sensor_class(self, msgtype):
        return self.SENSOR_TYPES.get(short_type_name(msgtype).lower())

    def _plan(
        self,
        msgtype: str,
        fmt: str | None = None,
        *,
        rosbag2: bool = False,
        has_deserializer: bool = True,
        embedded: bool = False,
    ) -> DecodePlan:
        key = (msgtype, is_json_format(fmt), bool(rosbag2), bool(has_deserializer), bool(embedded))
        plan = self._plans.get(key)
        if plan is None:
            plan = self._make_plan(msgtype, *key[1:])
            self._plans[key] = plan
        return plan

    def _make_plan(self, msgtype, json_format, rosbag2, has_deserializer, embedded) -> DecodePlan:
        if json_format and not json_type_supported(msgtype):
            return _UNSUPPORTED

        cdr = cdr_decoder(msgtype) if rosbag2 else None
        if cdr is not None and embedded and has_deserializer and msgtype in CUSTOM_CDR_TYPES:
            # Decode custom types by the definition the bag embeds; the
            # fixed-field-order decoder only matches one schema version.
            cdr = None

        sensor_cls = self._sensor_class(msgtype)
        if sensor_cls is not None:
            if (
                not has_deserializer
                and isinstance(sensor_cls, type)
                and issubclass(sensor_cls, BaseSensor)
                and not rosbags_available()
            ):
                # BaseSensor falls back to the rosbags typestore for CDR payloads.
                sensor_cls = None
            else:
                is_available = getattr(sensor_cls, "is_available", None)
                if callable(is_available) and not is_available():
                    sensor_cls = None

        if json_format:
            # JSON payloads use the JSON decoder; a stray CDR payload on the
            # topic can still use the CDR/sensor decoders.
            return DecodePlan(True, cdr, sensor_cls)
        if cdr is None and sensor_cls is None:
            return _UNSUPPORTED
        return DecodePlan(True, cdr, sensor_cls)

    def _type_supported(
        self,
        msgtype: str,
        fmt: str | None = None,
        *,
        rosbag2: bool = False,
        has_deserializer: bool = True,
    ) -> bool:
        """Whether messages of `msgtype` decode to arrays on this read path."""

        return self._plan(msgtype, fmt, rosbag2=rosbag2, has_deserializer=has_deserializer).supported

    def _connection_supported(self, connection, formats: dict | None = None) -> bool:
        msgtype = getattr(connection, "msgtype", None)
        if not isinstance(msgtype, str) or not self._topic_wanted(connection.topic):
            return False
        fmt = serialization_format(connection, formats)
        return self._type_supported(msgtype, fmt, rosbag2=fmt is not None)

    def _build_sensor(self, sensor_cls, rawdata, msgtype, deserializer):
        if deserializer is None:
            sensor = sensor_cls(rawdata, msgtype)
        else:
            sensor = sensor_cls(rawdata, msgtype, deserializer=deserializer)
        if hasattr(sensor, "max_points"):
            sensor.max_points = self.max_points
        return sensor

    def _decode_message(
        self,
        topic,
        msgtype,
        rawdata,
        log_time_ns,
        *,
        rosbag2,
        deserializer=None,
        fmt=None,
        embedded=False,
    ):
        """Decode one payload into a message dict.

        Returns None for types without a decoder; raises one of DECODE_ERRORS
        for malformed payloads (including JSON payloads of a type without a
        JSON decoder). `rosbag2` marks CDR/JSON payloads (anything but ROS1),
        which can use the dedicated decoders in `cdr`.
        """

        plan = self._plan(
            msgtype, fmt, rosbag2=rosbag2, has_deserializer=deserializer is not None, embedded=embedded
        )
        if not isinstance(rawdata, bytes):
            rawdata = bytes(rawdata)

        if rosbag2 and looks_like_json(rawdata):
            decoded = decode_rosbridge_json_message(rawdata, msgtype)
            if decoded is None:
                raise ValueError(f"No rosbridge JSON decoder for {msgtype}")
        elif plan.cdr is not None:
            decoded = plan.cdr(rawdata)
        elif plan.sensor is not None:
            decoded = None
        else:
            if self._debug:
                _logger.debug("Message type not supported: %s", msgtype)
            if is_json_format(fmt) and plan.supported:
                raise ValueError(f"{msgtype} payload on a JSON topic is not JSON")
            return None

        point_count = None
        if decoded is not None:
            data, name, timestamp = decoded.data, decoded.name, decoded.timestamp
            frame_id, extra = decoded.frame_id, decoded.extra
            if name == "PointCloud2":
                data, point_count = finite_points(data, self.max_points)
        else:
            sensor = self._build_sensor(plan.sensor, rawdata, msgtype, deserializer)
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

    def _reader_messages(self, reader_context, formats: dict | None = None, stats: dict | None = None):
        """Yield decoded messages from an (unopened or EnteredReader) rosbags reader.

        `formats` maps topics to the serialization format recorded in the
        storage itself (see `serialization_format`). When given, `stats`
        receives ``expected`` (message count the reader reports for the
        selected connections) and ``read`` (raw messages it actually yielded).
        """

        if stats is not None:
            stats.update(expected=0, read=0)
        skipped = SkippedMessages()
        try:
            with reader_context as reader:
                merge_default_types(reader)
                # AnyReader.deserialize knows whether payloads are ROS1- or
                # CDR-serialized; without it ROS1 .bag payloads would be
                # mis-parsed as CDR.
                deserializer = getattr(reader, "deserialize", None)
                connections = [
                    connection
                    for connection in reader.connections
                    if self._connection_supported(connection, formats)
                ]
                if not connections:
                    return
                if stats is not None:
                    stats["expected"] = sum(_message_count(connection) for connection in connections)
                for connection, timestamp, rawdata in iter_reader_messages(reader, connections):
                    if stats is not None:
                        stats["read"] += 1
                    topic = connection.topic
                    if not self._topic_wanted(topic):
                        continue
                    fmt = serialization_format(connection, formats)
                    try:
                        message = self._decode_message(
                            topic,
                            connection.msgtype,
                            rawdata,
                            timestamp,
                            rosbag2=fmt is not None,
                            deserializer=deserializer,
                            fmt=fmt,
                            embedded=has_embedded_definition(connection),
                        )
                    except DECODE_ERRORS as exc:
                        skipped.record(topic, connection.msgtype, exc)
                        continue
                    if message is not None:
                        yield message
        finally:
            skipped.summarize()

    def messages(self):
        """Yield NumPy-oriented messages from a ROS source."""

        yield from self._reader_messages(self.reader())
