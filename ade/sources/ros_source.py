from __future__ import annotations

import logging

from .base_source import BaseSource

_logger = logging.getLogger(__name__)


class RosSource(BaseSource):
    """Shared message conversion for ROS bag-like sources."""

    SENSOR_TYPES = {}

    def _sensor_for(self, connection, rawdata, deserializer=None):
        type_name = connection.msgtype.rsplit("/", 1)[1].lower()
        sensor_cls = self.SENSOR_TYPES.get(type_name)
        if sensor_cls is None:
            if self._debug:
                _logger.debug("Message type not supported: %s", type_name)
            return None
        if deserializer is None:
            return sensor_cls(rawdata, connection.msgtype)
        return sensor_cls(rawdata, connection.msgtype, deserializer=deserializer)

    def messages(self):
        """Yield NumPy-oriented messages from a ROS source."""

        with self.reader() as reader:
            # AnyReader.deserialize knows whether payloads are ROS1- or
            # CDR-serialized; without it ROS1 .bag payloads would be
            # mis-parsed as CDR.
            deserializer = getattr(reader, "deserialize", None)
            for connection, timestamp, rawdata in reader.messages():
                sensor = self._sensor_for(connection, rawdata, deserializer=deserializer)
                if sensor is None:
                    continue

                npified, class_name, ts = sensor.numpyify()
                message = {
                    "data": npified,
                    "timestamp": ts,
                    "topic": connection.topic,
                    "name": class_name,
                }
                frame_id = getattr(sensor, "frame_id", None)
                if frame_id is not None:
                    message["frame_id"] = frame_id
                yield message
