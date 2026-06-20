from __future__ import annotations

import logging

from .base_source import BaseSource

_logger = logging.getLogger(__name__)


class RosSource(BaseSource):
    """Shared message conversion for ROS bag-like sources."""

    SENSOR_TYPES = {}

    def _sensor_for(self, connection, rawdata):
        type_name = connection.msgtype.rsplit("/", 1)[1].lower()
        sensor_cls = self.SENSOR_TYPES.get(type_name)
        if sensor_cls is None:
            if self._debug:
                _logger.debug("Message type not supported: %s", type_name)
            return None
        return sensor_cls(rawdata, connection.msgtype)

    def messages(self):
        """Yield NumPy-oriented messages from a ROS source."""

        with self.reader() as reader:
            for connection, timestamp, rawdata in reader.messages():
                sensor = self._sensor_for(connection, rawdata)
                if sensor is None:
                    continue

                npified, class_name, ts = sensor.numpyify()
                yield {
                    "data": npified,
                    "timestamp": ts,
                    "topic": connection.topic,
                    "name": class_name,
                }
