"""Lazy MCAP playback with per-encoding and per-schema payload decoders."""

from __future__ import annotations

import os

import numpy as np

from ..recording import ENCODING, _require_mcap, decode_message
from .ros_source import RosSource, normalize_topics, validate_max_points


class McapSource:
    """Replay in recorded file order, without loading or sorting the recording.

    Each call to ``get_message()`` opens a fresh stream. Playback is as fast as
    the consumer requests messages; it does not sleep to match timestamps or
    tail a file still being written. Use ``DataSources`` for automatic routing.

    ADE NumPy and supported ROS 2/CDR sensor types are decoded automatically.
    ``decoders`` maps an encoding or ``(encoding, schema_name)`` to a callable
    ``(schema, channel, record) -> dict`` containing ``data`` and optional ADE
    metadata. Schema-specific hooks take precedence. Unknown encodings fail
    explicitly; filter their topics or supply a hook to read a mixed file.
    """

    supports_topic_filter = True

    def __init__(self, data_path, *, topics=None, max_points=None, decoders=None):
        _require_mcap()
        from mcap.reader import make_reader

        validate_max_points(max_points)
        self.data_path = os.fspath(data_path)
        self.topics = normalize_topics(topics)
        self.max_points = max_points
        self.decoders = dict(decoders or {})
        for key, decoder in self.decoders.items():
            if not (isinstance(key, str) or isinstance(key, tuple) and len(key) == 2
                    and all(isinstance(part, str) for part in key)) or not callable(decoder):
                raise TypeError("decoders maps encoding or (encoding, schema name) to callable")
        self._counts = None
        self._all_topics = []
        self._all_types = {}
        self._ros = RosSource(self.data_path)
        self._ros_schemas = {}
        with open(self.data_path, "rb") as stream:
            make_reader(stream).get_header()

    def _decoder(self, schema, channel):
        name = schema.name if schema is not None else ""
        decoder = self.decoders.get((channel.message_encoding, name))
        if decoder is None:
            decoder = self.decoders.get(channel.message_encoding)
        if decoder is not None:
            return decoder
        if channel.message_encoding == ENCODING:
            return lambda schema, channel, record: decode_message(record.data)
        if channel.message_encoding == "cdr" and schema is not None:
            return self._decode_cdr
        raise ValueError(
            f"No MCAP decoder for topic {channel.topic!r}, encoding {channel.message_encoding!r}, "
            f"schema {name!r}; pass mcap_decoders to DataSources (or decoders to McapSource)"
        )

    def _decode_cdr(self, schema, channel, record):
        from .cdr import CUSTOM_CDR_TYPES
        from .db3_source import EmbeddedTypes

        key = (schema.id, schema.name, schema.encoding, schema.data)
        embedded = self._ros_schemas.get(key)
        if embedded is None:
            definitions = []
            if schema.encoding in ("ros2msg", "ros2idl"):
                definitions.append((schema.name, schema.encoding, schema.data.decode("utf-8")))
            embedded = self._ros_schemas[key] = EmbeddedTypes(definitions)
        message = self._ros._decode_message(
            channel.topic, schema.name, record.data, record.log_time,
            rosbag2=True, fmt="cdr", deserializer=embedded.deserialize,
            embedded=schema.name in CUSTOM_CDR_TYPES and embedded.has(schema.name),
        )
        if message is None:
            raise ValueError(
                f"No ADE array conversion for ROS type {schema.name!r} on {channel.topic!r}; "
                "supply a schema-specific mcap_decoders hook (sensor types may require the ros/image extras)"
            )
        return message

    def data_exists(self):
        return os.path.isfile(self.data_path)

    def get_data_path(self):
        return self.data_path

    def _read_counts(self):
        if self._counts is not None:
            return self._counts
        from mcap.reader import make_reader

        with open(self.data_path, "rb") as stream:
            summary = make_reader(stream).get_summary()
        if summary is not None:
            self._all_topics = list(dict.fromkeys(channel.topic for channel in summary.channels.values()))
            self._all_types = {
                channel.topic: (summary.schemas[channel.schema_id].name if channel.schema_id in summary.schemas
                                else channel.message_encoding)
                for channel in summary.channels.values()
            }
        counts = {topic: 0 for topic in self._all_topics if self.topics is None or topic in self.topics}
        if summary is not None and summary.statistics is not None and self.max_points is None:
            for channel_id, channel in summary.channels.items():
                if self.topics is None or channel.topic in self.topics:
                    self._decoder(summary.schemas.get(channel.schema_id), channel)
                    counts[channel.topic] = counts.get(channel.topic, 0) + summary.statistics.channel_message_counts.get(channel_id, 0)
        else:
            for message in self.messages():
                topic = message["topic"]
                counts[topic] = counts.get(topic, 0) + 1
        self._counts = counts
        if not self._all_topics:
            self._all_topics = list(counts)
        return counts

    def get_topics(self):
        return list(self._read_counts())

    def get_count(self, topic):
        return self._read_counts().get(topic, 0)

    def get_topic_types(self):
        self._read_counts()
        return dict(self._all_types)

    def messages(self, *, with_record_index=False):
        from mcap.reader import NonSeekingReader

        with open(self.data_path, "rb") as stream:
            reader = NonSeekingReader(stream, validate_crcs=True)
            self._records_read = 0
            for record_index, (schema, channel, record) in enumerate(reader.iter_messages(log_time_order=False)):
                self._records_read += 1
                if self.topics is not None and channel.topic not in self.topics:
                    continue
                message = self._decoder(schema, channel)(schema, channel, record)
                if not isinstance(message, dict) or "data" not in message:
                    raise ValueError(f"MCAP decoder for {channel.topic!r} must return a dict containing 'data'")
                message = dict(message)
                message["data"] = np.asarray(message["data"])
                if with_record_index:
                    message["_mcap_record_index"] = record_index
                if schema is not None:
                    import hashlib
                    message.setdefault("schema_sha256", hashlib.sha256(schema.data).hexdigest())
                    message.setdefault("schema_name", schema.name)
                    message.setdefault("schema_encoding", schema.encoding)
                message["topic"] = channel.topic
                message.setdefault("name", schema.name if schema is not None else channel.topic)
                message.setdefault("timestamp", record.log_time / 1e9)
                message.setdefault("timestamp_ns", record.log_time)
                message.setdefault("publish_time_ns", record.publish_time)
                message.setdefault("sequence", record.sequence)
                if self.max_points is not None and message.get("point_count", 0) > self.max_points:
                    continue
                yield message

    def get_message(self):
        yield from self.messages()
