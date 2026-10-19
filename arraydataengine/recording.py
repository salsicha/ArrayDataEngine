"""Incremental MCAP recording of ADE message dictionaries.

The ``arraydataengine/numpy/v1`` profile permits mixed channels. NumPy channels
are schema-less with ``ade-numpy-v1`` encoding: a little-endian uint32 JSON byte length, UTF-8
JSON metadata (all message fields except ``data``), then one pickle-free NPY
array. Each message carries its own dtype and shape, including ragged scans.
Other channels retain their original encoding and schema (e.g. ROS 2 CDR).
ROS tooling needs a custom decoder for the ADE NumPy channels.
"""

from __future__ import annotations

import io
import json
import logging
import math
import operator
import os
import struct
import threading

import numpy as np

PROFILE = "arraydataengine/numpy/v1"
ENCODING = "ade-numpy-v1"
_MAGIC = b"\x89MCAP0\r\n"


def _require_mcap():
    try:
        from mcap.writer import CompressionType, Writer
    except ImportError as exc:
        raise ImportError('MCAP recording requires: pip install "arraydataengine[mcap]"') from exc
    return CompressionType, Writer


def is_ade_mcap(path) -> bool:
    """Peek only at the MCAP Header profile, without requiring the extra."""
    try:
        with open(path, "rb") as stream:
            prefix = stream.read(21)
            if len(prefix) != 21 or prefix[:9] != _MAGIC + b"\x01":
                return False
            record_size, profile_size = struct.unpack("<QI", prefix[9:])
            return (
                profile_size == len(PROFILE)
                and record_size >= profile_size + 8
                and stream.read(profile_size) == PROFILE.encode("utf-8")
            )
    except OSError:
        return False


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"MCAP metadata must be JSON-compatible, got {type(value).__name__}")


def _uint(value, bits, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a uint{bits}") from exc
    if isinstance(value, (bool, np.bool_)) or not 0 <= result < 2**bits:
        raise ValueError(f"{name} must be a uint{bits}")
    return result


def _encode_message(message):
    topic = message["topic"]
    if not isinstance(topic, str) or not topic:
        raise ValueError("topic must be a non-empty string")
    timestamp = float(message["timestamp"])
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite, non-negative seconds")
    log_time = message.get("timestamp_ns")
    if log_time is None:
        if timestamp >= 2**64 / 1e9:
            raise ValueError("timestamp exceeds the MCAP uint64 nanosecond range")
        log_time = round(timestamp * 1e9)
    log_time = _uint(log_time, 64, "timestamp_ns")
    publish_time = _uint(message.get("publish_time_ns", log_time), 64, "publish_time_ns")
    sequence = _uint(message.get("sequence", 0), 32, "sequence")
    array = np.asarray(message["data"])
    if array.dtype.hasobject:
        raise ValueError("MCAP recording does not support object arrays")
    metadata = {key: value for key, value in message.items() if key != "data"}
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError("message metadata keys must be strings")
    for key in ("name", "frame_id"):
        if isinstance(metadata.get(key), bytes):
            metadata[key] = metadata[key].decode("utf-8")
    metadata["timestamp"] = timestamp
    header = json.dumps(metadata, default=_json_value, allow_nan=False).encode("utf-8")
    payload = io.BytesIO()
    payload.write(struct.pack("<I", len(header)))
    payload.write(header)
    np.save(payload, array, allow_pickle=False)
    return topic, log_time, publish_time, sequence, payload.getvalue()


def decode_message(payload: bytes) -> dict:
    if len(payload) < 4:
        raise ValueError("Truncated ADE MCAP message")
    header_size = struct.unpack_from("<I", payload)[0]
    boundary = 4 + header_size
    if boundary >= len(payload):
        raise ValueError("Truncated ADE MCAP metadata or array")
    metadata = json.loads(payload[4:boundary])
    if not isinstance(metadata, dict):
        raise ValueError("ADE MCAP metadata must be an object")
    stream = io.BytesIO(payload[boundary:])
    # Reject archives and pickle payloads; this encoding is exactly one NPY.
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("ADE MCAP payload must contain an NPY array")
    stream.seek(0)
    metadata["data"] = np.load(stream, allow_pickle=False)
    return metadata


class McapRecorder:
    """Write messages as they arrive; no message count or fixed shape needed.

    Existing files are refused unless ``overwrite=True``. ``write`` calls are
    serialized for use from sensor callbacks. ``record`` consumes an iterable
    synchronously. Close (or use a context manager) to finish indexes and flush
    the final chunk. ``flush(durable=True)`` also fsyncs the file.

    Metadata must be JSON-compatible; NumPy metadata arrays/scalars become
    lists/scalars. ``timestamp`` is seconds; optional ``timestamp_ns`` and
    ``publish_time_ns`` preserve exact MCAP times, and ``sequence`` is uint32.
    """

    def __init__(self, path, *, compression="zstd", chunk_size=1024 * 1024, overwrite=False):
        CompressionType, Writer = _require_mcap()
        codecs = {"zstd": CompressionType.ZSTD, "lz4": CompressionType.LZ4, "none": CompressionType.NONE}
        if compression not in codecs:
            raise ValueError("compression must be 'zstd', 'lz4', or 'none'")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._lock = threading.Lock()
        self._channels = {}
        self._schemas = {}
        self._closed = False
        self._stream = open(path, "wb" if overwrite else "xb")
        try:
            self._writer = Writer(self._stream, compression=codecs[compression], chunk_size=chunk_size)
            self._writer.start(profile=PROFILE, library="arraydataengine")
        except BaseException:
            self._stream.close()
            raise

    def _check_open(self):
        if self._closed:
            raise ValueError("MCAP recorder is closed")

    def write(self, message: dict) -> None:
        """Append one ADE message, owning its serialized bytes before returning."""
        topic, log_time, publish_time, sequence, payload = _encode_message(message)
        self.write_encoded(topic, payload, timestamp_ns=log_time, publish_time_ns=publish_time,
                           sequence=sequence, message_encoding=ENCODING)

    def write_encoded(self, topic, payload, *, timestamp_ns, message_encoding,
                      publish_time_ns=None, sequence=0, schema_name=None,
                      schema_encoding="", schema_data=b"") -> None:
        """Record original serialized bytes, alongside NumPy channels if desired.

        For example, CDR with a ``ros2msg`` schema or Protobuf with its descriptor
        schema can be recorded without converting the payload to NumPy first.
        Decoding such channels requires a matching reader decoder. The caller
        supplies the encoding/schema; this method does not validate wire data.
        """
        if not isinstance(topic, str) or not topic:
            raise ValueError("topic must be a non-empty string")
        if not isinstance(message_encoding, str) or not message_encoding:
            raise ValueError("message_encoding must be a non-empty string")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        payload = bytes(payload)
        timestamp_ns = _uint(timestamp_ns, 64, "timestamp_ns")
        publish_time_ns = _uint(timestamp_ns if publish_time_ns is None else publish_time_ns, 64, "publish_time_ns")
        sequence = _uint(sequence, 32, "sequence")
        if not isinstance(schema_encoding, str) or not isinstance(schema_data, (bytes, bytearray, memoryview)):
            raise TypeError("schema_encoding must be a string and schema_data must be bytes-like")
        schema_data = bytes(schema_data)
        if schema_name is None:
            if schema_encoding or schema_data:
                raise ValueError("schema_name is required when supplying a schema")
        elif not isinstance(schema_name, str) or not schema_name:
            raise ValueError("schema_name must be a non-empty string")
        if schema_data and not schema_encoding:
            raise ValueError("schema_encoding is required for non-empty schema_data")
        with self._lock:
            self._check_open()
            schema_id = 0
            if schema_name is not None:
                key = (schema_name, schema_encoding, schema_data)
                if key not in self._schemas:
                    self._schemas[key] = self._writer.register_schema(*key)
                schema_id = self._schemas[key]
            channel_key = (topic, message_encoding, schema_id)
            if channel_key not in self._channels:
                self._channels[channel_key] = self._writer.register_channel(topic, message_encoding, schema_id)
            self._writer.add_message(self._channels[channel_key], timestamp_ns, payload, publish_time_ns, sequence)

    def record(self, messages) -> int:
        """Consume a generator/iterable; return the number written on exhaustion."""
        count = 0
        for message in messages:
            self.write(message)
            count += 1
        return count

    def flush(self, *, durable=False) -> None:
        """Finish the current chunk; optionally sync to disk. Recording continues."""
        with self._lock:
            self._check_open()
            self._writer.flush()
            if durable:
                os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._writer.finish()
            finally:
                self._closed = True
                self._stream.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            logging.getLogger(__name__).exception("Could not finalize MCAP after a recording error")
        return False
