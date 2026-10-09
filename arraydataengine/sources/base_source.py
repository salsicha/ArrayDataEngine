from __future__ import annotations

import inspect
from pathlib import Path
from numbers import Real

AnyReader = None


class _LowerSuffixPath(type(Path())):
    """Path whose `suffix` is lower case.

    rosbags picks the bag format from the exact suffix (".bag", ".mcap",
    ".db3"); this lets it open files named e.g. ``REC.BAG`` or ``REC.MCAP``.
    """

    @property
    def suffix(self):
        return super().suffix.lower()


def rosbags_path(path) -> Path:
    path = Path(path)
    suffix = path.suffix
    return path if suffix == suffix.lower() else _LowerSuffixPath(path)


def accepts_parameter(func, name: str) -> bool:
    """True when `func` can be called with keyword argument `name`."""

    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def combine_metadata(parts) -> dict:
    """Merge per-storage metadata dicts ({topics, counts, types, start, end}) in order."""

    topics = []
    counts = {}
    types = {}
    start = None
    end = None
    for part in parts:
        for topic, msgtype in part.get("types", {}).items():
            types.setdefault(topic, msgtype)
        for topic in part["topics"]:
            if topic not in counts:
                topics.append(topic)
                counts[topic] = 0
            counts[topic] += part["counts"].get(topic, 0)
        part_start = part.get("start")
        part_end = part.get("end")
        if isinstance(part_start, Real):
            start = part_start if start is None else min(start, part_start)
        if isinstance(part_end, Real):
            end = part_end if end is None else max(end, part_end)
    duration = None if start is None or end is None else (end - start) * 1e-9
    return {
        "topics": topics,
        "counts": counts,
        "types": types,
        "start": start,
        "end": end,
        "duration": duration,
    }


class BaseSource:
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path, debug=False):
        """Constructor

        """
        self.data_path = data_path
        self._debug = debug
        self._metadata_cache = None

    def reader(self, paths=None):
        global AnyReader
        if AnyReader is None:
            from rosbags.highlevel import AnyReader as Reader

            AnyReader = Reader
        paths = [rosbags_path(path) for path in (paths if paths is not None else [self.data_path])]
        if accepts_parameter(AnyReader, "default_typestore"):
            # rosbag2 recordings made with Humble or older embed no message
            # definitions; AnyReader needs a default typestore to decode them.
            from ..sensors.base_sensor import _get_typestore

            return AnyReader(paths, default_typestore=_get_typestore())
        return AnyReader(paths)

    def clear_metadata_cache(self) -> None:
        self._metadata_cache = None

    def _connection_supported(self, connection, formats: dict | None = None) -> bool:
        """Whether get_topics/get_count should report this reader connection."""

        return True

    def _reader_metadata(self, reader_context=None, formats: dict | None = None) -> dict:
        """Topics, counts, and time range of a rosbags reader.

        `formats` maps topics to the serialization format stored in the bag's
        storage files (see RosSource). The end time is the latest message's
        log time (rosbags reports one nanosecond past it).
        """

        topics = []
        counts = {}
        types = {}
        start = None
        end = None

        with (self.reader() if reader_context is None else reader_context) as reader:
            for connection in reader.connections:
                msgtype = getattr(connection, "msgtype", None)
                if isinstance(msgtype, str):
                    types.setdefault(connection.topic, msgtype)
                if not self._connection_supported(connection, formats):
                    continue
                topic = connection.topic
                if topic not in counts:
                    topics.append(topic)
                    counts[topic] = 0
                counts[topic] += connection.msgcount

            start = getattr(reader, "start_time", None)
            end = getattr(reader, "end_time", None)

        if isinstance(start, Real) and isinstance(end, Real):
            if end > start:
                # rosbags' end_time is one nanosecond after the latest message;
                # match the SQLite reader, which reports the latest log time.
                end -= 1
            elif end < start:
                # rosbags reports (2**63 - 1, 0) for recordings without messages.
                start = end = None

        return {
            "topics": topics,
            "counts": counts,
            "types": types,
            "start": start if isinstance(start, Real) else None,
            "end": end if isinstance(end, Real) else None,
        }

    def _metadata(self):
        if self._metadata_cache is None:
            self._metadata_cache = combine_metadata([self._reader_metadata()])
        return self._metadata_cache

    def get_topics(self):
        return list(self._metadata()["topics"])


    def get_count(self, axis: str) -> int:
        return self._metadata()["counts"].get(axis, 0)


    def get_duration(self):
        duration = self._metadata()["duration"]
        if duration is None:
            raise ValueError(f"Duration is not available for {self.data_path}")
        return duration


    def get_data_path(self):
        return self.data_path
