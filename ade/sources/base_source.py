from __future__ import annotations

from pathlib import Path
from numbers import Real

AnyReader = None


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

    def reader(self):
        global AnyReader
        if AnyReader is None:
            from rosbags.highlevel import AnyReader as Reader

            AnyReader = Reader
        return AnyReader([Path(self.data_path)])

    def clear_metadata_cache(self) -> None:
        self._metadata_cache = None

    def _metadata(self):
        if self._metadata_cache is not None:
            return self._metadata_cache

        topics = []
        counts = {}
        duration = None

        with self.reader() as reader:
            for connection in reader.connections:
                topic = connection.topic
                if topic not in counts:
                    topics.append(topic)
                    counts[topic] = 0
                counts[topic] += connection.msgcount

            start = getattr(reader, "start_time", None)
            end = getattr(reader, "end_time", None)
            if isinstance(start, Real) and isinstance(end, Real):
                duration = (end - start) * 1e-9

        self._metadata_cache = {
            "topics": topics,
            "counts": counts,
            "duration": duration,
        }
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
    
