from __future__ import annotations

from pathlib import Path

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

    def reader(self):
        global AnyReader
        if AnyReader is None:
            from rosbags.highlevel import AnyReader as Reader

            AnyReader = Reader
        return AnyReader([Path(self.data_path)])

    def get_topics(self):
        topics = []
        with self.reader() as reader:
            for conn in reader.connections:
                topics.append(conn.topic)
        return topics


    def get_duration(self):
        with self.reader() as reader:
            bag_duration = (reader.end_time - reader.start_time) * 1e-9
        return bag_duration


    def get_data_path(self):
        return self.data_path
    
