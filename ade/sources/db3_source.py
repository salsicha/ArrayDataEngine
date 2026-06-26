from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from .ros_source import RosSource
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor
from ..sensors.imu_sensor import IMUSensor
from ..sensors.odom_sensor import OdomSensor
from ..sensors.nav_sensor import NavSensor
from ..sensors.pose_sensor import PoseSensor


class DB3Source(RosSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """

    SENSOR_TYPES = {
        "pointcloud2": PointCloudSensor,
        "image": ImageSensor,
        "imu": IMUSensor,
        "odometry": OdomSensor,
        "navsatfix": NavSensor,
        "pose": PoseSensor,
        "posestamped": PoseSensor,
    }


    def __init__(self, data_path: str):
        """Constructor

        """
        super().__init__(data_path)
        self.input_path = data_path

        if os.path.isdir(data_path):
            self.data_path = data_path
        else:
            self.data_path = os.path.dirname(data_path) or "."

    def messages(self):
        if not self._standalone_db3_input():
            try:
                yield from super().messages()
                return
            except ModuleNotFoundError as exc:
                if exc.name != "rosbags":
                    raise

        from .cdr import decode_supported_cdr_message

        for db_path in self._db3_paths():
            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT messages.timestamp, topics.name, topics.type, messages.data
                    FROM messages
                    JOIN topics ON topics.id = messages.topic_id
                    ORDER BY messages.timestamp ASC, messages.id ASC
                    """
                )
                for fallback_timestamp, topic, msgtype, rawdata in rows:
                    decoded = decode_supported_cdr_message(bytes(rawdata), str(msgtype))
                    if decoded is None:
                        continue
                    timestamp = decoded.timestamp
                    if timestamp == 0.0:
                        timestamp = int(fallback_timestamp) * 1e-9
                    message = {
                        "data": decoded.data,
                        "timestamp": timestamp,
                        "topic": str(topic),
                        "name": decoded.name,
                    }
                    if decoded.frame_id is not None:
                        message["frame_id"] = decoded.frame_id
                    yield message


    def _metadata(self):
        if self._standalone_db3_input():
            metadata = self._sqlite_metadata()
            if metadata is not None:
                self._metadata_cache = metadata
                return metadata

        try:
            return super()._metadata()
        except ModuleNotFoundError as exc:
            if exc.name != "rosbags":
                raise
            metadata = self._sqlite_metadata()
            if metadata is None:
                raise
            self._metadata_cache = metadata
            return metadata


    def _sqlite_metadata(self):
        db_paths = self._db3_paths()
        if not db_paths:
            return None

        topics = []
        counts = {}
        start = None
        end = None
        for db_path in db_paths:
            try:
                with sqlite3.connect(db_path) as connection:
                    topic_rows = connection.execute(
                        "select id, name from topics order by id"
                    ).fetchall()
                    id_to_topic = {}
                    for topic_id, topic in topic_rows:
                        id_to_topic[int(topic_id)] = str(topic)
                        if topic not in counts:
                            topics.append(str(topic))
                            counts[str(topic)] = 0

                    for topic_id, count in connection.execute(
                        "select topic_id, count(*) from messages group by topic_id"
                    ):
                        topic = id_to_topic.get(int(topic_id))
                        if topic is not None:
                            counts[topic] += int(count)

                    min_ts, max_ts = connection.execute(
                        "select min(timestamp), max(timestamp) from messages"
                    ).fetchone()
                    if min_ts is not None:
                        start = int(min_ts) if start is None else min(start, int(min_ts))
                    if max_ts is not None:
                        end = int(max_ts) if end is None else max(end, int(max_ts))
            except sqlite3.Error:
                return None

        duration = None if start is None or end is None else (end - start) * 1e-9
        return {"topics": topics, "counts": counts, "duration": duration}


    def _standalone_db3_input(self) -> bool:
        input_path = Path(self.input_path)
        if not input_path.is_file() or input_path.suffix.lower() != ".db3":
            return False
        metadata_path = input_path.parent / "metadata.yaml"
        referenced = self._metadata_db3_paths(metadata_path)
        return input_path not in referenced


    def _db3_paths(self):
        input_path = Path(self.input_path)
        if input_path.is_file() and input_path.suffix.lower() == ".db3":
            metadata_path = input_path.parent / "metadata.yaml"
            referenced = self._metadata_db3_paths(metadata_path)
            if input_path in referenced:
                return [str(path) for path in referenced]
            return [str(input_path)]

        if os.path.isdir(self.data_path):
            metadata_paths = self._metadata_db3_paths(Path(self.data_path) / "metadata.yaml")
            if metadata_paths:
                return [str(path) for path in metadata_paths]
            return [
                os.path.join(self.data_path, name)
                for name in sorted(os.listdir(self.data_path))
                if name.lower().endswith(".db3")
            ]
        if str(self.data_path).lower().endswith(".db3"):
            return [self.data_path]
        return []


    def _metadata_db3_paths(self, metadata_path: Path):
        if not metadata_path.exists():
            return []
        paths = []
        for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith("- ") or ".db3" not in line:
                continue
            value = line[2:].strip()
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("''", "'")
            candidate = metadata_path.parent / value
            if candidate.exists():
                paths.append(candidate)
        return paths


    def data_exists(self) -> bool:
        if not os.path.isdir(self.data_path):
            return False

        if os.path.exists(os.path.join(self.data_path, "metadata.yaml")):
            return True

        return any(name.lower().endswith(".db3") for name in os.listdir(self.data_path))
