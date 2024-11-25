from __future__ import annotations

from rosbags.highlevel import AnyReader
from pathlib import Path
import os
from .base_source import BaseSource
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor
from ..sensors.imu_sensor import IMUSensor
from ..sensors.odom_sensor import OdomSensor
from ..sensors.nav_sensor import NavSensor


class DB3Source(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path: str):
        """Constructor

        """
        super().__init__(data_path)

        self.data_path = os.path.dirname(data_path)


    def data_exists(self) -> bool:
        if os.path.isdir(self.data_path):
            return True
        return False


    def get_count(self, axis: str) -> int:
        count = 0
        with AnyReader([Path(self.data_path)]) as reader:
            for connection in reader.connections:
                if connection.topic == axis:
                    count = count + connection.msgcount
        return count


    def messages(self, source) -> dict | None:
        '''Messages from data source
        Yields dictionary:
        - "data": numpy array
        - "timestamp"
        - "topic"
        - "name"
        '''

        with AnyReader([Path(source.get_data_path())]) as reader:
            for connection, timestamp, rawdata in reader.messages():

                type_name = connection.msgtype.rsplit('/', 1)[1]

                ### TODO:
                # type_name: imuptp
                # type: novatel_oem7_msgs/msg/ImuPTP
                # topic: /novatel/oem7/imu/data_raw
                # ...
                # type_name: a429rxparray
                # type: altadt_msgs/msg/A429RXPArray
                # topic: /altadt/raw_data
                # ...
                ### plus many other INS topics...

                # PointCloud2 msgs don't need to be converted to native ROS format
                if type_name.lower() == "pointcloud2":
                    sensor = PointCloudSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "image":
                    sensor = ImageSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "imu":
                    sensor = IMUSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "odometry":
                    sensor = OdomSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "navsatfix":
                    sensor = NavSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "vector3stamped":
                    continue
                    # sensor = Vec3Sensor(rawdata, connection.msgtype)
                else:
                    if self._debug:
                        print(f"Message type not supported: {type_name}")
                    continue

                npified, class_name, ts = sensor.numpyify()

                try:
                    # Yield: numpy array of data, timestamp, msg topic, class name
                    yield {"data": npified, \
                            "timestamp": ts, \
                            "topic": connection.topic, \
                            "name": class_name}
                except StopIteration:
                    print("End of source")
                    return

