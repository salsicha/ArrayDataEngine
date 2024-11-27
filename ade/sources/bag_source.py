from __future__ import annotations

from rosbags.highlevel import AnyReader

from pathlib import Path

from .base_source import BaseSource
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor


class BagSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path: str):
        """Constructor

        """
        super().__init__(data_path)

        self.data_path = data_path


    def messages(self, source):
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

                # PointCloud2 msgs don't need to be converted to native ROS format
                if type_name.lower() == "pointcloud2":
                    sensor = PointCloudSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "image":
                    sensor = ImageSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "imu":
                    continue
                    # sensor = ImuSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "navsatfix":
                    continue
                    # sensor = NavSatSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "gps":
                    continue
                    # sensor = GPSSensor(rawdata, connection.msgtype)
                elif type_name.lower() == "ins":
                    continue
                    # sensor = InsSensor(rawdata, connection.msgtype)
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
