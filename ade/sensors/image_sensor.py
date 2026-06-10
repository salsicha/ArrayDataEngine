from __future__ import annotations

import numpy as np
from .base_sensor import BaseSensor


class ImageSensor(BaseSensor):
    """Image Sensor Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, rawdata, msgtype):
        """Constructor

        """
        super().__init__(rawdata, msgtype)


    def numpyify(self) -> tuple:
        msg = self.deserialize()
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        ts = sec + nanosec * 1e-9

        encoding = msg.encoding
        
        # In rosbags, msg.data might be bytes, memoryview, or numpy array.
        # np.frombuffer handles any buffer-like object efficiently.
        if encoding in ('mono16', '16UC1'):
            data = np.frombuffer(msg.data, dtype=np.uint16)
            npified = data.reshape((msg.height, msg.width))
        elif encoding in ('rgb8', 'bgr8', 'rgb', 'bgr'):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            npified = data.reshape((msg.height, msg.width, 3))
        elif encoding in ('rgba8', 'bgra8'):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            npified = data.reshape((msg.height, msg.width, 4))
        elif encoding in ('mono8', '8UC1'):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            npified = data.reshape((msg.height, msg.width))
        else:
            # Fallback
            data = np.frombuffer(msg.data, dtype=np.uint8)
            npified = data

        return npified, msg.__class__.__name__, ts