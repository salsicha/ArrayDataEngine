import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ade.sensors.image_sensor import ImageSensor
from ade.sensors.imu_sensor import IMUSensor


def test_image_sensor_numpyify():
    # Setup mock message structure that mimics a rosbags deserialized Image message
    mock_msg = MagicMock()
    mock_msg.header.stamp.sec = 1620000000
    mock_msg.header.stamp.nanosec = 500000000
    mock_msg.__class__.__name__ = "Image"
    
    # rgb8 encoding image
    mock_msg.encoding = "rgb8"
    mock_msg.height = 4
    mock_msg.width = 4
    mock_msg.data = np.arange(4 * 4 * 3, dtype=np.uint8).tobytes()

    sensor = ImageSensor(rawdata=b"", msgtype="sensor_msgs/msg/Image")
    
    with patch.object(ImageSensor, "deserialize", return_value=mock_msg):
        npified, name, ts = sensor.numpyify()
        
        assert name == "Image"
        assert ts == 1620000000.5
        assert npified.shape == (4, 4, 3)
        assert npified.dtype == np.uint8
        assert npified[0, 0, 0] == 0
        assert npified[3, 3, 2] == 47


def test_image_sensor_numpyify_with_padded_rows():
    mock_msg = MagicMock()
    mock_msg.header.stamp.sec = 1
    mock_msg.header.stamp.nanosec = 500000000
    mock_msg.__class__.__name__ = "Image"
    mock_msg.encoding = "mono8"
    mock_msg.height = 2
    mock_msg.width = 2
    mock_msg.step = 4
    mock_msg.is_bigendian = False
    mock_msg.data = bytes([1, 2, 99, 99, 3, 4, 99, 99])

    sensor = ImageSensor(rawdata=b"", msgtype="sensor_msgs/msg/Image")

    with patch.object(ImageSensor, "deserialize", return_value=mock_msg):
        npified, name, ts = sensor.numpyify()

        assert name == "Image"
        assert ts == 1.5
        assert np.array_equal(npified, np.array([[1, 2], [3, 4]], dtype=np.uint8))


def test_imu_sensor_numpyify():
    mock_msg = MagicMock()
    mock_msg.header.stamp.sec = 1620000000
    mock_msg.header.stamp.nanosec = 100000000
    mock_msg.__class__.__name__ = "Imu"
    
    mock_msg.orientation.x = 1.0
    mock_msg.orientation.y = 2.0
    mock_msg.orientation.z = 3.0
    mock_msg.orientation.w = 4.0
    
    mock_msg.orientation_covariance = [0.1] * 9
    
    mock_msg.angular_velocity.x = 0.5
    mock_msg.angular_velocity.y = 0.6
    mock_msg.angular_velocity.z = 0.7
    mock_msg.angular_velocity_covariance = [0.2] * 9
    
    mock_msg.linear_acceleration.x = 9.8
    mock_msg.linear_acceleration.y = 0.1
    mock_msg.linear_acceleration.z = -0.1
    mock_msg.linear_acceleration_covariance = [0.3] * 9

    sensor = IMUSensor(rawdata=b"", msgtype="sensor_msgs/msg/Imu")
    
    with patch.object(IMUSensor, "deserialize", return_value=mock_msg):
        npified, name, ts = sensor.numpyify()
        
        assert name == "Imu"
        assert ts == 1620000000.1
        assert npified.shape == (6, 4)
        
        # Verify specific rows mapping
        # Orientation: x, y, z, w
        assert np.allclose(npified[0], np.array([1.0, 2.0, 3.0, 4.0]))
        # Angular velocity: x, y, z, 0
        assert np.allclose(npified[2], np.array([0.5, 0.6, 0.7, 0.0]))
        # Linear acceleration: x, y, z, 0
        assert np.allclose(npified[4], np.array([9.8, 0.1, -0.1, 0.0]))
