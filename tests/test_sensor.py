import pytest
import sys
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from ade.sensors.image_sensor import ImageSensor
from ade.sensors.imu_sensor import IMUSensor
from ade.sensors.pointcloud2_sensor import PointCloudSensor


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


def test_image_sensor_numpyify_mono16_endianness():
    values = np.array([[1, 256], [1025, 65535]], dtype=np.uint16)

    for is_bigendian, raw in [
        (False, values.astype("<u2").tobytes()),
        (True, values.astype(">u2").tobytes()),
    ]:
        mock_msg = MagicMock()
        mock_msg.header.stamp.sec = 2
        mock_msg.header.stamp.nanosec = 0
        mock_msg.__class__.__name__ = "Image"
        mock_msg.encoding = "mono16"
        mock_msg.height = 2
        mock_msg.width = 2
        mock_msg.step = 4
        mock_msg.is_bigendian = is_bigendian
        mock_msg.data = raw

        sensor = ImageSensor(rawdata=b"", msgtype="sensor_msgs/msg/Image")
        with patch.object(ImageSensor, "deserialize", return_value=mock_msg):
            npified, name, ts = sensor.numpyify()

        assert name == "Image"
        assert ts == 2.0
        assert npified.dtype == np.uint16
        assert np.array_equal(npified, values)


def test_image_sensor_raises_for_invalid_step_and_short_data():
    sensor = ImageSensor(rawdata=b"", msgtype="sensor_msgs/msg/Image")

    small_step = MagicMock()
    small_step.header.stamp.sec = 0
    small_step.header.stamp.nanosec = 0
    small_step.encoding = "rgb8"
    small_step.height = 1
    small_step.width = 2
    small_step.step = 5
    small_step.is_bigendian = False
    small_step.data = bytes(6)

    with patch.object(ImageSensor, "deserialize", return_value=small_step):
        with pytest.raises(ValueError, match="step"):
            sensor.numpyify()

    short_data = MagicMock()
    short_data.header.stamp.sec = 0
    short_data.header.stamp.nanosec = 0
    short_data.encoding = "rgb8"
    short_data.height = 2
    short_data.width = 2
    short_data.step = 6
    short_data.is_bigendian = False
    short_data.data = bytes(6)

    with patch.object(ImageSensor, "deserialize", return_value=short_data):
        with pytest.raises(ValueError, match="expected at least"):
            sensor.numpyify()


def test_pointcloud_sensor_pads_points_and_rejects_oversized_clouds(monkeypatch):
    msg = MagicMock()
    msg.header.stamp.sec = 3
    msg.header.stamp.nanosec = 500000000
    msg.__class__.__name__ = "PointCloud2"
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    fake_rnp = SimpleNamespace(
        point_cloud2=SimpleNamespace(point_cloud2_to_array=lambda msg: {"xyz": xyz})
    )
    monkeypatch.setitem(sys.modules, "ros2_numpy", fake_rnp)

    sensor = PointCloudSensor(rawdata=b"", msgtype="sensor_msgs/msg/PointCloud2", max_points=4)
    with patch.object(PointCloudSensor, "deserialize", return_value=msg):
        npified, name, ts = sensor.numpyify()

    assert name == "PointCloud2"
    assert ts == 3.5
    assert npified.shape == (4, 3)
    assert np.allclose(npified[:2], xyz)
    assert np.allclose(npified[2:], np.zeros((2, 3), dtype=np.float32))

    oversized = PointCloudSensor(rawdata=b"", msgtype="sensor_msgs/msg/PointCloud2", max_points=1)
    with patch.object(PointCloudSensor, "deserialize", return_value=msg):
        with pytest.raises(ValueError, match="exceeds max_points"):
            oversized.numpyify()


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
