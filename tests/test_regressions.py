"""Regression tests for the 2026-07 code-review fixes."""

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ade.buffer import DataBuffer
from ade.sensors.image_sensor import ImageSensor
from ade.sensors.imu_sensor import IMUSensor
from ade.sensors.odom_sensor import OdomSensor
from ade.sources.cdr import CDRReader, decode_pointcloud2
from ade.ops import (
    apply_transform,
    augment_point_cloud,
    augment_trajectory,
    hillshade,
    image_pyramid,
    mosaic_dem_tiles,
    navsat_to_enu,
    navsat_to_trajectory,
    normalize_image,
    pose_to_matrix,
    poses_to_matrices,
    slope_aspect,
    statistical_outlier_filter,
    terrain_normals,
    topic_pipeline,
    topic_view,
    FrameGraph,
)


class StreamSource:
    def __init__(self, count=5):
        self.count = count

    def get_topics(self):
        return ["t"]

    def get_count(self, topic):
        return self.count

    def get_message(self):
        for i in range(self.count):
            yield {
                "topic": "t",
                "timestamp": 100.0 + i,
                "name": b"n",
                "data": np.array([float(i)]),
            }


# --- sensors -----------------------------------------------------------------


def test_imu_sensor_extracts_covariance_diagonal():
    msg = MagicMock()
    msg.header.stamp.sec = 1
    msg.header.stamp.nanosec = 0
    msg.__class__.__name__ = "Imu"
    msg.orientation.x = msg.orientation.y = msg.orientation.z = 0.0
    msg.orientation.w = 1.0
    msg.orientation_covariance = list(range(9))
    msg.angular_velocity.x = msg.angular_velocity.y = msg.angular_velocity.z = 0.0
    msg.angular_velocity_covariance = list(range(10, 19))
    msg.linear_acceleration.x = msg.linear_acceleration.y = msg.linear_acceleration.z = 0.0
    msg.linear_acceleration_covariance = list(range(20, 29))

    sensor = IMUSensor(rawdata=b"", msgtype="sensor_msgs/msg/Imu")
    with patch.object(IMUSensor, "deserialize", return_value=msg):
        npified, _, _ = sensor.numpyify()

    # 3x3 row-major diagonals are indices 0, 4, 8
    assert np.allclose(npified[1, :3], [0, 4, 8])
    assert np.allclose(npified[3, :3], [10, 14, 18])
    assert np.allclose(npified[5, :3], [20, 24, 28])


def test_odom_sensor_extracts_covariance_diagonal():
    msg = MagicMock()
    msg.header.stamp.sec = 1
    msg.header.stamp.nanosec = 0
    msg.__class__.__name__ = "Odometry"
    pose = msg.pose.pose
    pose.position.x = pose.position.y = pose.position.z = 0.0
    pose.orientation.x = pose.orientation.y = pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    msg.pose.covariance = list(range(36))
    twist = msg.twist.twist
    twist.linear.x = twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = twist.angular.z = 0.0
    msg.twist.covariance = list(range(100, 136))

    sensor = OdomSensor(rawdata=b"", msgtype="nav_msgs/msg/Odometry")
    with patch.object(OdomSensor, "deserialize", return_value=msg):
        npified, _, _ = sensor.numpyify()

    # 6x6 row-major diagonals are 0, 7, 14 (position) and 21, 28, 35 (rotation)
    assert np.allclose(npified[1, :3], [0, 7, 14])
    assert np.allclose(npified[3, :3], [21, 28, 35])
    assert np.allclose(npified[5, :3], [100, 107, 114])
    assert np.allclose(npified[7, :3], [121, 128, 135])


def test_image_sensor_honors_integer_bigendian_flag():
    values = np.array([[1, 256], [1025, 65535]], dtype=np.uint16)
    msg = MagicMock()
    msg.header.stamp.sec = 2
    msg.header.stamp.nanosec = 0
    msg.__class__.__name__ = "Image"
    msg.encoding = "mono16"
    msg.height = 2
    msg.width = 2
    msg.step = 4
    msg.is_bigendian = 1  # rosbags yields uint8 as int, not bool
    msg.data = values.astype(">u2").tobytes()

    sensor = ImageSensor(rawdata=b"", msgtype="sensor_msgs/msg/Image")
    with patch.object(ImageSensor, "deserialize", return_value=msg):
        npified, _, _ = sensor.numpyify()

    assert np.array_equal(npified, values)


# --- CDR decoding ------------------------------------------------------------

_F32 = 7
_XYZ_FIELDS = [("x", 0, _F32, 1), ("y", 4, _F32, 1), ("z", 8, _F32, 1)]


def _build_pointcloud2(height, width, point_step, row_step, payload, fields=None, encaps=b"\x00\x01\x00\x00"):
    fields = _XYZ_FIELDS if fields is None else fields
    buf = bytearray(encaps)

    def align(size):
        remainder = (len(buf) - 4) % size
        if remainder:
            buf.extend(b"\x00" * (size - remainder))

    def u32(value):
        align(4)
        buf.extend(struct.pack("<I", value))

    def i32(value):
        align(4)
        buf.extend(struct.pack("<i", value))

    def u8(value):
        buf.extend(struct.pack("<B", value))

    def string(value):
        raw = value.encode() + b"\x00"
        u32(len(raw))
        buf.extend(raw)

    i32(100)
    u32(500000000)
    string("map")
    u32(height)
    u32(width)
    u32(len(fields))
    for name, offset, datatype, count in fields:
        string(name)
        u32(offset)
        u8(datatype)
        u32(count)
    u8(0)  # is_bigendian
    u32(point_step)
    u32(row_step)
    u32(len(payload))
    buf.extend(payload)
    u8(1)  # is_dense follows the payload with no padding
    return bytes(buf)


def test_decode_pointcloud2_empty_cloud():
    decoded = decode_pointcloud2(_build_pointcloud2(0, 0, 16, 0, b""))
    assert decoded.data.shape == (0, 3)
    assert decoded.data.dtype == np.float32


def test_decode_pointcloud2_unaligned_point_step():
    # xyz float32 + uint8 intensity: point_step 13 leaves no trailing padding
    payload = b"".join(struct.pack("<fffB", i, i * 2.0, i * 3.0, i) for i in range(3))
    decoded = decode_pointcloud2(_build_pointcloud2(1, 3, 13, 39, payload))
    assert np.allclose(decoded.data, [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
    assert decoded.timestamp == 100.5
    assert decoded.frame_id == "map"


def test_decode_pointcloud2_respects_row_step():
    row0 = struct.pack("<ffffff", 1, 1, 1, 2, 2, 2) + b"\xAA" * 8
    row1 = struct.pack("<ffffff", 3, 3, 3, 4, 4, 4) + b"\xAA" * 8
    decoded = decode_pointcloud2(_build_pointcloud2(2, 2, 12, 32, row0 + row1))
    assert np.allclose(decoded.data, [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])


def test_cdr_reader_detects_pl_cdr_le():
    reader = CDRReader(b"\x00\x03\x00\x00" + struct.pack("<i", 5))
    assert reader.read_int32() == 5


# --- buffers -----------------------------------------------------------------


def test_numpy_buffer_get_size():
    buf = DataBuffer(StreamSource(), buffer_depth=3, axis="t", use_db=False)
    assert buf.get_size() == 1
    buf.roll_buffer("t")
    assert buf.get_size() == 2


def test_topic_view_excludes_unfilled_slots():
    buf = DataBuffer(StreamSource(), buffer_depth=4, axis="t", use_db=False, preload=2)
    view = buf.topic_view("t")
    assert len(view) == 2
    assert np.allclose(view.ts, [100.0, 101.0])
    assert np.allclose(buf.map_topic("t", lambda v: v * 2)["ts"], [100.0, 101.0])


def test_get_data_terminates_on_non_restartable_source():
    shared = iter(
        [{"topic": "t", "timestamp": 100.0, "name": b"n", "data": np.array([1.0])}]
    )
    buf = DataBuffer(lambda: shared, buffer_depth=2, topics=["t"], axis="t", use_db=False, preload=0)
    yielded = sum(1 for _ in buf.get_data("t"))
    assert yielded == 1


def test_numpy_buffer_rejects_unsupported_subscript():
    buf = DataBuffer(StreamSource(), buffer_depth=3, axis="t", use_db=False)
    with pytest.raises(TypeError):
        buf[1.5]


def test_tiledb_readonly_reopen_does_not_write(tmp_path):
    tiledb = pytest.importorskip("tiledb")
    group_uri = str(tmp_path / "grp") + "/"

    first = DataBuffer(
        StreamSource(), buffer_depth=5, data_uri=group_uri, axis="t", use_db=True, preload=0
    )
    first.roll_buffer("t")
    first.roll_buffer("t")
    first.close(closed=False)

    with DataBuffer(None, data_uri=group_uri, axis="t", use_db=True) as reopened:
        assert reopened.buffer_impl.read_only is True
        reopened.get_time_range("t", 0.0, 1e12)

    with tiledb.open(group_uri + "t__timestamps") as array:
        assert bool(array.meta["closed"]) is False


# --- geometry / point cloud --------------------------------------------------


def test_poses_to_matrices_batched():
    poses = np.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.7071068, 0.7071068],
        ]
    )
    matrices = poses_to_matrices(poses)
    assert matrices.shape == (2, 4, 4)
    assert np.allclose(matrices[0][:3, :3], np.eye(3))
    assert np.allclose(matrices[1], pose_to_matrix(poses[1]))
    # N=4 used to broadcast into garbage instead of raising
    identity = poses_to_matrices(np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (4, 1)))
    assert np.allclose(identity, np.eye(4))


def test_frame_graph_copies_caller_transform():
    transform = np.eye(4)
    graph = FrameGraph()
    graph.add_transform("a", "b", transform)
    transform[:3, 3] = [100.0, 0.0, 0.0]
    assert np.allclose(graph.lookup_transform("a", "b")[:3, 3], [0.0, 0.0, 0.0])


def test_apply_transform_integer_points():
    transform = np.eye(4)
    transform[:3, 3] = 0.5
    result = apply_transform(np.array([[1, 2, 3]], dtype=np.int64), transform)
    assert np.allclose(result, [[1.5, 2.5, 3.5]])


def test_statistical_outlier_filter_keeps_single_point():
    point = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(statistical_outlier_filter(point), point)


# --- DEM ---------------------------------------------------------------------


def test_slope_aspect_matches_terrain_normals():
    north_rising = np.tile(np.arange(5.0), (5, 1)).T
    _, aspect = slope_aspect(north_rising)
    # Downslope faces south (compass pi), like the terrain normal
    assert np.isclose(abs(aspect[2, 2]), np.pi)
    assert np.allclose(terrain_normals(north_rising)[2, 2], [0.0, -0.7071068, 0.7071068])


def test_hillshade_sun_direction():
    west_facing = np.tile(np.arange(5.0), (5, 1))
    assert np.isclose(hillshade(west_facing, azimuth=270, altitude=45)[2, 2], 1.0)
    assert np.isclose(hillshade(west_facing, azimuth=180, altitude=45)[2, 2], 0.5)
    assert np.isclose(hillshade(west_facing, azimuth=90, altitude=45)[2, 2], 0.0)


def test_mosaic_dem_tiles_fills_missing_rows():
    tile = np.ones((2, 2))
    mosaic, lats, _ = mosaic_dem_tiles({"N37W122": tile, "N35W122": tile * 3}, return_index=True)
    assert mosaic.shape == (6, 2)
    assert lats.tolist() == [37, 36, 35]
    assert np.isnan(mosaic[2:4]).all()
    assert (mosaic[4:6] == 3).all()


# --- nav / image / ml --------------------------------------------------------


def test_navsat_to_trajectory_stream_without_timestamps():
    trajectory = navsat_to_trajectory(np.array([[37.0, -122.0, 10.0], [37.001, -122.0, 12.0]]))
    assert trajectory["ts"].tolist() == [0.0, 1.0]
    assert trajectory["position"].shape == (2, 3)


def test_normalize_image_integer_min_value():
    image = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    result = normalize_image(image, min_value=200, max_value=400)
    assert np.isclose(result[0, 0], -0.5)


def test_image_pyramid_keeps_spatial_axes():
    shapes = [level.shape for level in image_pyramid(np.zeros((8, 64, 64)), levels=6)]
    assert shapes == [(8, 64, 64), (8, 32, 32), (8, 16, 16), (8, 8, 8), (8, 4, 4), (8, 2, 2)]


def test_augment_point_cloud_empty_with_dropout():
    result = augment_point_cloud(np.empty((0, 3)), dropout_ratio=0.5)
    assert result.shape == (0, 3)


def test_augment_trajectory_recomputes_navsat():
    trajectory = navsat_to_trajectory(np.array([[37.0, -122.0, 10.0], [37.001, -122.0, 12.0]]))
    augmented = augment_trajectory(trajectory, translation=[100.0, 0.0, 0.0])
    reference = trajectory["reference"]
    round_trip = navsat_to_enu(
        augmented["navsat"][:, 0],
        augmented["navsat"][:, 1],
        augmented["navsat"][:, 2],
        reference["lat"],
        reference["lon"],
        reference["alt"],
    )
    assert np.allclose(round_trip, augmented["position"], atol=1e-6)


# --- ops core pipelines -------------------------------------------------------


def _float32_topic():
    return {
        "ts": np.array([1.0, 2.0]),
        "data": np.arange(6, dtype=np.float32).reshape(2, 3),
    }


def test_empty_lazy_collect_preserves_schema():
    collected = topic_pipeline(_float32_topic()).time_range(5.0, 6.0).collect()
    assert collected["data"].shape == (0, 3)
    assert collected["data"].dtype == np.float32


def test_lazy_collect_idless_schema_matches_eager():
    lazy = topic_pipeline(_float32_topic()).collect()
    eager = topic_view(_float32_topic()).as_dict()
    assert sorted(lazy.keys()) == sorted(eager.keys())
    assert "id" not in lazy


def test_map_typeerror_not_masked():
    def bad(data, ts, name):
        raise TypeError("real user bug")

    with pytest.raises(TypeError, match="real user bug"):
        topic_pipeline(_float32_topic()).map(bad).collect()


def test_map_internal_typeerror_not_retried():
    calls = []

    def two_arg(data, ts):
        calls.append(1)
        raise TypeError("inner failure")

    with pytest.raises(TypeError, match="inner failure"):
        topic_view(_float32_topic()).map(two_arg)
    assert len(calls) == 1


def test_pipeline_frame_id_filter_applies():
    topic = {
        "ts": np.array([1.0, 2.0, 3.0]),
        "data": np.array(
            [(b"map", 1.0), (b"odom", 2.0), (b"map", 3.0)],
            dtype=[("frame_id", "S16"), ("value", "<f8")],
        ),
    }
    collected = topic_pipeline(topic).frame_id("map").collect()
    assert collected["ts"].tolist() == [1.0, 3.0]
