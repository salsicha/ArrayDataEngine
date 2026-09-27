"""Regression tests for source/sensor read-path fixes."""

import logging
import os
import shutil
import sqlite3
import struct
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arraydataengine.source import DataSources
from arraydataengine.sources.cdr import CDRReader, decode_pose_stamped, pointcloud_xyz
from arraydataengine.sources.dem_source import DEMSource, _rebuild_earthdata_auth
from arraydataengine.sensors.image_sensor import ImageSensor

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_BAG = REPO / "example" / "mapeverything_0.bag"
LOG_NS = 1_700_000_000_123_456_789

CALIBRATION_MSG = """
std_msgs/Header header
uint32 schema_version
string source
string relative_pointcloud_topic
string overlay_mesh_source
string frame_id
uint32 relative_depth_width
uint32 relative_depth_height
uint32 image_width
uint32 image_height
float64 scale
float64 offset
string equation
string relative_depth_units
string metric_depth_units
string calibration_source
string metadata_json
"""
CALIBRATION_TYPE = "mapeverything_msgs/msg/DepthAnythingCalibration"


# -- helpers ------------------------------------------------------------------


def _typestore(ros1=False):
    pytest.importorskip("rosbags")
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    store = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_JAZZY)
    store.register(get_types_from_msg(CALIBRATION_MSG, CALIBRATION_TYPE))
    return store


def _header(store, sec, nanosec, frame, ros1=False):
    T = store.types
    stamp = T["builtin_interfaces/msg/Time"](sec=sec, nanosec=nanosec)
    if ros1:
        return T["std_msgs/msg/Header"](seq=0, stamp=stamp, frame_id=frame)
    return T["std_msgs/msg/Header"](stamp=stamp, frame_id=frame)


def _pose(store, sec, ros1=False):
    T = store.types
    return T["geometry_msgs/msg/PoseStamped"](
        header=_header(store, sec, 0, "map", ros1),
        pose=T["geometry_msgs/msg/Pose"](
            position=T["geometry_msgs/msg/Point"](x=1.0, y=2.0, z=3.0),
            orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def _cloud(store, xyz, dtype="<f4", datatype=7, ros1=False):
    T = store.types
    xyz = np.asarray(xyz, dtype=dtype)
    item = xyz.dtype.itemsize
    fields = [
        T["sensor_msgs/msg/PointField"](name=name, offset=i * item, datatype=datatype, count=1)
        for i, name in enumerate("xyz")
    ]
    return T["sensor_msgs/msg/PointCloud2"](
        header=_header(store, 100, 5, "lidar", ros1), height=1, width=xyz.shape[0], fields=fields,
        is_bigendian=False, point_step=3 * item, row_step=3 * item * xyz.shape[0],
        data=np.frombuffer(xyz.tobytes(), dtype=np.uint8), is_dense=False,
    )


def _calibration(store, ros1=False):
    return store.types[CALIBRATION_TYPE](
        header=_header(store, 50, 0, "camera_link", ros1), schema_version=1, source="da",
        relative_pointcloud_topic="/rel", overlay_mesh_source="mesh", frame_id="camera_optical",
        relative_depth_width=518, relative_depth_height=392, image_width=1920, image_height=1440,
        scale=2.0, offset=0.5, equation="m = s * r + o", relative_depth_units="rel",
        metric_depth_units="m", calibration_source="lidar", metadata_json="{}",
    )


def _edge_messages(store):
    T = store.types
    zero_pose = _pose(store, 0)
    rng = np.random.default_rng(0)
    return [
        ("/pose_zero_stamp", zero_pose, None),
        ("/pose_nohdr", zero_pose.pose, None),
        ("/cloud_nan", _cloud(store, [[1, 2, 3], [np.nan, 0, 0], [4, 5, 6]]), None),
        ("/cloud_big", _cloud(store, rng.uniform(-5, 5, size=(40, 3))), None),
        ("/cloud_f64", _cloud(store, [[4_500_000.123, 600_000.456, 12.3]], "<f8", 8), None),
        ("/cloud_bad", _cloud(store, [[1, 2, 3]]), b"\x00\x01\x00\x00\x01"),
        ("/calib", _calibration(store), None),
        ("/pose_after", _pose(store, 7), None),
    ]


def _write_rosbag2(path, store, messages, storage_plugin=None):
    from rosbags.rosbag2 import Writer

    kwargs = {} if storage_plugin is None else {"storage_plugin": storage_plugin}
    with Writer(path, version=9, **kwargs) as writer:
        connections = {}
        for index, (topic, msg, override) in enumerate(messages):
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msg.__msgtype__, typestore=store)
            raw = override if override is not None else store.serialize_cdr(msg, msg.__msgtype__)
            writer.write(connections[topic], LOG_NS + index, raw)
    return path


def _sqlite_bag(path, rows, fmt="cdr"):
    """Minimal rosbag2 SQLite storage: rows are (topic, type, log_ns, payload)."""

    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, "
            "serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, "
            "timestamp INTEGER NOT NULL, data BLOB NOT NULL)"
        )
        topics = {}
        for topic, msgtype, log_ns, payload in rows:
            if topic not in topics:
                topics[topic] = len(topics) + 1
                connection.execute(
                    "INSERT INTO topics VALUES (?, ?, ?, ?, '')", (topics[topic], topic, msgtype, fmt)
                )
            connection.execute(
                "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
                (topics[topic], log_ns, payload),
            )
    connection.close()
    return path


def _comparable(messages):
    out = []
    for message in messages:
        row = {key: value for key, value in message.items() if key != "data"}
        row["data"] = (message["data"].shape, str(message["data"].dtype), message["data"].tolist())
        out.append(row)
    return out


# -- #1 / #3 / #4 / #7: one policy on every read path ------------------------------


def test_example_bag_databuffer_on_lidar_axis():
    if not EXAMPLE_BAG.exists():
        pytest.skip("example bag not available")
    from arraydataengine.buffer import DataBuffer

    buffer = DataBuffer(
        DataSources(str(EXAMPLE_BAG)), axis="/mapping/pointcloud/lidar", buffer_depth=10, preload=True
    )
    window = buffer.get_buffer()["/mapping/pointcloud/lidar"]
    assert window["data"].shape == (10, 30000, 3)
    assert window["data"].dtype == np.float32


def test_rosbags_and_sqlite_paths_yield_identical_messages(tmp_path, caplog):
    store = _typestore()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, _edge_messages(store))
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    shutil.copy(next(bag_dir.glob("*.db3")), standalone / "edge.db3")

    with caplog.at_level(logging.WARNING):
        via_rosbags = list(DataSources(str(bag_dir), max_points=20).get_message())
        via_sqlite = list(DataSources(str(standalone / "edge.db3"), max_points=20).get_message())

    assert _comparable(via_rosbags) == _comparable(via_sqlite)
    by_topic = {message["topic"]: message for message in via_rosbags}

    # Oversized and malformed clouds are skipped; later messages still arrive.
    assert "/cloud_big" not in by_topic and "/cloud_bad" not in by_topic
    assert "/pose_after" in by_topic
    assert "exceeds max_points=20" in caplog.text
    # Short type names and log-time fallback for zero / missing stamps.
    assert by_topic["/pose_zero_stamp"]["name"] == "PoseStamped"
    assert by_topic["/pose_nohdr"]["name"] == "Pose"
    assert by_topic["/pose_zero_stamp"]["timestamp"] == pytest.approx(LOG_NS * 1e-9)
    assert by_topic["/pose_nohdr"]["timestamp"] == pytest.approx((LOG_NS + 1) * 1e-9)
    # Clouds: NaN rows dropped, zero padded to max_points, valid count reported.
    cloud = by_topic["/cloud_nan"]
    assert cloud["data"].shape == (20, 3) and cloud["point_count"] == 2
    np.testing.assert_array_equal(cloud["data"][:2], [[1, 2, 3], [4, 5, 6]])
    assert not cloud["data"][2:].any()
    # float64 xyz keeps its precision.
    f64 = by_topic["/cloud_f64"]["data"]
    assert f64.dtype == np.float64
    assert f64[0].tolist() == [4_500_000.123, 600_000.456, 12.3]
    # Calibration decodes on both paths; the body frame_id cannot clobber the header's.
    calibration = by_topic["/calib"]
    assert calibration["name"] == "DepthAnythingCalibration"
    assert calibration["frame_id"] == "camera_link"
    assert calibration["calibration_frame_id"] == "camera_optical"
    assert calibration["scale"] == 2.0 and calibration["offset"] == 0.5
    np.testing.assert_array_equal(calibration["data"], [2.0, 0.5, 518, 392, 1920, 1440])


def test_max_points_is_configurable_through_sources(tmp_path):
    store = _typestore()
    cloud = np.random.default_rng(1).uniform(-5, 5, size=(40, 3))
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/points", _cloud(store, cloud), None)])

    source = DataSources(str(bag_dir), max_points=64)
    [message] = list(source.get_message())
    assert message["data"].shape == (64, 3)
    assert message["point_count"] == 40
    with pytest.raises(ValueError, match="max_points"):
        DataSources(str(bag_dir), max_points=0)


def test_ros1_bag_uses_same_names_and_calibration_decoder(tmp_path):
    store = _typestore(ros1=True)
    from rosbags.rosbag1 import Writer

    path = tmp_path / "rec.bag"
    calibration = _calibration(store, ros1=True)
    pose = _pose(store, 3, ros1=True)
    with Writer(path) as writer:
        for index, (topic, msg) in enumerate((("/calib", calibration), ("/pose", pose))):
            connection = writer.add_connection(topic, msg.__msgtype__, typestore=store)
            writer.write(connection, LOG_NS + index, store.serialize_ros1(msg, msg.__msgtype__))

    source = DataSources(str(path))
    assert source.get_topics() == ["/calib", "/pose"]
    by_topic = {message["topic"]: message for message in source.get_message()}
    assert by_topic["/pose"]["name"] == "PoseStamped"
    assert by_topic["/calib"]["frame_id"] == "camera_link"
    assert by_topic["/calib"]["calibration_frame_id"] == "camera_optical"
    np.testing.assert_array_equal(by_topic["/calib"]["data"], [2.0, 0.5, 518, 392, 1920, 1440])


# -- #2: rosbag2 without embedded definitions ------------------------------------


def test_rosbag2_without_message_definitions_uses_default_typestore(tmp_path):
    store = _typestore()
    bag_dir = _write_rosbag2(tmp_path / "humble", store, [("/pose", _pose(store, 5), None)])
    with sqlite3.connect(next(bag_dir.glob("*.db3"))) as connection:
        connection.execute("DELETE FROM message_definitions")
    connection.close()

    source = DataSources(str(bag_dir))
    assert source.get_topics() == ["/pose"]
    [message] = list(source.get_message())
    assert message["data"].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]


# -- #5: image encodings ----------------------------------------------------------


def _image(encoding, height, width, step, data, is_bigendian=0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id="cam"),
        encoding=encoding, height=height, width=width, step=step,
        is_bigendian=is_bigendian, data=data,
    )


def _numpyify_image(msg):
    sensor = ImageSensor(b"", "sensor_msgs/msg/Image", deserializer=lambda raw, msgtype: msg)
    return sensor.numpyify()[0]


def test_image_sensor_generic_encodings_honor_step_and_endianness():
    depth = np.arange(6, dtype="<f4").reshape(2, 3) + 0.5
    padded = b"".join(row.tobytes() + b"\xee" * 4 for row in depth)
    out = _numpyify_image(_image("32FC1", 2, 3, 16, padded))
    assert out.dtype == np.float32 and out.shape == (2, 3)
    np.testing.assert_array_equal(out, depth)

    signed = np.array([[-2, 3], [400, -32768]], dtype=np.int16)
    out = _numpyify_image(_image("16SC1", 2, 2, 4, signed.astype(">i2").tobytes(), is_bigendian=1))
    assert out.dtype == np.int16 and out.dtype.isnative
    np.testing.assert_array_equal(out, signed)

    doubles = np.array([[1.25, -2.5]], dtype="<f8")
    out = _numpyify_image(_image("64FC1", 1, 2, 16, doubles.tobytes()))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, doubles)

    assert _numpyify_image(_image("8UC3", 2, 2, 6, bytes(12))).shape == (2, 2, 3)
    assert _numpyify_image(_image("bayer_rggb8", 2, 4, 4, bytes(8))).shape == (2, 4)
    assert _numpyify_image(_image("yuv422", 2, 2, 4, bytes(8))).shape == (2, 2, 2)
    with pytest.raises(ValueError, match="Unsupported image encoding"):
        _numpyify_image(_image("nv21", 2, 2, 2, bytes(6)))


def test_image_topic_through_bag(tmp_path):
    store = _typestore()
    depth = np.arange(6, dtype="<f4").reshape(2, 3)
    image = store.types["sensor_msgs/msg/Image"](
        header=_header(store, 100, 7, "cam"), height=2, width=3, encoding="32FC1", is_bigendian=0,
        step=12, data=np.frombuffer(depth.tobytes(), dtype=np.uint8),
    )
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/depth", image, None)])
    [message] = list(DataSources(str(bag_dir)).get_message())
    np.testing.assert_array_equal(message["data"], depth)


# -- #6: topics/counts match what is yielded -----------------------------------


def test_counts_match_yielded_messages_for_example_bag():
    if not EXAMPLE_BAG.exists():
        pytest.skip("example bag not available")
    pytest.importorskip("cv2")
    source = DataSources(str(EXAMPLE_BAG))
    yielded = {}
    for message in source.get_message():
        yielded[message["topic"]] = yielded.get(message["topic"], 0) + 1
    assert {topic: source.get_count(topic) for topic in source.get_topics()} == yielded
    assert "/mapping/camera/camera_info" not in source.get_topics()
    assert source.get_topic_types()["/mapping/camera/camera_info"] == "sensor_msgs/msg/CameraInfo"


def test_unsupported_json_and_cdr_topics_are_not_counted(tmp_path):
    import json

    imu = json.dumps({"msg": {"header": {"stamp": {"sec": 1, "nanosec": 0}}}}).encode()
    pose = json.dumps({"msg": {"header": {"stamp": {"sec": 2, "nanosec": 0}}, "pose": {}}}).encode()
    path = _sqlite_bag(
        tmp_path / "json.db3",
        [("/imu", "sensor_msgs/msg/Imu", 1, imu), ("/pose", "geometry_msgs/msg/PoseStamped", 2, pose)],
        fmt="rosbridge_json",
    )
    source = DataSources(str(path))
    assert source.get_topics() == ["/pose"]
    assert [message["topic"] for message in source.get_message()] == ["/pose"]


def test_compressed_image_is_decoded_and_counted(tmp_path):
    cv2 = pytest.importorskip("cv2")
    store = _typestore()
    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    ok, png = cv2.imencode(".png", pixels)
    assert ok
    msg = store.types["sensor_msgs/msg/CompressedImage"](
        header=_header(store, 3, 0, "cam"), format="png", data=png.reshape(-1)
    )
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/image/compressed", msg, None)])
    source = DataSources(str(bag_dir))
    assert source.get_count("/image/compressed") == 1
    [message] = list(source.get_message())
    assert message["name"] == "CompressedImage"
    np.testing.assert_array_equal(message["data"], pixels)


# -- #8: storage detection ----------------------------------------------------------


def _cdr_pose(sec):
    store = _typestore()
    return store.serialize_cdr(_pose(store, sec), "geometry_msgs/msg/PoseStamped")


def test_sqlite_bag_copied_alone_into_directory(tmp_path):
    path = _sqlite_bag(tmp_path / "rec.bag", [("/pose", "geometry_msgs/msg/PoseStamped", 1, _cdr_pose(4))])
    [message] = list(DataSources(str(path)).get_message())
    assert message["timestamp"] == 4.0


def test_missing_chunk_path_does_not_read_neighbouring_bag(tmp_path):
    store = _typestore()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store, 5), None)])
    with pytest.raises(FileNotFoundError):
        DataSources(str(bag_dir / "typo.db3"))


def test_directory_of_db3_without_metadata_reads_in_natural_order(tmp_path):
    for index in (10, 2, 1):
        _sqlite_bag(tmp_path / f"rec_{index}.db3", [("/pose", "geometry_msgs/msg/PoseStamped", index, _cdr_pose(index))])
    source = DataSources(str(tmp_path))
    assert source.get_count("/pose") == 3
    assert [message["timestamp"] for message in source.get_message()] == [1.0, 2.0, 10.0]


def test_bare_mcap_without_metadata(tmp_path):
    store = _typestore()
    from rosbags.rosbag2 import StoragePlugin

    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store, 6), None)], StoragePlugin.MCAP)
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(next(bag_dir.glob("*.mcap")), alone / "rec.mcap")
    source = DataSources(str(alone / "rec.mcap"))
    assert source.get_topics() == ["/pose"]
    assert [message["timestamp"] for message in source.get_message()] == [6.0]


def test_metadata_yaml_double_quoted_chunk_path_selects_directory(tmp_path):
    from arraydataengine.sources.db3_source import DB3Source, metadata_file_names

    (tmp_path / "metadata.yaml").write_text(
        'rosbag2_bagfile_information:\n  relative_file_paths:\n    - "rec 0.db3"\n'
    )
    assert metadata_file_names(tmp_path / "metadata.yaml") == ["rec 0.db3"]
    chunk = _sqlite_bag(tmp_path / "rec 0.db3", [("/pose", "geometry_msgs/msg/PoseStamped", 1, _cdr_pose(1))])
    assert DB3Source(str(chunk))._storage_units() == [("rosbag2", tmp_path)]


# -- #9 / #13: DEM ---------------------------------------------------------------------


def _hgt_zip(name, side=3):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{name}.hgt", np.arange(side * side, dtype=">i2").tobytes())
    return buffer.getvalue()


class _Response:
    def __init__(self, url, content, status=200):
        self.url = url
        self.content = content
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _RecordingSession:
    instances = []

    def __init__(self):
        self.auth = None
        self.calls = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(url, _hgt_zip(Path(url).name.split(".")[0]))

    request = get


def test_dem_downloads_once_over_https_without_resending_credentials(monkeypatch, tmp_path):
    _RecordingSession.instances = []
    monkeypatch.setenv("earthdata_username", "user")
    monkeypatch.setenv("earthdata_password", "secret")
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=_RecordingSession))

    source = DEMSource([37, 38], [122, 123], cache_dir=tmp_path)
    [message] = list(source.messages())
    [session] = _RecordingSession.instances
    assert len(session.calls) == 1  # no second fetch of the redirected URL
    assert "auth" not in session.calls[0][1]  # credentials only via session auth policy
    assert session.calls[0][0].startswith("https://")
    assert message["name"] == "N37W122"
    assert message["data"].dtype == np.int16 and message["data"].dtype.isnative
    assert message["data"].flags.writeable
    assert sorted(os.listdir(tmp_path)) == ["N37W122.hgt"]  # atomic write leaves no temp file

    with pytest.raises(ValueError, match="non-HTTPS"):
        list(DEMSource([37, 38], [122, 123], base_url="http://mirror.example/SRTM").messages())


def test_dem_redirects_only_keep_credentials_for_earthdata_login():
    def redirect(original, target):
        prepared = SimpleNamespace(url=target, headers={"Authorization": "Basic x"})
        _rebuild_earthdata_auth(prepared, SimpleNamespace(request=SimpleNamespace(url=original)))
        return "Authorization" in prepared.headers

    data = "https://e4ftl01.cr.usgs.gov/tile.zip"
    login = "https://urs.earthdata.nasa.gov/oauth/authorize"
    assert redirect(data, login)
    assert redirect(login, data)
    assert not redirect(data, "https://cdn.thirdparty.example/tile.zip")
    assert not redirect(data, "http://urs.earthdata.nasa.gov/oauth/authorize")


def test_dem_tile_names_bounds_and_topic_counts():
    assert DEMSource.tile_name(37, 122) == "N37W122"
    assert DEMSource.tile_name(-3, -5) == "S03E005"
    assert DEMSource.tile_name(0, 0) == "N00E000"
    reversed_bounds = DEMSource([38, 37], [123, 122])
    reversed_bounds._read_cached_hgt = lambda name: np.zeros(4, dtype=">i2").tobytes()
    assert reversed_bounds.get_count() == len(list(reversed_bounds.messages())) == 1
    assert reversed_bounds.get_count("images") == 1
    assert reversed_bounds.get_count("/unknown") == 0


# -- #10 / #11 / #15: decoder details ---------------------------------------------------


def test_json_ros1_stamps_and_leading_whitespace(tmp_path):
    import json

    ros1 = json.dumps({"msg": {"header": {"stamp": {"secs": 10, "nsecs": 500000000}, "frame_id": "map"}}}).encode()
    pretty = b"\n  " + json.dumps({"msg": {"header": {"stamp": {"sec": 11, "nanosec": 0}}}}).encode()
    path = _sqlite_bag(
        tmp_path / "json.db3",
        [("/a", "geometry_msgs/msg/PoseStamped", 1, ros1), ("/b", "geometry_msgs/msg/PoseStamped", 2, pretty)],
        fmt="rosbridge_json",
    )
    assert [message["timestamp"] for message in DataSources(str(path)).get_message()] == [10.5, 11.0]


def test_cdr_rejects_unsupported_encapsulations():
    body = struct.pack("<iI", 7, 9) + struct.pack("<I", 4) + b"map\x00" + struct.pack("<7d", *range(7))
    assert decode_pose_stamped(b"\x00\x01\x00\x00" + body).timestamp == pytest.approx(7 + 9e-9)
    for header, name in ((b"\x00\x07\x00\x00", "CDR2_LE"), (b"\x00\x03\x00\x00", "PL_CDR_LE")):
        with pytest.raises(ValueError, match=name):
            CDRReader(header + body)


def test_pointcloud_xyz_validates_layout():
    fields = [{"name": name, "offset": 4 * i, "datatype": 7} for i, name in enumerate("xyz")]
    payload = np.arange(9, dtype="<f4").tobytes()
    with pytest.raises(ValueError, match="point_step"):
        pointcloud_xyz(payload, fields, 1, 3, 0, 0, False)
    with pytest.raises(ValueError, match="does not fit"):
        pointcloud_xyz(payload, fields, 1, 3, 8, 24, False)
    with pytest.raises(ValueError, match="expected at least"):
        pointcloud_xyz(payload[:20], fields, 1, 3, 12, 36, False)
    np.testing.assert_array_equal(pointcloud_xyz(payload, fields, 1, 3, 12, 36, False), np.arange(9).reshape(3, 3))


# -- #12: metadata cache -------------------------------------------------------------


def test_standalone_db3_metadata_is_cached(monkeypatch, tmp_path):
    from arraydataengine.sources import db3_source

    path = _sqlite_bag(tmp_path / "rec.db3", [("/pose", "geometry_msgs/msg/PoseStamped", 1, _cdr_pose(1))])
    opened = []
    real_connect = db3_source.connect_readonly
    monkeypatch.setattr(db3_source, "connect_readonly", lambda p: opened.append(p) or real_connect(p))

    source = DataSources(str(path))
    for topic in source.get_topics():
        source.get_count(topic)
    source.get_count("/pose")
    assert len(opened) == 1

    connection = real_connect(path)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM messages")
    connection.close()


# -- #14: image extensions ---------------------------------------------------------------


def test_image_extensions_are_case_insensitive_and_accept_tif(tmp_path):
    cv2 = pytest.importorskip("cv2")
    cv2.imwrite(str(tmp_path / "a.tif"), np.zeros((2, 2), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "b.png"), np.zeros((2, 2), dtype=np.uint8))
    os.rename(tmp_path / "b.png", tmp_path / "B.PNG")

    assert DataSources(str(tmp_path / "*.tif")).get_count("images") == 1
    upper = DataSources(str(tmp_path / "*.PNG"))
    assert upper.get_count("images") == 1
    assert upper.get_count("/other") == 0
