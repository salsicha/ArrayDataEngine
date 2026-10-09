"""Regression tests for the second source/sensor review (ragged clouds, RGB images, ...)."""

import io
import json
import logging
import os
import shutil
import sqlite3
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arraydataengine.source import DataSources
from arraydataengine.sources.cdr import pointcloud_xyz

LOG_NS = 1_700_000_000_000_000_000
CAL_TYPE = "mapeverything_msgs/msg/DepthAnythingCalibration"
CAL_MSG = """std_msgs/Header header
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
CAL_V2_MSG = CAL_MSG.replace("float64 scale\n", "float64 scale\nfloat64 scale_stddev\n")


# -- helpers ------------------------------------------------------------------


def _store(ros1=False, cal_msg=CAL_MSG):
    pytest.importorskip("rosbags")
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    store = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_JAZZY)
    store.register(get_types_from_msg(cal_msg, CAL_TYPE))
    return store


def _header(store, sec=1, nanosec=0, frame="f", ros1=False):
    T = store.types
    stamp = T["builtin_interfaces/msg/Time"](sec=sec, nanosec=nanosec)
    if ros1:
        return T["std_msgs/msg/Header"](seq=0, stamp=stamp, frame_id=frame)
    return T["std_msgs/msg/Header"](stamp=stamp, frame_id=frame)


def _pose(store, sec=1, ros1=False):
    T = store.types
    return T["geometry_msgs/msg/PoseStamped"](
        header=_header(store, sec, 0, "map", ros1),
        pose=T["geometry_msgs/msg/Pose"](
            position=T["geometry_msgs/msg/Point"](x=1.0, y=2.0, z=3.0),
            orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def _cloud(store, xyz, ros1=False):
    T = store.types
    xyz = np.asarray(xyz, dtype="<f4")
    fields = [
        T["sensor_msgs/msg/PointField"](name=name, offset=4 * i, datatype=7, count=1)
        for i, name in enumerate("xyz")
    ]
    return T["sensor_msgs/msg/PointCloud2"](
        header=_header(store, 2, 0, "lidar", ros1), height=1, width=xyz.shape[0], fields=fields,
        is_bigendian=False, point_step=12, row_step=12 * xyz.shape[0],
        data=np.frombuffer(xyz.tobytes(), dtype=np.uint8), is_dense=False,
    )


def _image(store, encoding, height, width, step, data, ros1=False):
    return store.types["sensor_msgs/msg/Image"](
        header=_header(store, 3, 0, "cam", ros1), height=height, width=width, encoding=encoding,
        is_bigendian=0, step=step, data=np.frombuffer(bytes(data), dtype=np.uint8),
    )


def _compressed(store, fmt, payload, ros1=False):
    return store.types["sensor_msgs/msg/CompressedImage"](
        header=_header(store, 4, 0, "cam", ros1), format=fmt, data=np.frombuffer(bytes(payload), dtype=np.uint8)
    )


def _navsat(store, status, ros1=False):
    T = store.types
    return T["sensor_msgs/msg/NavSatFix"](
        header=_header(store, 5, 0, "gps", ros1),
        status=T["sensor_msgs/msg/NavSatStatus"](status=status, service=1),
        latitude=37.5, longitude=-122.25, altitude=10.0,
        position_covariance=np.arange(9, dtype=np.float64), position_covariance_type=2,
    )


def _calibration(store, **extra):
    return store.types[CAL_TYPE](
        header=_header(store, 50, 0, "camera_link"), schema_version=extra.pop("schema_version", 1),
        source="da", relative_pointcloud_topic="/rel", overlay_mesh_source="mesh", frame_id="optical",
        relative_depth_width=518, relative_depth_height=392, image_width=1920, image_height=1440,
        scale=2.0, offset=0.5, equation="e", relative_depth_units="rel", metric_depth_units="m",
        calibration_source="lidar", metadata_json="{}", **extra,
    )


def _write_rosbag2(path, store, messages, storage_plugin=None):
    """messages: (topic, msg) or (topic, msgtype, raw_payload)."""

    from rosbags.rosbag2 import Writer

    kwargs = {} if storage_plugin is None else {"storage_plugin": storage_plugin}
    with Writer(path, version=9, **kwargs) as writer:
        connections = {}
        for index, item in enumerate(messages):
            topic = item[0]
            if len(item) == 2:
                msgtype, raw = item[1].__msgtype__, store.serialize_cdr(item[1], item[1].__msgtype__)
            else:
                msgtype, raw = item[1], item[2]
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msgtype, typestore=store)
            writer.write(connections[topic], LOG_NS + index, raw)
    return path


def _write_ros1(path, store, messages):
    from rosbags.rosbag1 import Writer

    with Writer(path) as writer:
        connections = {}
        for index, (topic, msg) in enumerate(messages):
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msg.__msgtype__, typestore=store)
            writer.write(connections[topic], LOG_NS + index, store.serialize_ros1(msg, msg.__msgtype__))
    return path


def _lone_copy(bag_dir, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(next(Path(bag_dir).glob("*.db3")), target)
    return target


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


def _by_topic(messages):
    out = {}
    for message in messages:
        out.setdefault(message["topic"], []).append(message)
    return out


def _png_header_only(width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 10)) + chunk(b"IEND", b"")


# -- RAGGED_SPEC section 1: unpadded clouds, point_count, max_points ------------


def test_point_clouds_are_unpadded_and_max_points_defaults_to_no_limit(tmp_path, caplog):
    store = _store()
    big = np.random.default_rng(0).uniform(-5, 5, size=(40, 3))
    messages = [("/small", _cloud(store, [[1, 2, 3], [np.nan, 0, 0], [4, 5, 6]])), ("/big", _cloud(store, big))]
    bag_dir = _write_rosbag2(tmp_path / "bag", store, messages)
    lone = _lone_copy(bag_dir, tmp_path / "lone" / "rec.db3")
    ros1 = _write_ros1(tmp_path / "rec.bag", _store(ros1=True), [
        (topic, _cloud(_store(ros1=True), msg_xyz, ros1=True))
        for topic, msg_xyz in (("/small", [[1, 2, 3], [np.nan, 0, 0], [4, 5, 6]]), ("/big", big))
    ])

    for path in (bag_dir, lone, ros1):
        source = DataSources(str(path))
        assert source.max_points is None and source.source.max_points is None
        by_topic = {m["topic"]: m for m in source.get_message()}
        small = by_topic["/small"]
        assert small["data"].shape == (2, 3) and small["point_count"] == 2
        np.testing.assert_array_equal(small["data"], [[1, 2, 3], [4, 5, 6]])
        assert by_topic["/big"]["data"].shape == (40, 3) and by_topic["/big"]["point_count"] == 40

        limited = DataSources(str(path), max_points=10)
        assert limited.max_points == 10 and limited.source.max_points == 10
        with caplog.at_level(logging.WARNING):
            topics = [m["topic"] for m in limited.get_message()]
        assert topics == ["/small"]  # the 40-point cloud is skipped, not truncated
        assert "exceeds max_points=10" in caplog.text
        caplog.clear()


def test_synthetic_source_clouds_are_unpadded():
    from arraydataengine.sources.synthetic_source import SyntheticSource

    source = SyntheticSource(duration=1.0)
    assert source.max_points == 120
    clouds = [m for m in source.messages() if m["topic"] == "/points"]
    assert clouds and all(m["data"].shape == (m["point_count"], 3) for m in clouds)
    assert all(np.abs(m["data"]).sum(axis=1).all() for m in clouds)  # no zero padding rows


# -- 1: cv2.error from CompressedImage --------------------------------------------


def test_undecodable_compressed_image_is_skipped_not_fatal(tmp_path, caplog):
    cv2 = pytest.importorskip("cv2")
    store = _store()
    ok, good = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    payloads = [good.tobytes(), _png_header_only(40000, 40000), good.tobytes()]
    bag_dir = _write_rosbag2(
        tmp_path / "bag", store, [("/cam/compressed", _compressed(store, "png", p)) for p in payloads]
    )
    lone = _lone_copy(bag_dir, tmp_path / "lone.db3")
    for path in (bag_dir, lone):
        source = DataSources(str(path))
        with caplog.at_level(logging.WARNING):
            got = list(source.get_message())
        assert source.get_count("/cam/compressed") == 3
        assert len(got) == 2
        assert "Unable to decode compressed image payload" in caplog.text


# -- 2: corrupt PointCloud2 header ----------------------------------------------------


def test_corrupt_cloud_dimensions_fail_before_allocating(tmp_path):
    fields = [{"name": name, "offset": 4 * i, "datatype": 7} for i, name in enumerate("xyz")]
    with pytest.raises(ValueError, match="expected at least"):
        pointcloud_xyz(bytes(12), fields, 65536, 65536, 12, 12 * 65536, False)

    store = _store()
    T = store.types
    bad = T["sensor_msgs/msg/PointCloud2"](
        header=_header(store), height=65536, width=65536,
        fields=[T["sensor_msgs/msg/PointField"](name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate("xyz")],
        is_bigendian=False, point_step=12, row_step=12 * 65536, data=np.zeros(12, np.uint8), is_dense=True,
    )
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store)), ("/cloud", bad), ("/pose", _pose(store))])
    assert [m["topic"] for m in DataSources(str(bag_dir)).get_message()] == ["/pose", "/pose"]


# -- 3: rosbridge JSON payloads behind a "cdr" metadata.yaml ------------------------


def test_json_recorder_bag_counts_match_on_directory_and_file_paths(tmp_path, caplog):
    imu = {"op": "publish", "msg": {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "imu"}}}
    pose = {"op": "publish", "msg": {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "map"},
                                     "pose": {"position": {"x": 1, "y": 2, "z": 3}, "orientation": {"w": 1}}}}
    rows = []
    for i in range(3):
        rows.append(("/imu", "sensor_msgs/msg/Imu", 1000 + i, json.dumps(imu).encode()))
        rows.append(("/pose", "geometry_msgs/msg/PoseStamped", 1000 + i, json.dumps(pose).encode()))
    bag_dir = tmp_path / "jsondir"
    bag_dir.mkdir()
    _sqlite_bag(bag_dir / "rec_0.db3", rows, fmt="rosbridge_json")
    topics_yaml = "".join(
        f"    - topic_metadata: {{name: {name}, type: {msgtype}, serialization_format: cdr, "
        "offered_qos_profiles: [], type_description_hash: ''}\n      message_count: 3\n"
        for name, msgtype in (("/imu", "sensor_msgs/msg/Imu"), ("/pose", "geometry_msgs/msg/PoseStamped"))
    )
    (bag_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  version: 9\n  storage_identifier: sqlite3\n"
        "  duration: {nanoseconds: 2}\n  starting_time: {nanoseconds_since_epoch: 1000}\n"
        "  message_count: 6\n  topics_with_message_count:\n" + topics_yaml +
        "  compression_format: ''\n  compression_mode: ''\n  relative_file_paths: [rec_0.db3]\n"
        "  files: []\n  custom_data: ~\n  ros_distro: rolling\n"
    )
    lone = _lone_copy(bag_dir, tmp_path / "lone" / "rec_0.db3")
    for path in (bag_dir, lone):
        source = DataSources(str(path))
        yielded = [m["topic"] for m in source.get_message()]
        assert source.get_topics() == ["/pose"]  # no JSON decoder for Imu on either path
        assert {t: source.get_count(t) for t in ("/imu", "/pose")} == {"/imu": 0, "/pose": 3}
        assert yielded == ["/pose"] * 3

    # A JSON payload on a topic declared CDR, for a type without a JSON
    # decoder, is skipped with a warning instead of vanishing silently.
    mixed = _sqlite_bag(tmp_path / "mixed.db3", [
        ("/imu", "sensor_msgs/msg/Imu", 1, json.dumps(imu).encode()),
        ("/pose", "geometry_msgs/msg/PoseStamped", 2, json.dumps(pose).encode()),
    ])
    with caplog.at_level(logging.WARNING):
        assert [m["topic"] for m in DataSources(str(mixed)).get_message()] == ["/pose"]
    assert "No rosbridge JSON decoder for sensor_msgs/msg/Imu" in caplog.text


def test_malformed_json_payload_is_skipped(tmp_path, caplog):
    good = json.dumps({"msg": {"header": {"stamp": {"sec": 2, "nanosec": 0}}, "pose": {}}}).encode()
    path = _sqlite_bag(tmp_path / "json.db3", [
        ("/pose", "geometry_msgs/msg/PoseStamped", 1, b'{"msg": {"header": '),
        ("/pose", "geometry_msgs/msg/PoseStamped", 2, b"{not json}"),
        ("/pose", "geometry_msgs/msg/PoseStamped", 3, good),
    ], fmt="rosbridge_json")
    with caplog.at_level(logging.WARNING):
        assert [m["timestamp"] for m in DataSources(str(path)).get_message()] == [2.0]
    assert "Malformed rosbridge JSON" in caplog.text


# -- 4: DEM credentials ------------------------------------------------------------------


def test_dem_redirects_send_credentials_only_to_tile_and_login_hosts():
    from arraydataengine.sources.dem_source import _rebuild_earthdata_auth

    tile = "https://tile.example/dir/"

    def keeps(target, allow_insecure_http=False, tile_url=tile):
        prepared = SimpleNamespace(url=target, headers={"Authorization": "Basic x"})
        response = SimpleNamespace(request=SimpleNamespace(url="https://urs.earthdata.nasa.gov/oauth"))
        _rebuild_earthdata_auth(prepared, response, allow_insecure_http=allow_insecure_http, tile_url=tile_url)
        return "Authorization" in prepared.headers

    assert keeps("https://urs.earthdata.nasa.gov/oauth/authorize")
    assert keeps("https://tile.example/dir/N37W122.zip?code=1")
    assert keeps("https://tile.example:443/dir/x.zip")
    assert not keeps("https://attacker.example/steal")  # redirected *from* the login host
    assert not keeps("https://tile.example:8443/dir/x.zip")  # same name, other port
    assert not keeps("http://tile.example/dir/x.zip")
    assert not keeps("https://urs.earthdata.nasa.gov:444/x")
    assert not keeps("https://urs.earthdata.nasa.gov.evil.example/x")
    assert not keeps("http://urs.earthdata.nasa.gov/x", allow_insecure_http=True)
    assert keeps("http://mirror.example/x.zip", allow_insecure_http=True, tile_url="http://mirror.example/")
    assert not keeps("http://mirror.example/x.zip", tile_url="http://mirror.example/")


def test_dem_session_redirect_chain_end_to_end(monkeypatch):
    requests = pytest.importorskip("requests")
    from requests.adapters import BaseAdapter
    from requests.models import Response
    from arraydataengine.sources.dem_source import DEMSource

    tile_url = "https://tile.example/dir/N37W122.SRTMGL1.hgt.zip"
    routes = {
        tile_url: "https://urs.earthdata.nasa.gov/oauth/authorize",
        "https://urs.earthdata.nasa.gov/oauth/authorize": "https://attacker.example/steal",
    }
    seen = []

    class Adapter(BaseAdapter):
        def send(self, request, **kwargs):
            seen.append((request.url, request.headers.get("Authorization") is not None))
            response = Response()
            response.request, response.url, response.raw = request, request.url, io.BytesIO(b"")
            if request.url in routes:
                response.status_code, response.headers["Location"], response._content = 302, routes[request.url], b""
            else:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr("N37W122.hgt", np.zeros(4, dtype=">i2").tobytes())
                response.status_code, response._content = 200, buffer.getvalue()
            return response

        def close(self):
            pass

    monkeypatch.setenv("earthdata_username", "user")
    monkeypatch.setenv("earthdata_password", "pw")
    source = DEMSource([37, 38], [122, 123], base_url="https://tile.example/dir")
    context, session = source._open_session()
    session.mount("https://", Adapter())
    assert source._download_hgt(session, "N37W122") == np.zeros(4, dtype=">i2").tobytes()
    context.__exit__(None, None, None)
    assert seen == [
        (tile_url, True),
        ("https://urs.earthdata.nasa.gov/oauth/authorize", True),
        ("https://attacker.example/steal", False),
    ]


class _TileSession:
    def __init__(self, responses):
        self.responses = responses
        self.auth = None
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        self.calls.append(url)
        name = Path(url).name.split(".")[0]
        status, content = self.responses[name]
        return SimpleNamespace(
            status_code=status, content=content,
            raise_for_status=lambda: None if status < 400 else (_ for _ in ()).throw(RuntimeError(status)),
        )


def _hgt_zip(name, payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{name}.hgt", payload)
    return buffer.getvalue()


def test_dem_missing_tile_is_skipped_and_bad_tiles_are_not_cached(monkeypatch, tmp_path, caplog):
    from arraydataengine.sources.dem_source import DEMSource

    good = np.arange(4, dtype=">i2").tobytes()
    session = _TileSession({"N37W122": (404, b""), "N37W123": (200, _hgt_zip("N37W123", good))})
    monkeypatch.setenv("earthdata_username", "user")
    monkeypatch.setenv("earthdata_password", "pw")
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=lambda: session))

    with caplog.at_level(logging.WARNING):
        messages = list(DEMSource([37, 38], [122, 124], cache_dir=tmp_path).messages())
    assert [m["name"] for m in messages] == ["N37W123"]
    assert "N37W122" in caplog.text and "404" in caplog.text
    assert sorted(os.listdir(tmp_path)) == ["N37W123.hgt"]

    # An invalid download raises before it is cached ...
    session.responses = {"N38W122": (200, _hgt_zip("N38W122", b"\x00\x01\x02"))}
    with pytest.raises(ValueError, match="square"):
        list(DEMSource([38, 39], [122, 123], cache_dir=tmp_path).messages())
    assert not (tmp_path / "N38W122.hgt").exists()
    # ... and a corrupt cached tile is downloaded again.
    (tmp_path / "N38W122.hgt").write_bytes(b"\x00")
    session.responses = {"N38W122": (200, _hgt_zip("N38W122", good))}
    [message] = list(DEMSource([38, 39], [122, 123], cache_dir=tmp_path).messages())
    assert message["data"].shape == (2, 2)
    assert (tmp_path / "N38W122.hgt").read_bytes() == good


# -- 5: compressed_depth_image_transport ------------------------------------------------


def _compressed_depth(depth_png_values, encoding, quant=(0.0, 0.0)):
    cv2 = pytest.importorskip("cv2")
    ok, png = cv2.imencode(".png", depth_png_values)
    assert ok
    return struct.pack("<iff", 0, *quant) + png.tobytes(), f"{encoding}; compressedDepth png"


def test_compressed_depth_payloads_are_decoded(tmp_path):
    from arraydataengine.sensors.compressed_image_sensor import decode_compressed_image_with_encoding

    raw16 = np.array([[0, 1000], [2500, 65535]], dtype=np.uint16)
    payload, fmt = _compressed_depth(raw16, "16UC1")
    image, encoding = decode_compressed_image_with_encoding(payload, fmt)
    assert encoding == "16UC1" and image.dtype == np.uint16
    np.testing.assert_array_equal(image, raw16)

    # 32FC1: inverse-depth quantization as compressed_depth_image_transport writes it.
    depth_quantization, depth_max = 100.0, 10.0
    quant_a = depth_quantization * (depth_quantization + 1.0)
    quant_b = 1.0 - quant_a / depth_max
    depth = np.array([[1.0, 2.5], [np.nan, 8.0]], dtype=np.float32)
    inverse = np.where(np.isfinite(depth), quant_a / np.nan_to_num(depth, nan=1.0) + quant_b, 0)
    payload, fmt = _compressed_depth(inverse.astype(np.uint16), "32FC1", (quant_a, quant_b))

    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/depth/compressedDepth", _compressed(store, fmt, payload))])
    [message] = list(DataSources(str(bag_dir)).get_message())
    assert message["encoding"] == "32FC1" and message["data"].dtype == np.float32
    np.testing.assert_allclose(message["data"], depth, rtol=2e-3, equal_nan=True)


# -- 6: DepthAnythingCalibration with an embedded (newer) definition -----------------------


def test_calibration_decodes_by_embedded_definition(tmp_path):
    store = _store(cal_msg=CAL_V2_MSG)
    message = _calibration(store, schema_version=2, scale_stddev=0.1)
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/calib", message)])
    lone = _lone_copy(bag_dir, tmp_path / "lone" / "rec.db3")
    for path in (bag_dir, lone):
        [decoded] = list(DataSources(str(path)).get_message())
        np.testing.assert_array_equal(decoded["data"], [2.0, 0.5, 518, 392, 1920, 1440])
        assert decoded["offset"] == 0.5 and decoded["equation"] == "e"
        assert decoded["scale_stddev"] == 0.1 and decoded["schema_version"] == 2
        assert decoded["frame_id"] == "camera_link" and decoded["calibration_frame_id"] == "optical"

    # Without embedded definitions the fixed-layout CDR decoder is the fallback.
    store_v1 = _store()
    old_dir = _write_rosbag2(tmp_path / "old", store_v1, [("/calib", _calibration(store_v1))])
    with sqlite3.connect(next(old_dir.glob("*.db3"))) as connection:
        connection.execute("DELETE FROM message_definitions")
    connection.close()
    [decoded] = list(DataSources(str(old_dir)).get_message())
    assert decoded["offset"] == 0.5 and "scale_stddev" not in decoded


# -- 7 / 9: read-only SQLite connections ----------------------------------------------------


def test_connect_readonly_falls_back_when_sqlite_rejects_the_uri(monkeypatch, tmp_path):
    from arraydataengine.sources import db3_source

    path = _sqlite_bag(tmp_path / "rec.db3", [("/pose", "geometry_msgs/msg/PoseStamped", 1, b"x")])
    # Windows UNC paths become file://server/share/... which SQLite refuses.
    monkeypatch.setattr(type(path), "as_uri", lambda self: "file://server/share/rec.db3")
    connection = db3_source.connect_readonly(path)
    try:
        assert connection.execute("SELECT count(*) FROM messages").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM messages")
    finally:
        connection.close()
    assert not (tmp_path / "missing.db3").exists()
    with pytest.raises(sqlite3.Error):
        db3_source.connect_readonly(tmp_path / "missing.db3")
    assert not (tmp_path / "missing.db3").exists()  # never created by the plain fallback


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as non-root")
def test_wal_database_in_read_only_directory(tmp_path):
    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store))])
    folder = tmp_path / "ro"
    db = _lone_copy(bag_dir, folder / "w.db3")
    connection = sqlite3.connect(db)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    connection.close()
    os.chmod(folder, 0o555)
    try:
        source = DataSources(str(db))
        assert source.get_topics() == ["/pose"]
        assert len(list(source.get_message())) == 1
    finally:
        os.chmod(folder, 0o755)


# -- 8: partial / unparsable embedded definitions ------------------------------------------------


def test_partial_definitions_and_unknown_encoding(tmp_path, caplog):
    store = _store()
    T = store.types
    imu = T["sensor_msgs/msg/Imu"](
        header=_header(store, 6, 0, "imu"), orientation=T["geometry_msgs/msg/Quaternion"](x=0, y=0, z=0, w=1),
        orientation_covariance=np.zeros(9), angular_velocity=T["geometry_msgs/msg/Vector3"](x=1, y=2, z=3),
        angular_velocity_covariance=np.zeros(9), linear_acceleration=T["geometry_msgs/msg/Vector3"](x=4, y=5, z=6),
        linear_acceleration_covariance=np.zeros(9),
    )
    messages = [("/pose", _pose(store)), ("/imu", imu), ("/nav", _navsat(store, 0))]

    partial = _write_rosbag2(tmp_path / "partial", store, messages)
    with sqlite3.connect(next(partial.glob("*.db3"))) as connection:
        connection.execute(
            "DELETE FROM message_definitions WHERE topic_type IN ('sensor_msgs/msg/Imu', 'sensor_msgs/msg/NavSatFix')"
        )
    connection.close()

    unknown = _write_rosbag2(tmp_path / "unknown", store, messages)
    with sqlite3.connect(next(unknown.glob("*.db3"))) as connection:
        connection.execute("UPDATE message_definitions SET encoding = 'unknown' WHERE topic_type = 'sensor_msgs/msg/Imu'")
    connection.close()

    for path in (partial, unknown):
        source = DataSources(str(path))
        with caplog.at_level(logging.WARNING):
            by_topic = _by_topic(source.get_message())
        assert {t: len(v) for t, v in by_topic.items()} == {"/pose": 1, "/imu": 1, "/nav": 1}
        assert {t: source.get_count(t) for t in by_topic} == {"/pose": 1, "/imu": 1, "/nav": 1}
        np.testing.assert_array_equal(by_topic["/imu"][0]["data"][2], [1, 2, 3, 0])
    assert "reading its SQLite storage directly" in caplog.text


# -- improvements ---------------------------------------------------------------------------------


def test_decoder_plan_is_cached_per_type(tmp_path, monkeypatch):
    from arraydataengine.sources.ros_source import RosSource

    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store, sec)) for sec in range(1, 21)])
    calls = []
    original = RosSource._make_plan
    monkeypatch.setattr(RosSource, "_make_plan", lambda self, *args: calls.append(args) or original(self, *args))
    for path in (bag_dir, _lone_copy(bag_dir, tmp_path / "lone.db3")):
        calls.clear()
        source = DataSources(str(path))
        assert len(list(source.get_message())) == 20
        assert len(calls) <= 2  # listing + decoding, not once per message


def test_topics_filter_skips_decoding_other_topics(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import arraydataengine.sensors.compressed_image_sensor as compressed

    store = _store()
    ok, png = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    messages = [("/pose", _pose(store)), ("/cam", _compressed(store, "png", png.tobytes())), ("/pose", _pose(store, 2))]
    bag_dir = _write_rosbag2(tmp_path / "bag", store, messages)
    ros1_store = _store(ros1=True)
    ros1 = _write_ros1(tmp_path / "rec.bag", ros1_store, [
        ("/pose", _pose(ros1_store, ros1=True)), ("/cam", _compressed(ros1_store, "png", png.tobytes(), ros1=True)),
    ])

    def fail(*args, **kwargs):
        raise AssertionError("filtered topic was decoded")

    monkeypatch.setattr(compressed, "decode_compressed_image_with_encoding", fail)
    for path in (bag_dir, _lone_copy(bag_dir, tmp_path / "lone.db3"), ros1):
        for topics in (["/pose"], "/pose"):
            source = DataSources(str(path), topics=topics)
            assert source.topics == ("/pose",)
            assert source.get_topics() == ["/pose"]
            assert source.get_count("/cam") == 0
            assert {m["topic"] for m in source.get_message()} == {"/pose"}
            assert "/cam" in source.get_topic_types()

    images = tmp_path / "imgs"
    images.mkdir()
    cv2.imwrite(str(images / "a.png"), np.zeros((2, 2), np.uint8))
    with pytest.raises(ValueError, match="not found"):  # a typo fails fast instead of yielding nothing
        DataSources(str(images / "*.png"), topics=["/other"])
    assert len(list(DataSources(str(images / "*.png"), topics="images").get_message())) == 1


def test_natural_key_ignores_non_decimal_digits():
    from arraydataengine.sources.db3_source import natural_key
    from arraydataengine.sources.img_source import _natural_key

    names = ["rec_10.db3", "rec_².db3", "²", "rec_2.db3"]
    for key in (natural_key, _natural_key):
        assert sorted(names, key=key)[:2] == ["rec_2.db3", "rec_10.db3"]


def test_extensions_are_case_insensitive_and_file_inputs_keep_their_path(tmp_path):
    store = _store()
    from rosbags.rosbag2 import StoragePlugin

    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store))])
    upper_db3 = _lone_copy(bag_dir, tmp_path / "upper" / "REC.DB3")
    mcap_dir = _write_rosbag2(tmp_path / "mcap", store, [("/pose", _pose(store))], StoragePlugin.MCAP)
    upper_mcap = tmp_path / "mcapalone" / "REC.MCAP"
    upper_mcap.parent.mkdir()
    shutil.copy(next(mcap_dir.glob("*.mcap")), upper_mcap)
    ros1_store = _store(ros1=True)
    upper_bag = _write_ros1(tmp_path / "REC.BAG", ros1_store, [("/pose", _pose(ros1_store, ros1=True))])

    for path in (upper_db3, upper_mcap, upper_bag):
        source = DataSources(str(path))
        assert [m["topic"] for m in source.get_message()] == ["/pose"]
        assert source.source.get_data_path() == str(path)
        assert source.get_data_path() == str(path)
    assert DataSources(str(bag_dir)).get_data_path() == str(bag_dir)


def test_image_paths_with_glob_metacharacters_and_rgb_order(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 200  # red
    literal = tmp_path / "frame[1].png"
    cv2.imwrite(str(literal), rgb[..., ::-1])

    [message] = list(DataSources(str(literal)).get_message())
    assert message["name"] == "frame[1].png"
    np.testing.assert_array_equal(message["data"], rgb)
    assert message["encoding"] == "rgb8"
    [bgr] = list(DataSources(str(literal), color_order="bgr").get_message())
    np.testing.assert_array_equal(bgr["data"], rgb[..., ::-1])
    assert bgr["encoding"] == "bgr8"
    with pytest.raises(ValueError, match="color_order"):
        DataSources(str(literal), color_order="hsv")

    gray = tmp_path / "depth.png"
    cv2.imwrite(str(gray), np.full((2, 2), 4000, np.uint16))
    [depth] = list(DataSources(str(gray)).get_message())
    assert depth["encoding"] == "mono16" and depth["data"].dtype == np.uint16


def test_scalar_relative_file_paths_in_metadata(tmp_path):
    from arraydataengine.sources.db3_source import metadata_file_names

    (tmp_path / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  relative_file_paths: rec_0.db3\n  files: {path: rec_0.db3}\n"
    )
    assert metadata_file_names(tmp_path / "metadata.yaml") == ["rec_0.db3"]


def test_rosbags_and_sqlite_durations_match(tmp_path):
    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store, sec)) for sec in (1, 2, 3)])
    lone = _lone_copy(bag_dir, tmp_path / "lone.db3")
    ros1_store = _store(ros1=True)
    ros1 = _write_ros1(tmp_path / "r.bag", ros1_store, [("/pose", _pose(ros1_store, s, ros1=True)) for s in (1, 2, 3)])
    durations = [DataSources(str(p)).source.get_duration() for p in (bag_dir, lone, ros1)]
    assert durations == [pytest.approx(2e-9, abs=1e-12)] * 3


# -- A: RGB images with an encoding key ------------------------------------------------------


def test_colour_images_are_rgb_with_encoding_on_every_path(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rgba = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    ok, png = cv2.imencode(".png", rgb[..., ::-1])
    json_png = json.dumps({"msg": {"header": {"stamp": {"sec": 9, "nanosec": 0}}, "format": "png",
                                   "data": __import__("base64").b64encode(png.tobytes()).decode()}}).encode()

    def messages(store, ros1=False):
        return [
            ("/rgb8", _image(store, "rgb8", 2, 3, 9, rgb.tobytes(), ros1)),
            ("/bgr8", _image(store, "bgr8", 2, 3, 9, rgb[..., ::-1].tobytes(), ros1)),
            ("/bgra8", _image(store, "bgra8", 2, 3, 12, rgba[..., [2, 1, 0, 3]].tobytes(), ros1)),
            ("/mono8", _image(store, "mono8", 2, 3, 3, bytes(6), ros1)),
            ("/compressed", _compressed(store, "png", png.tobytes(), ros1)),
        ]

    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, messages(store))
    ros1_store = _store(ros1=True)
    ros1 = _write_ros1(tmp_path / "r.bag", ros1_store, messages(ros1_store, ros1=True))
    json_bag = _sqlite_bag(tmp_path / "json.db3", [("/compressed", "sensor_msgs/msg/CompressedImage", 1, json_png)],
                           fmt="rosbridge_json")

    expected = {"/rgb8": (rgb, "rgb8"), "/bgr8": (rgb, "rgb8"), "/bgra8": (rgba, "rgba8"),
                "/mono8": (np.zeros((2, 3), np.uint8), "mono8"), "/compressed": (rgb, "rgb8")}
    for path in (bag_dir, _lone_copy(bag_dir, tmp_path / "lone.db3"), ros1, json_bag):
        for message in DataSources(str(path)).get_message():
            data, encoding = expected[message["topic"]]
            np.testing.assert_array_equal(message["data"], data)
            assert message["encoding"] == encoding


# -- B: NavSatFix status and covariance -------------------------------------------------------


def test_navsatfix_status_and_covariance_on_every_path(tmp_path):
    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/fix", _navsat(store, 0)), ("/fix", _navsat(store, -1))])
    ros1_store = _store(ros1=True)
    ros1 = _write_ros1(tmp_path / "r.bag", ros1_store, [
        ("/fix", _navsat(ros1_store, 0, ros1=True)), ("/fix", _navsat(ros1_store, -1, ros1=True)),
    ])
    fix = {"header": {"stamp": {"sec": 5, "nanosec": 0}, "frame_id": "gps"}, "latitude": 37.5,
           "longitude": -122.25, "altitude": 10.0, "position_covariance": list(range(9))}
    json_bag = _sqlite_bag(tmp_path / "json.db3", [
        ("/fix", "sensor_msgs/msg/NavSatFix", 1, json.dumps({"msg": dict(fix, status={"status": 0, "service": 1})}).encode()),
        ("/fix", "sensor_msgs/msg/NavSatFix", 2, json.dumps({"msg": dict(fix, status={"status": -1, "service": 1})}).encode()),
    ], fmt="rosbridge_json")

    for path in (bag_dir, _lone_copy(bag_dir, tmp_path / "lone.db3"), ros1, json_bag):
        got = list(DataSources(str(path)).get_message())
        assert [m["status"] for m in got] == [0, -1]
        for message in got:
            np.testing.assert_array_equal(message["data"], [37.5, -122.25, 10.0])
            assert message["position_covariance"].shape == (3, 3)
            np.testing.assert_array_equal(message["position_covariance"], np.arange(9.0).reshape(3, 3))
            assert isinstance(message["status"], int)


def test_rosbags_yielding_nothing_falls_back_to_sqlite(tmp_path, monkeypatch, caplog):
    """Newer rosbags versions yield no messages when metadata.yaml and the
    storage disagree (e.g. on QoS or format); the SQLite storage is read instead."""

    from arraydataengine.sources import ros_source

    store = _store()
    bag_dir = _write_rosbag2(tmp_path / "bag", store, [("/pose", _pose(store, sec)) for sec in (1, 2)])
    monkeypatch.setattr(ros_source, "iter_reader_messages", lambda reader, connections: iter(()))
    source = DataSources(str(bag_dir))
    with caplog.at_level(logging.WARNING):
        assert [m["timestamp"] for m in source.get_message()] == [1.0, 2.0]
        assert [m["timestamp"] for m in source.get_message()] == [1.0, 2.0]
    assert caplog.text.count("rosbags yielded none of its 2 listed messages") == 1


def test_datasources_forwards_dem_base_url_and_http_opt_in(tmp_path):
    from arraydataengine.source import DataSources

    source = DataSources("DEM", bounds=[[37, 38], [122, 123]], cache_dir=tmp_path,
                         base_url="http://mirror.example/srtm", allow_insecure_http=True)
    assert source.source.base_url == "http://mirror.example/srtm"
    assert source.source.allow_insecure_http is True
    default = DataSources("DEM", bounds=[[37, 38], [122, 123]], cache_dir=tmp_path)
    assert default.source.allow_insecure_http is False


def test_datasources_rejects_unknown_topic_names(tmp_path):
    from arraydataengine.source import DataSources

    from pathlib import Path

    db3 = Path(__file__).resolve().parents[1] / "example" / "mapeverything_0.db3"
    with open(db3, "rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs"):
            pytest.skip("example recording is a Git LFS pointer")
    with pytest.raises(ValueError, match=r"/mapping/pos'\] not found.*available topics: \[.*'/mapping/pose'"):
        DataSources(str(db3), topics=["/mapping/pose", "/mapping/pos"])
    assert DataSources(str(db3), topics=["/mapping/pose"]).get_topics() == ["/mapping/pose"]

    from PIL import Image
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(tmp_path / "f0.png")
    assert DataSources(str(tmp_path / "*.png"), topics=["images"]).get_topics() == ["images"]
    with pytest.raises(ValueError, match="not found"):
        DataSources(str(tmp_path / "*.png"), topics=["camera"])


def test_image_glob_with_wildcard_extension_reads_only_images(tmp_path):
    from PIL import Image
    from arraydataengine.source import DataSources

    for i in range(3):
        Image.fromarray(np.full((4, 5), i, np.uint16)).save(tmp_path / f"f{i}.tiff")
    (tmp_path / "f0.tiff.aux.xml").write_text("<PAMDataset/>")  # sidecar a *.tif* glob also matches
    source = DataSources(str(tmp_path / "*.tif*"))
    frames = [m["data"] for m in source.get_message()]
    assert source.get_count("images") == 3 and [int(f[0, 0]) for f in frames] == [0, 1, 2]
    with pytest.raises(ValueError, match="not supported"):
        DataSources(str(tmp_path / "*.xml*"))
