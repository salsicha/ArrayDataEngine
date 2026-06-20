import os
import tempfile
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from ade.source import DataSources
from ade.sources.img_source import ImgSource
from ade.sources.bag_source import BagSource
from ade.sources.db3_source import DB3Source
from ade.sources.dem_source import DEMSource
from ade.sources import base_source


def test_img_source():
    temp_dir = tempfile.mkdtemp()
    img_path1 = os.path.join(temp_dir, "frame_000.png")
    img_path2 = os.path.join(temp_dir, "frame_001.png")

    try:
        # Create fake images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path1, img)
        cv2.imwrite(img_path2, img)

        # Instantiate ImgSource
        source = ImgSource(os.path.join(temp_dir, "*.png"), period=0.1, file_type=".png")
        
        assert source.data_exists() is True
        assert source.get_count() == 2
        assert source.get_duration() == 0.2
        assert source.get_topics() == ["images"]

        messages = list(source.messages())
        assert len(messages) == 2
        assert messages[0]["topic"] == "images"
        assert messages[0]["name"] == "frame_000.png"
        assert messages[0]["data"].shape == (100, 100, 3)

    finally:
        shutil.rmtree(temp_dir)


def test_synthetic_image_pipeline_through_data_sources_and_buffer(tmp_path):
    from ade.buffer import DataBuffer

    for i in range(5):
        image = np.full((4, 4), i, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), image)

    source = DataSources(str(tmp_path / "*.png"), period=0.25)
    buffer = DataBuffer(source, buffer_depth=3, axis="images", preload=0)

    for _ in range(5):
        buffer.roll_buffer("images")

    window = buffer.get_buffer()["images"]
    assert np.allclose(window["ts"], np.array([0.5, 0.75, 1.0]))
    assert [name.decode() for name in window["id"]] == ["frame_002.png", "frame_003.png", "frame_004.png"]
    assert [int(frame[0, 0]) for frame in window["data"]] == [2, 3, 4]
    assert int(buffer[-1][0, 0]) == 4

    time_range = buffer.get_time_range("images", start=0.75, end=1.0)
    assert np.allclose(time_range["ts"], np.array([0.75, 1.0]))
    assert [int(frame[0, 0]) for frame in time_range["data"]] == [3, 4]

    recent = buffer.get_last_seconds("images", seconds=0.25)
    assert np.allclose(recent["ts"], np.array([0.75, 1.0]))
    assert [int(frame[0, 0]) for frame in recent["data"]] == [3, 4]


@patch("ade.sources.base_source.AnyReader")
def test_bag_source_mocked(mock_any_reader):
    # Setup mock reader behaviour
    mock_reader_instance = MagicMock()
    mock_reader_instance.end_time = 2000000000
    mock_reader_instance.start_time = 1000000000

    mock_conn = MagicMock()
    mock_conn.topic = "/camera/image"
    mock_conn.msgcount = 10
    mock_reader_instance.connections = [mock_conn]

    mock_any_reader.return_value.__enter__.return_value = mock_reader_instance

    source = BagSource("fake_bag_file.bag")

    # Test get_duration
    duration = source.get_duration()
    assert duration == 1.0  # (2e9 - 1e9) * 1e-9 = 1.0

    # Test get_count
    count = source.get_count("/camera/image")
    assert count == 10


@patch("ade.sources.base_source.AnyReader")
def test_bag_source_caches_metadata(mock_any_reader):
    mock_reader_instance = MagicMock()
    mock_reader_instance.end_time = 2000000000
    mock_reader_instance.start_time = 1000000000

    mock_conn = MagicMock()
    mock_conn.topic = "/camera/image"
    mock_conn.msgcount = 10
    mock_reader_instance.connections = [mock_conn]

    mock_any_reader.return_value.__enter__.return_value = mock_reader_instance

    source = BagSource("fake_bag_file.bag")

    assert source.get_topics() == ["/camera/image"]
    assert source.get_count("/camera/image") == 10
    assert source.get_duration() == 1.0
    assert mock_any_reader.call_count == 1


class _SyntheticReader:
    def __init__(self, connections, messages):
        self.connections = connections
        self._messages = messages
        self.start_time = 10
        self.end_time = 40

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def messages(self):
        yield from self._messages


class _SyntheticRosSensor:
    def __init__(self, rawdata, msgtype):
        self.rawdata = rawdata
        self.msgtype = msgtype

    def numpyify(self):
        value = self.rawdata[0]
        class_name = self.msgtype.rsplit("/", 1)[1]
        return np.array([value], dtype=np.float64), class_name, 100.0 + value * 0.1


def test_synthetic_db3_multitopic_stream_filters_and_counts(monkeypatch, tmp_path):
    image_conn = SimpleNamespace(topic="/camera/image", msgcount=2, msgtype="sensor_msgs/msg/Image")
    imu_conn = SimpleNamespace(topic="/imu", msgcount=1, msgtype="sensor_msgs/msg/Imu")
    unsupported_conn = SimpleNamespace(topic="/diagnostics", msgcount=1, msgtype="custom_msgs/msg/Unsupported")
    messages = [
        (image_conn, 10, bytes([0])),
        (unsupported_conn, 20, bytes([99])),
        (imu_conn, 30, bytes([5])),
        (image_conn, 40, bytes([1])),
    ]

    monkeypatch.setattr(base_source, "AnyReader", lambda paths: _SyntheticReader(
        [image_conn, imu_conn, unsupported_conn],
        messages,
    ))
    monkeypatch.setitem(DB3Source.SENSOR_TYPES, "image", _SyntheticRosSensor)
    monkeypatch.setitem(DB3Source.SENSOR_TYPES, "imu", _SyntheticRosSensor)

    db3_path = tmp_path / "synthetic_0.db3"
    db3_path.write_text("dummy")
    source = DB3Source(str(db3_path))

    assert source.get_topics() == ["/camera/image", "/imu", "/diagnostics"]
    assert source.get_count("/camera/image") == 2
    assert source.get_count("/imu") == 1
    assert np.isclose(source.get_duration(), 3.0e-8)

    output = list(source.messages())
    assert [msg["topic"] for msg in output] == ["/camera/image", "/imu", "/camera/image"]
    assert [msg["name"] for msg in output] == ["Image", "Imu", "Image"]
    assert [float(msg["data"][0]) for msg in output] == [0.0, 5.0, 1.0]


def test_db3_source_chunk_path_uses_containing_directory(monkeypatch, tmp_path):
    captured_paths = []
    connection = SimpleNamespace(topic="/camera/image", msgcount=1, msgtype="sensor_msgs/msg/Image")

    def reader_factory(paths):
        captured_paths.append(paths)
        return _SyntheticReader([connection], [])

    monkeypatch.setattr(base_source, "AnyReader", reader_factory)
    (tmp_path / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    chunk_path = tmp_path / "recording_0.db3"
    chunk_path.write_text("dummy")

    source = DB3Source(str(chunk_path))
    assert source.get_topics() == ["/camera/image"]
    assert captured_paths == [[Path(tmp_path)]]


@patch("ade.sources.base_source.AnyReader")
def test_db3_source_mocked(mock_any_reader):
    mock_reader_instance = MagicMock()
    mock_reader_instance.connections = [MagicMock(topic="/camera/image", msgcount=5)]
    mock_any_reader.return_value.__enter__.return_value = mock_reader_instance

    temp_dir = tempfile.mkdtemp()
    db3_path = os.path.join(temp_dir, "fake_db3_file.db3")
    
    try:
        # Create a dummy file and directory structure
        with open(db3_path, "w") as f:
            f.write("dummy")

        source = DB3Source(db3_path)
        assert source.data_exists() is True
        assert source.get_count("/camera/image") == 5

    finally:
        shutil.rmtree(temp_dir)


@patch("ade.sources.base_source.AnyReader")
def test_db3_source_split_directory_mocked(mock_any_reader):
    mock_reader_instance = MagicMock()
    mock_reader_instance.connections = [MagicMock(topic="/camera/image", msgcount=8)]
    mock_any_reader.return_value.__enter__.return_value = mock_reader_instance

    temp_dir = tempfile.mkdtemp()

    try:
        with open(os.path.join(temp_dir, "metadata.yaml"), "w") as f:
            f.write("rosbag2_bagfile_information: {}\n")
        with open(os.path.join(temp_dir, "recording_0.db3"), "w") as f:
            f.write("dummy")
        with open(os.path.join(temp_dir, "recording_1.db3"), "w") as f:
            f.write("dummy")

        source = DB3Source(temp_dir)
        assert source.data_path == temp_dir
        assert source.data_exists() is True
        assert source.get_count("/camera/image") == 8

    finally:
        shutil.rmtree(temp_dir)


@patch("ade.sources.base_source.AnyReader")
def test_data_sources_accepts_split_db3_directory(mock_any_reader):
    mock_reader_instance = MagicMock()
    mock_reader_instance.connections = [MagicMock(topic="/camera/image", msgcount=8)]
    mock_any_reader.return_value.__enter__.return_value = mock_reader_instance

    temp_dir = tempfile.mkdtemp()

    try:
        with open(os.path.join(temp_dir, "metadata.yaml"), "w") as f:
            f.write("rosbag2_bagfile_information: {}\n")
        with open(os.path.join(temp_dir, "recording_0.db3"), "w") as f:
            f.write("dummy")
        with open(os.path.join(temp_dir, "recording_1.db3"), "w") as f:
            f.write("dummy")

        source = DataSources(temp_dir)
        assert isinstance(source.source, DB3Source)
        assert source.get_count("/camera/image") == 8

    finally:
        shutil.rmtree(temp_dir)


def _dem_zip_bytes(tile_name, side=4):
    hgt = np.arange(side * side, dtype=">i2").reshape((side, side)).tobytes()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr(f"{tile_name}.hgt", hgt)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, url, content):
        self.url = url
        self.content = content

    def raise_for_status(self):
        return None


class _FakeSession:
    created = 0
    calls = []

    def __init__(self):
        type(self).created += 1
        self.auth = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        self.calls.append(("request", method, url, kwargs.get("timeout")))
        return _FakeResponse(url, b"")

    def get(self, url, auth=None, **kwargs):
        self.calls.append(("get", "get", url, kwargs.get("timeout")))
        tile_name = Path(url).name.split(".")[0]
        return _FakeResponse(url, _dem_zip_bytes(tile_name))


def test_synthetic_dem_source_reuses_session_and_decodes_tiles(monkeypatch):
    _FakeSession.created = 0
    _FakeSession.calls = []
    monkeypatch.setenv("earthdata_username", "user")
    monkeypatch.setenv("earthdata_password", "password")
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=_FakeSession))

    source = DEMSource([1, 2], [1, 3], timeout=7.0)
    messages = list(source.messages())

    assert _FakeSession.created == 1
    assert [msg["name"] for msg in messages] == ["N1W1", "N1W2"]
    assert [msg["data"].shape for msg in messages] == [(4, 4), (4, 4)]
    assert all(call[3] == 7.0 for call in _FakeSession.calls)
