import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from ade.source import DataSources
from ade.sources.img_source import ImgSource
from ade.sources.bag_source import BagSource
from ade.sources.db3_source import DB3Source


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
