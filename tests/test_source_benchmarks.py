import io
import sys
import time
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np

from ade.buffer import DataBuffer
from ade.ops import topic_pipeline
from ade.sources import base_source
from ade.sources.bag_source import BagSource
from ade.sources.db3_source import DB3Source
from ade.sources.dem_source import DEMSource
from ade.sources.img_source import ImgSource


MESSAGE_COUNT = 200
IMAGE_SHAPE = (64, 64, 3)
DEM_SIDE = 32
PIPELINE_MESSAGE_COUNT = 50_000
TILEDB_PIPELINE_MESSAGE_COUNT = 200
PIPELINE_CHUNK_SIZE = 256
REPEATS = 3


def _time_best(func, repeats=REPEATS):
    best = None
    best_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        best_result = func()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best, best_result


def _print_benchmark(name, count, elapsed):
    rate = count / elapsed if elapsed else float("inf")
    us_per_message = elapsed / count * 1_000_000 if count else 0.0
    print(f"BENCHMARK {name}: {count} messages in {elapsed:.6f}s, {rate:.0f} msg/s, {us_per_message:.1f} us/msg")


class _FakeSensor:
    def __init__(self, rawdata, msgtype):
        self.rawdata = rawdata
        self.msgtype = msgtype

    def numpyify(self):
        return np.zeros(IMAGE_SHAPE, dtype=np.uint8), "Image", 1.0


class _PipelineSource:
    def __init__(self, count):
        self.count = count

    def get_topics(self):
        return ["sensor_topic"]

    def get_count(self, topic):
        return self.count

    def get_message(self):
        for i in range(self.count):
            yield {
                "topic": "sensor_topic",
                "timestamp": float(i) * 0.01,
                "name": f"frame_{i}",
                "data": np.array([float(i), float(i) * 2.0, 1.0, -1.0], dtype=np.float64),
            }


class _FakeReader:
    def __init__(self, connections, message_count):
        self.connections = connections
        self.message_count = message_count
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def messages(self):
        connection = self.connections[0]
        for i in range(self.message_count):
            yield connection, self.start_time + i, b"raw"


def _patch_reader(monkeypatch, topic="/camera/image", message_count=MESSAGE_COUNT):
    connection = SimpleNamespace(topic=topic, msgcount=message_count, msgtype="sensor_msgs/msg/Image")

    def reader_factory(paths):
        return _FakeReader([connection], message_count)

    monkeypatch.setattr(base_source, "AnyReader", reader_factory)
    return connection


def test_benchmark_img_source_messages(tmp_path):
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    for i in range(MESSAGE_COUNT):
        cv2.imwrite(str(tmp_path / f"frame_{i:04d}.png"), image)

    source = ImgSource(str(tmp_path / "*.png"), period=0.1, file_type=".png")

    def read_all():
        return sum(1 for _ in source.messages())

    elapsed, count = _time_best(read_all)
    assert count == MESSAGE_COUNT
    _print_benchmark("ImgSource.messages", count, elapsed)


def test_benchmark_bag_source_messages(monkeypatch):
    _patch_reader(monkeypatch)
    monkeypatch.setitem(BagSource.SENSOR_TYPES, "image", _FakeSensor)

    source = BagSource("fake_bag_file.bag")

    def read_all():
        return sum(1 for _ in source.messages())

    elapsed, count = _time_best(read_all)
    assert count == MESSAGE_COUNT
    _print_benchmark("BagSource.messages", count, elapsed)


def test_benchmark_db3_source_messages(monkeypatch, tmp_path):
    _patch_reader(monkeypatch)
    monkeypatch.setitem(DB3Source.SENSOR_TYPES, "image", _FakeSensor)

    db3_path = tmp_path / "fake_db3_file.db3"
    db3_path.write_text("dummy")
    source = DB3Source(str(db3_path))

    def read_all():
        return sum(1 for _ in source.messages())

    elapsed, count = _time_best(read_all)
    assert count == MESSAGE_COUNT
    _print_benchmark("DB3Source.messages", count, elapsed)


def _dem_zip_bytes():
    hgt = np.arange(DEM_SIDE * DEM_SIDE, dtype=">i2").reshape((DEM_SIDE, DEM_SIDE)).tobytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for n in range(1, 3):
            for w in range(1, 3):
                zip_file.writestr(f"N{n}W{w}.hgt", hgt)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, url, content):
        self.url = url
        self.content = content

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, content):
        self.content = content
        self.auth = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        return _FakeResponse(url, self.content)

    def get(self, url, auth=None, **kwargs):
        return _FakeResponse(url, self.content)


def test_benchmark_dem_source_messages(monkeypatch):
    content = _dem_zip_bytes()
    monkeypatch.setenv("earthdata_username", "user")
    monkeypatch.setenv("earthdata_password", "password")
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=lambda: _FakeSession(content)))

    source = DEMSource([1, 3], [1, 3])

    def read_all():
        return sum(1 for _ in source.messages())

    elapsed, count = _time_best(read_all)
    assert count == 4
    _print_benchmark("DEMSource.messages", count, elapsed)


def test_benchmark_topic_pipeline_iter_chunks():
    topic = {
        "id": np.arange(PIPELINE_MESSAGE_COUNT, dtype=np.int64),
        "ts": np.arange(PIPELINE_MESSAGE_COUNT, dtype=np.float64) * 0.01,
        "data": np.arange(PIPELINE_MESSAGE_COUNT * 4, dtype=np.float64).reshape(PIPELINE_MESSAGE_COUNT, 4),
    }
    pipeline = (
        topic_pipeline(topic, topic="sensor_topic")
        .time_range(50.0, 450.0)
        .index_range(0, 20_000, 2)
        .map(lambda data: data * 0.5)
    )

    def process_all():
        return sum(len(chunk) for chunk in pipeline.iter_chunks(chunk_size=PIPELINE_CHUNK_SIZE))

    elapsed, count = _time_best(process_all)
    assert count == 10_000
    _print_benchmark("TopicPipeline.iter_chunks", count, elapsed)


def test_benchmark_tiledb_topic_pipeline_time_range(tmp_path):
    group_uri = str(tmp_path / "pipeline_benchmark_group")
    with DataBuffer(
        data_source=_PipelineSource(TILEDB_PIPELINE_MESSAGE_COUNT),
        buffer_depth=TILEDB_PIPELINE_MESSAGE_COUNT,
        data_uri=group_uri,
        axis="sensor_topic",
        use_db=True,
        preload=0,
    ) as buffer:
        buffer.load_data_db("sensor_topic")

        pipeline = (
            buffer.topic("sensor_topic")
            .time_range(0.5, 1.49)
            .map(lambda data: data + 1.0)
        )

        def process_all():
            return sum(len(chunk) for chunk in pipeline.iter_chunks(chunk_size=32))

        elapsed, count = _time_best(process_all)

    assert count == 100
    _print_benchmark("TileDB.TopicPipeline.time_range", count, elapsed)
