import io
import sys
import time
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np

from ade.buffer import DataBuffer
from ade.ops import (
    dead_reckon_trajectory,
    dem_to_point_cloud,
    estimate_normals,
    knn_search,
    normalize_images,
    resample_imu,
    resample_navsat,
    resample_odometry,
    resize_images,
    rgb_to_gray,
    roughness_map,
    terrain_normals,
    topic_pipeline,
    traversability_map,
    voxel_downsample,
)
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
SYNTHETIC_IMAGE_COUNT = 64
SYNTHETIC_IMAGE_SHAPE = (96, 96, 3)
SYNTHETIC_POINT_COUNT = 1_024
SYNTHETIC_NAV_COUNT = 2_000
SYNTHETIC_DEM_SIDE = 128
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


def _print_benchmark(name, count, elapsed, unit="items"):
    rate = count / elapsed if elapsed else float("inf")
    us_per_item = elapsed / count * 1_000_000 if count else 0.0
    singular_unit = unit[:-1] if unit.endswith("s") else unit
    print(
        f"BENCHMARK {name}: {count} {unit} in {elapsed:.6f}s, "
        f"{rate:.0f} {unit}/s, {us_per_item:.1f} us/{singular_unit}"
    )


def _identity_quaternions(count):
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1))


def _synthetic_images():
    rng = np.random.default_rng(123)
    return rng.integers(
        0,
        256,
        size=(SYNTHETIC_IMAGE_COUNT, *SYNTHETIC_IMAGE_SHAPE),
        dtype=np.uint8,
    )


def _synthetic_points():
    rng = np.random.default_rng(456)
    xyz = rng.normal(size=(SYNTHETIC_POINT_COUNT, 3)).astype(np.float64)
    intensity = rng.random((SYNTHETIC_POINT_COUNT, 1))
    return np.hstack((xyz, intensity))


def _synthetic_timestamps(count=SYNTHETIC_NAV_COUNT):
    return np.arange(count, dtype=np.float64) * 0.01


def _synthetic_imu_topic():
    timestamps = _synthetic_timestamps()
    data = np.zeros((timestamps.size, 6, 4), dtype=np.float64)
    data[:, 0, :4] = _identity_quaternions(timestamps.size)
    data[:, 1, :3] = 0.01
    data[:, 2, 2] = 0.02
    data[:, 3, :3] = 0.02
    data[:, 4, 0] = 0.1
    data[:, 4, 2] = 9.80665
    data[:, 5, :3] = 0.03
    return {"ts": timestamps, "data": data}


def _synthetic_odometry_topic():
    timestamps = _synthetic_timestamps()
    data = np.zeros((timestamps.size, 8, 4), dtype=np.float64)
    data[:, 0, 0] = timestamps
    data[:, 0, 1] = timestamps * 0.5
    data[:, 1, :3] = 0.01
    data[:, 2, :4] = _identity_quaternions(timestamps.size)
    data[:, 3, :3] = 0.02
    data[:, 4, :3] = np.array([1.0, 0.5, 0.0])
    data[:, 5, :3] = 0.03
    data[:, 6, 2] = 0.01
    data[:, 7, :3] = 0.04
    return {"ts": timestamps, "data": data}


def _synthetic_navsat_topic():
    timestamps = _synthetic_timestamps()
    index = np.arange(timestamps.size, dtype=np.float64)
    data = np.column_stack((
        37.0 + index * 1e-6,
        -122.0 + index * 1e-6,
        10.0 + index * 0.01,
    ))
    return {"ts": timestamps, "data": data}


def _synthetic_dem():
    axis = np.linspace(-1.0, 1.0, SYNTHETIC_DEM_SIDE)
    xx, yy = np.meshgrid(axis, axis)
    return 25.0 + 3.0 * xx + 2.0 * yy + np.sin(xx * np.pi) * np.cos(yy * np.pi)


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


def test_benchmark_synthetic_image_operations():
    images = _synthetic_images()

    def process_all():
        resized = resize_images(images, (48, 48))
        normalized = normalize_images(resized, min_value=0, max_value=255)
        return rgb_to_gray(normalized)

    elapsed, gray = _time_best(process_all)
    assert gray.shape == (SYNTHETIC_IMAGE_COUNT, 48, 48)
    assert gray.dtype == np.float64
    _print_benchmark("ImageOps.synthetic", SYNTHETIC_IMAGE_COUNT, elapsed, unit="frames")


def test_benchmark_synthetic_point_cloud_operations():
    points = _synthetic_points()

    def process_all():
        downsampled = voxel_downsample(points, voxel_size=0.2)
        distances, indices = knn_search(points, points[::16, :3], k=8)
        normals = estimate_normals(points[:512], k=8, orient_toward=np.array([0.0, 0.0, 10.0]))
        return downsampled, distances, indices, normals

    elapsed, (downsampled, distances, indices, normals) = _time_best(process_all)
    assert downsampled.shape[1] == points.shape[1]
    assert distances.shape == indices.shape == (64, 8)
    assert normals.shape == (512, 3)
    _print_benchmark("PointCloudOps.synthetic", SYNTHETIC_POINT_COUNT, elapsed, unit="points")


def test_benchmark_synthetic_imu_operations():
    imu = _synthetic_imu_topic()
    timestamps = imu["ts"]
    targets = np.linspace(timestamps[0], timestamps[-1], timestamps.size // 2)
    position = np.zeros((timestamps.size, 3), dtype=np.float64)
    linear_velocity = np.tile(np.array([0.1, 0.0, 0.0]), (timestamps.size, 1))

    def process_all():
        resampled = resample_imu(
            imu,
            target_timestamps=targets,
            position=position,
            linear_velocity=linear_velocity,
        )
        return dead_reckon_trajectory(resampled)

    elapsed, trajectory = _time_best(process_all)
    assert trajectory["pose"].shape == (targets.size, 7)
    assert np.isfinite(trajectory["position"]).all()
    _print_benchmark("IMUOps.synthetic", timestamps.size, elapsed, unit="samples")


def test_benchmark_synthetic_odometry_operations():
    odometry = _synthetic_odometry_topic()
    timestamps = odometry["ts"]
    targets = np.linspace(timestamps[0], timestamps[-1], timestamps.size // 2)

    def process_all():
        resampled = resample_odometry(odometry, target_timestamps=targets)
        return dead_reckon_trajectory(resampled)

    elapsed, trajectory = _time_best(process_all)
    assert trajectory["pose"].shape == (targets.size, 7)
    assert np.isfinite(trajectory["linear_velocity"]).all()
    _print_benchmark("OdometryOps.synthetic", timestamps.size, elapsed, unit="samples")


def test_benchmark_synthetic_navsat_operations():
    navsat = _synthetic_navsat_topic()
    timestamps = navsat["ts"]
    targets = np.linspace(timestamps[0], timestamps[-1], timestamps.size // 2)

    def process_all():
        return resample_navsat(
            navsat,
            target_timestamps=targets,
            ref_lat=37.0,
            ref_lon=-122.0,
            ref_alt=10.0,
        )

    elapsed, trajectory = _time_best(process_all)
    assert trajectory["pose"].shape == (targets.size, 7)
    assert trajectory["navsat"].shape == (targets.size, 3)
    _print_benchmark("NavSatOps.synthetic", timestamps.size, elapsed, unit="samples")


def test_benchmark_synthetic_dem_operations():
    elevation = _synthetic_dem()

    def process_all():
        normals = terrain_normals(elevation, resolution=2.0)
        roughness = roughness_map(elevation, window_size=5)
        traversability = traversability_map(
            elevation,
            resolution=2.0,
            max_slope_degrees=35.0,
            max_roughness=1.0,
            roughness_window=5,
        )
        points = dem_to_point_cloud(elevation[::2, ::2], resolution=4.0)
        return normals, roughness, traversability, points

    elapsed, (normals, roughness, traversability, points) = _time_best(process_all)
    assert normals.shape == (*elevation.shape, 3)
    assert roughness.shape == traversability.shape == elevation.shape
    assert points.shape == ((SYNTHETIC_DEM_SIDE // 2) ** 2, 3)
    _print_benchmark("DEMOps.synthetic", elevation.size, elapsed, unit="cells")


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
