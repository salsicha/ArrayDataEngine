<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/CyberPhysics/main/icon.png"
    height="70" align="right" alt="Array Data Engine logo" />
</a>

# Array Data Engine

Array Data Engine is a Python package for turning heterogeneous sensor and array data into a consistent NumPy-first stream. It can read image sequences, ROS bag files, ROS 2 `.db3` recordings or split rosbag2 directories, and DEM tiles, then keep recent context in memory or persist complete streams to TileDB.

The project is aimed at robotics and perception workflows where algorithms need synchronized windows of image, point cloud, navigation, odometry, IMU, and terrain-like array data.

## Features

- NumPy-oriented message dictionaries with `data`, `timestamp`, `topic`, and `name` fields.
- Source adapters for image folders, `.bag`, `.db3`, split rosbag2 directories, and DEM data.
- Cached ROS topic metadata, so repeated topic and count lookups do not reopen the same bag.
- Rolling in-memory buffers for recent context windows.
- Optional TileDB-backed storage for datasets larger than memory, with timestamps stored in queryable sidecar arrays.
- Lazy dataset-level selection by topic, time, message index, frame id, geographic bounds, and spatial bounds.
- Lazy optional imports so lightweight workflows do not need the full ROS, DEM, or visualization stack.
- Notebook examples for demos, iterators, terrain, INS, MOT, and TileDB workflows.

## Installation

For editable local development:

```bash
python -m pip install -e .
```

Install optional feature groups as needed:

```bash
python -m pip install -e ".[dev,image,tiledb]"
python -m pip install -e ".[ros]"
python -m pip install -e ".[dem]"
python -m pip install -e ".[visualization]"
```

The full Docker/notebook environment still uses `requirements.txt`, which includes the heavier ROS, ML, notebook, and visualization dependencies.

## Supported Sources

| Input | Adapter | Topic(s) | Notes |
| --- | --- | --- | --- |
| `*.png`, `*.jpg`, `*.jpeg`, `*.tiff` | `ImgSource` | `images` | Reads sorted image paths from a glob. |
| `.bag` | `BagSource` | Bag topics | Supports image and point cloud messages today. |
| `.db3` file or rosbag2 directory | `DB3Source` | Bag topics | Supports single and split ROS 2 bags with image, point cloud, IMU, odometry, and navsat messages. |
| `"DEM"` | `DEMSource` | `images` | Downloads SRTM HGT tiles using Earthdata credentials. |

Each yielded message has this shape:

```python
{
    "data": np.ndarray,
    "timestamp": 123.456,
    "topic": "images",
    "name": "frame_000.png",
}
```

## Quick Start

Read an image sequence directly:

```python
from ade.source import DataSources

source = DataSources("/data/frames/*.png", period=0.1)

print(source.get_topics())        # ["images"]
print(source.get_count("images")) # number of matched frames

for message in source.get_message():
    image = message["data"]
    timestamp = message["timestamp"]
    print(timestamp, image.shape)
```

Read a ROS bag, ROS 2 database, or split rosbag2 directory:

```python
from ade.source import DataSources

bag_source = DataSources("/data/recording.bag")
ros2_source = DataSources("/data/rosbag2/split_recording/")

for topic in ros2_source.get_topics():
    print(topic, ros2_source.get_count(topic))
```

For split ROS 2 bags, pass the directory that contains `metadata.yaml` and the chunk files. Passing one `.db3` chunk still works; the reader uses the containing directory.

## Rolling Buffers

`DataBuffer` keeps a recent window for each topic and exposes NumPy-like indexing on the selected axis topic.

```python
from ade.buffer import DataBuffer
from ade.source import DataSources

axis = "images"
source = DataSources("/data/frames/*.png", period=0.1)

buffer = DataBuffer(
    data_source=source,
    buffer_depth=10,
    axis=axis,
    use_db=False,
)

# The constructor preloads one axis message by default.
buffer.roll_buffer(axis)
latest_image = buffer[-1]

window = buffer.get_buffer()
image_window = window[axis]["data"]
timestamps = window[axis]["ts"]
```

Use `copy=False` when you want to avoid extra in-memory copies during read-only inspection:

```python
window = buffer.get_buffer(copy=False)
```

Use `preload=True` to fill the entire window during construction, or `preload=0` to create the buffer without reading from the source.

```python
buffer = DataBuffer(source, buffer_depth=10, axis="images", preload=True)
```

Slice buffered data by timestamp:

```python
time_range = buffer.get_time_range(axis, start=12.0, end=12.5)
index_range = buffer.get_index_range(axis, start=100, stop=200, step=2)
recent = buffer.get_last_seconds(axis, seconds=0.5)

images = time_range["data"]
timestamps = time_range["ts"]
```

Time ranges are inclusive and return the same `{"id", "ts", "data"}` shape for both in-memory and TileDB-backed buffers.

## Array Operations

The `ade.ops` package provides NumPy-first helpers for common robotics data operations. They work on topic arrays returned by `DataBuffer.get_buffer()`, `get_time_range()`, and `get_last_seconds()`.

```python
from ade.ops import map_topic, select_time_range, voxel_downsample

window = buffer.get_buffer()
images = select_time_range(window["images"], start=12.0, end=12.5)
normalized = map_topic(images, lambda frame: frame.astype("float32") / 255.0)

points = window["/points"]["data"][-1]
reduced_points = voxel_downsample(points, voxel_size=0.25)
```

SE(3) coordinate-frame helpers work across common robotics arrays:

```python
from ade.ops import apply_transform, transform_navsat, transform_odometry

points_in_map = apply_transform(points_in_lidar, lidar_to_map)
odom_in_map = transform_odometry(odom_message_array, odom_to_map)
gps_in_map_frame = transform_navsat(gps_samples, enu_transform, ref_lat=37.0, ref_lon=-122.0, ref_alt=10.0)
```

Use `FrameGraph` when transforms need to be composed by frame name. Static transforms are used directly; time-varying transforms are interpolated by timestamp, including rotation SLERP.

```python
from ade.ops import FrameGraph

frames = FrameGraph()
frames.add_static_transform("base_link", "odom", base_to_odom)
frames.add_time_varying_transform(
    "lidar",
    "base_link",
    timestamps=lidar_tf_timestamps,
    transforms=lidar_to_base_samples,
)

points_in_odom = frames.transform_points(points_in_lidar, "lidar", "odom", timestamp=12.5)
```

Projection helpers connect point clouds, depth images, RGB images, DEM grids, and camera frames with pinhole intrinsics.

```python
from ade.ops import colorize_points, points_to_depth_image, project_dem_to_image, rgbd_to_points

rgbd_cloud = rgbd_to_points(depth_image, rgb_image, fx=525.0, fy=525.0, cx=319.5, cy=239.5)
colored_lidar = colorize_points(points_in_camera, rgb_image, fx=525.0, fy=525.0, cx=319.5, cy=239.5)
rendered_depth = points_to_depth_image(points_in_camera, image_shape=rgb_image.shape[:2], fx=525.0, fy=525.0, cx=319.5, cy=239.5)
dem_pixels, dem_mask = project_dem_to_image(elevation, fx=525.0, fy=525.0, cx=319.5, cy=239.5, image_shape=rgb_image.shape[:2])
```

IMU, odometry, and NavSat arrays can be normalized into one trajectory representation with `pose` as `[x, y, z, qx, qy, qz, qw]` and `trajectory` as pose plus linear and angular velocity.

```python
from ade.ops import imu_to_trajectory, navsat_to_trajectory, odometry_to_trajectory

imu_traj = imu_to_trajectory(window["/imu"])
odom_traj = odometry_to_trajectory(window["/odom"])
gps_traj = navsat_to_trajectory(window["/gps"], ref_lat=37.0, ref_lon=-122.0, ref_alt=10.0)
```

For large datasets, use the lazy topic pipeline. It records operations and only executes them when chunks, rows, reductions, windows, or explicit collection are requested.

```python
import numpy as np

pipeline = (
    buffer.topic("images")
    .time_range(12.0, 20.0)
    .index_range(0, 1_000)
    .map(lambda frame: frame.astype(np.float32) / 255.0)
    .filter(lambda frame, ts, name: frame.mean() > 0.05)
)

for chunk in pipeline.iter_chunks(chunk_size=32):
    process(chunk.data, chunk.timestamps)

summary = pipeline.reduce(lambda acc, frame: acc + frame.mean(), initial=0.0)
small_result = pipeline.collect(chunk_size=32, max_rows=1_000)
```

`collect()` is intentionally guarded. By default it refuses results above 512 MiB; pass `max_rows`, `max_bytes`, `out=`, or `allow_large=True` when materializing a large bounded result is intentional.

Use `DataBuffer.dataset()` when the query spans multiple topics. Topic, timestamp, and message-index constraints are applied before later row filters. On TileDB-backed buffers, leading timestamp and index ranges are pushed down before payload arrays are read.

```python
selection = (
    buffer.dataset()
    .select_topics("/camera/image", "/points", "/gps")
    .time_range(12.0, 20.0)
    .index_range(0, 10_000)
    .frame_id("map")
)

for topic, chunk in selection.iter_chunks(chunk_size=64):
    process(topic, chunk.data, chunk.timestamps)
```

Geographic bounds expect latitude/longitude values by default in columns `(0, 1)`, matching NavSat arrays shaped like `[latitude, longitude, altitude]`. Spatial bounds work with XYZ vectors or point arrays and keep a message when any point falls inside the bounds.

```python
gps_samples = (
    buffer.dataset(["/gps"])
    .geographic_bounds(36.9, -122.3, 37.8, -121.7)
    .collect(max_rows=50_000)
)

point_chunks = (
    buffer.dataset(["/points"])
    .spatial_bounds(min_bound=[-20, -20, -2], max_bound=[20, 20, 5])
    .iter_chunks(chunk_size=16)
)
```

Topic alignment helpers cover exact timestamp joins, nearest-neighbor joins, bounded-tolerance joins, fixed-rate resampling, and rolling window joins.

```python
from ade.ops import align_topic, resample_topic, rolling_window_join

imu_at_image_times = align_topic(images, imu, mode="bounded", tolerance=0.02)
gps_10hz = resample_topic(gps, rate_hz=10.0)
recent_points = rolling_window_join(images, points, seconds=0.25)
```

`TopicView` is still available when the data is already in memory and you want eager, metadata-aware operations. It keeps message ids, timestamps, data, topic name, frame id, source URI, dtype, shape, and time bounds together while exposing the same operations as methods.

```python
from ade.ops import topic_view

view = topic_view(window["images"], topic="images")
normalized = view.map(lambda frame: frame.astype(np.float32) / 255.0).as_dict()
```

`DataBuffer` also keeps eager convenience wrappers for smaller results and compatibility:

```python
normalized = buffer.map_topic("images", lambda frame: frame.astype("float32") / 255.0)
normalized = buffer.map_topic("images", lambda frame: frame.astype("float32") / 255.0, chunk_size=16)
recent_windows = list(buffer.window_topic("images", size=5))
```

Initial operation coverage includes topic selection, map/filter/reduce/window helpers, nearest-time alignment, SE(3) transforms, frame graphs, camera projection helpers, bounds cropping, point cloud downsampling/search/normals/outlier filters/clustering/plane segmentation, image/depth utilities, navsat ENU conversion, quaternion interpolation, trajectory speed, and DEM/raster helpers.

## TileDB Persistence

Set `use_db=True` to persist messages to a TileDB group. This is intended for full-source ingest and larger-than-memory datasets.

```python
from ade.buffer import DataBuffer
from ade.source import DataSources

axis = "/camera/image"
source = DataSources("/data/rosbag2/split_recording/")

with DataBuffer(
    data_source=source,
    data_uri="/tmp/tiledb/my_dataset/",
    axis=axis,
    use_db=True,
) as buffer:
    buffer.load_data_db(axis)
    print(buffer.get_group_uri())
```

Using `DataBuffer` as a context manager closes TileDB arrays cleanly and marks completed topic arrays as closed.

TileDB-backed topics keep timestamps, message names, frame ids, and per-message spatial bounds in sidecar arrays. Time, index, frame-id, and spatial-bounds constraints from lazy topic or dataset queries are pushed down before data chunks are read, so pipelines avoid loading unselected message payloads.

Existing TileDB groups can be reopened without the original source:

```python
reopened = DataBuffer(
    data_source=None,
    data_uri="/tmp/tiledb/my_dataset/",
    axis=axis,
    use_db=True,
)

for chunk in reopened.topic(axis).time_range(12.0, 20.0).iter_chunks(chunk_size=64):
    process(chunk.data)
```

Interrupted ingests can be resumed by constructing a new `DataBuffer` with the same source and `data_uri`. Stored per-topic counts are loaded from TileDB metadata, previously written messages are skipped during source replay, and remaining messages are appended.

## DEM Tiles

DEM sources use NASA Earthdata credentials from environment variables:

```bash
export earthdata_username="..."
export earthdata_password="..."
```

Then request north/west tile ranges:

```python
from ade.source import DataSources

north = [37, 39]
west = [122, 124]

source = DataSources("DEM", bounds=[north, west])

for tile in source.get_message():
    print(tile["name"], tile["data"].shape)
```

## Benchmarks

Run the source adapter and lazy pipeline benchmarks with:

```bash
python -m pytest tests/test_source_benchmarks.py -q -s
```

Results below were measured on 2026-06-20 with Python 3.14.5 on arm64. Each result is the best of three runs. Bag, DB3, and DEM use mocked readers/network responses, so those rows measure adapter overhead without requiring ROS bag files or Earthdata access.

| Source | Workload | Messages | Elapsed | Throughput | Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `ImgSource.messages` | temporary 64x64 PNG files read through OpenCV | 200 | 0.008944s | 22,362 msg/s | 44.7 us/msg |
| `BagSource.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000160s | 1,253,596 msg/s | 0.8 us/msg |
| `DB3Source.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000160s | 1,249,024 msg/s | 0.8 us/msg |
| `DEMSource.messages` | mocked Earthdata zip response with 32x32 HGT tiles | 4 | 0.000073s | 54,795 msg/s | 18.2 us/msg |
| `TopicPipeline.iter_chunks` | lazy in-memory time/index pushdown and row map over 50k synthetic samples | 10,000 | 0.019936s | 501,609 msg/s | 2.0 us/msg |
| `TileDB.TopicPipeline.time_range` | lazy TileDB time-range pushdown and row map over a temp persisted topic | 100 | 0.130066s | 769 msg/s | 1300.7 us/msg |

## Development

Run the test suite:

```bash
python -m pytest -q
```

Release packaging and Python registry publishing steps are tracked in [TODO.md](TODO.md).

Build and run the notebook container:

```bash
./build.sh
docker compose up jupyter
```

Notebook examples live in `notebooks/`.
