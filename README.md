<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/CyberPhysics/main/icon.png"
    height="70" align="right" alt="Array Data Engine logo" />
</a>

# Array Data Engine

Array Data Engine is a Python package for turning heterogeneous sensor and array data into a consistent NumPy-first stream. It can read image sequences, ROS bag files, ROS 2 `.db3` recordings, and DEM tiles, then keep recent context in memory or persist complete streams to TileDB.

The project is aimed at robotics and perception workflows where algorithms need synchronized windows of image, point cloud, navigation, odometry, IMU, and terrain-like array data.

## Features

- NumPy-oriented message dictionaries with `data`, `timestamp`, `topic`, and `name` fields.
- Source adapters for image folders, `.bag`, `.db3`, and DEM data.
- Rolling in-memory buffers for recent context windows.
- Optional TileDB-backed storage for datasets larger than memory.
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
| `.db3` | `DB3Source` | Bag topics | Supports image, point cloud, IMU, odometry, and navsat messages. |
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

Read a ROS bag or ROS 2 database:

```python
from ade.source import DataSources

bag_source = DataSources("/data/recording.bag")
db3_source = DataSources("/data/rosbag2/recording_0.db3")

for topic in db3_source.get_topics():
    print(topic, db3_source.get_count(topic))
```

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

Use `preload=True` to fill the entire window during construction, or `preload=0` to create the buffer without reading from the source.

```python
buffer = DataBuffer(source, buffer_depth=10, axis="images", preload=True)
```

Slice buffered data by timestamp:

```python
time_range = buffer.get_time_range(axis, start=12.0, end=12.5)
recent = buffer.get_last_seconds(axis, seconds=0.5)

images = time_range["data"]
timestamps = time_range["ts"]
```

Time ranges are inclusive and return the same `{"id", "ts", "data"}` shape for both in-memory and TileDB-backed buffers.

## TileDB Persistence

Set `use_db=True` to persist messages to a TileDB group. This is intended for full-source ingest and larger-than-memory datasets.

```python
from ade.buffer import DataBuffer
from ade.source import DataSources

axis = "/camera/image"
source = DataSources("/data/rosbag2/recording_0.db3")

buffer = DataBuffer(
    data_source=source,
    data_uri="/tmp/tiledb/my_dataset/",
    axis=axis,
    use_db=True,
)

buffer.load_data_db(axis)
print(buffer.get_group_uri())
```

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

Run the source adapter benchmarks with:

```bash
python -m pytest tests/test_source_benchmarks.py -q -s
```

Results below were measured on 2026-06-19 with Python 3.14.5 on arm64. Each result is the best of three runs. Bag, DB3, and DEM use mocked readers/network responses, so the benchmarks measure adapter overhead without requiring ROS bag files or Earthdata access.

| Source | Workload | Messages | Elapsed | Throughput | Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `ImgSource.messages` | temporary 64x64 PNG files read through OpenCV | 200 | 0.010868s | 18,402 msg/s | 54.3 us/msg |
| `BagSource.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000205s | 973,828 msg/s | 1.0 us/msg |
| `DB3Source.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000191s | 1,046,435 msg/s | 1.0 us/msg |
| `DEMSource.messages` | mocked Earthdata zip response with 32x32 HGT tiles | 4 | 0.000082s | 48,780 msg/s | 20.5 us/msg |

## Development

Run the test suite:

```bash
python -m pytest -q
```

Build and run the notebook container:

```bash
./build.sh
docker compose up jupyter
```

Notebook examples live in `notebooks/`.
