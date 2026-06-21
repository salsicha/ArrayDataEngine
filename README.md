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
from ade.ops import map_topic, random_downsample, select_time_range, voxel_downsample

window = buffer.get_buffer()
images = select_time_range(window["images"], start=12.0, end=12.5)
normalized = map_topic(images, lambda frame: frame.astype("float32") / 255.0)

points = window["/points"]["data"][-1]
reduced_points = voxel_downsample(points, voxel_size=0.25)
preview_points = random_downsample(points, count=10_000, seed=42)
```

Image and depth sequences can be resized, cropped, padded, normalized, color-converted, dtype-converted, backprojected, converted to normals, motion-aligned, and fused as NumPy stacks.

```python
import numpy as np

from ade.ops import align_images, convert_color, convert_image_dtype, crop_images, depth_to_normals, frame_to_frame_optical_flow, fuse_rgbd_frames, image_gradients, image_mask, image_pyramid, iter_rgbd_frame_points, local_statistics, motion_compensated_rolling_windows, normalize_images, open_mask, resize_images

frames = window["images"]["data"]
depth_frames = window["depth"]["data"]
small = resize_images(frames, shape=(240, 320))
roi = crop_images(small, row_start=40, row_stop=200, col_start=80, col_stop=260)
float_roi = normalize_images(convert_image_dtype(roi, np.float32), per_image=True)
gray_roi = convert_color(roi, "rgb_to_gray")
foreground = open_mask(image_mask(gray_roi, min_value=32), size=3)
edges = image_gradients(gray_roi, method="sobel")
pyramid = image_pyramid(gray_roi[0], levels=4)
local = local_statistics(gray_roi, size=5, statistics=("mean", "std"))
flow = frame_to_frame_optical_flow(gray_roi, max_shift=8)
aligned_frames = align_images(gray_roi, max_shift=8)
for aligned_window in motion_compensated_rolling_windows(gray_roi, window_size=5, max_shift=8):
    process(aligned_window)
normals = depth_to_normals(depth_frames[0], fx=525.0, fy=525.0, cx=319.5, cy=239.5)
for rgbd_chunk in iter_rgbd_frame_points(depth_frames, frames, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
    process(rgbd_chunk)
rgbd_cloud = fuse_rgbd_frames(depth_frames[:10], frames[:10], fx=525.0, fy=525.0, cx=319.5, cy=239.5)
```

Point-cloud downsampling includes voxel-grid averaging, every-k uniform sampling, seeded random sampling by count or ratio, and farthest-point sampling.

```python
from ade.ops import connected_components, curvature_descriptors, farthest_point_downsample, hybrid_search, multi_scale_icp, nearest_neighbor_distance_stats, segment_ground, to_open3d_point_cloud, uniform_downsample

uniform_points = uniform_downsample(points, every_k=4)
keypoints = farthest_point_downsample(points, count=2_048)
shape_features = curvature_descriptors(points, k=16)
spacing = nearest_neighbor_distance_stats(points, k=4)
distances, indices, counts = hybrid_search(points, query_points, radius=0.5, max_neighbors=32)
components = connected_components(points, radius=0.5, min_component_size=20)
ground, obstacles, ground_mask = segment_ground(points, distance_threshold=0.15)
registration = multi_scale_icp(scan_a, scan_b, voxel_sizes=(1.0, 0.5, 0.25))
open3d_cloud = to_open3d_point_cloud(points, color_columns=(3, 4, 5))
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
from ade.ops import backproject_pixels, camera_model, colorize_points, depth_to_point_grid, distort_pixels, points_to_depth_image, project_camera_points, project_dem_to_image, rectify_image, rgbd_to_points, scale_camera_matrix

camera = camera_model(fx=525.0, fy=525.0, cx=319.5, cy=239.5, image_shape=rgb_image.shape[:2], distortion=distortion_coeffs)
small_camera_matrix = scale_camera_matrix(camera.camera_matrix, scale_x=0.5)
rectified_rgb = rectify_image(rgb_image, camera.camera_matrix, distortion=camera.distortion)
distorted_pixels = distort_pixels(raw_pixels, camera.camera_matrix, camera.distortion)
camera_points = backproject_pixels(distorted_pixels, depth_values, camera.camera_matrix, distortion=camera.distortion)
projected_pixels, visible = project_camera_points(points_in_camera, camera)
organized_depth_points = depth_to_point_grid(depth_image, fx=525.0, fy=525.0, cx=319.5, cy=239.5)
rgbd_cloud = rgbd_to_points(depth_image, rgb_image, fx=525.0, fy=525.0, cx=319.5, cy=239.5)
colored_lidar = colorize_points(points_in_camera, rgb_image, fx=525.0, fy=525.0, cx=319.5, cy=239.5)
rendered_depth = points_to_depth_image(points_in_camera, image_shape=rgb_image.shape[:2], fx=525.0, fy=525.0, cx=319.5, cy=239.5)
dem_pixels, dem_mask = project_dem_to_image(elevation, fx=525.0, fy=525.0, cx=319.5, cy=239.5, image_shape=rgb_image.shape[:2])
```

Crop/select helpers cover row masks, axis-aligned bounds, oriented 3D bounds, and geographic bounding boxes.

```python
from ade.ops import crop_bounds, crop_geographic_bounds, crop_oriented_bounds, select_mask

valid_points = select_mask(points, valid_mask)
nearby_points = crop_bounds(points, min_bound=[-10, -10, -2], max_bound=[10, 10, 3])
vehicle_box = crop_oriented_bounds(points, center=pose_xyz, extent=[8.0, 4.0, 3.0], rotation=pose_rotation)
gps_window = crop_geographic_bounds(gps_samples, min_lat=36.9, min_lon=-122.3, max_lat=37.8, max_lon=-121.7)
```

IMU, odometry, and NavSat arrays can be normalized into one trajectory representation with `pose` as `[x, y, z, qx, qy, qz, qw]` and `trajectory` as pose plus linear and angular velocity. Resampling uses SLERP for orientation and linear interpolation for position, velocity, acceleration, and covariance fields. Quaternion/Euler conversion, gravity compensation, bias correction, WGS84-to-local ENU/NED conversions, trajectory smoothing, differentiation, integration, dead reckoning, covariance propagation, and quality/status masks cover common navigation preprocessing.

```python
from ade.ops import (
    compensate_imu_gravity,
    correct_imu_bias,
    add_trajectory_quality_mask,
    dead_reckon_trajectory,
    differentiate_trajectory,
    euler_to_quaternion,
    imu_to_trajectory,
    integrate_trajectory,
    local_to_navsat,
    mask_trajectory,
    navsat_to_local,
    navsat_to_trajectory,
    odometry_to_trajectory,
    quaternion_to_euler,
    resample_imu,
    resample_navsat,
    resample_odometry,
    resample_trajectory,
    smooth_trajectory,
    propagate_trajectory_covariance,
)

corrected_imu = correct_imu_bias(compensate_imu_gravity(window["/imu"]), sample_slice=slice(0, 50))
imu_traj = imu_to_trajectory(corrected_imu)
odom_traj = odometry_to_trajectory(window["/odom"])
gps_traj = navsat_to_trajectory(window["/gps"], ref_lat=37.0, ref_lon=-122.0, ref_alt=10.0)
gps_ned, gps_ref = navsat_to_local(window["/gps"], frame="ned", return_reference=True)
gps_roundtrip = local_to_navsat(gps_ned, gps_ref["lat"], gps_ref["lon"], gps_ref["alt"], frame=gps_ref["frame"])
orientation = euler_to_quaternion(roll=0.0, pitch=0.0, yaw=1.57)
roll_pitch_yaw = quaternion_to_euler(odom_traj["orientation"])

odom_50hz = resample_trajectory(odom_traj, period=0.02)
imu_at_image_times = resample_imu(window["/imu"], target_timestamps=image_timestamps)
odom_at_image_times = resample_odometry(window["/odom"], target_timestamps=image_timestamps)
gps_at_image_times = resample_navsat(window["/gps"], target_timestamps=image_timestamps, ref_lat=37.0, ref_lon=-122.0, ref_alt=10.0)

smoothed_odom = smooth_trajectory(odom_50hz, window_size=5)
derived_odom = differentiate_trajectory(smoothed_odom)
integrated_odom = integrate_trajectory(derived_odom, initial_position=odom_traj["position"][0])
dead_reckoned = dead_reckon_trajectory(odom_at_image_times, initial_position=odom_traj["position"][0])
covar_odom = propagate_trajectory_covariance(dead_reckoned, process_noise={"position": [0.02, 0.02, 0.05]})
quality_odom = add_trajectory_quality_mask(covar_odom, covariance_limits={"position": [1.0, 1.0, 2.0]})
trusted_odom = mask_trajectory(quality_odom, quality_odom["quality_mask"], drop=True)
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

Independent topic chunks can be processed concurrently with `max_workers`. The parallel path preserves output order and requires stateful `index_range()` filters to be placed before row maps/filters so they can be pushed down before chunks are scheduled.

```python
for chunk in pipeline.iter_chunks(chunk_size=32, max_workers=4):
    process(chunk.data, chunk.timestamps)
```

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

Use `topic_workers` when collecting independent topics concurrently. `max_workers` still controls per-topic chunk workers:

```python
selected = selection.collect(chunk_size=64, max_workers=2, topic_workers=3, max_rows=30_000)
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

Use `source_pipeline()` when you want operations to run while messages stream from a `DataSources` object, before full topics are loaded. The same pipeline can write to an in-memory `DataBuffer` or persist directly to TileDB. Long-running source and topic pipelines accept progress callbacks, cancellation tokens, and mutable checkpoint dictionaries that can be saved and reused to resume from the last processed row.

```python
import json

from ade.ops import CancellationToken, PipelineCancelled, source_pipeline, voxel_downsample
from ade.source import DataSources

source = DataSources("/data/rosbag2/split_recording/")
checkpoint = {}
tiledb_checkpoint = {}
cancel_token = CancellationToken()

def report(progress):
    print(progress.processed, progress.emitted, progress.topic)

pipeline = (
    source_pipeline(source)
    .select_topics("/points")
    .time_range(12.0, 20.0)
    .map(lambda msg: {**msg, "data": voxel_downsample(msg["data"], voxel_size=0.1)})
    .filter(lambda msg: msg["data"].shape[0] > 0)
)

try:
    rolling_buffer = pipeline.to_buffer(
        buffer_depth=128,
        progress_callback=report,
        cancel_token=cancel_token,
        checkpoint=checkpoint,
        progress_interval=100,
    )
except PipelineCancelled:
    with open("filtered_points.checkpoint.json", "w") as checkpoint_file:
        json.dump(checkpoint, checkpoint_file)

tiledb_buffer = pipeline.persist_to_tiledb("/tmp/tiledb/filtered_points/", checkpoint=tiledb_checkpoint)
```

TileDB persistence uses `source.get_count(topic)` to size the destination arrays, then records the actual filtered message count as metadata.

Initial operation coverage includes source-level streaming pipelines, progress reporting, cancellation, resumable checkpoints, topic selection, map/filter/reduce/window helpers, nearest-time alignment, SE(3) transforms, frame graphs, camera projection helpers, camera intrinsics/distortion/rectification utilities, mask and bounds cropping, point cloud downsampling/sampling/KNN-radius-hybrid search/normals/covariance descriptors/distance stats/outlier filters/clustering/connected components/plane and ground segmentation/ICP registration/Open3D adapters, image/depth sequence transforms, morphology, gradients, pyramids, local image statistics, frame-to-frame optical flow, image alignment, motion-compensated rolling windows, valid-depth masks, depth backprojection, depth normals, RGB-D fusion, navsat ENU/NED conversion, quaternion/Euler conversion, gravity compensation, bias correction, trajectory resampling/smoothing/differentiation/integration/dead reckoning/covariance propagation/quality masks, trajectory speed, and DEM/raster helpers for terrain gradients, normals, roughness, traversability, local patches, point clouds, and meshes.

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

DEM helper functions operate on NumPy windows, so they can be used on individual tiles, cropped patches, or lazy pipeline chunks:

```python
from ade.ops import (
    dem_to_mesh,
    dem_to_point_cloud,
    roughness_map,
    sample_elevation,
    terrain_normals,
    terrain_patch,
    traversability_map,
)

tile = next(source.get_message())
elevation = tile["data"].astype("float64")

normals = terrain_normals(elevation, resolution=30.0)
roughness = roughness_map(elevation, window_size=5)
traversability = traversability_map(
    elevation,
    resolution=30.0,
    max_slope_degrees=25.0,
    max_roughness=1.0,
)

vehicle_patch = terrain_patch(elevation, center=(150.0, 240.0), size=(64, 64), resolution=30.0)
vehicle_elevation = sample_elevation(elevation, x=[150.0], y=[240.0], resolution=30.0)
terrain_points = dem_to_point_cloud(vehicle_patch, resolution=30.0)
terrain_mesh = dem_to_mesh(vehicle_patch, resolution=30.0)
```

## Benchmarks

Run the source adapter, synthetic operation, lazy pipeline, and TileDB benchmarks with:

```bash
python -m pytest tests/test_source_benchmarks.py -q -s
```

Results below were measured on 2026-06-21 with Python 3.14.5 on arm64. Each result is the best of three runs. Bag, DB3, and DEM use mocked readers/network responses, so those rows measure adapter overhead without requiring ROS bag files or Earthdata access.

| Benchmark | Workload | Items | Elapsed | Throughput | Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `ImgSource.messages` | temporary 64x64 PNG files read through OpenCV | 200 | 0.006395s | 31,274 items/s | 32.0 us/item |
| `BagSource.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000125s | 1,600,000 items/s | 0.6 us/item |
| `DB3Source.messages` | mocked `AnyReader` and image sensor conversion | 200 | 0.000124s | 1,609,657 items/s | 0.6 us/item |
| `DEMSource.messages` | mocked Earthdata zip response with 32x32 HGT tiles | 4 | 0.000048s | 82,616 items/s | 12.1 us/item |
| `ImageOps.synthetic` | resize, normalize, and grayscale 96x96 RGB image frames | 64 | 0.001120s | 57,153 frames/s | 17.5 us/frame |
| `PointCloudOps.synthetic` | voxel downsample, KNN search, and normal estimation on XYZ+intensity points | 1,024 | 0.010514s | 97,395 points/s | 10.3 us/point |
| `IMUOps.synthetic` | resample synthetic IMU samples and dead-reckon the trajectory | 2,000 | 0.027607s | 72,445 samples/s | 13.8 us/sample |
| `OdometryOps.synthetic` | resample synthetic odometry samples and dead-reckon the trajectory | 2,000 | 0.027660s | 72,306 samples/s | 13.8 us/sample |
| `NavSatOps.synthetic` | convert and resample synthetic WGS84 NavSat samples into local trajectory arrays | 2,000 | 0.012711s | 157,349 samples/s | 6.4 us/sample |
| `DEMOps.synthetic` | terrain normals, roughness, traversability, and point-cloud conversion over a DEM grid | 16,384 | 0.006353s | 2,579,024 cells/s | 0.4 us/cell |
| `TopicPipeline.iter_chunks` | lazy in-memory time/index pushdown and row map over 50k synthetic samples | 10,000 | 0.031245s | 320,053 items/s | 3.1 us/item |
| `TileDB.TopicPipeline.time_range` | lazy TileDB time-range pushdown and row map over a temp persisted topic | 100 | 0.192457s | 520 items/s | 1924.6 us/item |

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
