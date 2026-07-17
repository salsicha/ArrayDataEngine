# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-16

### Added

- **`ade` command-line interface**: `ade info` (topics, counts, duration,
  optional per-topic rate/jitter/value stats), `ade topics`, `ade export`
  (topic to portable `.npz`), `ade ingest` (any source into a TileDB group,
  zero-padding ragged point-cloud topics), `ade viewer` (interactive HTML
  point-cloud viewer, no open3d needed), and `ade demo` (a full synthetic
  showcase: stitched point-cloud map, TUM trajectory, and NPZ export with no
  input files).
- **`SyntheticSource`**: deterministic multi-topic synthetic data (IMU,
  odometry, point clouds, navsat, images) simulating a circular trajectory
  with geometrically consistent scans — try the library with zero data files.
- **rosbridge JSON payload support**: `.db3` bags recorded through
  rosbridge/foxglove (JSON envelopes instead of CDR) now decode PoseStamped,
  PointCloud2 (base64 data), NavSatFix, and DepthAnythingCalibration
  messages; malformed payloads are skipped with a warning instead of making
  the whole bag unreadable.
- **MCAP support**: rosbag2 directories with MCAP storage work end-to-end,
  and bare `.mcap` file paths route to the bag reader.
- **Topic statistics**: `describe_topic` / `describe_dataset` /
  `format_describe` — counts, time range, rate, inter-message jitter, data
  schema, and value statistics.
- **Trajectory interchange**: `write_tum_trajectory` / `read_tum_trajectory`
  (TUM format, compatible with `evo` and RGB-D tooling) and
  `write_kitti_trajectory` (KITTI odometry format).
- **Topic persistence**: `save_topic_npz` / `load_topic_npz` — portable,
  pickle-free `.npz` round-trip for buffered topics with metadata.

## [0.1.0] - 2026-07-16

First public release on PyPI.

### Added

- `DataSources` adapters for image globs, ROS1 `.bag`, ROS 2 `.db3` (single
  chunks and split rosbag2 directories), and SRTM DEM tiles.
- `DataBuffer` rolling in-memory buffers (NumPy backend) and persistent
  TileDB-backed storage with resumable ingest.
- `ade.ops`: NumPy-first operations for topics and datasets — lazy pipelines
  with map/filter/reduce, time/index/frame/spatial selection pushdown,
  chunked iteration, checkpoint/cancel/resume, point-cloud processing
  (voxel/outlier filters, ICP registration, ground segmentation), navigation
  math (IMU integration, ENU/NED, navsat conversion), DEM utilities
  (mosaic, hillshade, slope/aspect, traversability), image operations, and
  small ML dataset helpers.
- Point-cloud (native/embedded/HTML) and video visualizers.
- Two rounds of full-codebase review fixes (~60 correctness bugs) with a
  regression test suite (123 tests).

[0.2.0]: https://github.com/salsicha/ArrayDataEngine/releases/tag/v0.2.0
[0.1.0]: https://github.com/salsicha/ArrayDataEngine/releases/tag/v0.1.0
