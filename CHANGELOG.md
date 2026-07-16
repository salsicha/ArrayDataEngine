# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/salsicha/ArrayDataEngine/releases/tag/v0.1.0
