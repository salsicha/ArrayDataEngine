# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions 0.1.0 and 0.2.0 were repository development milestones. They were not
published to PyPI or tagged as GitHub releases. Version 0.3.0 is the planned
first registry release.

## 0.3.0 - Unreleased

### Added

- **Apache Arrow / Parquet storage backend** — the new default for
  persistent stores. Each topic is a directory of Parquet fragments with a
  `fixed_shape_tensor` data column plus timestamp/name/frame-id/spatial-AABB
  columns and a JSON manifest for resume. Staged writes and bounded-readahead
  streaming keep both ingest and scans within a fixed memory budget
  regardless of dataset size. Measured against the TileDB backend on a
  1.1 GB point-cloud workload: ~14x faster ingest, ~9x faster full scans,
  ~30x faster time-range reads, and ~45% less disk.
- `DataBuffer(backend=...)`: `"memory"`, `"arrow"`, or `"tiledb"`. When
  omitted, `use_db=True` selects Arrow for new stores and auto-detects
  existing TileDB stores so previously written datasets keep working.
- `DataBuffer(backend_options=...)` exposes the Arrow tuning knobs:
  `flush_bytes` (default 32 MB), `row_group_bytes` (16 MB), `compression`
  (`"zstd"`), `batch_readahead`/`fragment_readahead` (1), `use_threads`
  (False). Documented in the README and `arraydataengine.buffers.arrow_buffer`.
- `ade ingest --backend {arrow,tiledb}` (default arrow) and
  `SourcePipeline.to_buffer(backend=...)`; `persist_to_tiledb()` keeps
  writing TileDB as its name promises.
- `arrow` optional dependency extra (`pip install "arraydataengine[arrow]"`).

### Changed

- The canonical import package is now `arraydataengine`; the distribution no
  longer ships the conflicting `ade` import package. The `ade` CLI command is
  unchanged, and `ArrayDataEngine` remains as a convenience facade.
- Persisted Arrow topics are directly readable by Polars, DuckDB, pandas,
  and any other Parquet consumer.
- The Arrow backend rejects `buffer[i] = ...` in-place writes (immutable
  fragments); use `backend="tiledb"` when cell updates are needed.

## 0.2.0 (development milestone) - 2026-07-16

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

## 0.1.0 (development milestone) - 2026-07-16

Initial packaged development milestone; not published to PyPI.

### Added

- `DataSources` adapters for image globs, ROS1 `.bag`, ROS 2 `.db3` (single
  chunks and split rosbag2 directories), and SRTM DEM tiles.
- `DataBuffer` rolling in-memory buffers (NumPy backend) and persistent
  TileDB-backed storage with resumable ingest.
- `arraydataengine.ops`: NumPy-first operations for topics and datasets — lazy
  pipelines with map/filter/reduce, time/index/frame/spatial selection pushdown,
  chunked iteration, checkpoint/cancel/resume, point-cloud processing
  (voxel/outlier filters, ICP registration, ground segmentation), navigation
  math (IMU integration, ENU/NED, navsat conversion), DEM utilities
  (mosaic, hillshade, slope/aspect, traversability), image operations, and
  small ML dataset helpers.
- Point-cloud (native/embedded/HTML) and video visualizers.
- Two rounds of full-codebase review fixes (~60 correctness bugs) with a
  regression test suite.
