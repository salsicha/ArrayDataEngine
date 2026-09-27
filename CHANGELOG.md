# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions 0.1.0 and 0.2.0 were repository development milestones. They were not
published to PyPI or tagged as GitHub releases. Version 0.3.0 was the first
registry release.

## Unreleased

### Added

- `max_points` option on `DataSources`, `BagSource`, and `DB3Source`; point-cloud
  messages carry `point_count`, the number of real points before zero padding.
- `sensor_msgs/CompressedImage` decoding for CDR and rosbridge JSON payloads
  (requires OpenCV), and `get_topic_types()` on ROS sources.
- `DataBuffer.close_completed()`, `DEMSource(allow_insecure_http=...)`, and
  `.tif` image inputs.
- `north_up` on every DEM helper that maps rows to `y`; `resolution` accepts
  `(dx, dy)`; `mosaic_dem_tiles` merges shared SRTM edges and masks voids.
- `twist_frame` on `transform_odometry` and `odometry_to_trajectory`,
  `scale_camera_matrix(half_pixel_centers=...)`, `distortion=` on
  `colorize_points`, `sample_image_at_points`, `points_to_depth_image`, and
  `rgbd_to_points`, `refine=` on plane/ground segmentation, `channel_axis=` on
  `image_gradients` and `augment_image`, and `from_initial=` on
  `propagate_trajectory_covariance`.
- `PipelineCancelled.partial` holds the rows a cancelled `collect()` gathered.
- `describe_topic` reports `non_monotonic_count` and `nonfinite_ts_count`.
- `ade --debug` (or `ADE_DEBUG=1`) shows tracebacks; `ade export` stores
  per-message `lengths` for zero-padded ragged topics.

### Changed

- Every ROS read path returns the same messages: short type names, log-time
  fallback for zero stamps, and point clouds with non-finite rows dropped and
  zero-padded to `(max_points, 3)` in the field dtype (float64 stays float64).
  Oversized or undecodable messages are skipped with a warning instead of
  ending the stream.
- `get_topics()`/`get_count()` list only topics whose type can be decoded.
- Unknown image encodings raise instead of returning raw bytes; OpenCV-style
  (`32FC1`, `16SC1`, ...), Bayer, and YUV422 encodings are decoded.
- The DepthAnythingCalibration body `frame_id` is exposed as
  `calibration_frame_id` instead of overwriting the header frame.
- `transform_odometry` leaves the body-frame twist unchanged by default, and
  `odometry_to_trajectory` rotates it into the parent frame, matching
  `nav_msgs/Odometry` semantics.
- NavSat conversions use exact WGS84 geodetic/ECEF/ENU math.
- `verify_loop_closures` always gates registration (2×`voxel_size`, or 3× the
  median point spacing when no distance is given); `inlier_rmse` is
  point-to-point for every ICP method.
- Undistortion iterates to a tolerance and returns NaN outside the lens
  model's domain.
- DEM aspect is a compass bearing in `[0, 2π)` with NaN for flat cells, and
  `sample_grid` returns NaN outside the grid instead of clamping.
- Memory and TileDB buffers reject messages whose shape or dtype differs from
  the topic's instead of broadcasting or casting them.
- TileDB batches writes, compresses attributes with Zstd, and consolidates with
  bounded buffers on close: ingesting the example bag takes 6.5 s and 119 MB
  instead of ~100 s and 900 MB.
- Pipelines without row operations pass whole chunks through and copy data
  once; `align_nearest`, quaternion interpolation, neighbour searches (SciPy
  `cKDTree` when available, memory-bounded chunks otherwise), normals, and
  DEM meshing are vectorized.
- `ade ingest --backend` auto-detects an existing store (new stores still
  default to Arrow) and refuses non-store directories; `ade demo` sizes its
  buffer to the data.
- Removed unused packages from the `ros` and `ml` extras; the `visualization`
  extra now includes `opencv-python`.

### Removed

- Unused `TopicSource` and `ros2_numpy`-based sensor code paths.

### Fixed

- Arrow stores publish fragments and manifests atomically, record committed
  fragments, and reconcile legacy fragment counts when resuming interrupted ingests.
- Arrow reads preserve per-message frame IDs through maps and collection, and
  frame filters exclude messages with missing frames even in older stores.
- Arrow appends reject dtype changes before staging incompatible messages.
- ROS1 and rosbag2 point-cloud readers use the built-in NumPy decoder instead
  of requiring the undeclared `ros2_numpy` dependency.
- Distinct escaped storage paths and TileDB group member names prevent topic
  collisions; existing stores retain their original locations when reopened.
- Persistent source pipelines resume from checkpoints without skipping rows twice,
  including filtered, multi-topic ingests.
- Standalone SQLite ROS bags use the existing sensor converters for standard CDR
  messages unsupported by the lightweight decoder, including IMU, image,
  odometry, and NavSatFix messages (requires the `ros` extra).
- Arrow appends copy staged arrays so reusing an input array cannot change
  messages waiting to be flushed.
- Source pipelines can append into empty Arrow buffers without invoking TileDB
  initialization.
- In-memory buffers retain each message's frame ID through ring wrapping,
  views, maps, filters, windows, and sequential or parallel topic collection.
- Leaving a `DataBuffer` context only marks complete topics closed, and Arrow
  resumes partial topics that older versions marked closed.
- Resuming a persistent pipeline checks the checkpoint against the stored
  per-topic counts, so rows lost before a flush are replayed instead of skipped.
- Checkpoints stay in step with delivered rows after errors, early stops,
  cancellation, and parallel execution.
- Arrow reads no longer flush or rewrite manifests, stale handles refuse to
  overwrite another writer's commits, unsupported payloads are rejected on
  append, `close()` attempts every topic, single-row reads load only the
  needed row groups, and the directory is fsynced after publishing.
- TileDB returns empty results for queries matching no rows, handles
  out-of-order timestamps, keeps per-message frame IDs, indexes subscripts by
  message count, opens in existing empty directories, accepts direct appends
  and callable sources, and never deletes written arrays when reopening.
- Spatial pushdown handles organized point clouds; the memory backend returns
  empty results for topics not yet received and stores non-ASCII names.
- Buffering the example ROS1 bag's variable-size point clouds no longer fails.
- rosbag2 bags without embedded message definitions (Humble and older), lone
  SQLite `.bag` files, bare `.mcap` files, and `.db3` directories without
  `metadata.yaml` open; a mistyped chunk path raises instead of reading a
  neighbouring bag.
- Earthdata credentials are sent only over HTTPS to the login or tile host;
  DEM tiles download once, cache writes are atomic, and southern/eastern tile
  names are correct.
- XCDR2/PL_CDR payloads and malformed PointCloud2 layouts raise clear errors;
  ROS1 rosbridge stamps and whitespace-prefixed JSON are handled.
- Callables receive `ts`/`id` only when they require them, so
  `map(np.linalg.norm)` and functions with optional parameters work.
- Pipelines accept scalar, string, object, and differently shaped map outputs;
  `DatasetQuery.collect()` keeps `frame_ids`; progress callbacks fire at every
  interval in parallel mode.
- Alignment, resampling, and rolling joins handle unsorted or NaN timestamps;
  linear resampling honours `tolerance` and keeps the last grid point.
- `save_topic_npz` returns the written `.npz` path, stores ragged and string
  data without pickle, and round-trips frame IDs.
- Phase-correlation shift estimation applies a Hann window and works on
  natural, non-periodic images.
- `navsat_to_enu` wraps longitudes across ±180°; default NavSat references skip
  NaN and no-fix samples; zero quaternions become NaN instead of raising; nav
  helpers accept `DataBuffer.get_buffer()` topic arrays.
- `point_to_plane_icp` converges for clouds far from the origin;
  `multi_scale_icp` reports gated final metrics and actual iterations;
  nearest-neighbour search no longer needs O(Q·N) memory; a single NaN point no
  longer breaks outlier filtering, ICP, downsampling, or normals.
- Distorted projection no longer folds far off-axis points back into the image;
  `CameraModel` supports `==` and hashing; `voxel_downsample` keeps fractional
  means for integer input.
- `convert_image_dtype` preserves zero and round-trips signed integers;
  `normalize_image` ignores NaN; `dem_to_mesh` faces point up;
  `traversability_map` stays in `[0, 1]`.
- Multi-worker PyTorch iterable datasets no longer duplicate samples; time
  splits sort by timestamp; NaN groups are kept; collation schemas and integer
  dtypes are stable.
- `ade info --messages` handles variable-size clouds and shows frame IDs; `ade`
  rejects non-positive `--stride`/`--limit`/`--duration` and reports missing
  files, unsupported message types, and missing extras as one-line errors.
- The HTML point-cloud viewer drops NaN/Inf points and fits the view to the
  data; `Visualizer("image").show()` no longer requires an argument or writes
  `filename.gif` into the working directory; colormaps load through
  `matplotlib.colormaps`.

### Security

- The Docker image fetches the ROS apt key over verified TLS and no longer
  ships a fixed or empty Jupyter token.
- Removed the unrelated PyPI `tk` package from `requirements.txt`.

## 0.3.1 - 2026-08-14

### Fixed

- Restored the public facade on the canonical `arraydataengine` package, so
  `import arraydataengine as ADE` exposes `DataBuffer`, `DataSources`,
  `Visualizer`, the `ops` module, and the documented top-level operations.

## 0.3.0 - 2026-08-11

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
