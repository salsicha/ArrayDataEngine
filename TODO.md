# TODO

## Large Sensor Array Operations

Open3D is a useful model for this project: keep the API small, composable, NumPy-first, and fast enough for very large robotics datasets. The operations below should work on in-memory buffers first, then TileDB-backed arrays where practical.

Reference categories: [Open3D point cloud tutorial](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html), [Open3D PointCloud API](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html), and [Open3D ICP registration tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).

## Prioritized Next Work

1. [x] P0 - Finish navigation quality: covariance propagation plus quality/status masks.
2. [x] P0 - Add operation pipelines that stream directly from `DataSources`, write to `DataBuffer`, and persist to TileDB without materializing full topics.
3. [x] P1 - Add progress reporting, cancellation, and resumable operation checkpoints for long source and topic pipelines.
4. [x] P1 - Finish DEM terrain operations: terrain patches, roughness/traversability, and DEM-to-point-cloud/mesh conversion.
5. [x] P2 - Add optional parallel execution for independent chunks/topics.
6. [x] P2 - Add benchmark tests for core operations on synthetic image, point cloud, IMU, odometry, navsat, DEM, and TileDB workloads.
7. [x] P2 - Add ML-ready exports, deterministic splits, augmentations, and mixed-rate collation.
8. [x] P3 - Finish DEM tile reprojection, resampling, and cache support.
9. [ ] P3 - Work through the package-and-publish checklist for TestPyPI and PyPI.

## Backlog

- [x] Define a common operation interface for buffered topics:
  - [x] Add `map(topic, fn)`, `filter(topic, predicate)`, `reduce(topic, fn)`, and `window(topic, size|seconds)` helpers.
  - [x] Support eager NumPy output and lazy/chunked iteration for larger-than-memory arrays.
  - [x] Preserve message metadata: `timestamp`, `topic`, `name`, frame id, shape, dtype, and source URI.
  - [x] Add consistent `copy`, `out`, and `chunk_size` options for memory-sensitive workflows.
- [x] Add dataset-level selection and indexing:
  - [x] Select by topic, timestamp range, message index range, frame id, geographic bounds, and spatial bounds.
  - [x] Add timestamp range and message index range selection helpers.
  - [x] Add nearest-time lookup and bounded nearest alignment helpers.
  - [x] Add generic numeric time-series interpolation helpers.
  - [x] Add topic alignment modes: exact timestamp, nearest neighbor, bounded tolerance, fixed-rate resampling, and rolling window joins.
  - [x] Add persistent secondary indexes for TileDB-backed timestamp and message-name queries.
  - [x] Add persistent secondary indexes for frame id and spatial bounds queries.
- [x] Add geometry and coordinate-frame operations:
  - [x] Apply SE(3) transforms to point clouds, odometry poses, navsat-derived local coordinates, and DEM grids.
  - [x] Add SE(3) transform helpers for XYZ point arrays.
  - [x] Convert IMU, odometry, and navsat streams into common pose/trajectory arrays.
  - [x] Add frame graph support for static and time-varying transforms.
  - [x] Add projection helpers between point clouds, depth images, RGB images, DEM tiles, and camera frames.
  - [x] Add crop/select helpers for axis-aligned bounds, oriented bounds, masks, and geographic bounding boxes.
  - [x] Add axis-aligned XYZ bounds cropping with mask output.
- [x] Add point cloud operations:
  - [x] Downsample by voxel grid, uniform sampling, random sampling, and farthest-point sampling.
  - [x] Add voxel-grid downsampling.
  - [x] Estimate normals, local covariance, curvature-like descriptors, and nearest-neighbor distance statistics.
  - [x] Add normal estimation.
  - [x] Remove outliers with statistical and radius-based filters.
  - [x] Cluster and segment with DBSCAN, plane fitting, connected components, and ground/non-ground separation.
  - [x] Add DBSCAN clustering and RANSAC-style plane fitting.
  - [x] Add nearest-neighbor search with KNN, radius search, and hybrid search.
  - [x] Add KNN and radius search.
  - [x] Add registration helpers for point-to-point ICP, point-to-plane ICP, multi-scale ICP, and odometry-seeded registration.
  - [x] Add conversion adapters to and from Open3D point clouds when `open3d` is installed.
- [x] Add image and depth operations:
  - [x] Resize, crop, pad, normalize, color convert, and dtype convert image sequences.
  - [x] Add resize-nearest, pad, normalize, and RGB-to-gray helpers.
  - [x] Add masks, morphology, thresholding, gradients, pyramids, and local statistics.
  - [x] Add depth-image operations: valid-depth masks, backprojection to point clouds, depth-to-normal, and RGB-D fusion.
  - [x] Add valid-depth masks and depth backprojection to point clouds.
  - [x] Add frame-to-frame optical flow, image alignment, and motion-compensated rolling windows.
  - [x] Add camera model utilities for intrinsics, distortion, rectification, and projection.
- [x] Add IMU, odometry, and navsat operations:
  - [x] Resample and interpolate orientation, angular velocity, linear acceleration, position, velocity, and covariance.
  - [x] Add generic numeric time-series interpolation.
  - [x] Add quaternion normalization, SLERP, Euler conversion, gravity compensation, and bias correction helpers.
  - [x] Add quaternion normalization and SLERP.
  - [x] Convert WGS84 navsat samples to local ENU/NED frames and back.
  - [x] Add approximate WGS84 to local ENU conversion and inverse conversion.
  - [x] Add trajectory smoothing, differentiation, integration, and dead-reckoning helpers.
  - [x] Add covariance propagation and quality/status masks for navigation streams.
- [x] Add DEM and raster operations:
  - [x] Mosaic, crop, reproject, resample, and cache DEM tiles.
  - [x] Add mosaic, crop, bilinear sampling, and nearest sampling helpers.
  - [x] Compute slope, aspect, hillshade, normals, gradients, roughness, and traversability maps.
  - [x] Add slope, aspect, and hillshade helpers.
  - [x] Sample elevation at navsat/trajectory points and generate local terrain patches around a vehicle pose.
  - [x] Add raster grid sampling helper.
  - [x] Convert DEM windows to point clouds, meshes, or height grids for fusion with sensor topics.
- [ ] Add large-array execution features:
  - [x] Add chunked operation execution for buffered topic arrays that do not fit in memory.
  - [x] Add lazy buffered-topic pipelines with explicit `collect()`, `iter_chunks()`, `iter_rows()`, `reduce()`, and sliding-window execution.
  - [x] Push lazy buffered-topic time and index constraints into TileDB before reading data chunks.
  - [x] Add guarded materialization limits for explicit `collect()` calls.
  - [x] Reopen existing TileDB datasets without the original source.
  - [x] Resume partial TileDB ingest by replaying the source and skipping stored per-topic offsets.
  - [x] Add operation pipelines that can stream from `DataSources`, write to `DataBuffer`, and persist to TileDB.
  - [x] Add optional parallel execution for independent chunks/topics.
  - [x] Add progress reporting, cancellation, and resumable operation checkpoints.
  - [x] Add benchmark tests for core operations on synthetic image, point cloud, IMU, odometry, navsat, DEM, and TileDB workloads.
- [x] Add ML-ready dataset operations:
  - [x] Export topic windows to PyTorch, NumPy, and plain iterator datasets.
  - [x] Add deterministic train/validation/test splits by time, sequence, geography, or source file.
  - [x] Add augmentation operations for images, point clouds, trajectories, and DEM patches.
  - [x] Add batch collation for variable-size point clouds and mixed-rate sensor windows.

## Package And Publish To Python Registries

- [ ] Confirm the package metadata in `pyproject.toml`:
  - [ ] Package name is correct for the registry: `arraydataengine`.
  - [ ] Version is bumped for the release.
  - [ ] Description, README, license, authors, URLs, classifiers, and `requires-python` are accurate.
  - [ ] Optional dependency groups cover supported installs: `dev`, `image`, `ros`, `dem`, `tiledb`, `visualization`, `notebook`, and `ml`.
- [ ] Add release tooling if it is not already installed:

  ```bash
  python -m pip install --upgrade build twine
  ```

- [ ] Run the pre-release checks from a clean working tree:

  ```bash
  python -m pytest -q
  python -m compileall -q ade tests
  git diff --check
  ```

- [ ] Build the source distribution and wheel:

  ```bash
  python -m build
  ```

- [ ] Validate the built artifacts:

  ```bash
  python -m twine check dist/*
  python -m pip install --force-reinstall dist/*.whl
  python -m pytest -q
  ```

- [ ] Publish to TestPyPI first:

  ```bash
  python -m twine upload --repository testpypi dist/*
  ```

- [ ] Verify the TestPyPI install in a fresh virtual environment:

  ```bash
  python -m venv /tmp/ade-testpypi
  /tmp/ade-testpypi/bin/python -m pip install --upgrade pip
  /tmp/ade-testpypi/bin/python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arraydataengine
  /tmp/ade-testpypi/bin/python -c "import ade; print(ade.__file__)"
  ```

- [ ] Create and push the release commit and tag:

  ```bash
  git add pyproject.toml README.md TODO.md
  git commit -m "Release vX.Y.Z"
  git tag vX.Y.Z
  git push origin main --tags
  ```

- [ ] Publish the same checked artifacts to PyPI:

  ```bash
  python -m twine upload dist/*
  ```

- [ ] Verify the PyPI install in a fresh virtual environment:

  ```bash
  python -m venv /tmp/ade-pypi
  /tmp/ade-pypi/bin/python -m pip install --upgrade pip
  /tmp/ade-pypi/bin/python -m pip install arraydataengine
  /tmp/ade-pypi/bin/python -c "import ade; print(ade.__file__)"
  ```

- [ ] Create a GitHub release from the pushed tag and attach the generated `dist/` artifacts.
- [ ] Record the released version, PyPI URL, TestPyPI URL, and release notes in the project README or GitHub release notes.
