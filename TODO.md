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
  - [x] Add loop closure candidate search and ICP verification for datasets with point cloud and pose streams.
  - [x] Calibrate relative point clouds to accurate metric point clouds and apply the fitted scale/offset.
  - [x] Add conversion adapters to and from Open3D point clouds when `open3d` is installed.
- [x] Add image and depth operations:
  - [x] Resize, crop, pad, normalize, color convert, and dtype convert image sequences.
  - [x] Add resize-nearest, pad, normalize, and RGB-to-gray helpers.
  - [x] Add masks, morphology, thresholding, gradients, pyramids, and local statistics.
  - [x] Add depth-image operations: valid-depth masks, backprojection to point clouds, depth-to-normal, and RGB-D fusion.
  - [x] Add valid-depth masks and depth backprojection to point clouds.
  - [x] Calibrate relative depth images to accurate metric point clouds and apply the fitted scale/offset.
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

Automated release safeguards:

- [x] Build exactly one sdist and wheel and run strict metadata/content checks.
- [x] Verify the canonical package, facade, CLI, annotated tag/version match,
  and dated changelog entry.
- [x] Smoke-test the built wheel on Python 3.12, 3.13, and 3.14 before upload.
- [x] Separate unprivileged build/test jobs from OIDC publishing jobs.
- [x] Restrict production publishing to matching `v*` tags and a protected
  `pypi` environment with required review.
- [x] Provide an explicit protected TestPyPI Trusted Publishing rehearsal.
- [x] Pin every external action to a full commit SHA and let Dependabot maintain
  those pins.
- [x] Require SHA-pinned actions, read-only default workflow permissions, and
  immutable GitHub releases in repository settings.

One-time registry setup:

- [ ] Configure the pending Trusted Publisher on PyPI for `publish.yml` and
  environment `pypi`.
- [ ] Optionally configure the separate pending Trusted Publisher on TestPyPI
  for environment `testpypi`.
- [ ] Confirm both registry accounts use 2FA and no legacy registry-token
  secrets remain in GitHub.

For every release:

- [ ] Confirm `pyproject.toml` metadata and supported dependency extras are
  accurate, then bump the version.
- [ ] Replace the current changelog's `Unreleased` marker with the release
  date.
- [ ] Run the local release checks from a clean working tree:

  ```bash
  python -m pip install --upgrade build twine
  python scripts/check_release.py --tag vX.Y.Z
  python -m pytest -q --strict-config --strict-markers tests/
  python -m compileall -q arraydataengine ArrayDataEngine scripts tests
  rm -rf dist/
  python -m build
  python -m twine check --strict dist/*
  python scripts/check_dist.py dist
  git diff --check
  git status --short
  ```

- [ ] Optionally run **Actions → release → Run workflow** with TestPyPI enabled,
  approve `testpypi`, and verify an exact-version install.
- [ ] Commit the intentional release files, wait for the `tests` workflow,
  create an annotated tag, and explicitly push the branch and tag:

  ```bash
  git tag -a vX.Y.Z -m "arraydataengine X.Y.Z"
  git push origin main
  git push origin vX.Y.Z
  ```

- [ ] Inspect the release workflow's build and Python 3.12–3.14 verification
  jobs, then approve the `pypi` deployment.
- [ ] Verify the exact PyPI version in a fresh environment:

  ```bash
  python -m venv /tmp/arraydataengine-pypi
  /tmp/arraydataengine-pypi/bin/python -m pip install --upgrade pip
  /tmp/arraydataengine-pypi/bin/python -m pip install "arraydataengine==X.Y.Z"
  /tmp/arraydataengine-pypi/bin/python -c \
      "import arraydataengine; print(arraydataengine.__version__)"
  ```

- [ ] Create the immutable GitHub release from the existing tag and record the
  PyPI/TestPyPI URLs and release notes.
