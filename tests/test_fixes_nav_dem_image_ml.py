"""Regression tests for the nav / DEM / image / ML review fixes."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from arraydataengine import DataBuffer
from arraydataengine.ops import dem, image, ml, nav
from arraydataengine.sources.synthetic_source import SyntheticSource


# --- helpers ---------------------------------------------------------------------

WGS84_A = 6378137.0
WGS84_E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)


def _ecef(lat, lon, alt):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([
        (n + alt) * np.cos(lat) * np.cos(lon),
        (n + alt) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - WGS84_E2) + alt) * np.sin(lat),
    ], axis=-1)


def _reference_enu(lat, lon, alt, lat0, lon0, alt0):
    delta = _ecef(lat, lon, alt) - _ecef(lat0, lon0, alt0)
    la, lo = np.deg2rad(lat0), np.deg2rad(lon0)
    rotation = np.array([
        [-np.sin(lo), np.cos(lo), 0.0],
        [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
    ])
    return delta @ rotation.T


def _yaw_quaternions(yaw):
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2.0), np.cos(yaw / 2.0)], axis=-1)


def _blurred_noise(shape, sigma, seed):
    rng = np.random.default_rng(seed)
    field = rng.random(shape)
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    field = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, field)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, field)


def _synthetic_buffer_window():
    source = SyntheticSource(
        topics=[
            {"name": "/imu", "kind": "imu", "rate": 20.0},
            {"name": "/odom", "kind": "odometry", "rate": 20.0},
            {"name": "/gps", "kind": "navsat", "rate": 20.0},
        ],
        duration=1.0,
    )
    count = source.get_count("/imu")
    buffer = DataBuffer(source, buffer_depth=count, axis="/imu", use_db=False, preload=0)
    for _ in range(count):
        buffer.roll_buffer("/imu")
    return buffer.get_buffer()


# --- nav ----------------------------------------------------------------------------


def test_odometry_twist_is_rotated_into_parent_frame():
    # 1 m/s forward (body x) with a 0.5 rad/s yaw rate: a circle of radius 2.
    ts = np.arange(0.0, np.pi, 0.01)
    yaw = 0.5 * ts
    odom = np.zeros((ts.size, 8, 4))
    odom[:, 0, 0] = np.sin(yaw) / 0.5
    odom[:, 0, 1] = (1.0 - np.cos(yaw)) / 0.5
    odom[:, 2] = _yaw_quaternions(yaw)
    odom[:, 4, 0] = 1.0
    odom[:, 5, :3] = [0.1, 0.2, np.nan]  # unknown z variance must not leak into x/y
    odom[:, 6, 2] = 0.5
    trajectory = nav.odometry_to_trajectory({"data": odom, "ts": ts})
    assert np.allclose(trajectory["linear_velocity"][:, 0], np.cos(yaw))
    assert np.allclose(trajectory["linear_velocity"][:, 1], np.sin(yaw))
    assert np.allclose(trajectory["angular_velocity"][:, 2], 0.5)  # stays body-frame
    assert np.isfinite(trajectory["linear_velocity_covariance"][:, :2]).all()
    quarter = ts.size // 2
    assert np.isclose(trajectory["linear_velocity_covariance"][quarter, 0],
                      0.1 * np.cos(yaw[quarter]) ** 2 + 0.2 * np.sin(yaw[quarter]) ** 2)

    # README usage: default (world-frame) dead reckoning now tracks the truth.
    dead_reckoned = nav.dead_reckon_trajectory(trajectory, initial_position=trajectory["position"][0])
    assert np.abs(dead_reckoned["position"] - trajectory["position"]).max() < 1e-3

    parent = nav.odometry_to_trajectory({"data": odom, "ts": ts}, twist_frame="parent")
    assert np.allclose(parent["linear_velocity"][:, 0], 1.0)
    resampled = nav.resample_odometry({"data": odom, "ts": ts}, target_timestamps=ts[:3], twist_frame="parent")
    assert np.allclose(resampled["linear_velocity"][:, 0], 1.0)
    with pytest.raises(ValueError, match="twist_frame"):
        nav.odometry_to_trajectory(odom, twist_frame="body")


def test_navsat_enu_matches_exact_wgs84_and_round_trips():
    rng = np.random.default_rng(3)
    for lat0 in (0.0, 37.0, 60.0):
        lat = lat0 + rng.uniform(-0.5, 0.5, 50)
        lon = -122.0 + rng.uniform(-0.5, 0.5, 50)
        alt = rng.uniform(-50.0, 500.0, 50)
        enu = nav.navsat_to_enu(lat, lon, alt, lat0, -122.0, 10.0)
        assert np.abs(enu - _reference_enu(lat, lon, alt, lat0, -122.0, 10.0)).max() < 1e-6
        back = nav.enu_to_navsat(enu, lat0, -122.0, 10.0)
        # Round trip to well under a micrometre (degrees -> metres ~ 1.1e5).
        assert np.abs(back[:, :2] - np.column_stack((lat, lon))).max() * 1.2e5 < 1e-6
        assert np.abs(back[:, 2] - alt).max() < 1e-6

    # One degree of latitude is ~110.9 km at 37N on WGS84, not the spherical 111.3 km.
    north = nav.navsat_to_enu(37.5, -122.0, 0.0, 36.5, -122.0, 0.0)
    assert abs(np.hypot(north[0], north[1]) - 110_990.0) < 50.0


def test_navsat_longitude_wraps_across_antimeridian():
    enu = nav.navsat_to_enu(0.0, -179.9999, 0.0, 0.0, 179.9999, 0.0)
    assert np.isclose(enu[0], 22.26, atol=0.01)
    back = nav.enu_to_navsat(np.array([30.0, 0.0, 0.0]), 0.0, 179.9999, 0.0)
    assert -180.0 <= back[1] < 180.0
    assert np.isclose(back[1], -179.99963, atol=1e-5)


def test_zero_quaternions_become_nan_instead_of_raising():
    count = 5
    imu = np.zeros((count, 6, 4))
    imu[:, 1, 0] = -1.0  # ROS: orientation_covariance[0] = -1 -> no orientation
    imu[:, 2, :3] = [0.0, 0.0, 0.1]
    imu[:, 4, :3] = [0.0, 0.0, 9.81]
    topic = {"data": imu, "ts": np.arange(count) * 0.01}
    trajectory = nav.imu_to_trajectory(topic)
    assert np.isnan(trajectory["orientation"]).all()
    assert np.allclose(trajectory["angular_velocity"][:, 2], 0.1)
    resampled = nav.resample_imu(topic, period=0.005)
    assert np.isnan(resampled["orientation"]).all()
    compensated = nav.compensate_imu_gravity(imu)
    assert np.isnan(compensated[:, 4, :3]).all()

    # A valid quaternion flagged with covariance -1 is also "no estimate".
    flagged = imu.copy()
    flagged[:, 0, 3] = 1.0
    assert np.isnan(nav.imu_to_trajectory(flagged)["orientation"]).all()

    odom = np.zeros((2, 8, 4))
    odom[1, 2, 3] = 1.0
    trajectory = nav.odometry_to_trajectory({"data": odom, "ts": np.array([0.0, 1.0])})
    assert np.isnan(trajectory["orientation"][0]).all()
    assert np.allclose(trajectory["orientation"][1], [0.0, 0.0, 0.0, 1.0])


def test_navsat_reference_skips_invalid_leading_fixes():
    gps = np.array([[np.nan, np.nan, np.nan], [37.0, -122.0, 10.0], [37.0001, -122.0, 10.0]])
    trajectory = nav.navsat_to_trajectory({"data": gps, "ts": np.array([0.0, 1.0, 2.0])})
    assert trajectory["reference"] == {"lat": 37.0, "lon": -122.0, "alt": 10.0}
    assert np.allclose(trajectory["position"][1], 0.0)

    no_fix = np.array([[36.0, -121.0, 5.0], [37.0, -122.0, 10.0]])
    trajectory = nav.navsat_to_trajectory({"data": no_fix, "ts": np.array([0.0, 1.0]), "status": np.array([-1, 0])})
    assert trajectory["reference"]["lat"] == 37.0
    local, reference = nav.navsat_to_local({"data": no_fix, "status": np.array([-1, 0])}, return_reference=True)
    assert reference["lat"] == 37.0 and np.allclose(local[1], 0.0)
    assert np.isfinite(nav.navsat_to_local(gps)[1:]).all()


def test_trajectory_speed_is_second_order_and_rejects_duplicates():
    ts = np.array([0.0, 0.1, 1.0, 1.1, 3.0])
    positions = np.column_stack([ts ** 2, np.zeros_like(ts), np.zeros_like(ts)])
    speed = nav.trajectory_speed(ts, positions)
    assert np.allclose(speed[1:-1], 2.0 * ts[1:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        nav.trajectory_speed(np.array([0.0, 1.0, 1.0, 2.0]), np.zeros((4, 3)))

    gps = nav.enu_to_navsat(positions, 37.0, -122.0, 0.0)
    trajectory = nav.navsat_to_trajectory({"data": gps, "ts": ts}, ref_lat=37.0, ref_lon=-122.0, ref_alt=0.0)
    assert np.allclose(trajectory["linear_velocity"][1:-1, 0], 2.0 * ts[1:-1], atol=1e-6)


def test_interpolate_timeseries_requires_increasing_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        nav.interpolate_timeseries(np.array([0.0, 2.0, 1.0]), np.array([0.0, 2.0, 1.0]), np.array([1.5]))


def test_vectorized_quaternion_helpers_match_scalar_reference():
    rng = np.random.default_rng(0)
    count = 40
    ts = np.cumsum(rng.uniform(0.01, 0.1, count))
    quaternions = rng.normal(size=(count, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions[10:13] = quaternions[10]  # exercise the near-identical (lerp) branch
    targets = np.concatenate([np.linspace(ts[0] - 0.5, ts[-1] + 0.5, 97), ts[:5], [np.nan]])

    expected = []
    for target in targets[:-1]:
        if target <= ts[0]:
            expected.append(quaternions[0])
        elif target >= ts[-1]:
            expected.append(quaternions[-1])
        else:
            upper = int(np.searchsorted(ts, target, side="right"))
            fraction = (target - ts[upper - 1]) / (ts[upper] - ts[upper - 1])
            expected.append(nav.slerp(quaternions[upper - 1], quaternions[upper], fraction))
    result = nav.interpolate_quaternions(ts, quaternions, targets)
    assert np.allclose(result[:-1], nav.normalize_quaternion(np.asarray(expected)), atol=1e-12)
    assert np.isnan(result[-1]).all()

    omega = nav.angular_velocity_from_quaternions(ts, quaternions)
    intervals = []
    for index in range(count - 1):
        delta = nav._quaternion_multiply(nav._quaternion_conjugate(quaternions[index]), quaternions[index + 1])
        intervals.append(nav._rotation_vector_from_quaternion(delta) / (ts[index + 1] - ts[index]))
    intervals = np.asarray(intervals)
    assert np.allclose(omega[1:-1], 0.5 * (intervals[:-1] + intervals[1:]), atol=1e-12)
    assert np.allclose(omega[0], intervals[0], atol=1e-12)

    rates = rng.normal(size=(count, 3))
    integrated = nav.integrate_orientations(ts, rates, initial_orientation=quaternions[0])
    reference = [quaternions[0]]
    for index in range(1, count):
        step = nav._quaternion_from_rotation_vector(0.5 * (rates[index - 1] + rates[index]) * (ts[index] - ts[index - 1]))
        reference.append(nav.normalize_quaternion(nav._quaternion_multiply(reference[-1], step)))
    assert np.allclose(integrated, np.asarray(reference), atol=1e-12)


def test_propagate_covariance_from_initial_avoids_double_counting():
    trajectory = {
        "ts": np.array([0.0, 1.0, 2.0]),
        "position": np.zeros((3, 3)),
        "orientation": np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        "position_covariance": np.array([[0.5] * 3, [0.7] * 3, [0.9] * 3]),
    }
    default = nav.propagate_trajectory_covariance(trajectory, process_noise={"position": 0.1})
    assert np.allclose(default["position_covariance"][:, 0], [0.5, 0.8, 1.1])
    initial = nav.propagate_trajectory_covariance(trajectory, process_noise={"position": 0.1}, from_initial=True)
    assert np.allclose(initial["position_covariance"][:, 0], [0.5, 0.6, 0.7])


def test_nav_ops_accept_structured_buffer_topics():
    window = _synthetic_buffer_window()
    imu_topic = window["/imu"]
    assert imu_topic.dtype.names is not None  # DataBuffer.get_buffer() structured arrays

    compensated = nav.compensate_imu_gravity(imu_topic)
    assert compensated.dtype == imu_topic.dtype
    assert np.allclose(compensated["data"][:, 4, 2], 9.81 - 9.80665)
    assert np.array_equal(compensated["ts"], imu_topic["ts"])
    corrected = nav.correct_imu_bias(compensated, sample_slice=slice(0, 5))
    assert corrected["data"].shape == imu_topic["data"].shape

    imu_trajectory = nav.imu_to_trajectory(imu_topic)
    assert np.allclose(imu_trajectory["ts"], imu_topic["ts"])
    assert imu_trajectory["orientation"].shape == (imu_topic.shape[0], 4)

    odom_trajectory = nav.odometry_to_trajectory(window["/odom"])
    assert odom_trajectory["position"].shape == (window["/odom"].shape[0], 3)

    gps_trajectory = nav.navsat_to_trajectory(window["/gps"])
    assert np.isfinite(gps_trajectory["position"]).all()
    assert nav.navsat_to_local(window["/gps"]).shape == (window["/gps"].shape[0], 3)
    dispatched = nav.sensor_to_trajectory(window["/odom"], kind="odometry")
    assert np.allclose(dispatched["position"], odom_trajectory["position"], equal_nan=True)


# --- DEM ----------------------------------------------------------------------------


def test_sample_grid_handles_nan_and_out_of_range_coordinates():
    elevation = np.arange(9.0).reshape(3, 3)
    assert np.isnan(dem.sample_grid(elevation, np.array([np.nan]), np.array([1.0]))).all()
    assert np.isnan(dem.sample_grid(elevation, np.array([np.nan]), np.array([np.nan]), bilinear=False)).all()
    assert np.isnan(dem.sample_elevation(elevation, x=[1e6], y=[1e6])).all()
    assert np.isnan(dem.sample_elevation_at_navsat(elevation, np.array([[np.nan, np.nan, np.nan]]), 37.0, -122.0)).all()
    # Within half a cell of the border the edge value is still used.
    assert np.allclose(dem.sample_grid(elevation, np.array([-0.4, 2.4]), np.array([0.0, 2.0])), [0.0, 8.0])
    # A NaN neighbour with zero weight does not poison an exact node sample.
    with_hole = np.array([[1.0, 2.0], [np.nan, 4.0]])
    assert dem.sample_grid(with_hole, np.array([0.0]), np.array([0.0]))[0] == 1.0
    assert np.isclose(dem.sample_grid(with_hole, np.array([0.0]), np.array([0.5]))[0], 1.5)
    # Integer rasters: nearest sampling promotes to float so misses can be NaN.
    nearest = dem.sample_grid(np.arange(4, dtype=np.int16).reshape(2, 2), np.array([1.0, 7.0]), np.array([1.0, 0.0]), bilinear=False)
    assert nearest.dtype == np.float64 and nearest[0] == 3.0 and np.isnan(nearest[1])


def test_north_up_convention_for_dem_ops():
    rows = 5
    native = np.tile(np.arange(rows, dtype=np.float64)[:, None] * 25.0, (1, 4))  # rises toward +y (north)
    north_up = native[::-1].copy()  # HGT layout: row 0 = north

    assert np.allclose(dem.terrain_normals(north_up, north_up=True), dem.terrain_normals(native)[::-1])
    assert dem.terrain_normals(north_up, north_up=True)[2, 1, 1] < 0.0  # rises north -> normal tilts south
    dz_dx, dz_dy = dem.terrain_gradients(north_up, north_up=True)
    assert np.allclose(dz_dy, 25.0) and np.allclose(dz_dx, 0.0)
    assert np.allclose(
        dem.sample_elevation(north_up, x=[1.0, 2.0], y=[0.5, 3.0], north_up=True),
        dem.sample_elevation(native, x=[1.0, 2.0], y=[0.5, 3.0]),
    )
    patch, origin = dem.terrain_patch(north_up, center=(1.0, 3.0), size=3, north_up=True, return_origin=True)
    native_patch, native_origin = dem.terrain_patch(native, center=(1.0, 3.0), size=3, return_origin=True)
    assert np.allclose(patch, native_patch[::-1]) and origin == native_origin

    points = dem.dem_to_point_cloud(north_up, north_up=True)
    assert np.allclose(points[0], [0.0, rows - 1, 100.0])  # row 0 is the northernmost row
    native_points = dem.dem_to_point_cloud(native)
    assert np.allclose(np.sort(points[:, 1] * 1000 + points[:, 2]), np.sort(native_points[:, 1] * 1000 + native_points[:, 2]))

    # reproject_raster with GIS bounds (west, south, east, north) on a north-up tile.
    northern = dem.reproject_raster(
        north_up,
        src_bounds=(-122.0, 37.0, -121.0, 38.0),
        dst_bounds=(-122.0, 37.75, -121.0, 38.0),
        shape=(2, 2),
        north_up=True,
    )
    assert np.allclose(northern[:, 0], [100.0, 75.0])

    navsat = nav.enu_to_navsat(np.array([2.0, 1.0, 0.0]), 37.0, -122.0, 0.0)
    assert np.isclose(dem.sample_elevation_at_navsat(north_up, navsat, 37.0, -122.0, 0.0, north_up=True), 25.0)
    nav_patch = dem.terrain_patch_at_navsat(north_up, navsat, 37.0, -122.0, 0.0, size=1, north_up=True)
    assert np.allclose(nav_patch, 25.0)


def test_dem_to_mesh_faces_point_up_and_match_loop():
    def face_normals(mesh):
        v, f = mesh["vertices"], mesh["faces"]
        return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])

    flat = dem.dem_to_mesh(np.zeros((2, 2)))
    assert flat["faces"].tolist() == [[0, 1, 2], [1, 3, 2]]
    assert (face_normals(flat)[:, 2] > 0).all()

    rng = np.random.default_rng(1)
    terrain = rng.random((6, 7))
    terrain[2, 3] = np.nan
    for north_up in (False, True):
        mesh = dem.dem_to_mesh(terrain, resolution=(2.0, 3.0), north_up=north_up)
        assert (face_normals(mesh)[:, 2] > 0).all()
        assert mesh["faces"].shape == (2 * (5 * 6 - 4), 3)
    with_nan = dem.dem_to_mesh(terrain, include_nan=True)
    assert with_nan["faces"].shape == (2 * 5 * 6, 3)


def test_mosaic_dem_tiles_merges_srtm_seams_and_voids():
    def tile(lon, samples=5):
        return np.tile((lon + np.arange(samples) / (samples - 1)) * 1000.0, (samples, 1))

    mosaic = dem.mosaic_dem_tiles({"N37W122": tile(-122), "N37W121": tile(-121)}, overlap=1)
    assert mosaic.shape == (5, 9)
    assert np.allclose(mosaic[0], np.linspace(-122000.0, -120000.0, 9))
    assert dem.mosaic_dem_tiles({"N37W122": tile(-122), "N37W121": tile(-121)}).shape == (5, 10)  # generic default

    # Standard SRTM3 tiles (1201 x 1201) share their seam by default.
    north = np.zeros((1201, 1201), dtype=">i2")
    south = np.ones((1201, 1201), dtype=">i2")
    north[0, 0] = -32768
    south[0, 0] = -32768  # void on the shared seam row
    stacked = dem.mosaic_dem_tiles({"N37W122": north, "N36W122": south})
    assert stacked.shape == (2401, 1201)
    assert np.isnan(stacked[0, 0])  # void -> fill_value
    assert stacked[1200, 0] == 0.0 and stacked[1200, 1] == 1.0  # a seam void keeps the neighbour's sample

    voids = dem.mosaic_dem_tiles({"N37W122": np.array([[1, -32768], [3, 4]], dtype=np.int16)})
    assert np.isnan(voids[0, 1]) and voids[1, 1] == 4.0
    assert dem.mosaic_dem_tiles({"N37W122": np.array([[1, -32768]], dtype=np.int16)}, nodata=None)[0, 1] == -32768
    with pytest.raises(ValueError, match="overlap"):
        dem.mosaic_dem_tiles({"N37W122": np.ones((2, 2))}, overlap=2)


def test_mosaic_dem_tiles_accepts_buffered_dem_topic():
    dtype = np.dtype([("ts", "<f8"), ("id", "S256"), ("data", "<i2", (2, 2))])
    topic = np.zeros(2, dtype=dtype)
    topic["id"] = [b"N37W122", b"N37W121"]
    topic["data"][0] = 1
    topic["data"][1] = 2
    mosaic = dem.mosaic_dem_tiles(topic)
    assert np.array_equal(mosaic, [[1, 1, 2, 2], [1, 1, 2, 2]])
    as_dict = {"ts": topic["ts"], "id": np.array(["N37W122", "N37W121"], dtype=object), "data": topic["data"]}
    assert np.array_equal(dem.mosaic_dem_tiles(as_dict), mosaic)


def test_aspect_is_compass_bearing_and_flat_is_nan():
    west_facing = np.tile(np.array([0.0, 1.0, 2.0]), (3, 1))
    _, aspect = dem.slope_aspect(west_facing)
    assert np.allclose(aspect, 1.5 * np.pi)
    _, flat_aspect = dem.slope_aspect(np.ones((3, 3)))
    assert np.isnan(flat_aspect).all()
    rng = np.random.default_rng(2)
    _, random_aspect = dem.slope_aspect(rng.random((20, 20)))
    assert ((random_aspect >= 0.0) & (random_aspect < 2.0 * np.pi)).all()
    shade = dem.hillshade(np.ones((3, 3)))
    assert np.allclose(shade, np.sin(np.deg2rad(45.0)))


def test_traversability_stays_in_unit_range_next_to_nan():
    elevation = np.ones((5, 5))
    elevation[2, 2] = np.nan
    score = dem.traversability_map(elevation, max_roughness=1.0)
    assert np.isfinite(score).all()
    assert ((score >= 0.0) & (score <= 1.0)).all()
    assert score[2, 2] == 0.0 and score[2, 1] == 0.0 and score[0, 0] == 1.0


def test_roughness_matches_windowed_reference_and_resolution_pairs():
    rng = np.random.default_rng(4)
    elevation = rng.normal(size=(12, 9)).cumsum(axis=0) * 10.0 + 2000.0
    elevation[3, 3] = np.nan
    elevation[6:9, 5:8] = np.nan
    for window in (2, 3, 5):
        before = window // 2
        padded = np.pad(elevation, ((before, window - 1 - before),) * 2, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
            expected = np.nanstd(windows.reshape(windows.shape[:2] + (-1,)), axis=-1)
        assert np.allclose(dem.roughness_map(elevation, window_size=window), expected, equal_nan=True, atol=1e-9)
    assert np.allclose(dem.roughness_map(np.ones((4, 4))), 0.0)

    ramp = np.add.outer(np.arange(4.0) * 3.0, np.arange(5.0) * 2.0)  # dz/dy per row 3, dz/dx per col 2
    dz_dx, dz_dy = dem.terrain_gradients(ramp, resolution=(2.0, 3.0))
    assert np.allclose(dz_dx, 1.0) and np.allclose(dz_dy, 1.0)
    assert np.allclose(dem.sample_elevation(ramp, x=[4.0], y=[6.0], resolution=(2.0, 3.0)), ramp[2, 2])
    with pytest.raises(ValueError, match="positive"):
        dem.slope_aspect(ramp, resolution=(1.0, 0.0))


# --- image --------------------------------------------------------------------------


def test_phase_correlation_on_smooth_non_periodic_crops():
    base = _blurred_noise((200, 200), sigma=4.0, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(6):
        dy, dx = (int(value) for value in rng.integers(-8, 9, size=2))
        reference = base[50:178, 50:178]
        moving = base[50 - dy:178 - dy, 50 - dx:178 - dx]
        assert np.array_equal(image.estimate_image_shift(reference, moving), [dy, dx])

    # Circular shifts (the synthetic case) still resolve exactly.
    noise = np.random.default_rng(2).random((64, 64))
    assert np.array_equal(image.estimate_image_shift(noise, np.roll(noise, (3, -5), axis=(0, 1))), [3, -5])


def test_convert_image_dtype_signed_round_trips():
    values = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    as_float = image.convert_image_dtype(values, np.float32)
    assert np.allclose(as_float, [-1.0, -1.0 / 32767, 0.0, 1.0 / 32767, 1.0])
    # Everything except the most negative value (which shares -1.0) round-trips.
    assert image.convert_image_dtype(as_float, np.int16).tolist() == [-32767, -1, 0, 1, 32767]
    int8 = np.array([-128, 0, 64, 127], dtype=np.int8)
    assert image.convert_image_dtype(int8, np.uint8).tolist() == [0, 0, 129, 255]
    assert image.convert_image_dtype(image.convert_image_dtype(int8, np.uint8), np.int8).tolist() == [0, 0, 64, 127]
    widened = image.convert_image_dtype(values, np.int32)
    assert widened[2] == 0 and widened[0] == np.iinfo(np.int32).min
    assert np.array_equal(image.convert_image_dtype(widened, np.int16), values)
    assert image.convert_image_dtype(np.array([-1.0, 0.0, 1.0]), np.uint8).tolist() == [0, 0, 255]


def test_convert_image_dtype_matches_skimage_conventions():
    util = pytest.importorskip("skimage.util")
    for dtype in (np.uint8, np.uint16, np.int8, np.int16):
        info = np.iinfo(dtype)
        values = np.linspace(info.min, info.max, 257).astype(dtype)
        assert np.allclose(image.convert_image_dtype(values, np.float64), util.img_as_float64(values))
    unsigned = np.arange(256, dtype=np.uint8)
    assert np.array_equal(image.convert_image_dtype(unsigned, np.uint16), util.img_as_uint(unsigned))
    assert np.array_equal(image.convert_image_dtype(util.img_as_float64(unsigned), np.uint8), unsigned)


def test_normalize_image_ignores_nan():
    normalized = image.normalize_image(np.array([[1.0, np.nan], [3.0, 5.0]]))
    assert np.allclose(normalized, [[0.0, np.nan], [0.5, 1.0]], equal_nan=True)
    per_image = image.normalize_images(np.array([[[1.0, np.nan], [3.0, 5.0]], [[2.0, 4.0], [np.nan, 6.0]]]), per_image=True)
    assert np.allclose(per_image, [[[0.0, np.nan], [0.5, 1.0]], [[0.0, 0.5], [np.nan, 1.0]]], equal_nan=True)


def test_channel_axis_resolves_ambiguous_gray_sequences():
    sequence = np.random.default_rng(5).random((5, 8, 3))  # gray (N, H, W) with W == 3
    assert image.image_gradients(sequence)["dx"].shape == (5, 8)  # auto: treated as RGB
    explicit = image.image_gradients(sequence, channel_axis=None)
    assert explicit["dx"].shape == (5, 8, 3)
    assert np.allclose(image.image_gradients(sequence, spatial_axes=(1, 2))["dx"], explicit["dx"])
    per_frame = np.stack([image.image_gradients(frame, channel_axis=None)["dx"] for frame in sequence])
    assert np.allclose(explicit["dx"], per_frame)

    # RGB sequences keep working even when the width looks like a channel count.
    rgb_sequence = np.random.default_rng(6).random((2, 6, 4, 3))
    gradients = image.image_gradients(rgb_sequence)["dx"]
    assert gradients.shape == (2, 6, 4)
    assert np.allclose(gradients[1], image.image_gradients(rgb_sequence[1])["dx"])
    channels_first = np.moveaxis(rgb_sequence[0], -1, 0)
    assert np.allclose(image.image_gradients(channels_first, channel_axis=0)["dx"], gradients[0])

    gray = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    assert np.array_equal(ml.augment_image(gray, flip_horizontal=True, channel_axis=None), gray[:, :, ::-1])
    assert np.array_equal(ml.augment_image(gray, flip_horizontal=True), gray[:, ::-1, :])  # auto keeps HWC reading
    patches = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
    assert np.array_equal(ml.augment_dem_patch(patches, flip_horizontal=True), patches[:, :, ::-1])


def test_translate_image_rejects_nan_fill_for_integer_images():
    with pytest.raises(ValueError, match="fill_value"):
        image.translate_image(np.ones((3, 3), np.uint8), (1, 0), fill_value=np.nan)
    shifted = image.translate_image(np.ones((3, 3)), (1, 0), fill_value=np.nan)
    assert np.isnan(shifted[0]).all()


def test_local_std_matches_windowed_reference():
    rng = np.random.default_rng(7)
    frames = rng.normal(size=(2, 9, 11)) * 50.0 + 1000.0
    frames[0, 4, 4] = np.nan
    stats = image.local_statistics(frames, size=(3, 5), statistics=("mean", "std"), spatial_axes=(1, 2))
    padded = np.pad(frames, ((0, 0), (1, 1), (2, 2)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 5), axis=(1, 2))
    assert np.allclose(stats["std"], windows.std(axis=(-2, -1)), equal_nan=True)
    assert np.allclose(stats["mean"], windows.mean(axis=(-2, -1)), equal_nan=True)
    assert np.isnan(stats["std"][0, 4, 4]) and np.isfinite(stats["std"][1]).all()
    uint8 = (rng.random((6, 7, 3)) * 255).astype(np.uint8)
    windows8 = np.lib.stride_tricks.sliding_window_view(np.pad(uint8, ((1, 1), (1, 1), (0, 0)), mode="edge"), (3, 3), axis=(0, 1))
    assert np.allclose(image.local_std(uint8, size=3), windows8.astype(np.float64).std(axis=(-2, -1)))


# --- ML -----------------------------------------------------------------------------


def test_torch_iterable_dataset_shards_across_workers(monkeypatch):
    torch = pytest.importorskip("torch")
    dataset = ml.to_torch_dataset([{"x": np.array([index])} for index in range(7)], iterable=True)
    assert sorted(int(sample["x"][0]) for sample in dataset) == list(range(7))

    class _Worker:
        def __init__(self, worker_id, num_workers):
            self.id = worker_id
            self.num_workers = num_workers

    seen = []
    for worker_id in range(3):
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda worker_id=worker_id: _Worker(worker_id, 3))
        seen.extend(int(sample["x"][0]) for sample in dataset)
    assert sorted(seen) == list(range(7))


def test_split_indices_group_nan_and_stay_deterministic():
    groups = np.array([1.0, np.nan, np.nan, 2.0, 3.0])
    splits = ml.deterministic_split_indices(5, fractions=(0.5, 0.5), names=("a", "b"), groups=groups)
    covered = np.sort(np.concatenate(list(splits.values())))
    assert covered.tolist() == [0, 1, 2, 3, 4]
    assert any({1, 2} <= set(indices.tolist()) for indices in splits.values())  # NaNs stay together

    # Existing (non-NaN) grouped splits are unchanged: units in first-appearance order.
    labels = np.array(["b", "a", "b", "c", "a", "c"], dtype=object)
    grouped = ml.deterministic_split_indices(6, fractions=(0.34, 0.33, 0.33), groups=labels)
    assert grouped["train"].tolist() == [0, 2]
    assert grouped["val"].tolist() == [1, 4]
    assert grouped["test"].tolist() == [3, 5]
    many = ml.deterministic_split_indices(20_000, groups=np.arange(20_000) // 2)
    assert sum(indices.size for indices in many.values()) == 20_000

    first = ml.deterministic_split_indices(10, shuffle=True)
    second = ml.deterministic_split_indices(10, shuffle=True)
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_split_topic_by_time_sorts_timestamps():
    topic = {
        "id": np.array(list("abcdef"), dtype=object),
        "ts": np.array([5.0, 0.0, 4.0, 1.0, 3.0, 2.0]),
        "data": np.zeros((6, 3)),
    }
    splits = ml.split_topic(topic, by="time", fractions=(0.5, 0.5), names=("train", "test"))
    assert sorted(splits["train"]["ts"].tolist()) == [0.0, 1.0, 2.0]
    assert sorted(splits["test"]["ts"].tolist()) == [3.0, 4.0, 5.0]
    assert splits["train"]["id"].tolist() == ["b", "d", "f"]  # original row order within a split


def test_collate_schema_is_stable_and_dtype_preserving():
    uniform = ml.collate_samples([{"x": np.ones((3, 2)), "label": 0}, {"x": np.ones((3, 2)), "label": 1}])
    ragged = ml.collate_samples([{"x": np.ones((2, 2)), "label": 0}, {"x": np.ones((3, 2)), "label": 1}])
    assert sorted(uniform) == sorted(ragged) == ["label", "x", "x_lengths", "x_mask"]
    assert uniform["x_lengths"].tolist() == [3, 3] and uniform["x_mask"].all()

    stamps = np.array([1_700_000_000_123_456_789, 1_700_000_000_123_456_790], dtype=np.int64)
    batch = ml.collate_samples([{"ts": stamps}, {"ts": stamps[:1]}])
    assert batch["ts"].dtype == np.int64
    assert int(batch["ts"][0, 1]) == int(stamps[1]) and batch["ts"][1, 1] == 0
    assert ml.collate_samples([{"ts": stamps}, {"ts": stamps[:1]}], pad_value=np.nan)["ts"].dtype == np.float64

    with pytest.raises(ValueError, match="dimensions"):
        ml.collate_samples([{"x": np.ones((2,))}, {"x": np.full((2, 3), 2.0)}])
    unpadded = ml.collate_samples([{"x": np.ones((2,))}, {"x": np.ones((3,))}], pad=False)
    assert unpadded["x"].dtype == object and "x_lengths" not in unpadded


def test_ml_windows_accept_structured_buffer_topics():
    window = _synthetic_buffer_window()
    imu = window["/imu"]
    structured = list(ml.iter_ml_windows(imu, size=4))
    as_dict = list(ml.iter_ml_windows({"ts": imu["ts"], "data": imu["data"], "topic": "topic"}, size=4))
    assert len(structured) == len(as_dict) > 0
    assert all(np.array_equal(a["data"], b["data"]) for a, b in zip(structured, as_dict))
    splits = ml.split_topic(window["/imu"], by="time", fractions=(0.5, 0.5), names=("a", "b"))
    assert splits["a"]["ts"].max() <= splits["b"]["ts"].min()
