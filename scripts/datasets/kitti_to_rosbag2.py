"""Convert one KITTI raw drive into a ROS 2 bag for testing the notebooks.

KITTI raw drives (https://www.cvlibs.net/datasets/kitti/raw_data.php, CC BY-NC-SA 3.0)
carry real GPS/IMU (OXTS), Velodyne HDL-64 and camera data. This writes them as
standard ROS 2 messages so the INS, LOAM and demo notebooks can be pointed at a
genuine recording through their ``ADE_*`` overrides:

====================================  ===========================  =========
topic                                 type                         rate
====================================  ===========================  =========
``/kitti/oxts/imu``                   sensor_msgs/msg/Imu          10 Hz
``/kitti/oxts/gps/fix``               sensor_msgs/msg/NavSatFix    10 Hz
``/kitti/oxts/odom``                  nav_msgs/msg/Odometry        10 Hz
``/kitti/oxts/pose``                  geometry_msgs/msg/PoseStamped 10 Hz
``/kitti/velo/pointcloud``            sensor_msgs/msg/PointCloud2  10 Hz
``/kitti/velo/pose``                  geometry_msgs/msg/PoseStamped 10 Hz
``/kitti/camera_color_left/image/compressed``  sensor_msgs/msg/CompressedImage  10 Hz
====================================  ===========================  =========

Poses are in a local ENU frame (``map``) anchored at the first OXTS fix, built with
the KITTI devkit's Mercator projection; ``/kitti/velo/pose`` is the Velodyne
pose (``T_map_imu @ T_imu_velo``), i.e. ground truth for lidar odometry. Odometry
twists are in the vehicle (child) frame, as ROS defines them.

Usage::

    python scripts/datasets/kitti_to_rosbag2.py \
        --drive DATA/2011_09_26/2011_09_26_drive_0001_sync \
        --calib DATA/2011_09_26 --out DATA/kitti_0001_rosbag2 [--storage mcap]

``--storage ros1`` writes a ROS 1 bag instead (``--out`` is then a ``.bag`` file),
with the same topics, using the Noetic message definitions.
"""
from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

EARTH_RADIUS = 6378137.0  # KITTI devkit Mercator scale (WGS84 equatorial radius)


def read_timestamps(path: Path) -> list[int]:
    """KITTI timestamps ('2011-09-26 13:02:25.964389445') as integer nanoseconds."""
    stamps = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        date, frac = line.strip().split(".")
        whole = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        epoch = int((whole - datetime(1970, 1, 1)).total_seconds())
        stamps.append(epoch * 1_000_000_000 + int(frac.ljust(9, "0")[:9]))
    return stamps


def read_calib(path: Path) -> np.ndarray:
    """4x4 transform from a KITTI calib file with R: and T: rows."""
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            try:
                values[key.strip()] = np.array([float(v) for v in rest.split()])
            except ValueError:
                continue
    transform = np.eye(4)
    transform[:3, :3] = values["R"].reshape(3, 3)
    transform[:3, 3] = values["T"]
    return transform


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to an (x, y, z, w) quaternion with w >= 0."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return q if q[3] >= 0 else -q


def oxts_poses(packets: np.ndarray) -> list[np.ndarray]:
    """T_map_imu per packet: KITTI devkit Mercator position + R = Rz(yaw) Ry(pitch) Rx(roll)."""
    scale = math.cos(math.radians(packets[0, 0]))
    poses, origin = [], None
    for lat, lon, alt, roll, pitch, yaw in packets[:, :6]:
        tx = scale * math.radians(lon) * EARTH_RADIUS
        ty = scale * EARTH_RADIUS * math.log(math.tan(math.radians(90.0 + lat) / 2.0))
        pose = np.eye(4)
        pose[:3, :3] = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        pose[:3, 3] = (tx, ty, alt)
        if origin is None:
            origin = pose[:3, 3].copy()
        pose[:3, 3] -= origin
        poses.append(pose)
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", type=Path, required=True, help="the *_sync drive directory")
    parser.add_argument("--calib", type=Path, required=True, help="the date directory holding calib_*.txt")
    parser.add_argument("--out", type=Path, required=True,
                        help="output rosbag2 directory, or .bag file with --storage ros1 (replaced)")
    parser.add_argument("--storage", choices=("sqlite3", "mcap", "ros1"), default="sqlite3")
    parser.add_argument("--no-images", action="store_true", help="skip the camera topic")
    args = parser.parse_args()

    from rosbags.typesys import Stores, get_typestore

    ros1 = args.storage == "ros1"
    store = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    serialize = store.serialize_ros1 if ros1 else store.serialize_cdr
    sequence = iter(range(1 << 31))
    T = store.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]
    Vector3, Quaternion, Point = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Point"]
    Pose, PoseStamped = T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/PoseStamped"]
    PoseCov, Twist, TwistCov = (T["geometry_msgs/msg/PoseWithCovariance"], T["geometry_msgs/msg/Twist"],
                                T["geometry_msgs/msg/TwistWithCovariance"])
    Imu, NavSatFix, NavSatStatus = T["sensor_msgs/msg/Imu"], T["sensor_msgs/msg/NavSatFix"], T["sensor_msgs/msg/NavSatStatus"]
    Odometry, PointCloud2, PointField = T["nav_msgs/msg/Odometry"], T["sensor_msgs/msg/PointCloud2"], T["sensor_msgs/msg/PointField"]
    CompressedImage = T["sensor_msgs/msg/CompressedImage"]

    def header(stamp_ns: int, frame: str):
        stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
        if ros1:  # ROS 1 headers carry a sequence number
            return Header(seq=next(sequence), stamp=stamp, frame_id=frame)
        return Header(stamp=stamp, frame_id=frame)

    def pose_msg(transform: np.ndarray):
        q = quaternion_xyzw(transform[:3, :3])
        return Pose(position=Point(x=float(transform[0, 3]), y=float(transform[1, 3]), z=float(transform[2, 3])),
                    orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))

    drive = args.drive
    oxts_stamps = read_timestamps(drive / "oxts" / "timestamps.txt")
    oxts_files = sorted((drive / "oxts" / "data").glob("*.txt"))
    packets = np.array([[float(v) for v in f.read_text().split()] for f in oxts_files])
    poses = oxts_poses(packets)
    t_velo_imu = read_calib(args.calib / "calib_imu_to_velo.txt")  # p_velo = T * p_imu
    t_imu_velo = np.linalg.inv(t_velo_imu)

    if args.out.is_dir():
        shutil.rmtree(args.out)
    elif args.out.exists():
        args.out.unlink()
    if ros1:
        from rosbags.rosbag1 import Writer as Ros1Writer

        writer_context = Ros1Writer(args.out)
    else:
        from rosbags.rosbag2 import StoragePlugin, Writer

        plugin = StoragePlugin.MCAP if args.storage == "mcap" else StoragePlugin.SQLITE3
        writer_context = Writer(args.out, version=9, storage_plugin=plugin)
    with writer_context as writer:
        def connection(topic, msgtype):
            return writer.add_connection(topic, msgtype, typestore=store)

        imu_c = connection("/kitti/oxts/imu", "sensor_msgs/msg/Imu")
        fix_c = connection("/kitti/oxts/gps/fix", "sensor_msgs/msg/NavSatFix")
        odom_c = connection("/kitti/oxts/odom", "nav_msgs/msg/Odometry")
        pose_c = connection("/kitti/oxts/pose", "geometry_msgs/msg/PoseStamped")
        velo_pose_c = connection("/kitti/velo/pose", "geometry_msgs/msg/PoseStamped")
        cloud_c = connection("/kitti/velo/pointcloud", "sensor_msgs/msg/PointCloud2")
        image_c = None if args.no_images else connection(
            "/kitti/camera_color_left/image/compressed", "sensor_msgs/msg/CompressedImage")

        def write(conn, stamp, msg, msgtype):
            writer.write(conn, stamp, serialize(msg, msgtype))

        for stamp, packet, pose in zip(oxts_stamps, packets, poses):
            # KITTI's f/l/u fields are parallel to the earth's surface (heading frame);
            # ROS Imu and Odometry twist are in the body frame: x/y/z fields, and the
            # ENU velocity rotated into the vehicle frame.
            (lat, lon, alt, _roll, _pitch, _yaw, vn, ve, _vf, _vl, vu, ax, ay, az, _af, _al, _au,
             wx, wy, wz, _wf, _wl, _wu, pos_acc, vel_acc, *_rest) = packet
            q = quaternion_xyzw(pose[:3, :3])
            orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
            body_velocity = pose[:3, :3].T @ np.array([ve, vn, vu])
            covariance = np.zeros(9)
            imu = Imu(header=header(stamp, "imu_link"), orientation=orientation,
                      orientation_covariance=np.diag([1e-4] * 3).ravel(),
                      angular_velocity=Vector3(x=float(wx), y=float(wy), z=float(wz)),
                      angular_velocity_covariance=covariance,
                      linear_acceleration=Vector3(x=float(ax), y=float(ay), z=float(az)),
                      linear_acceleration_covariance=covariance)
            write(imu_c, stamp, imu, "sensor_msgs/msg/Imu")
            fix = NavSatFix(header=header(stamp, "gps_link"),
                            status=NavSatStatus(status=NavSatStatus.STATUS_FIX, service=NavSatStatus.SERVICE_GPS),
                            latitude=float(lat), longitude=float(lon), altitude=float(alt),
                            position_covariance=np.diag([pos_acc ** 2] * 3).ravel(),
                            position_covariance_type=NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN)
            write(fix_c, stamp, fix, "sensor_msgs/msg/NavSatFix")
            odom = Odometry(
                header=header(stamp, "map"), child_frame_id="imu_link",
                pose=PoseCov(pose=pose_msg(pose), covariance=np.diag([pos_acc ** 2] * 3 + [1e-4] * 3).ravel()),
                twist=TwistCov(twist=Twist(linear=Vector3(x=float(body_velocity[0]), y=float(body_velocity[1]),
                                                          z=float(body_velocity[2])),
                                           angular=Vector3(x=float(wx), y=float(wy), z=float(wz))),
                               covariance=np.diag([vel_acc ** 2] * 3 + [1e-4] * 3).ravel()))
            write(odom_c, stamp, odom, "nav_msgs/msg/Odometry")
            write(pose_c, stamp, PoseStamped(header=header(stamp, "map"), pose=pose_msg(pose)),
                  "geometry_msgs/msg/PoseStamped")
            write(velo_pose_c, stamp, PoseStamped(header=header(stamp, "map"), pose=pose_msg(pose @ t_imu_velo)),
                  "geometry_msgs/msg/PoseStamped")

        velo_stamps = read_timestamps(drive / "velodyne_points" / "timestamps.txt")
        fields = [PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, name in enumerate(("x", "y", "z", "intensity"))]
        for stamp, path in zip(velo_stamps, sorted((drive / "velodyne_points" / "data").glob("*.bin"))):
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            cloud = PointCloud2(header=header(stamp, "velo_link"), height=1, width=points.shape[0], fields=fields,
                                is_bigendian=False, point_step=16, row_step=16 * points.shape[0],
                                data=points.view(np.uint8).ravel(), is_dense=True)
            write(cloud_c, stamp, cloud, "sensor_msgs/msg/PointCloud2")

        if image_c is not None:
            camera = drive / "image_02"
            for stamp, path in zip(read_timestamps(camera / "timestamps.txt"), sorted((camera / "data").glob("*.png"))):
                image = CompressedImage(header=header(stamp, "camera_color_left"), format="png",
                                        data=np.frombuffer(path.read_bytes(), dtype=np.uint8))
                write(image_c, stamp, image, "sensor_msgs/msg/CompressedImage")

    print(f"wrote {args.out} ({len(oxts_files)} OXTS packets, {len(velo_stamps)} scans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
