from __future__ import annotations

import numpy as np

DEFAULT_TOPICS = (
    {"name": "/imu", "kind": "imu", "rate": 50.0},
    {"name": "/odom", "kind": "odometry", "rate": 20.0},
    {"name": "/points", "kind": "pointcloud", "rate": 5.0},
)

_SUPPORTED_KINDS = {"imu", "odometry", "pointcloud", "navsat", "image", "scalar"}


class SyntheticSource:
    """Deterministic synthetic multi-topic source for demos and tests.

    Simulates a robot driving one loop of a circle (radius `radius` meters)
    over `duration` seconds. All topics are geometrically consistent with that
    trajectory: odometry reports the pose on the circle, the IMU reports the
    matching yaw rate and centripetal acceleration, point clouds observe a
    fixed set of world landmarks from the sensor frame (so stitching scans
    with the odometry poses reconstructs the landmark map), and navsat
    positions convert the trajectory to WGS84 around a reference coordinate.

    Message layouts match the ROS-backed sources: IMU messages are ``(6, 4)``
    arrays, odometry ``(8, 4)``, point clouds ``(N, 3)``, navsat
    ``[lat, lon, alt]``, images ``(H, W)`` float64.

    Args:
        topics: iterable of dicts with keys ``name``, ``kind``
            (imu | odometry | pointcloud | navsat | image | scalar), ``rate``
            in Hz, and optional per-kind parameters (``points`` for
            pointcloud, ``shape`` for image). Defaults to imu + odometry +
            pointcloud.
        duration: seconds of data (one full loop of the circle).
        radius: circle radius in meters.
        seed: RNG seed; identical seeds yield identical messages.
    """

    def __init__(self, topics=None, duration: float = 5.0, radius: float = 10.0, seed: int = 0):
        if duration <= 0:
            raise ValueError("duration must be positive")
        if radius <= 0:
            raise ValueError("radius must be positive")

        self.duration = float(duration)
        self.radius = float(radius)
        self.seed = int(seed)
        self.topics = [dict(topic) for topic in (DEFAULT_TOPICS if topics is None else topics)]

        for topic in self.topics:
            if "name" not in topic or "rate" not in topic:
                raise ValueError("each topic needs at least 'name' and 'rate'")
            kind = topic.setdefault("kind", "scalar")
            if kind not in _SUPPORTED_KINDS:
                raise ValueError(f"unsupported kind {kind!r}; expected one of {sorted(_SUPPORTED_KINDS)}")
            if topic["rate"] <= 0:
                raise ValueError("topic rate must be positive")

        rng = np.random.default_rng(self.seed)
        # Fixed world landmarks observed by every point-cloud scan.
        self._landmarks = rng.uniform(-1.5 * self.radius, 1.5 * self.radius, size=(160, 3))
        self._landmarks[:, 2] = rng.uniform(0.0, 2.0, size=160)

    # -- trajectory ---------------------------------------------------------

    def _pose_at(self, t: float) -> tuple[np.ndarray, float]:
        """World position and yaw of the simulated robot at time t."""
        omega = 2.0 * np.pi / self.duration
        angle = omega * t
        position = np.array([
            self.radius * np.cos(angle),
            self.radius * np.sin(angle),
            0.0,
        ])
        yaw = angle + np.pi / 2.0  # tangent to the circle
        return position, yaw

    def _omega(self) -> float:
        return 2.0 * np.pi / self.duration

    # -- message payloads ---------------------------------------------------

    def _imu_message(self, t: float) -> np.ndarray:
        _, yaw = self._pose_at(t)
        omega = self._omega()
        speed = self.radius * omega
        centripetal = speed * omega
        quat = [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]
        return np.array([
            quat,
            [1e-6, 1e-6, 1e-6, 0.0],
            [0.0, 0.0, omega, 0.0],
            [1e-6, 1e-6, 1e-6, 0.0],
            [0.0, centripetal, 9.81, 0.0],
            [1e-4, 1e-4, 1e-4, 0.0],
        ])

    def _odometry_message(self, t: float) -> np.ndarray:
        position, yaw = self._pose_at(t)
        omega = self._omega()
        speed = self.radius * omega
        quat = [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]
        return np.array([
            [position[0], position[1], position[2], 0.0],
            [1e-4, 1e-4, 1e-4, 0.0],
            quat,
            [1e-6, 1e-6, 1e-6, 0.0],
            [speed, 0.0, 0.0, 0.0],
            [1e-4, 1e-4, 1e-4, 0.0],
            [0.0, 0.0, omega, 0.0],
            [1e-6, 1e-6, 1e-6, 0.0],
        ])

    def _pointcloud_message(self, t: float, rng: np.random.Generator, max_range: float, points: int) -> np.ndarray:
        position, yaw = self._pose_at(t)
        offsets = self._landmarks - position
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        local = offsets.copy()
        local[:, 0] = offsets[:, 0] * cos_yaw - offsets[:, 1] * sin_yaw
        local[:, 1] = offsets[:, 0] * sin_yaw + offsets[:, 1] * cos_yaw
        in_range = np.linalg.norm(local[:, :2], axis=1) <= max_range
        visible = local[in_range]
        if visible.shape[0] > points:
            visible = visible[:points]
        visible = visible + rng.normal(0.0, 0.01, size=visible.shape)
        # Buffers need a fixed per-topic message shape (like PointCloudSensor,
        # which pads to max_points); zero-pad the scan to exactly `points`.
        scan = np.zeros((points, 3), dtype=np.float64)
        scan[: visible.shape[0]] = visible
        return scan

    def _navsat_message(self, t: float) -> np.ndarray:
        from ..ops.nav import enu_to_navsat

        position, _ = self._pose_at(t)
        return enu_to_navsat(position.reshape(1, 3), 37.0, -122.0, 10.0)[0]

    def _image_message(self, t: float, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        rows = np.linspace(0.0, 1.0, height)[:, None]
        cols = np.linspace(0.0, 1.0, width)[None, :]
        image = rows * cols
        spot_col = int((t / self.duration) * (width - 1))
        image[:, spot_col] = 2.0
        return image

    # -- source interface ---------------------------------------------------

    def get_topics(self):
        return [topic["name"] for topic in self.topics]

    def get_count(self, axis: str) -> int:
        for topic in self.topics:
            if topic["name"] == axis:
                return int(np.floor(self.duration * topic["rate"])) + 1
        return 0

    def get_duration(self) -> float:
        return self.duration

    def data_exists(self) -> bool:
        return True

    def get_data_path(self) -> str:
        return f"synthetic://seed={self.seed}"

    def messages(self):
        rng = np.random.default_rng(self.seed + 1)
        schedule = []
        for topic in self.topics:
            count = self.get_count(topic["name"])
            times = np.arange(count) / topic["rate"]
            for order, t in enumerate(times):
                schedule.append((float(t), topic["name"], topic, order))
        schedule.sort(key=lambda item: (item[0], item[1]))

        for t, name, topic, order in schedule:
            kind = topic["kind"]
            if kind == "imu":
                data, frame = self._imu_message(t), "imu_link"
            elif kind == "odometry":
                data, frame = self._odometry_message(t), "odom"
            elif kind == "pointcloud":
                data = self._pointcloud_message(
                    t, rng, topic.get("max_range", 2.0 * self.radius), topic.get("points", 120)
                )
                frame = "lidar"
            elif kind == "navsat":
                data, frame = self._navsat_message(t), "gps"
            elif kind == "image":
                data, frame = self._image_message(t, tuple(topic.get("shape", (32, 48)))), "camera"
            else:  # scalar
                data, frame = np.array([np.sin(2.0 * np.pi * t / self.duration)]), None

            message = {
                "data": data,
                "timestamp": t,
                "topic": name,
                "name": f"{kind}_{order}",
            }
            if frame is not None:
                message["frame_id"] = frame
            yield message

    def get_message(self):
        yield from self.messages()
