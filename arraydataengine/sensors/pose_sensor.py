from __future__ import annotations

import numpy as np

from .base_sensor import BaseSensor


class PoseSensor(BaseSensor):
    """Convert ROS Pose and PoseStamped messages to ADE XYZ + XYZW pose arrays."""

    def numpyify(self) -> tuple:
        msg = self.deserialize()
        self._capture_header_metadata(msg)

        pose = getattr(msg, "pose", msg)
        position = pose.position
        orientation = pose.orientation
        npified = np.array(
            [
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ],
            dtype=np.float64,
        )

        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            ts = 0.0
        else:
            ts = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return npified, msg.__class__.__name__, ts
