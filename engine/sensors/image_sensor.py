from __future__ import annotations

import numpy as np
import ros2_numpy as rnp

from .base_sensor import BaseSensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from rosbags.serde import deserialize_cdr


class ImageSensor(BaseSensor):
    """Point Cloud Sensor Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, rawdata, msgtype):
        """Constructor

        """
        super().__init__(rawdata, msgtype)