from __future__ import annotations

import os

from .base_source import BaseSource

import rclpy


class TopicSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path):
        """Constructor

        """
        super().__init__(data_path)

        self.data_path = os.path.dirname(data_path)
