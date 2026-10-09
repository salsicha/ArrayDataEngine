from __future__ import annotations

from .base_source import BaseSource

from glob import glob

import cv2
import os
import re

_COLOR_ORDERS = ("rgb", "bgr")


def _natural_key(path: str):
    """Sort frame_2 before frame_10 by comparing embedded numbers numerically."""
    # isdecimal, not isdigit: superscripts such as "²" are digits int() rejects.
    return [int(part) if part.isdecimal() else part for part in re.split(r"(\d+)", path)]


class ImgSource(BaseSource):
    """Image files matching a path or glob, yielded on the synthetic "images" topic.

    Colour images are returned in RGB(A) order like every ROS source
    (`color_order="bgr"` keeps OpenCV's BGR order); each message carries an
    ``encoding`` key describing the array (``rgb8``, ``mono8``, ``mono16``, ...).
    A path whose glob matches nothing is tried as a literal file name, so
    names with glob metacharacters such as ``frame[1].png`` work.
    """


    def __init__(self, data_path, period, file_type, color_order: str = "rgb"):
        """Constructor

        """

        super().__init__(data_path)

        color_order = str(color_order).lower()
        if color_order not in _COLOR_ORDERS:
            raise ValueError(f"color_order must be one of {_COLOR_ORDERS}, got {color_order!r}")

        self.period = period
        self.data_path = data_path
        self.file_type = file_type
        self.color_order = color_order

        images = glob(data_path)
        if not images and os.path.isfile(data_path):
            images = [data_path]
        images.sort(key=_natural_key)

        self.images = images


    def get_count(self, axis=None):
        # Only the synthetic "images" topic exists; other topics have no messages.
        if axis is not None and str(axis).lower() != "images":
            return 0
        return len(self.images)


    def get_duration(self):
        """Duration of recording

        """

        return self.get_count() * self.period


    def get_topics(self):
        topics = ["images"]
        return topics


    def data_exists(self):
        return len(self.images) > 0


    def messages(self, source=None):
        '''Messages from data source
        Yields dictionary:
        - "data": numpy array (colour images in `color_order`)
        - "timestamp"
        - "topic": "images" for an img source
        - "name": file name
        - "encoding": layout of "data", e.g. "rgb8" or "mono16"
        '''

        from ..sensors.image_sensor import array_encoding, bgr_to_rgb

        for count, img in enumerate(self.images):
            # Yield: numpy array of data, timestamp, msg topic, filename
            # "images" topic is fake, but is needed in the buffer

            data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if data is None:
                raise ValueError(f"Unable to read image: {img}")
            if self.color_order == "rgb":
                data = bgr_to_rgb(data)

            yield {"data": data, \
                    "timestamp": count * self.period, \
                    "topic": "images", \
                    "name": os.path.basename(img), \
                    "encoding": array_encoding(data, self.color_order)}
