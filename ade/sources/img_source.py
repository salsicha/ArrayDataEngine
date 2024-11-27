from __future__ import annotations

from rosbags.highlevel import AnyReader

from pathlib import Path

from .base_source import BaseSource

from glob import glob

import cv2
import fnmatch
import os


class ImgSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path, period, file_type):
        """Constructor

        """

        super().__init__(data_path)

        self.period = period
        self.data_path = data_path
        self.file_type = file_type

        images = glob(f"{data_path}")
        images.sort()

        self.images = images 


    def get_count(self, axis="Images"):
        img_count = len(fnmatch.filter(os.listdir(os.path.dirname(self.data_path)), '*'+self.file_type))
        return img_count


    def get_duration(self):
        """Duration of recording
        
        """

        img_count = len(fnmatch.filter(os.listdir(os.path.dirname(self.data_path)), '*'+self.file_type))
        duration = img_count * self.period
        return duration


    def get_topics(self):
        topics = ["images"]
        return topics


    def data_exists(self):
        if len(self.images) > 0:
            return True
        return False


    def messages(self, source=None):
        '''Messages from data source
        Yields dictionary:
        - "data": numpy array
        - "timestamp"
        - "topic": "images" for an img source
        - "name": file name
        '''

        for count, img in enumerate(self.images):
            # Yield: numpy array of data, timestamp, msg topic, filename
            # "images" topic is fake, but is needed in the buffer

            yield {"data": cv2.imread(img, cv2.IMREAD_UNCHANGED), \
                    "timestamp": count * self.period, \
                    "topic": "images", \
                    "name": os.path.basename(img)}
