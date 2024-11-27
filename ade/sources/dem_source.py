from __future__ import annotations

from pathlib import Path

from .base_source import BaseSource
from ..sensors.pointcloud2_sensor import PointCloudSensor
from ..sensors.image_sensor import ImageSensor

import requests
from io import BytesIO
import zipfile
import numpy as np
import rasterio
import rasterio.merge as merge

import os


class DEMSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, north: list[int], west: list[int]):
        """Constructor

        """
        super().__init__("")

        self.north = north
        self.west = west


    def get_count(self, axis="Images"):
        """Image count
        
        """
        img_count = (self.north[-1] - self.north[0]) * (self.west[-1] - self.west[0])

        return img_count


    def get_duration(self):
        """Duration of recording
        
        """

        return 0


    def get_topics(self):
        return ["images"]


    def data_exists(self):
        return True


    def messages(self, source=None):
        '''Messages from data source
        Yields dictionary:
        - "data": numpy array
        - "timestamp"
        - "topic": "images" for an img source
        - "name": file name
        '''

        output_images = []
        names = []

        for n in range(self.north[0], self.north[-1]):
            
            for w in range(self.west[0], self.west[-1]):

                url = f"https://e4ftl01.cr.usgs.gov//DP109/SRTM/SRTMGL1.003/2000.02.11/N{n}W{w}.SRTMGL1.hgt.zip"
                
                username = os.getenv("earthdata_username")
                password = os.getenv("earthdata_password")

                with requests.Session() as session:
                        session.auth = (username, password)
                        r1 = session.request('get', url)
                        r = session.get(r1.url, auth=(username, password))
                        if r.ok:
                            bytes_data = BytesIO(r.content)
                            zip_file = zipfile.ZipFile(bytes_data)
                            
                            hgt_content = zip_file.read(f"N{n}W{w}.hgt")
                            side = int(np.sqrt(len(hgt_content) / 2))

                            dem = np.frombuffer(hgt_content, dtype='>i2').reshape((side, side))
                            name = f"N{n}W{w}"

                            yield {"data": dem, \
                                    "timestamp": 0, \
                                    "topic": "images", \
                                    "name": name}

