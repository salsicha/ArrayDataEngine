from __future__ import annotations

from .base_source import BaseSource

from io import BytesIO
import zipfile
import numpy as np

import os


class DEMSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, north: list[int], west: list[int], timeout: float = 30.0):
        """Constructor

        """
        super().__init__("")

        self.north = north
        self.west = west
        self.timeout = timeout


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

        import requests

        username = os.getenv("earthdata_username")
        password = os.getenv("earthdata_password")
        if not username or not password:
            raise RuntimeError("earthdata_username and earthdata_password must be set for DEM downloads")

        with requests.Session() as session:
            session.auth = (username, password)

            for n in range(self.north[0], self.north[-1]):

                for w in range(self.west[0], self.west[-1]):

                    url = f"https://e4ftl01.cr.usgs.gov//DP109/SRTM/SRTMGL1.003/2000.02.11/N{n}W{w}.SRTMGL1.hgt.zip"

                    r1 = session.request('get', url, timeout=self.timeout)
                    r = session.get(r1.url, auth=(username, password), timeout=self.timeout)
                    r.raise_for_status()
                    bytes_data = BytesIO(r.content)
                    with zipfile.ZipFile(bytes_data) as zip_file:
                        hgt_content = zip_file.read(f"N{n}W{w}.hgt")

                    side = int(np.sqrt(len(hgt_content) / 2))

                    dem = np.frombuffer(hgt_content, dtype='>i2').reshape((side, side))
                    name = f"N{n}W{w}"

                    yield {"data": dem, \
                            "timestamp": 0, \
                            "topic": "images", \
                            "name": name}
