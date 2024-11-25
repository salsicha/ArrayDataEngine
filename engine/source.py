
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import os

from .sources.bag_source import BagSource
from .sources.db3_source import DB3Source
from .sources.img_source import ImgSource
from .sources.dem_source import DEMSource


class DataSources:
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path, period=0.1, bounds=[[0, 0], [0, 0]]):
        """Constructor

        """

        # db3/bag file extension
        self.file_type = os.path.splitext(data_path)[-1]

        img_types = [".png", ".jpg", ".jpeg", ".tiff"]

        # Check file extension in [".bag", ".db3", ".png"]
        if self.file_type == ".bag":
            self.source = BagSource(data_path)
        elif self.file_type == ".db3":
            # AnyReader needs just the path to the folder for db3 files
            self.source = DB3Source(data_path)
        elif self.file_type in img_types:
            self.source = ImgSource(data_path, period, self.file_type)
        elif data_path == "DEM":
            self.source = DEMSource(bounds[0], bounds[1])
        else:
            raise Exception(f"{self.file_type} is not supported file type: [.bag, .db3, .png, .jpg, .jpeg]")

        if not self.source.data_exists():
            raise Exception(f"No data found!")


    def get_topics(self):
        return self.source.get_topics()


    def get_count(self, axis):
        return self.source.get_count(axis)


    def get_message(self):
        for result in self.source.messages(self.source):
            try:
                yield result
            except StopIteration:
                print("End of source")
                return
