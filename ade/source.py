from __future__ import annotations

import os


class DataSources:
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_path, period=0.1, bounds=None):
        """Constructor

        """

        # db3/bag file extension
        self.file_type = os.path.splitext(data_path)[-1]
        bounds = [[0, 0], [0, 0]] if bounds is None else bounds

        img_types = [".png", ".jpg", ".jpeg", ".tiff"]

        # Check file extension in [".bag", ".db3", ".png"]
        if self.file_type == ".bag":
            from .sources.bag_source import BagSource

            self.source = BagSource(data_path)
        elif self.file_type == ".db3" or self._is_rosbag2_dir(data_path):
            from .sources.db3_source import DB3Source

            self.source = DB3Source(data_path)
        elif self.file_type in img_types:
            from .sources.img_source import ImgSource

            self.source = ImgSource(data_path, period, self.file_type)
        elif data_path == "DEM":
            from .sources.dem_source import DEMSource

            self.source = DEMSource(bounds[0], bounds[1])
        else:
            raise ValueError(
                f"{self.file_type} is not supported file type: [.bag, .db3, rosbag2 directory, .png, .jpg, .jpeg, .tiff]"
            )

        if not self.source.data_exists():
            raise FileNotFoundError(f"No data found for {data_path}")


    @staticmethod
    def _is_rosbag2_dir(data_path):
        if not os.path.isdir(data_path):
            return False

        if os.path.exists(os.path.join(data_path, "metadata.yaml")):
            return True

        return any(name.lower().endswith(".db3") for name in os.listdir(data_path))


    def get_topics(self):
        return self.source.get_topics()


    def get_count(self, axis):
        return self.source.get_count(axis)


    def get_message(self):
        yield from self.source.messages()
