from __future__ import annotations

import os

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class DataSources:
    """Open a recording, image glob, or DEM region as a message source.

    Args:
        data_path: ROS1 ``.bag``, rosbag2 ``.db3``/``.mcap`` file or bag
            directory, an image file or glob (``.png``, ``.jpg``, ``.jpeg``,
            ``.tif``, ``.tiff``), or ``"DEM"`` for SRTM tiles within `bounds`.
            Extensions are matched case-insensitively.
        period: seconds between images of an image source.
        bounds: ``[[north_min, north_max], [west_min, west_max]]`` for DEM.
        cache_dir, refresh_cache, timeout: DEM download options.
        max_points: point clouds with more finite points than this are
            skipped with a warning; ``None`` (default) means no limit.
            Clouds are always yielded unpadded as ``(N, 3)`` with
            ``point_count`` = N.
        topics: only read these topics (a name or an iterable of names);
            other topics are neither decoded nor listed by get_topics().
        color_order: channel order of colour images from an image source,
            ``"rgb"`` (default, like every ROS source) or ``"bgr"``.
    """


    def __init__(
        self,
        data_path,
        period=0.1,
        bounds=None,
        cache_dir=None,
        refresh_cache: bool = False,
        timeout: float = 30.0,
        max_points: int | None = None,
        topics=None,
        color_order: str = "rgb",
    ):
        """Constructor"""

        from .sources.ros_source import normalize_topics, validate_max_points

        data_path = os.fspath(data_path)
        validate_max_points(max_points)
        self._topics = normalize_topics(topics)

        # db3/bag file extension
        self.file_type = os.path.splitext(data_path)[-1]
        extension = self.file_type.lower()
        bounds = [[0, 0], [0, 0]] if bounds is None else bounds

        # Image globs such as "*.PNG" match regardless of extension case.
        is_image = extension in _IMAGE_EXTENSIONS
        ros_options = {"max_points": max_points, "topics": topics}

        if extension == ".bag" and self._is_sqlite_rosbag(data_path):
            from .sources.db3_source import DB3Source

            self.source = DB3Source(data_path, **ros_options)
        elif extension == ".bag":
            from .sources.bag_source import BagSource

            self.source = BagSource(data_path, **ros_options)
        elif extension in (".db3", ".mcap") or self._is_rosbag2_dir(data_path):
            from .sources.db3_source import DB3Source

            self.source = DB3Source(data_path, **ros_options)
        elif is_image:
            from .sources.img_source import ImgSource

            self.source = ImgSource(data_path, period, self.file_type, color_order=color_order)
        elif data_path == "DEM":
            from .sources.dem_source import DEMSource

            self.source = DEMSource(
                bounds[0],
                bounds[1],
                timeout=timeout,
                cache_dir=cache_dir,
                refresh_cache=refresh_cache,
            )
        else:
            raise ValueError(
                f"{self.file_type} is not supported file type: "
                "[.bag, .db3, .mcap, rosbag2 directory, .png, .jpg, .jpeg, .tif, .tiff]"
            )

        # ROS sources filter topics while reading; others are filtered here.
        self._filter_topics = self._topics is not None and not getattr(
            self.source, "supports_topic_filter", False
        )

        if not self.source.data_exists():
            raise FileNotFoundError(f"No data found for {data_path}")


    @property
    def max_points(self) -> int | None:
        """Point-cloud row limit of the underlying source (None = no limit)."""

        return getattr(self.source, "max_points", None)


    @property
    def topics(self) -> tuple[str, ...] | None:
        """Topics this source was restricted to (None = all)."""

        return self._topics


    @staticmethod
    def _is_sqlite_rosbag(data_path):
        if not os.path.isfile(data_path):
            return False
        try:
            with open(data_path, "rb") as handle:
                return handle.read(16) == b"SQLite format 3\x00"
        except OSError:
            return False


    @staticmethod
    def _is_rosbag2_dir(data_path):
        if not os.path.isdir(data_path):
            return False

        if os.path.exists(os.path.join(data_path, "metadata.yaml")):
            return True

        return any(name.lower().endswith((".db3", ".mcap")) for name in os.listdir(data_path))


    def get_topics(self):
        topics = self.source.get_topics()
        if self._filter_topics:
            topics = [topic for topic in topics if topic in self._topics]
        return topics


    def get_count(self, axis):
        if self._filter_topics and axis not in self._topics:
            return 0
        return self.source.get_count(axis)


    def get_topic_types(self) -> dict:
        """topic -> message type for ROS sources (including undecodable topics)."""

        getter = getattr(self.source, "get_topic_types", None)
        return getter() if callable(getter) else {}


    def get_data_path(self):
        getter = getattr(self.source, "get_data_path", None)
        return getter() if callable(getter) else getattr(self.source, "data_path", None)


    def get_message(self):
        if not self._filter_topics:
            yield from self.source.messages()
            return
        for message in self.source.messages():
            if message.get("topic") in self._topics:
                yield message
