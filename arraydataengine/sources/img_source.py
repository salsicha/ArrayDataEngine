from __future__ import annotations

from .base_source import BaseSource

from glob import glob

import cv2
import os
import re

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_TIFF_EXTENSIONS = (".tif", ".tiff")

_COLOR_ORDERS = ("rgb", "bgr")
_FLIR_MODES = ("raw", "celsius", "kelvin", "fahrenheit")


def _natural_key(path: str):
    """Sort frame_2 before frame_10 by comparing embedded numbers numerically."""
    # isdecimal, not isdigit: superscripts such as "²" are digits int() rejects.
    return [int(part) if part.isdecimal() else part for part in re.split(r"(\d+)", path)]


def _tiff_page_count(path: str) -> int:
    """Pages in a (possibly multi-page) TIFF; 1 when it cannot be told."""

    try:
        return max(1, int(cv2.imcount(path)))
    except (cv2.error, AttributeError):
        pass
    try:
        from PIL import Image

        with Image.open(path) as image:
            return max(1, int(getattr(image, "n_frames", 1)))
    except Exception:
        return 1


def _read_tiff_page(path: str, page: int):
    """One page of a multi-page TIFF at its native depth, in OpenCV channel order."""

    try:
        ok, mats = cv2.imreadmulti(path, page, 1, flags=cv2.IMREAD_UNCHANGED)
        if ok and mats:
            return mats[0]
    except (cv2.error, TypeError):  # OpenCV without the ranged overload
        pass
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        image.seek(page)
        data = np.array(image)
    if data.ndim == 3 and data.shape[-1] in (3, 4):  # PIL is RGB(A); match cv2's BGR(A)
        data = data[..., [2, 1, 0, *range(3, data.shape[-1])]]
    return data


class ImgSource(BaseSource):
    """Image files matching a path or glob, yielded on the synthetic "images" topic.

    Colour images are returned in RGB(A) order like every ROS source
    (`color_order="bgr"` keeps OpenCV's BGR order); each message carries an
    ``encoding`` key describing the array (``rgb8``, ``mono8``, ``mono16``, ...).
    A path whose glob matches nothing is tried as a literal file name, so
    names with glob metacharacters such as ``frame[1].png`` work.

    Images keep their native bit depth (16-bit thermal frames stay ``uint16``
    with encoding ``mono16``). A multi-page TIFF stack yields one message per
    page, named ``<file>#<page>`` (pages count from 0).

    `flir` reads FLIR radiometric JPEGs (R-JPEG) for their thermal data
    instead of the palette image: ``"raw"`` yields the sensor counts
    (``uint16``, ``mono16``); ``"celsius"``, ``"kelvin"`` or ``"fahrenheit"``
    yield the object temperature (``float32``, ``32FC1``) computed with each
    file's own Planck constants and measurement parameters. Each message then
    also carries ``"flir"``: that calibration and camera information
    (`arraydataengine.ops.FlirCalibration.from_dict` reads it back).
    """


    def __init__(self, data_path, period, file_type, color_order: str = "rgb", flir: str | None = None):
        """Constructor

        """

        super().__init__(data_path)

        color_order = str(color_order).lower()
        if color_order not in _COLOR_ORDERS:
            raise ValueError(f"color_order must be one of {_COLOR_ORDERS}, got {color_order!r}")
        if flir is not None:
            flir = str(flir).lower()
            if flir not in _FLIR_MODES:
                raise ValueError(f"flir must be one of {_FLIR_MODES} or None, got {flir!r}")

        self.period = period
        self.data_path = data_path
        self.file_type = file_type
        self.color_order = color_order
        self.flir = flir

        images = glob(data_path)
        if any(char in os.path.basename(data_path) for char in "*?["):
            # A wildcard pattern (e.g. "*.tif*") may also match sidecar files.
            images = [path for path in images if os.path.splitext(path)[1].lower() in _IMAGE_EXTENSIONS]
        if not images and os.path.isfile(data_path):
            images = [data_path]
        images.sort(key=_natural_key)

        self.images = images
        # (file, page) per message; page is None for a single-image file.
        self.frames = []
        for path in images:
            pages = _tiff_page_count(path) if os.path.splitext(path)[1].lower() in _TIFF_EXTENSIONS else 1
            self.frames.extend([(path, None)] if pages == 1 else [(path, page) for page in range(pages)])


    def get_count(self, axis=None):
        # Only the synthetic "images" topic exists; other topics have no messages.
        if axis is not None and str(axis).lower() != "images":
            return 0
        return len(self.frames)


    def get_duration(self):
        """Duration of recording

        """

        return self.get_count() * self.period


    def get_topics(self):
        topics = ["images"]
        return topics


    def data_exists(self):
        return len(self.frames) > 0


    def messages(self, source=None):
        '''Messages from data source
        Yields dictionary:
        - "data": numpy array (colour images in `color_order`)
        - "timestamp"
        - "topic": "images" for an img source
        - "name": file name (``<file>#<page>`` for a page of a multi-page TIFF)
        - "encoding": layout of "data", e.g. "rgb8" or "mono16"
        - "flir": calibration and camera information (FLIR radiometric mode only)
        '''

        if self.flir is not None:
            yield from self._flir_messages()
            return

        from ..sensors.image_sensor import array_encoding, bgr_to_rgb

        for count, (img, page) in enumerate(self.frames):
            # Yield: numpy array of data, timestamp, msg topic, filename
            # "images" topic is fake, but is needed in the buffer

            data = cv2.imread(img, cv2.IMREAD_UNCHANGED) if page is None else _read_tiff_page(img, page)
            if data is None:
                raise ValueError(f"Unable to read image: {img}" + ("" if page is None else f" page {page}"))
            if self.color_order == "rgb":
                data = bgr_to_rgb(data)

            yield {"data": data, \
                    "timestamp": count * self.period, \
                    "topic": "images", \
                    "name": os.path.basename(img) if page is None else f"{os.path.basename(img)}#{page}", \
                    "encoding": array_encoding(data, self.color_order)}


    def _flir_messages(self):
        import numpy as np

        from ..ops.flir import read_flir_radiometric

        for count, (img, _) in enumerate(self.frames):
            image = read_flir_radiometric(img)
            if self.flir == "raw":
                data, encoding = image.raw, "mono16"
            else:
                data, encoding = image.temperature(unit=self.flir).astype(np.float32), "32FC1"
            yield {"data": data,
                   "timestamp": count * self.period,
                   "topic": "images",
                   "name": os.path.basename(img),
                   "encoding": encoding,
                   "flir": image.metadata()}
