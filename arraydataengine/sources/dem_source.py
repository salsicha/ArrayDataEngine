from __future__ import annotations

import contextlib
from functools import partial
from io import BytesIO
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import zipfile

import os

import numpy as np

from .base_source import BaseSource

EARTHDATA_AUTH_HOST = "urs.earthdata.nasa.gov"


def _ordered_range(bounds) -> list[int]:
    first, last = int(bounds[0]), int(bounds[-1])
    return [min(first, last), max(first, last)]


def _rebuild_earthdata_auth(prepared_request, response, allow_insecure_http: bool = False) -> None:
    """requests `Session.rebuild_auth` replacement (NASA Earthdata pattern).

    Keeps the Authorization header only on same-host redirects and on
    redirects to or from urs.earthdata.nasa.gov, and never over plain HTTP
    unless explicitly allowed. Any other redirect (e.g. to a CDN) drops the
    credentials.
    """

    headers = prepared_request.headers
    if "Authorization" not in headers:
        return
    original = urlparse(response.request.url)
    redirect = urlparse(prepared_request.url)
    insecure = redirect.scheme.lower() != "https" and not allow_insecure_http
    trusted = (
        original.hostname == redirect.hostname
        or redirect.hostname == EARTHDATA_AUTH_HOST
        or original.hostname == EARTHDATA_AUTH_HOST
    )
    if insecure or not trusted:
        del headers["Authorization"]


class DEMSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """

    DEFAULT_BASE_URL = "https://e4ftl01.cr.usgs.gov//DP109/SRTM/SRTMGL1.003/2000.02.11/"

    def __init__(
        self,
        north: list[int],
        west: list[int],
        timeout: float = 30.0,
        cache_dir: str | os.PathLike | None = None,
        refresh_cache: bool = False,
        base_url: str | None = None,
        allow_insecure_http: bool = False,
    ):
        """Constructor

        `north` is a latitude range and `west` a longitude range in degrees
        west (negative values are east); reversed ranges are normalized.
        Earthdata credentials are only sent over HTTPS unless
        `allow_insecure_http=True`.
        """
        super().__init__("")

        self.north = _ordered_range(north)
        self.west = _ordered_range(west)
        self.timeout = timeout
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.refresh_cache = bool(refresh_cache)
        self.base_url = base_url or os.getenv("ADE_DEM_BASE_URL") or self.DEFAULT_BASE_URL
        self.allow_insecure_http = bool(allow_insecure_http)


    @staticmethod
    def tile_name(north: int, west: int) -> str:
        """SRTM tile name for the tile whose south-west corner is (north, -west).

        e.g. (37, 122) -> "N37W122", (-3, -5) -> "S03E005".
        """

        lat = int(north)
        lon = -int(west)
        lat_part = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
        lon_part = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
        return lat_part + lon_part


    def get_count(self, axis=None):
        """Image count

        """
        if axis is not None and str(axis).lower() != "images":
            return 0
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

        session_context = None
        session = None
        credentials = None
        try:
            for n in range(self.north[0], self.north[-1]):
                for w in range(self.west[0], self.west[-1]):
                    # SRTM tile names are zero-padded: 2-digit lat, 3-digit lon
                    name = self.tile_name(n, w)
                    hgt_content = self._read_cached_hgt(name)
                    if hgt_content is None:
                        if session is None:
                            session_context, session, credentials = self._open_session()
                        hgt_content = self._download_hgt(session, credentials, name)
                        self._write_cached_hgt(name, hgt_content)

                    yield {
                        "data": self._decode_hgt(hgt_content),
                        "timestamp": 0,
                        "topic": "images",
                        "name": name,
                        "source_uri": str(self._cache_path(name)) if self.cache_dir is not None else None,
                    }
        finally:
            if session_context is not None:
                session_context.__exit__(None, None, None)

    def _open_session(self):
        import requests

        username = os.getenv("earthdata_username")
        password = os.getenv("earthdata_password")
        if not username or not password:
            raise RuntimeError("earthdata_username and earthdata_password must be set for DEM downloads")

        session_context = requests.Session()
        session = session_context.__enter__() if hasattr(session_context, "__enter__") else session_context
        session.auth = (username, password)
        # Earthdata login redirects through urs.earthdata.nasa.gov; keep the
        # credentials only for that hop instead of re-sending them to whatever
        # host the download finally redirects to.
        session.rebuild_auth = partial(_rebuild_earthdata_auth, allow_insecure_http=self.allow_insecure_http)
        return session_context, session, (username, password)

    def _tile_url(self, name: str) -> str:
        url = f"{self.base_url.rstrip('/')}/{name}.SRTMGL1.hgt.zip"
        if urlparse(url).scheme.lower() != "https" and not self.allow_insecure_http:
            raise ValueError(
                f"Refusing to send Earthdata credentials to non-HTTPS DEM URL {url!r}; "
                "use an https base_url or pass allow_insecure_http=True"
            )
        return url

    def _download_hgt(self, session, credentials: tuple[str, str], name: str) -> bytes:
        response = session.get(self._tile_url(name), timeout=self.timeout)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            return zip_file.read(f"{name}.hgt")

    def _decode_hgt(self, hgt_content: bytes) -> np.ndarray:
        side = int(np.sqrt(len(hgt_content) / 2))
        if side * side * 2 != len(hgt_content):
            raise ValueError("HGT payload length does not describe a square int16 tile")
        # HGT samples are big-endian; return a native, writable int16 array.
        return np.frombuffer(hgt_content, dtype=">i2").astype(np.int16).reshape((side, side))

    def _cache_path(self, name: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured")
        return self.cache_dir / f"{name}.hgt"

    def _read_cached_hgt(self, name: str) -> bytes | None:
        if self.cache_dir is None or self.refresh_cache:
            return None
        path = self._cache_path(name)
        if not path.exists():
            return None
        return path.read_bytes()

    def _write_cached_hgt(self, name: str, hgt_content: bytes) -> None:
        if self.cache_dir is None:
            return
        path = self._cache_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename so an interrupted write never
        # leaves a truncated tile that later reads would trust.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(hgt_content)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
