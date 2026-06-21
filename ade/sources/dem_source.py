from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import os

import numpy as np

from .base_source import BaseSource


class DEMSource(BaseSource):
    """Data Sources Class
    Attributes:
    Args:
    Returns:
    """

    def __init__(
        self,
        north: list[int],
        west: list[int],
        timeout: float = 30.0,
        cache_dir: str | os.PathLike | None = None,
        refresh_cache: bool = False,
    ):
        """Constructor

        """
        super().__init__("")

        self.north = north
        self.west = west
        self.timeout = timeout
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.refresh_cache = bool(refresh_cache)


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

        session_context = None
        session = None
        credentials = None
        try:
            for n in range(self.north[0], self.north[-1]):
                for w in range(self.west[0], self.west[-1]):
                    name = f"N{n}W{w}"
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
        return session_context, session, (username, password)

    def _download_hgt(self, session, credentials: tuple[str, str], name: str) -> bytes:
        username, password = credentials
        url = f"https://e4ftl01.cr.usgs.gov//DP109/SRTM/SRTMGL1.003/2000.02.11/{name}.SRTMGL1.hgt.zip"
        r1 = session.request("get", url, timeout=self.timeout)
        r = session.get(r1.url, auth=(username, password), timeout=self.timeout)
        r.raise_for_status()
        bytes_data = BytesIO(r.content)
        with zipfile.ZipFile(bytes_data) as zip_file:
            return zip_file.read(f"{name}.hgt")

    def _decode_hgt(self, hgt_content: bytes) -> np.ndarray:
        side = int(np.sqrt(len(hgt_content) / 2))
        if side * side * 2 != len(hgt_content):
            raise ValueError("HGT payload length does not describe a square int16 tile")
        return np.frombuffer(hgt_content, dtype=">i2").reshape((side, side))

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
        path.write_bytes(hgt_content)
