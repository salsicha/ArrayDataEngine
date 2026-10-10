"""DEMSource's authenticated SRTM download path against a synthetic NASA Earthdata.

scripts/datasets/mock_earthdata.py serves the real host names over local HTTPS
(Earthdata Login OAuth redirect, archive session cookie, pre-signed S3 hand-off),
so these tests run the library's unmodified default download path offline.
"""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

requests = pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mock_earthdata", ROOT / "scripts" / "datasets" / "mock_earthdata.py")
mock_earthdata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock_earthdata)

SAMPLES = 121  # small synthetic tiles keep the tests fast
TILES = ["N37W122", "N37W123", "N38W122", "N38W123"]

try:
    import cryptography  # noqa: F401
except ImportError:
    if shutil.which("openssl") is None:
        pytest.skip("needs the cryptography package or the openssl command for the mock TLS",
                    allow_module_level=True)


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("earthdata_username", mock_earthdata.SYNTHETIC_USER)
    monkeypatch.setenv("earthdata_password", mock_earthdata.SYNTHETIC_PASSWORD)
    monkeypatch.delenv("ADE_DEM_BASE_URL", raising=False)


def _expected(name):
    raw = np.frombuffer(mock_earthdata.synthetic_hgt(name, SAMPLES), dtype=">i2")
    return raw.reshape(SAMPLES, SAMPLES)


def test_default_download_path_completes_the_earthdata_login_flow(tmp_path, credentials):
    from arraydataengine import DataSources
    from arraydataengine.ops import mosaic_dem_tiles, sample_elevation_at_latlon

    with mock_earthdata.MockEarthdata(tiles=TILES, samples=SAMPLES) as mock:
        tiles = list(DataSources("DEM", bounds=[[37, 39], [122, 124]], cache_dir=tmp_path).get_message())
        log = list(mock.log)

        assert [t["name"] for t in tiles] == TILES
        for tile in tiles:
            np.testing.assert_array_equal(tile["data"], _expected(tile["name"]))
        assert sorted(p.name for p in tmp_path.iterdir()) == [f"{name}.hgt" for name in TILES]

        hosts = [entry[0] for entry in log]
        assert {mock_earthdata.TILE_HOST, mock_earthdata.URS_HOST, mock_earthdata.S3_HOST} <= set(hosts)
        # Credentials reach only the archive and the login host; S3 would reject them (400).
        assert {host for host, _, had_auth, _ in log if had_auth} <= {mock_earthdata.TILE_HOST, mock_earthdata.URS_HOST}
        assert all(status == 200 for host, _, _, status in log if host == mock_earthdata.S3_HOST)
        assert [status for host, path, _, status in log if path == "/oauth/authorize"] == [302]  # one login

        mosaic, lats, lons = mosaic_dem_tiles(tiles, return_index=True, overlap=1)
        for _, lat, lon, height in mock_earthdata.PEAKS:
            assert float(sample_elevation_at_latlon(mosaic, lat, lon, index=(lats, lons))) == pytest.approx(height, abs=1)

        before = len(mock.log)
        cached = list(DataSources("DEM", bounds=[[37, 39], [122, 124]], cache_dir=tmp_path).get_message())
        assert len(mock.log) == before and len(cached) == len(TILES)


def test_wrong_password_fails_at_the_login_host_and_caches_nothing(tmp_path, credentials, monkeypatch):
    from arraydataengine import DataSources

    monkeypatch.setenv("earthdata_password", "wrong")
    with mock_earthdata.MockEarthdata(tiles=TILES, samples=SAMPLES) as mock:
        with pytest.raises(requests.HTTPError, match="401"):
            list(DataSources("DEM", bounds=[[37, 38], [122, 123]], cache_dir=tmp_path).get_message())
        assert mock.log[-1][0] == mock_earthdata.URS_HOST and mock.log[-1][3] == 401
    assert list(tmp_path.iterdir()) == []


def test_tiles_the_archive_does_not_have_are_skipped(tmp_path, credentials, caplog):
    from arraydataengine import DataSources

    with mock_earthdata.MockEarthdata(tiles=["N37W122"], samples=SAMPLES):  # e.g. open ocean
        tiles = list(DataSources("DEM", bounds=[[37, 38], [122, 124]], cache_dir=tmp_path).get_message())
    assert [t["name"] for t in tiles] == ["N37W122"]
    assert "N37W123" in caplog.text


def test_hook_routes_another_interpreter_to_the_mock(tmp_path, credentials):
    import os
    import subprocess
    import sys

    hook_dir = tmp_path / "hook"
    mock_earthdata.write_hook(hook_dir)
    script = (
        "from arraydataengine import DataSources\n"
        f"tiles = list(DataSources('DEM', bounds=[[37, 38], [122, 123]], cache_dir={str(tmp_path / 'cache')!r}).get_message())\n"
        "print(tiles[0]['name'], tiles[0]['data'].shape)\n"
    )
    with mock_earthdata.MockEarthdata(tiles=TILES, samples=SAMPLES, intercept=False) as mock:
        env = {**os.environ, **mock.environment(),
               "PYTHONPATH": os.pathsep.join([str(hook_dir), str(ROOT), os.environ.get("PYTHONPATH", "")])}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"N37W122 ({SAMPLES}, {SAMPLES})"


def _tls_backends():
    backends = []
    try:
        import cryptography  # noqa: F401

        backends.append("cryptography")
    except ImportError:
        pass
    if shutil.which("openssl") is not None:
        backends.append("openssl")
    return backends


@pytest.mark.parametrize("backend", _tls_backends())
def test_mock_certificates_pass_strict_x509_verification(backend, monkeypatch):
    # Python 3.13+ verifies X.509 strictly by default; check it on every Python.
    import socket
    import ssl

    if backend == "openssl":
        monkeypatch.setattr(mock_earthdata, "_make_tls", mock_earthdata._make_tls_openssl)
    with mock_earthdata.MockEarthdata(intercept=False) as mock:
        context = ssl.create_default_context(cafile=str(mock.ca_file))
        context.verify_flags |= ssl.VERIFY_X509_STRICT | ssl.VERIFY_X509_PARTIAL_CHAIN
        for host in mock_earthdata.HOSTS:
            with socket.create_connection(("127.0.0.1", mock.port), timeout=10) as raw, \
                    context.wrap_socket(raw, server_hostname=host) as tls:
                assert tls.version() is not None
