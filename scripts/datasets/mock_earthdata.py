"""A synthetic NASA Earthdata service for testing authenticated SRTM downloads offline.

It reproduces the flow a real ``DataSources("DEM", ...)`` download goes through,
on the real host names, so the library code runs unmodified:

1. ``GET https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/<name>.SRTMGL1.hgt/<name>.SRTMGL1.hgt.zip``
   without a session cookie -> 302 to Earthdata Login (``app_type=401``).
2. ``GET https://urs.earthdata.nasa.gov/oauth/authorize?...`` -> 401 without valid
   Basic credentials, else 302 back to ``/login?code=...&state=...``.
3. ``GET https://data.lpdaac.earthdatacloud.nasa.gov/login?code=...`` -> sets the
   session cookie and 302s to the original tile URL.
4. With the cookie, the tile URL 307s to a pre-signed S3 URL on
   ``lp-prod-protected.s3.us-west-2.amazonaws.com``, which serves the zip and, like
   S3, rejects requests that also carry an ``Authorization`` header (400).
   Unknown tiles are 404.

Tiles are synthetic SRTMGL1 tiles (big-endian int16, 3601 x 3601 by default)
sampled from one continuous terrain function, so shared tile edges agree exactly.
The terrain has known summits (`PEAKS`) and an SRTM void, for checking results.

Interception is in-process: `MockEarthdata` serves HTTPS on 127.0.0.1 with a
throw-away certificate authority, routes the three host names there by patching
``socket.getaddrinfo``, and points ``REQUESTS_CA_BUNDLE`` at its CA. For another
process (e.g. a notebook kernel), `write_hook()` creates a ``sitecustomize.py``
that installs the same routing from environment variables.

Usage::

    python scripts/datasets/mock_earthdata.py --check        # full DEMSource flow, offline
    python scripts/datasets/mock_earthdata.py --check --samples 1201
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime
import hashlib
import hmac
import io
import os
import secrets
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np

TILE_HOST = "data.lpdaac.earthdatacloud.nasa.gov"
URS_HOST = "urs.earthdata.nasa.gov"
S3_HOST = "lp-prod-protected.s3.us-west-2.amazonaws.com"
HOSTS = (TILE_HOST, URS_HOST, S3_HOST)
TILE_PREFIX = "/lp-prod-protected/SRTMGL1.003/"
SYNTHETIC_USER, SYNTHETIC_PASSWORD = "synthetic-user", "synthetic-password"

# Known summits of the synthetic terrain: (name, lat, lon, elevation m).
PEAKS = (
    ("Synthetic Peak", 37.625, -121.625, 1450.0),  # on the 121/1201/3601-sample grids
    ("Twin Summit", 38.35, -122.45, 980.0),
)
VOID_BOX = (37.20, 37.23, -122.75, -122.72)  # lat0, lat1, lon0, lon1 set to -32768


def terrain(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Continuous synthetic elevation (metres) at latitude/longitude."""
    z = 120.0 + 80.0 * np.sin(np.radians(lat) * 400.0) * np.cos(np.radians(lon) * 300.0)
    for _, plat, plon, height in PEAKS:  # blend each summit in, so it peaks exactly at `height`
        weight = np.exp(-(((lat - plat) / 0.05) ** 2 + ((lon - plon) / 0.05) ** 2))
        z = z * (1.0 - weight) + height * weight
    return z


def tile_name(lat: int, lon: int) -> str:
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def synthetic_hgt(name: str, samples: int = 3601) -> bytes:
    """HGT bytes (big-endian int16, north row first) of tile `name`, e.g. N37W122."""
    lat0 = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon0 = int(name[4:7]) * (1 if name[3] == "E" else -1)
    step = 1.0 / (samples - 1)
    lats = lat0 + 1.0 - step * np.arange(samples)
    lons = lon0 + step * np.arange(samples)
    lat, lon = np.meshgrid(lats, lons, indexing="ij")
    z = np.rint(terrain(lat, lon))
    lat_a, lat_b, lon_a, lon_b = VOID_BOX
    z[(lat >= lat_a) & (lat <= lat_b) & (lon >= lon_a) & (lon <= lon_b)] = -32768
    return z.astype(">i2").tobytes()


def _make_tls(directory: Path) -> tuple[Path, Path, Path]:
    """CA certificate plus a server certificate/key for the mocked host names."""
    ca_pem, cert_pem, key_pem = directory / "ca.pem", directory / "server.pem", directory / "server.key"
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        return _make_tls_openssl(directory)
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ArrayDataEngine mock Earthdata CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
          .not_valid_after(now + datetime.timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                                       encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, TILE_HOST)]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(h) for h in HOSTS]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    ca_pem.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    return ca_pem, cert_pem, key_pem


def _make_tls_openssl(directory: Path) -> tuple[Path, Path, Path]:
    ca_key, ca_pem = directory / "ca.key", directory / "ca.pem"
    key_pem, csr, cert_pem, ext = directory / "server.key", directory / "server.csr", directory / "server.pem", directory / "ext.cnf"
    # Python 3.13+ verifies X.509 strictly: the CA needs key usage, the leaf key identifiers.
    ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
                   "extendedKeyUsage=serverAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n"
                   "subjectAltName=" + ",".join(f"DNS:{h}" for h in HOSTS) + "\n")

    def run(*args):
        subprocess.run(["openssl", *args], check=True, capture_output=True)

    run("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-days", "1",
        "-subj", "/CN=ArrayDataEngine mock Earthdata CA", "-keyout", str(ca_key), "-out", str(ca_pem),
        "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        "-addext", "subjectKeyIdentifier=hash")
    run("req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
        "-subj", f"/CN={TILE_HOST}", "-keyout", str(key_pem), "-out", str(csr))
    run("x509", "-req", "-in", str(csr), "-CA", str(ca_pem), "-CAkey", str(ca_key), "-CAcreateserial",
        "-days", "1", "-extfile", str(ext), "-out", str(cert_pem))
    return ca_pem, cert_pem, key_pem


class MockEarthdata:
    """In-process synthetic Earthdata (see the module docstring).

    ``with MockEarthdata(tiles=["N37W122", ...]) as mock:`` routes the real host
    names to it for this process; ``mock.log`` records every request as
    ``(host, path, had_authorization, status)``.
    """

    def __init__(self, tiles=None, samples: int = 3601, username: str = SYNTHETIC_USER,
                 password: str = SYNTHETIC_PASSWORD, intercept: bool = True):
        self.samples = samples
        self.tiles = None if tiles is None else set(tiles)
        self.username, self.password = username, password
        self.intercept = intercept
        self.log: list[tuple[str, str, bool, int]] = []
        self._codes: set[str] = set()
        self._sessions: set[str] = set()
        self._signing_key = secrets.token_bytes(16)
        self._zips: dict[str, bytes] = {}
        self._stack = contextlib.ExitStack()

    # -- lifecycle -------------------------------------------------------------------------
    def __enter__(self):
        workdir = Path(self._stack.enter_context(tempfile.TemporaryDirectory(prefix="ade-mock-earthdata-")))
        self.ca_file, cert, key = _make_tls(workdir)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                mock._handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self._stack.callback(self.server.server_close)
        self._stack.callback(self.server.shutdown)
        if self.intercept:
            self._stack.enter_context(route_hosts(HOSTS, self.port, self.ca_file))
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def environment(self) -> dict[str, str]:
        """Variables that let another process (with `write_hook`'s sitecustomize) use this mock."""
        return {"ADE_MOCK_EARTHDATA_PORT": str(self.port), "ADE_MOCK_EARTHDATA_HOSTS": ",".join(HOSTS),
                "REQUESTS_CA_BUNDLE": str(self.ca_file), "SSL_CERT_FILE": str(self.ca_file)}

    # -- request handling ------------------------------------------------------------------
    def _respond(self, handler, status, body=b"", headers=()):
        handler.send_response(status)
        for key, value in headers:
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        host = handler.headers.get("Host", "").split(":")[0]
        self.log.append((host, urlparse(handler.path).path, "Authorization" in handler.headers, status))

    def _signature(self, path: str) -> str:
        return hmac.new(self._signing_key, path.encode(), hashlib.sha256).hexdigest()

    def _tile_zip(self, name: str) -> bytes:
        if name not in self._zips:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"{name}.hgt", synthetic_hgt(name, self.samples))
            self._zips[name] = buffer.getvalue()
        return self._zips[name]

    def _handle(self, handler):
        url = urlparse(handler.path)
        query = parse_qs(url.query)
        host = handler.headers.get("Host", "").split(":")[0]
        if host == TILE_HOST and url.path.startswith(TILE_PREFIX):
            name = url.path.rsplit("/", 1)[-1].split(".", 1)[0]
            cookie = handler.headers.get("Cookie", "")
            if not any(f"accessToken={token}" in cookie for token in self._sessions):
                location = (f"https://{URS_HOST}/oauth/authorize?" + urlencode({
                    "client_id": "synthetic-lpdaac", "response_type": "code",
                    "redirect_uri": f"https://{TILE_HOST}/login", "state": url.path, "app_type": "401"}))
                return self._respond(handler, 302, headers=[("Location", location)])
            if self.tiles is not None and name not in self.tiles:
                return self._respond(handler, 404, b"Not Found")
            s3_path = url.path[len("/lp-prod-protected"):]
            signed = urlencode({"A-userid": self.username, "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                                "X-Amz-Expires": "3600", "X-Amz-Signature": self._signature(s3_path)})
            return self._respond(handler, 307, headers=[("Location", f"https://{S3_HOST}{s3_path}?{signed}")])
        if host == URS_HOST and url.path == "/oauth/authorize":
            expected = "Basic " + base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            if handler.headers.get("Authorization") != expected:
                return self._respond(handler, 401, b"Unauthorized",
                                     [("WWW-Authenticate", 'Basic realm="Please enter your Earthdata Login credentials"')])
            code = secrets.token_hex(8)
            self._codes.add(code)
            location = f"{query['redirect_uri'][0]}?" + urlencode({"code": code, "state": query["state"][0]})
            return self._respond(handler, 302, headers=[("Location", location)])
        if host == TILE_HOST and url.path == "/login":
            code = query.get("code", [""])[0]
            if code not in self._codes:
                return self._respond(handler, 403, b"invalid code")
            self._codes.discard(code)
            token = secrets.token_hex(8)
            self._sessions.add(token)
            return self._respond(handler, 302, headers=[
                ("Set-Cookie", f"accessToken={token}; Path=/; Secure; HttpOnly"),
                ("Location", f"https://{TILE_HOST}{query['state'][0]}")])
        if host == S3_HOST:
            if "Authorization" in handler.headers:  # S3 refuses two authentication mechanisms
                return self._respond(handler, 400, b"InvalidArgument: Only one auth mechanism allowed")
            if query.get("X-Amz-Signature", [""])[0] != self._signature(url.path):
                return self._respond(handler, 403, b"SignatureDoesNotMatch")
            name = url.path.rsplit("/", 1)[-1].split(".", 1)[0]
            return self._respond(handler, 200, self._tile_zip(name), [("Content-Type", "application/zip")])
        return self._respond(handler, 404, b"Not Found")


@contextlib.contextmanager
def route_hosts(hosts, port: int, ca_file):
    """Resolve `hosts` to 127.0.0.1:`port` and trust `ca_file`, for this process."""
    original = socket.getaddrinfo
    routed = set(hosts)

    def getaddrinfo(host, port_, *args, **kwargs):
        if host in routed:
            return original("127.0.0.1", port, *args, **kwargs)
        return original(host, port_, *args, **kwargs)

    saved = {key: os.environ.get(key) for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")}
    socket.getaddrinfo = getaddrinfo
    os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"] = str(ca_file)
    try:
        yield
    finally:
        socket.getaddrinfo = original
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


HOOK = '''\
# Written by scripts/datasets/mock_earthdata.py: routes the mocked NASA Earthdata
# host names to the local synthetic service for this interpreter.
import os as _os
import socket as _socket

_port = _os.environ.get("ADE_MOCK_EARTHDATA_PORT")
if _port:
    _hosts = set(_os.environ.get("ADE_MOCK_EARTHDATA_HOSTS", "").split(","))
    _original = _socket.getaddrinfo

    def _getaddrinfo(host, port, *args, **kwargs):
        if host in _hosts:
            return _original("127.0.0.1", int(_port), *args, **kwargs)
        return _original(host, port, *args, **kwargs)

    _socket.getaddrinfo = _getaddrinfo
'''


def write_hook(directory) -> Path:
    """Write a sitecustomize.py into `directory` (put it first on PYTHONPATH of the child process)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    path.write_text(HOOK)
    return path


def check(samples: int) -> int:
    """Run the library's real download path against the synthetic service and verify it."""
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from arraydataengine import DataSources
    from arraydataengine.ops import mosaic_dem_tiles, sample_elevation_at_latlon

    names = [tile_name(lat, lon) for lat in (37, 38) for lon in (-123, -122)]
    failures = []
    with MockEarthdata(tiles=names, samples=samples) as mock, tempfile.TemporaryDirectory() as cache:
        saved = {k: os.environ.get(k) for k in ("earthdata_username", "earthdata_password", "ADE_DEM_BASE_URL")}
        os.environ.pop("ADE_DEM_BASE_URL", None)
        try:
            os.environ["earthdata_username"], os.environ["earthdata_password"] = SYNTHETIC_USER, SYNTHETIC_PASSWORD
            tiles = list(DataSources("DEM", bounds=[[37, 39], [122, 124]], cache_dir=cache).get_message())
            print(f"downloaded {[t['name'] for t in tiles]} ({tiles[0]['data'].shape[0]} samples per side)")
            for tile in tiles:
                expected = np.frombuffer(synthetic_hgt(tile["name"], samples), dtype=">i2").reshape(samples, samples)
                if not np.array_equal(tile["data"], expected):
                    failures.append(f"{tile['name']} differs from the served tile")
            # Synthetic tiles share edge samples like SRTM at every size (auto-detection
            # only recognises the standard 1201/3601-sample tiles).
            mosaic, lats, lons = mosaic_dem_tiles(tiles, return_index=True, overlap=1)
            for name, lat, lon, height in PEAKS:
                z = float(sample_elevation_at_latlon(mosaic, lat, lon, index=(lats, lons)))
                print(f"  {name:15s} {z:7.1f} m (expected {height:.0f} m)")
                if abs(z - height) > 2.0:
                    failures.append(f"{name}: {z:.1f} m")
            leaked = sorted({host for host, _, had_auth, _ in mock.log if had_auth and host not in (TILE_HOST, URS_HOST)})
            if leaked:
                failures.append(f"credentials sent to {leaked}")
            print(f"  {len(mock.log)} requests; credentials only sent to the archive and login hosts: {not leaked}")
            before = len(mock.log)
            list(DataSources("DEM", bounds=[[37, 39], [122, 124]], cache_dir=cache).get_message())
            if len(mock.log) != before:
                failures.append("cached tiles were downloaded again")
            os.environ["earthdata_password"] = "wrong-password"
            try:
                list(DataSources("DEM", bounds=[[37, 38], [122, 123]], cache_dir=tempfile.mkdtemp()).get_message())
                failures.append("a wrong password did not fail")
            except Exception as exc:  # requests.HTTPError
                print(f"  wrong password -> {type(exc).__name__}: {str(exc)[:70]}")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    print("OK" if not failures else "FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="download through the library and verify")
    parser.add_argument("--samples", type=int, default=3601, help="samples per tile side (3601 = SRTMGL1)")
    args = parser.parse_args()
    if args.check:
        return check(args.samples)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
