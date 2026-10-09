"""Download and prepare the real-data test sets used to exercise the notebooks.

Everything goes under one data directory (default
``~/.cache/arraydataengine/test-data``), never into the repository:

=====================  ==========================================================  ========
directory              contents (source, license)                                  download
=====================  ==========================================================  ========
``srtm/cache``         1-arc-second SRTM-format (skadi) tiles N37-N38 / W122-W123   ~35 MB
                       (San Francisco Bay Area) from the AWS Open Data terrain-tiles
                       bucket, in DEMSource's cache layout. They merge SRTM and USGS
                       3DEP with NOAA ETOPO1 bathymetry offshore (public domain,
                       attribution requested: github.com/tilezen/joerd attribution.md)
``srtm/served``        the same tiles zipped as ``<name>.SRTMGL1.hgt.zip``, for     --
                       testing DEMSource's download path against a local server
``kitti/``             KITTI raw drive 2011_09_26_drive_0001 (GPS/IMU, Velodyne,    459 MB
                       camera) converted to rosbag2 (sqlite3, MCAP) and ROS 1 by
                       ``kitti_to_rosbag2.py`` (CC BY-NC-SA 3.0)
``cifar/``             CIFAR-10 python tarball where torchvision expects it        170 MB
                       (cs.toronto.edu)
``mot17/``             MOT17-05-FRCNN (837 frames, gt, det) from the Hugging Face   ~31 MB
                       mirror Morrison1025/MOT17 (CC BY-NC-SA 3.0)
``lwir_agc``,          synthetic FLIR Boson 640 sequences with blinking beacons     --
``lwir_raw16``         from ``make_lwir_sequence.py`` (no open dataset has them)
=====================  ==========================================================  ========

Downloads are verified against pinned checksums and skipped when already present.
Run ``scripts/run_notebooks.py --profile real --data <dir>`` afterwards.

Usage::

    python scripts/datasets/fetch_test_datasets.py [--dest DIR] [--only srtm kitti cifar mot17 lwir]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = Path.home() / ".cache" / "arraydataengine" / "test-data"

SRTM_TILES = {
    "N37W122": "7d28cdc456aef1d1fc8143deea5806b2d89e6bcd38c994f679aaf6845913d4d1",
    "N37W123": "420e904f5d8f06edc19130b7cdee44fb04ad8b13ef518e73195ce70bb9930cad",
    "N38W122": "921e495a7bca72efb9fdfc8142e5491c6a384f3117bdafe05de241d0b492b0ee",
    "N38W123": "340ab211e71a8c3752995394c6b1b88d3c05b1a5d711839ec6603ae9b96017a4",
}
SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{name}.hgt.gz"

KITTI_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
KITTI_FILES = {
    "2011_09_26_calib.zip": ("2011_09_26_calib.zip",
                             "e0108cfd000cf802c14ed94fba38601185c792e5640abba015a34b7a85b812e0"),
    "2011_09_26_drive_0001_sync.zip": ("2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip",
                                       "7827a821ddc0a973bdf66de3a005a8825bd37c304aa9750f3c018f80d5b18458"),
}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"  # the checksum torchvision verifies

MOT_BASE = "https://huggingface.co/datasets/Morrison1025/MOT17/resolve/main/train/MOT17-05-FRCNN"
MOT_FRAMES = 837
MOT_FILES = {
    "gt/gt.txt": "6da4933977d50fdba00811e33501a491d12c192ee577e800059e29779744b462",
    "det/det.txt": "3c763138f8471c141df0fbb00ddac2f2016de3c9a86ce92a1366a875bd14b6f6",
    "seqinfo.ini": None,
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, checksum: str | None = None, algorithm: str = "sha256", retries: int = 4) -> Path:
    """Download `url` to `path` (atomically), verify it, and skip valid existing files."""
    if path.exists() and (checksum is None or digest(path, algorithm) == checksum):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            break
        except OSError as exc:
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)
    if checksum is not None and digest(partial, algorithm) != checksum:
        partial.unlink()
        raise RuntimeError(f"checksum mismatch for {url}")
    partial.replace(path)
    return path


def fetch_srtm(dest: Path) -> None:
    root = dest / "srtm"
    for name, sha in SRTM_TILES.items():
        gz = download(SRTM_URL.format(lat=name[:3], name=name), root / f"{name}.hgt.gz", sha)
        hgt = root / "cache" / f"{name}.hgt"
        if not hgt.exists():
            hgt.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(gz, "rb") as src, open(hgt, "wb") as out:
                shutil.copyfileobj(src, out)
        served = root / "served" / f"{name}.SRTMGL1.hgt.zip"
        if not served.exists():
            served.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(served, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(hgt, f"{name}.hgt")
    print(f"srtm: {len(SRTM_TILES)} tiles in {root / 'cache'}")


def fetch_kitti(dest: Path) -> None:
    root = dest / "kitti"
    for name, (path, sha) in KITTI_FILES.items():
        archive = download(f"{KITTI_BASE}/{path}", root / name, sha)
        with zipfile.ZipFile(archive) as zf:
            missing = [m for m in zf.namelist() if not (root / m).exists()]
            if missing:
                zf.extractall(root)
    drive, calib = root / "2011_09_26" / "2011_09_26_drive_0001_sync", root / "2011_09_26"
    for storage, out in (("sqlite3", root / "kitti_0001_rosbag2"), ("mcap", root / "kitti_0001_mcap"),
                         ("ros1", root / "kitti_0001.bag")):
        if not ((out / "metadata.yaml").exists() or out.is_file()):
            subprocess.run([sys.executable, str(HERE / "kitti_to_rosbag2.py"), "--drive", str(drive),
                            "--calib", str(calib), "--out", str(out), "--storage", storage], check=True)
    print(f"kitti: rosbag2 at {root / 'kitti_0001_rosbag2'} and {root / 'kitti_0001_mcap'}, "
          f"ROS 1 bag at {root / 'kitti_0001.bag'}")


def fetch_cifar(dest: Path) -> None:
    path = download(CIFAR_URL, dest / "cifar" / "cifar-10-python.tar.gz", CIFAR_MD5, "md5")
    print(f"cifar: {path}")


def fetch_mot17(dest: Path) -> None:
    root = dest / "mot17" / "MOT17-05-FRCNN"
    for name, sha in MOT_FILES.items():
        download(f"{MOT_BASE}/{name}", root / name, sha)
    frames = [f"{i:06d}.jpg" for i in range(1, MOT_FRAMES + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda f: download(f"{MOT_BASE}/img1/{f}", root / "img1" / f), frames))
    count = len(list((root / "img1").glob("*.jpg")))
    if count != MOT_FRAMES:
        raise RuntimeError(f"mot17: expected {MOT_FRAMES} frames, found {count}")
    print(f"mot17: {count} frames in {root}")


def fetch_lwir(dest: Path) -> None:
    for name, extra in (("lwir_agc", []), ("lwir_raw16", ["--raw16"])):
        out = dest / name
        if not (out / "ground_truth.json").exists():
            subprocess.run([sys.executable, str(HERE / "make_lwir_sequence.py"), "--out", str(out),
                            "--frames", "60", "--fps", "30", *extra], check=True)
    print(f"lwir: synthetic sequences in {dest / 'lwir_agc'} and {dest / 'lwir_raw16'}")


STEPS = {"srtm": fetch_srtm, "kitti": fetch_kitti, "cifar": fetch_cifar, "mot17": fetch_mot17, "lwir": fetch_lwir}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--only", nargs="+", choices=sorted(STEPS), default=sorted(STEPS))
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        STEPS[name](args.dest)
    print(f"done: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
