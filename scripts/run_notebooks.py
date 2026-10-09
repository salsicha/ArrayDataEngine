"""Execute the example notebooks in isolated copies, optionally against real data.

Each notebook runs top to bottom in a fresh ``python3`` kernel inside a temporary
copy of ``notebooks/`` and ``example/`` (so nothing a notebook writes lands in the
repository). The first failing cell is reported with its source and error.

Profiles set the notebooks' ``ADE_*`` configuration variables:

* ``default`` – no overrides: the offline example recording / synthetic data.
* ``real`` – real datasets fetched by ``scripts/datasets/fetch_test_datasets.py``
  (KITTI GPS/IMU + Velodyne as rosbag2, MOT17-05, SRTM tiles, CIFAR-10) plus the
  synthetic Boson LWIR sequences.
* ``ros1`` – the same KITTI drive as a ROS 1 bag, to exercise the ROS 1 reader
  (the example recording's ``.bag`` copy is SQLite inside).
* ``dem-download`` – the DEM download branch of ``dem.ipynb`` and ``demo.ipynb`` against a local HTTP server
  serving the real SRTM tiles (dummy credentials, never sent anywhere else).

Usage::

    python scripts/run_notebooks.py                        # all notebooks, offline
    python scripts/run_notebooks.py --profile real --data ~/.cache/arraydataengine/test-data
    python scripts/run_notebooks.py dem.ipynb --env ADE_DEM_CACHE=/path/to/hgt
    python scripts/run_notebooks.py --save                 # refresh committed outputs (default profile only)

Requires ``nbclient`` (``pip install nbclient``) and an ``ipykernel`` named ``python3``.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
DEFAULT_DATA = Path.home() / ".cache" / "arraydataengine" / "test-data"
FORBIDDEN_OUTPUT = (str(Path.home()), tempfile.gettempdir() + "/")

KITTI = "{data}/kitti/kitti_0001_rosbag2"
KITTI_MCAP = "{data}/kitti/kitti_0001_mcap"
KITTI_ROS1 = "{data}/kitti/kitti_0001.bag"
KITTI_TOPICS = {
    "ADE_POSE_TOPIC": "/kitti/velo/pose",
    "ADE_LIDAR_TOPIC": "/kitti/velo/pointcloud",
    "ADE_CAMERA_TOPIC": "/kitti/camera_color_left/image/compressed",
}
MOT = "{data}/mot17/MOT17-05-FRCNN"

PROFILES: dict[str, dict[str, dict[str, str]]] = {
    "default": {},
    "real": {
        "bag_to_TileDB.ipynb": {"ADE_EXAMPLE_BAG": KITTI, **KITTI_TOPICS},
        "circular_buffer.ipynb": {"ADE_EXAMPLE_BAG": KITTI_MCAP, **KITTI_TOPICS, "ADE_POSE_TOPIC": "/kitti/oxts/pose"},
        "iterator_example.ipynb": {"ADE_EXAMPLE_BAG": KITTI, **KITTI_TOPICS},
        "ins.ipynb": {
            "ADE_INS_BAG": KITTI, "ADE_IMU_TOPIC": "/kitti/oxts/imu", "ADE_ODOM_TOPIC": "/kitti/oxts/odom",
            "ADE_NAVSAT_TOPIC": "/kitti/oxts/gps/fix", "ADE_EXAMPLE_BAG": KITTI, "ADE_POSE_TOPIC": "/kitti/oxts/pose",
        },
        "loam_example.ipynb": {
            "ADE_LIDAR_BAG": KITTI, "ADE_LIDAR_TOPIC": "/kitti/velo/pointcloud", "ADE_POSE_TOPIC": "/kitti/velo/pose",
            "ADE_SCANS_IN_MAP_FRAME": "0", "ADE_Y_UP": "0", "ADE_MAX_SCAN_POINTS": "30000",
            "ADE_VERTICAL_ANGLE_CORRECTION": "0.205",  # KITTI HDL-64 beam tilt (as KISS-ICP applies)
        },
        "mot.ipynb": {"ADE_MOT_FRAMES": MOT + "/img1/*.jpg", "ADE_MOT_GT": MOT + "/gt/gt.txt",
                      "ADE_MOT_DETS": MOT + "/det/det.txt"},
        "dem.ipynb": {"ADE_DEM_CACHE": "{data}/srtm/cache", "ADE_DEM_NORTH": "37,39", "ADE_DEM_WEST": "122,124",
                      "ADE_DEM_AOI": "37.8816,-121.9142"},
        "mlp.ipynb": {"ADE_TORCHVISION_ROOT": "{data}/cifar", "ADE_DOWNLOAD_CIFAR10": "1"},
        "tiff.ipynb": {"ADE_LWIR_GLOB": "{data}/lwir_raw16/*.tiff", "ADE_LWIR_TRUTH": "{data}/lwir_raw16/ground_truth.json"},
        "pulsed_encoded_beacons.ipynb": {"ADE_LWIR_DIR": "{data}/lwir_agc",
                                         "ADE_LWIR_TRUTH": "{data}/lwir_agc/ground_truth.json"},
        "demo.ipynb": {
            "ADE_LIDAR_BAG": KITTI, "ADE_LIDAR_TOPIC": "/kitti/velo/pointcloud", "ADE_POSE_TOPIC": "/kitti/velo/pose",
            "ADE_SCANS_IN_MAP_FRAME": "0", "ADE_LIDAR_MAX_POINTS": "30000",
            "ADE_IMAGE_SOURCE": MOT + "/img1/*.jpg",
            "ADE_INS_BAG": KITTI, "ADE_INS_IMU_TOPIC": "/kitti/oxts/imu", "ADE_INS_ODOM_TOPIC": "/kitti/oxts/odom",
            "ADE_INS_FIX_TOPIC": "/kitti/oxts/gps/fix",
            "ADE_DEM_CACHE": "{data}/srtm/cache", "ADE_DEM_NORTH": "37,39", "ADE_DEM_WEST": "122,124",
            "ADE_LWIR_GLOB": "{data}/lwir_agc/*.tiff", "ADE_LWIR_TRUTH": "{data}/lwir_agc/ground_truth.json",
        },
    },
    "ros1": {
        **{name: {"ADE_EXAMPLE_BAG": KITTI_ROS1, **KITTI_TOPICS}
           for name in ("bag_to_TileDB.ipynb", "circular_buffer.ipynb", "iterator_example.ipynb")},
        "loam_example.ipynb": {
            "ADE_LIDAR_BAG": KITTI_ROS1, "ADE_LIDAR_TOPIC": "/kitti/velo/pointcloud", "ADE_POSE_TOPIC": "/kitti/velo/pose",
            "ADE_SCANS_IN_MAP_FRAME": "0", "ADE_Y_UP": "0", "ADE_MAX_SCAN_POINTS": "30000",
            "ADE_VERTICAL_ANGLE_CORRECTION": "0.205",  # KITTI HDL-64 beam tilt (as KISS-ICP applies)
        },
        "ins.ipynb": {
            "ADE_INS_BAG": KITTI_ROS1, "ADE_IMU_TOPIC": "/kitti/oxts/imu", "ADE_ODOM_TOPIC": "/kitti/oxts/odom",
            "ADE_NAVSAT_TOPIC": "/kitti/oxts/gps/fix", "ADE_EXAMPLE_BAG": KITTI_ROS1, "ADE_POSE_TOPIC": "/kitti/oxts/pose",
        },
    },
    "dem-download": {
        name: {
            "earthdata_username": "local-test-user", "earthdata_password": "local-test-pass",
            "ADE_DEM_BASE_URL": "{srtm_server}", "ADE_DEM_ALLOW_HTTP": "1", "ADE_DEM_CACHE": "__dem_download_cache",
            "ADE_DEM_NORTH": "37,39", "ADE_DEM_WEST": "122,124", "ADE_DEM_AOI": "37.8816,-121.9142",
        }
        for name in ("dem.ipynb", "demo.ipynb")
    },
}


@contextlib.contextmanager
def srtm_server(directory: Path):
    """Serve `directory` over HTTP on 127.0.0.1 for the dem-download profile."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *args, **kwargs: None
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()


def output_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        parts.append(str(out.get("text", "")))
        parts.append(str(out.get("traceback", "")))
        parts.extend(str(v) for k, v in out.get("data", {}).items() if k.startswith("text/"))
    return "\n".join(parts)


def run(name: str, env: dict[str, str], timeout: int, save: bool, keep: Path | None) -> bool:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError

    root = Path(tempfile.mkdtemp(prefix=f"ade-nb-{Path(name).stem}-"))
    try:
        shutil.copytree(REPO / "example", root / "example")
        work = root / "notebooks"
        work.mkdir()
        shutil.copy2(NOTEBOOKS / name, work / name)
        nb = nbformat.read(work / name, as_version=4)
        saved_env = dict(os.environ)
        os.environ.update(env)
        for var in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(var, None)
        os.environ.setdefault("PYTHONHASHSEED", "0")
        started = time.monotonic()
        try:
            NotebookClient(nb, timeout=timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(work)}}).execute()
        except (CellExecutionError, CellTimeoutError) as exc:
            failing = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code" and any(
                o.get("output_type") == "error" for o in c.get("outputs", []))), None)
            print(f"FAIL  {name} (cell {failing}, {time.monotonic() - started:.0f}s)")
            if failing is not None:
                print("  " + nb.cells[failing].source[:800].replace("\n", "\n  "))
            print("  " + str(exc)[-2500:].replace("\n", "\n  "))
            return False
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
        elapsed = time.monotonic() - started
        nbformat.write(nb, work / name)
        leaks = sorted({i for i, c in enumerate(nb.cells) if c.cell_type == "code"
                        for f in FORBIDDEN_OUTPUT if f in output_text(c)})
        print(f"OK    {name} ({sum(c.cell_type == 'code' for c in nb.cells)} code cells, {elapsed:.0f}s, "
              f"{(work / name).stat().st_size // 1024} KB)" + (f"  [outputs mention local paths: cells {leaks}]" if leaks else ""))
        if save:
            if leaks:
                print(f"      not saved: outputs mention local paths")
                return False
            shutil.copy2(work / name, NOTEBOOKS / name)
            print(f"      saved {NOTEBOOKS / name}")
        return True
    finally:
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            target = keep / Path(name).stem
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(root, target, ignore=shutil.ignore_patterns("example"))
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", help="notebook names (default: all, or all in the profile)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="default")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="fetch_test_datasets.py --dest directory")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra override (repeatable)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    parser.add_argument("--save", action="store_true", help="write outputs back (default profile, no --env only)")
    parser.add_argument("--keep", type=Path, help="copy each executed notebook and its outputs here")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    names = args.notebooks or (sorted(profile) if profile else sorted(p.name for p in NOTEBOOKS.glob("*.ipynb")))
    names = [n if n.endswith(".ipynb") else n + ".ipynb" for n in names]
    if args.save and (args.profile != "default" or args.env):
        parser.error("--save only refreshes the committed (default profile, no --env) outputs")
    extra = dict(item.split("=", 1) for item in args.env)

    with contextlib.ExitStack() as stack:
        server = None
        if any("{srtm_server}" in v for env in profile.values() for v in env.values()):
            server = stack.enter_context(srtm_server(args.data / "srtm" / "served"))
        results = {}
        for name in names:
            env = {k: v.format(data=args.data, srtm_server=server) for k, v in profile.get(name, {}).items()}
            env.update(extra)
            results[name] = run(name, env, args.timeout, args.save, args.keep)
    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} notebooks passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
