"""Check the authenticated NASA Earthdata SRTM download path with YOUR credentials.

The notebooks and tests exercise DEMSource's download code against a local server
holding real SRTM tiles, but the Earthdata login step itself needs an account, so
this script is for you to run. It downloads tile N37W122 (San Francisco Bay Area,
about 25 MB) through ``DataSources("DEM", ...)`` into a temporary cache and checks:

* the tile decodes as 3601 x 3601 samples (SRTMGL1, 1 arc-second);
* the elevation at the Mount Diablo summit (1173 m) is within 30 m;
* if ``--compare`` points at the AWS copy fetched by ``fetch_test_datasets.py``
  (``<data>/srtm/cache``), the two tiles are identical.

Credentials come from the ``earthdata_username`` / ``earthdata_password``
environment variables and are never printed. Create a free account at
https://urs.earthdata.nasa.gov and approve the "LP DAAC Data Pool" application.

Usage::

    export earthdata_username=...  earthdata_password=...
    python scripts/datasets/check_earthdata_download.py [--compare ~/.cache/arraydataengine/test-data/srtm/cache]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", type=Path, help="directory with an AWS-mirrored N37W122.hgt to compare against")
    args = parser.parse_args()

    if not (os.environ.get("earthdata_username") and os.environ.get("earthdata_password")):
        print("Set earthdata_username and earthdata_password in the environment first.", file=sys.stderr)
        return 2

    from arraydataengine import DataSources
    from arraydataengine.ops import mosaic_dem_tiles, sample_elevation_at_latlon

    with tempfile.TemporaryDirectory(prefix="ade-earthdata-") as cache:
        source = DataSources("DEM", bounds=[[37, 38], [122, 123]], cache_dir=cache, timeout=120)
        tiles = list(source.get_message())
        if [t["name"] for t in tiles] != ["N37W122"]:
            print(f"FAIL: expected tile N37W122, got {[t['name'] for t in tiles]}")
            return 1
        tile = tiles[0]["data"]
        print(f"downloaded N37W122: {tile.shape} {tile.dtype}, elevation {tile.min()}..{tile.max()} m")
        if tile.shape != (3601, 3601):
            print("FAIL: expected a 3601 x 3601 SRTMGL1 tile")
            return 1
        mosaic, lats, lons = mosaic_dem_tiles(tiles, return_index=True)
        diablo = float(sample_elevation_at_latlon(mosaic, 37.8816, -121.9142, index=(lats, lons)))
        print(f"Mount Diablo summit: {diablo:.1f} m (published 1173 m)")
        ok = abs(diablo - 1173.0) <= 30.0
        if args.compare is not None:
            reference = args.compare / "N37W122.hgt"
            aws = np.fromfile(reference, dtype=">i2").reshape(3601, 3601)
            same = np.array_equal(aws, tile)
            print(f"identical to the AWS copy at {reference}: {same}")
            ok = ok and same
        print("OK: Earthdata download path works" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
