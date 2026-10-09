"""Generate a synthetic FLIR Boson 640 LWIR sequence with blinking IR beacons.

No small, openly downloadable thermal dataset contains pulsed IR beacons, so the
LWIR notebooks (`tiff.ipynb`, `pulsed_encoded_beacons.ipynb`, the beacon section
of `demo.ipynb`) are tested against this generator. It models what those
notebooks rely on in a real Boson capture:

* 512 x 640 frames, written either as 8-bit AGC TIFFs named like the camera
  software does (``<epoch_ms>_Boson_640_agc_<n>.tiff``) or as raw 16-bit counts
  (``--raw16``, ``<epoch_ms>_Boson_640_raw_<n>.tiff``);
* a smooth thermal scene (sky gradient, warm structures), column/row fixed-pattern
  noise, temporal noise, slow drift, and flat-field-correction (FFC) frames where
  the image briefly freezes;
* a slowly panning / jittering camera (integer pixel shifts, recorded as ground
  truth);
* beacons that blink as square waves at known frequencies, plus steady beacons.

Ground truth (beacon pixel positions in the first frame, frequencies, per-frame
camera offsets, frame period) is written to ``ground_truth.json`` next to the
frames.

Usage::

    python scripts/datasets/make_lwir_sequence.py --out DATA/lwir_boson --frames 60 --fps 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEIGHT, WIDTH = 512, 640


def scene(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Smooth background radiance in raw-count units (~22000 +/- a few hundred)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    base = 21800.0 + 250.0 * (yy / height)  # cooler sky at the top
    for _ in range(6):  # warm structures with sharp edges (good registration texture)
        y0, x0 = rng.integers(0, height), rng.integers(0, width)
        h, w = rng.integers(30, 140), rng.integers(30, 180)
        base[y0:y0 + h, x0:x0 + w] += rng.uniform(80.0, 260.0)
    blobs = np.zeros_like(base)
    for _ in range(25):
        cy, cx, s = rng.uniform(0, height), rng.uniform(0, width), rng.uniform(8, 40)
        blobs += rng.uniform(-60, 60) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    return base + blobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--raw16", action="store_true", help="write raw 16-bit counts instead of 8-bit AGC")
    parser.add_argument("--pan", type=float, default=0.6, help="camera pan in pixels per frame")
    args = parser.parse_args()

    from PIL import Image

    rng = np.random.default_rng(args.seed)
    margin = 48
    world = scene(rng, HEIGHT + 2 * margin, WIDTH + 2 * margin)
    column_fpn = rng.normal(0.0, 6.0, WIDTH)
    row_fpn = rng.normal(0.0, 3.0, HEIGHT)
    pixel_fpn = rng.normal(0.0, 4.0, (HEIGHT, WIDTH))

    # Beacons in world coordinates; frequencies chosen below Nyquist for --fps.
    nyquist = args.fps / 2.0
    blinking = [f for f in (1.0, 2.5, 5.0, 7.5) if f < nyquist]
    beacons = []
    for i, freq in enumerate(blinking + [0.0, 0.0]):
        beacons.append({
            "row": float(rng.uniform(margin + 60, HEIGHT + margin - 60)),
            "col": float(rng.uniform(margin + 60, WIDTH + margin - 60)),
            "frequency_hz": freq,  # 0 = steady
            "amplitude": float(rng.uniform(900.0, 1600.0)),
            "phase": float(rng.uniform(0.0, 1.0)),
        })

    yy, xx = np.mgrid[0:HEIGHT + 2 * margin, 0:WIDTH + 2 * margin].astype(np.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*_Boson_640_*.tiff"):
        old.unlink()
    start_ms = 1707777322740
    offsets, frame_files, last = [], [], None
    agc_low = agc_high = None
    for n in range(args.frames):
        t = n / args.fps
        dy = int(round(2.0 * np.sin(2 * np.pi * 0.3 * t)))  # vertical jitter
        dx = int(round(args.pan * n)) % margin              # slow pan
        offsets.append({"frame": n, "dy": dy, "dx": dx})
        ffc = n > 0 and n % 45 == 0  # flat-field correction: the camera repeats the last frame
        if ffc and last is not None:
            frame = last
        else:
            world_t = world + 15.0 * np.sin(2 * np.pi * 0.05 * t)  # slow drift
            for beacon in beacons:
                on = 1.0 if beacon["frequency_hz"] == 0.0 else float(
                    (t * beacon["frequency_hz"] + beacon["phase"]) % 1.0 < 0.5)
                world_t = world_t + on * beacon["amplitude"] * np.exp(
                    -((yy - beacon["row"]) ** 2 + (xx - beacon["col"]) ** 2) / (2 * 1.6 ** 2))
            r0, c0 = margin + dy, margin - margin // 2 + dx
            frame = world_t[r0:r0 + HEIGHT, c0:c0 + WIDTH]
            frame = frame + column_fpn[None, :] + row_fpn[:, None] + pixel_fpn
            frame = frame + rng.normal(0.0, 9.0, frame.shape)
            last = frame
        name = f"{start_ms + int(round(1000 * t))}_Boson_640_{'raw' if args.raw16 else 'agc'}_{n}.tiff"
        if args.raw16:
            image = np.clip(frame, 0, 65535).astype(np.uint16)
        else:
            if agc_low is None:  # Boson AGC: fixed linear map chosen on the first frame
                agc_low, agc_high = np.percentile(frame, 1), np.percentile(frame, 99.9) + 1200.0
            image = np.clip((frame - agc_low) / (agc_high - agc_low) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(args.out / name)
        frame_files.append(name)

    first = offsets[0]
    truth = {
        "frame_period_s": 1.0 / args.fps,
        "shape": [HEIGHT, WIDTH],
        "dtype": "uint16" if args.raw16 else "uint8",
        "ffc_frames": [n for n in range(args.frames) if n > 0 and n % 45 == 0],
        "frames": frame_files,
        "camera_offsets": offsets,
        "beacons_first_frame": [
            {"row": round(b["row"] - (margin + first["dy"]), 2),
             "col": round(b["col"] - (margin - margin // 2 + first["dx"]), 2),
             "frequency_hz": b["frequency_hz"]}
            for b in beacons
        ],
    }
    (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {args.frames} frames to {args.out} ({truth['dtype']}); beacons: "
          + ", ".join(f"{b['frequency_hz']} Hz @ ({b['row']:.0f},{b['col']:.0f})" for b in truth["beacons_first_frame"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
