#!/usr/bin/env python3
"""Write synthetic FLIR radiometric JPEGs (R-JPEG) with exact ground truth.

The files follow the layout of FLIR cameras' R-JPEGs (as documented by
ExifTool): a JPEG, here an 8-bit rendering of the thermal image, whose APP1
segments tagged ``FLIR`` carry a FLIR File Format (FFF) block with

* a RawData record: the 16-bit sensor counts, PNG-compressed (with the samples
  little-endian, as most FLIR cameras write them) or uncompressed;
* a CameraInfo record: Planck constants, emissivity, object distance,
  reflected/atmospheric/window temperatures (kelvin), relative humidity,
  atmospheric transmission coefficients, camera model, capture time.

Counts are rendered from a scene of known temperatures with
``arraydataengine.ops.flir_temperature_to_raw``, so each file comes with its
true temperature map (``truth/<name>.npy``, °C) and ``ground_truth.json``.

    python scripts/datasets/make_flir_rjpeg.py --out DIR [--frames 3] [--raw-format png|raw] [--size 160x120]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import struct
from pathlib import Path

import numpy as np

# Planck constants of the kind FLIR cameras store (Thermimage's defaults).
DEFAULT_CALIBRATION = {
    "planck_r1": 21106.77, "planck_b": 1501.0, "planck_f": 1.0, "planck_o": -7340.0, "planck_r2": 0.012545258,
    "emissivity": 0.95, "object_distance": 5.0, "reflected_temperature_c": 22.0,
    "atmospheric_temperature_c": 18.0, "window_temperature_c": 18.0, "window_transmission": 1.0,
    "relative_humidity": 0.45,
}
CAMERA_INFO_SIZE = 0x5B0
_PLANCK_FLOATS = {  # CameraInfo byte offsets (ExifTool's FLIR CameraInfo table)
    "emissivity": 0x20, "object_distance": 0x24, "window_transmission": 0x34,
    "planck_r1": 0x58, "planck_b": 0x5C, "planck_f": 0x60,
    "atmospheric_trans_alpha1": 0x70, "atmospheric_trans_alpha2": 0x74, "atmospheric_trans_beta1": 0x78,
    "atmospheric_trans_beta2": 0x7C, "atmospheric_trans_x": 0x80, "planck_r2": 0x30C,
}
_KELVIN_FLOATS = {"reflected_temperature_c": 0x28, "atmospheric_temperature_c": 0x2C, "window_temperature_c": 0x30}


def _calibration(calibration):
    from arraydataengine.ops import FlirCalibration

    if isinstance(calibration, FlirCalibration):
        return calibration
    return FlirCalibration.from_dict({**DEFAULT_CALIBRATION, **(calibration or {})})


def camera_info_record(calibration, *, order="<", camera_model="ADE synthetic R-JPEG", serial="0000000",
                       lens_model="FOL18", field_of_view=45.0, capture_time=None, humidity_percent=False) -> bytes:
    """A CameraInfo record; `order` is the record's byte order ("<" as cameras write it)."""
    c = dataclasses.asdict(_calibration(calibration))
    record = bytearray(CAMERA_INFO_SIZE)
    struct.pack_into(order + "H", record, 0, 2)  # byte-order mark
    for key, offset in _PLANCK_FLOATS.items():
        struct.pack_into(order + "f", record, offset, c[key])
    for key, offset in _KELVIN_FLOATS.items():
        struct.pack_into(order + "f", record, offset, c[key] + 273.15)
    humidity = c["relative_humidity"] * (100.0 if humidity_percent else 1.0)
    struct.pack_into(order + "f", record, 0x3C, humidity)
    struct.pack_into(order + "i", record, 0x308, int(round(c["planck_o"])))
    struct.pack_into(order + "f", record, 0x1B4, field_of_view)
    for offset, size, text in ((0xD4, 32, camera_model), (0x104, 16, serial), (0x170, 32, lens_model)):
        record[offset:offset + size] = text.encode("latin-1")[:size - 1].ljust(size, b"\x00")
    if capture_time is not None:
        seconds = int(capture_time)
        struct.pack_into(order + "IIh", record, 0x384, seconds, int(round((capture_time - seconds) * 1000)), 0)
    return bytes(record)


def _encode_png16(values: np.ndarray) -> bytes:
    try:
        import cv2

        ok, png = cv2.imencode(".png", values)
        if not ok:
            raise ValueError("PNG encoding failed")
        return png.tobytes()
    except ImportError:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(values.astype("<u2"), mode="I;16").save(buffer, format="PNG")
        return buffer.getvalue()


def raw_data_record(raw: np.ndarray, *, raw_format="png", png_little_endian=True, order="<") -> bytes:
    """A RawData record holding `raw` (uint16, H x W)."""
    raw = np.asarray(raw, dtype=np.uint16)
    height, width = raw.shape
    header = bytearray(0x20)
    struct.pack_into(order + "HHH", header, 0, 2, width, height)
    struct.pack_into(order + "HH", header, 12, width - 1, height - 1)
    if raw_format == "png":
        payload = _encode_png16(raw.byteswap() if png_little_endian else raw)
    elif raw_format == "raw":
        payload = raw.astype(order + "u2").tobytes()
    else:
        raise ValueError("raw_format must be 'png' or 'raw'")
    return bytes(header) + payload


def fff_block(records, *, order=">", free_entries=1) -> bytes:
    """An FFF block: header, record directory (with `free_entries` unused entries), records."""
    count = free_entries + len(records)
    directory = 0x40
    position = directory + 0x20 * count
    entries, body = [], b""
    for _ in range(free_entries):
        entries.append(bytes(0x20))
    for index, (kind, data) in enumerate(records, start=1):
        entries.append(struct.pack(order + "HHIIIIIII", kind, 0, 100, index, position + len(body), len(data), 0, 0, 0))
        body += data
    header = bytearray(0x40)
    header[0:4] = b"FFF\x00"
    header[4:20] = b"ADE synthetic".ljust(16, b"\x00")
    struct.pack_into(order + "IIII", header, 0x14, 100, directory, count, count + 1)
    return bytes(header) + b"".join(entries) + body


def flir_app1_segments(fff: bytes, chunk_size=65000) -> bytes:
    """`fff` split over JPEG APP1 segments: "FLIR\\0", 0x01, segment index, last index, data."""
    chunks = [fff[i:i + chunk_size] for i in range(0, len(fff), chunk_size)] or [b""]
    if len(chunks) > 256:
        raise ValueError("FFF block too large for 256 APP1 segments")
    out = b""
    for index, chunk in enumerate(chunks):
        payload = b"FLIR\x00\x01" + bytes([index, len(chunks) - 1]) + chunk
        out += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return out


def _encode_jpeg(image: np.ndarray) -> bytes:
    try:
        import cv2

        ok, jpeg = cv2.imencode(".jpg", image[..., ::-1] if image.ndim == 3 else image)
        if not ok:
            raise ValueError("JPEG encoding failed")
        return jpeg.tobytes()
    except ImportError:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        return buffer.getvalue()


def rjpeg_bytes(raw, calibration=None, visual=None, *, raw_format="png", png_little_endian=True,
                record_order="<", directory_order=">", chunk_size=65000, **camera) -> bytes:
    """A FLIR R-JPEG: `visual` (uint8 RGB/grey; default an AGC rendering of `raw`) with the FLIR data."""
    from arraydataengine.ops import colorize_thermal

    raw = np.asarray(raw, dtype=np.uint16)
    fff = fff_block([(0x01, raw_data_record(raw, raw_format=raw_format, png_little_endian=png_little_endian,
                                            order=record_order)),
                     (0x20, camera_info_record(calibration, order=record_order, **camera))], order=directory_order)
    jpeg = _encode_jpeg(colorize_thermal(raw, cmap="gray") if visual is None else np.asarray(visual, dtype=np.uint8))
    insert = 2
    if jpeg[2:4] == b"\xff\xe0":  # keep the JFIF APP0 segment first
        insert = 4 + struct.unpack(">H", jpeg[4:6])[0]
    return jpeg[:insert] + flir_app1_segments(fff, chunk_size) + jpeg[insert:]


def write_flir_rjpeg(path, raw, calibration=None, visual=None, **options) -> Path:
    path = Path(path)
    path.write_bytes(rjpeg_bytes(raw, calibration, visual, **options))
    return path


def scene_temperatures(shape=(120, 160), frame=0) -> np.ndarray:
    """A room at 20-23 °C with a hot pipe (65 °C), a person (34-36.5 °C) and a cold window (8 °C), in °C."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    scene = 20.0 + 3.0 * rows / shape[0]
    scene[int(0.15 * shape[0]):int(0.25 * shape[0]), :] = 65.0  # pipe
    scene[int(0.55 * shape[0]):int(0.9 * shape[0]), int(0.7 * shape[1]):int(0.9 * shape[1])] = 8.0  # window
    centre = (0.6 * shape[0], (0.25 + 0.05 * frame) * shape[1])
    person = np.exp(-(((rows - centre[0]) / (0.18 * shape[0])) ** 2 + ((cols - centre[1]) / (0.08 * shape[1])) ** 2))
    return np.where(person > 0.3, 34.0 + 2.5 * person, scene)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--raw-format", choices=("png", "raw"), default="png")
    parser.add_argument("--size", default="160x120", help="WIDTHxHEIGHT of the thermal sensor")
    args = parser.parse_args(argv)

    from arraydataengine.ops import flir_temperature_to_raw

    width, height = (int(v) for v in args.size.lower().split("x"))
    calibration = _calibration(None)
    (args.out / "truth").mkdir(parents=True, exist_ok=True)
    names = []
    for frame in range(args.frames):
        truth = scene_temperatures((height, width), frame)
        raw = flir_temperature_to_raw(truth, calibration)
        name = f"FLIR{frame:04d}.jpg"
        write_flir_rjpeg(args.out / name, raw, calibration, raw_format=args.raw_format,
                         capture_time=1_700_000_000.0 + frame)
        np.save(args.out / "truth" / f"{Path(name).stem}.npy", truth)
        names.append(name)
    (args.out / "ground_truth.json").write_text(json.dumps(
        {"files": names, "calibration": calibration.to_dict(), "truth": "truth/<name>.npy, degrees Celsius"},
        indent=1))
    print(f"wrote {len(names)} FLIR R-JPEGs ({width} x {height}, {args.raw_format} raw data) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
