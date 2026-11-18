"""FLIR radiometric JPEGs (R-JPEG) and FFF files: raw counts and their temperature.

FLIR handheld, fixed and drone cameras (Exx/Txxx, A-series, Vue Pro R, Duo Pro
R, the FLIR-built DJI Zenmuse XT/XT2, ...) save *radiometric JPEGs*: an ordinary
JPEG, the palette image a viewer shows, that carries in APP1 segments tagged
``FLIR`` a FLIR File Format (FFF) block with the raw 16-bit sensor counts and
the camera's calibration. ``.fff`` files hold the same block on its own.

Unlike the "TLinear" output of radiometric Boson and Lepton cores
(`radiometric_to_temperature`), these counts follow the radiance reaching the
sensor. Turning them into the object's temperature needs

* the camera's Planck constants (R1, R2, B, F, O), which map a blackbody's
  temperature to counts: ``raw = R1 / (R2 * (exp(B / T) - F)) - O``;
* the object's emissivity and the reflected apparent temperature (a surface
  with emissivity E emits E of a blackbody's radiance and reflects 1 - E of its
  surroundings');
* the atmosphere between object and camera (distance, air temperature and
  relative humidity, through the camera's atmospheric transmission
  coefficients), and an external IR window (temperature, transmission).

`flir_raw_to_temperature` removes the reflected, atmospheric and window
contributions and inverts the Planck curve, following the model FLIR's
software uses as documented by ExifTool and the Thermimage R package; the
object distance is split evenly between the object-window and window-camera
paths. The measurement parameters stored in the file are what the operator
set on the camera; change them with ``image.temperature(emissivity=0.95, ...)``
or `FlirCalibration.replace`.

DJI's own radiometric JPEGs (H20T, M30T, Mavic 2/3 Enterprise, ...) use a
different format that needs DJI's Thermal SDK and are not read here.
"""

from __future__ import annotations

import dataclasses
import io
import math
import os
import struct
from dataclasses import dataclass

import numpy as np

from .thermal import _from_kelvin, _temperature_unit, _to_kelvin

_FFF_MAGICS = (b"FFF\x00", b"AFF\x00")
_FLIR_APP1 = b"FLIR\x00"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_RECORD_RAW_DATA = 0x01
_RECORD_CAMERA_INFO = 0x20
_RAW_HEADER_SIZE = 0x20
_ZERO_CELSIUS = 273.15

# Byte offsets in the CameraInfo record (ExifTool's FLIR CameraInfo table).
# Temperatures are stored in kelvin, the humidity as a fraction (or percent).
_CAMERA_FLOATS = {
    "emissivity": 0x20,
    "object_distance": 0x24,
    "reflected_temperature": 0x28,
    "atmospheric_temperature": 0x2C,
    "window_temperature": 0x30,
    "window_transmission": 0x34,
    "relative_humidity": 0x3C,
    "planck_r1": 0x58,
    "planck_b": 0x5C,
    "planck_f": 0x60,
    "atmospheric_trans_alpha1": 0x70,
    "atmospheric_trans_alpha2": 0x74,
    "atmospheric_trans_beta1": 0x78,
    "atmospheric_trans_beta2": 0x7C,
    "atmospheric_trans_x": 0x80,
    "field_of_view": 0x1B4,
    "planck_r2": 0x30C,
}
_CAMERA_PLANCK_O = 0x308  # int32
_CAMERA_STRINGS = {"camera_model": (0xD4, 32), "camera_serial_number": (0x104, 16), "lens_model": (0x170, 32)}
_CAMERA_DATETIME = 0x384  # uint32 UTC seconds, uint32 milliseconds (low 16 bits), int16 time zone


@dataclass(frozen=True)
class FlirCalibration:
    """Radiometric calibration and measurement conditions of a FLIR image.

    The Planck constants (`planck_r1`, `planck_b`, `planck_f`, `planck_o`,
    `planck_r2`) and the atmospheric transmission coefficients come from the
    camera's factory calibration. The others describe the measurement and are
    the ones to change when re-analysing an image:

    * `emissivity` of the object's surface (0 < E <= 1);
    * `reflected_temperature_c`: apparent temperature of the surroundings the
      surface reflects, in °C;
    * `object_distance` in metres, `atmospheric_temperature_c` (°C) and
      `relative_humidity` (a fraction, 0-1) of the air in between;
    * an external IR window's `window_temperature_c` (°C) and
      `window_transmission` (1 when there is none).
    """

    planck_r1: float
    planck_b: float
    planck_f: float = 1.0
    planck_o: float = 0.0
    planck_r2: float = 1.0
    emissivity: float = 1.0
    object_distance: float = 1.0
    reflected_temperature_c: float = 20.0
    atmospheric_temperature_c: float = 20.0
    window_temperature_c: float = 20.0
    window_transmission: float = 1.0
    relative_humidity: float = 0.5
    atmospheric_trans_alpha1: float = 0.006569
    atmospheric_trans_alpha2: float = 0.01262
    atmospheric_trans_beta1: float = -0.002276
    atmospheric_trans_beta2: float = -0.00667
    atmospheric_trans_x: float = 1.9

    def __post_init__(self):
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{item.name} must be a number, got {value!r}") from None
            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite, got {value}")
            object.__setattr__(self, item.name, value)
        if self.planck_r1 <= 0 or self.planck_r2 <= 0 or self.planck_b <= 0:
            raise ValueError("planck_r1, planck_r2 and planck_b must be positive")
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError(f"emissivity must be in (0, 1], got {self.emissivity}")
        if not 0.0 < self.window_transmission <= 1.0:
            raise ValueError(f"window_transmission must be in (0, 1] (1 without a window), got {self.window_transmission}")
        if self.object_distance < 0:
            raise ValueError(f"object_distance must be >= 0 metres, got {self.object_distance}")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError(f"relative_humidity is a fraction in [0, 1], got {self.relative_humidity} "
                             "(divide a percentage by 100)")
        for name in ("reflected_temperature_c", "atmospheric_temperature_c", "window_temperature_c"):
            if getattr(self, name) <= -_ZERO_CELSIUS:
                raise ValueError(f"{name} is in degrees Celsius and must be above -273.15")

    def replace(self, **changes) -> "FlirCalibration":
        """A copy with some fields changed, e.g. ``replace(emissivity=0.95)``."""
        unknown = set(changes) - {item.name for item in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"unknown FlirCalibration fields: {sorted(unknown)}")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values) -> "FlirCalibration":
        """Build from a mapping such as a message's ``"flir"`` metadata (other keys are ignored)."""
        names = {item.name for item in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in dict(values).items() if key in names})


@dataclass(frozen=True, eq=False)
class FlirRadiometricImage:
    """The radiometric content of a FLIR R-JPEG or FFF file.

    `raw` is the ``(H, W)`` ``uint16`` sensor counts (its size is the thermal
    sensor's, not that of the JPEG's palette or visual image); `calibration`
    holds the Planck constants and the measurement parameters stored with it.
    `capture_time` is the camera's UTC timestamp (Unix seconds) when present.
    """

    raw: np.ndarray
    calibration: FlirCalibration
    camera_model: str = ""
    camera_serial_number: str = ""
    lens_model: str = ""
    field_of_view: float | None = None
    capture_time: float | None = None

    def temperature(self, unit: str = "celsius", **changes) -> np.ndarray:
        """Per-pixel object temperature, float64 in `unit`.

        Keyword arguments override calibration fields for this conversion,
        e.g. ``temperature(emissivity=0.95, reflected_temperature_c=18.0)``.
        """
        calibration = self.calibration.replace(**changes) if changes else self.calibration
        return flir_raw_to_temperature(self.raw, calibration, unit=unit)

    def metadata(self) -> dict:
        """JSON-compatible calibration and camera information (`FlirCalibration.from_dict` reads it back)."""
        return {
            **self.calibration.to_dict(),
            "camera_model": self.camera_model,
            "camera_serial_number": self.camera_serial_number,
            "lens_model": self.lens_model,
            "field_of_view": self.field_of_view,
            "capture_time": self.capture_time,
        }


# ----------------------------------------------------------------------------- conversion


def _planck_signal(kelvin, calibration: FlirCalibration):
    """Counts a blackbody at `kelvin` produces: R1 / (R2 (exp(B / T) - F)) - O."""
    c = calibration
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return c.planck_r1 / (c.planck_r2 * (np.exp(c.planck_b / kelvin) - c.planck_f)) - c.planck_o


def _path_transmission(calibration: FlirCalibration) -> float:
    """Atmospheric transmission over half the object distance (object-window, window-camera)."""
    c = calibration
    t = c.atmospheric_temperature_c
    water = c.relative_humidity * math.exp(1.5587 + 0.06939 * t - 0.00027816 * t ** 2 + 0.00000068455 * t ** 3)
    root = math.sqrt(c.object_distance / 2.0)
    return (c.atmospheric_trans_x * math.exp(-root * (c.atmospheric_trans_alpha1 + c.atmospheric_trans_beta1 * math.sqrt(water)))
            + (1.0 - c.atmospheric_trans_x)
            * math.exp(-root * (c.atmospheric_trans_alpha2 + c.atmospheric_trans_beta2 * math.sqrt(water))))


def _measurement_model(calibration: FlirCalibration) -> tuple[float, float]:
    """``raw = gain * object_signal + offset`` for the calibration's conditions."""
    c = calibration
    tau = _path_transmission(c)
    e, window = c.emissivity, c.window_transmission
    reflected = float(_planck_signal(c.reflected_temperature_c + _ZERO_CELSIUS, c))
    atmosphere = float(_planck_signal(c.atmospheric_temperature_c + _ZERO_CELSIUS, c))
    window_signal = float(_planck_signal(c.window_temperature_c + _ZERO_CELSIUS, c))
    gain = e * tau * window * tau  # the object's emission reaching the sensor
    offset = ((1.0 - e) * tau * window * tau * reflected  # surroundings reflected by the object
              + (1.0 - tau) * window * tau * atmosphere  # air between object and window
              + (1.0 - window) * tau * window_signal  # the window's own emission
              + (1.0 - tau) * atmosphere)  # air between window and camera
    return gain, offset


def flir_raw_to_temperature(raw, calibration: FlirCalibration, unit: str = "celsius") -> np.ndarray:
    """Object temperature from FLIR raw counts (R-JPEG / FFF), float64 in `unit`.

    Removes the reflected, atmospheric and window contributions described by
    `calibration`, then inverts the Planck curve. Counts that no temperature
    can produce under these conditions (e.g. below the reflected background
    for a low emissivity) give NaN.
    """

    unit = _temperature_unit(unit)
    c = calibration
    gain, offset = _measurement_model(c)
    signal = (np.asarray(raw, dtype=np.float64) - offset) / gain
    with np.errstate(divide="ignore", invalid="ignore"):
        kelvin = c.planck_b / np.log(c.planck_r1 / (c.planck_r2 * (signal + c.planck_o)) + c.planck_f)
    kelvin = np.where(np.isfinite(kelvin) & (kelvin > 0), kelvin, np.nan)
    return _from_kelvin(kelvin, unit)[()]


def flir_temperature_to_raw(temperature, calibration: FlirCalibration, unit: str = "celsius", dtype=np.uint16):
    """Counts a FLIR camera with `calibration` records for objects at `temperature`.

    The inverse of `flir_raw_to_temperature`, for simulation and test data.
    Integer `dtype` results are rounded and clipped to the type's range.
    """

    unit = _temperature_unit(unit)
    gain, offset = _measurement_model(calibration)
    raw = np.asarray(gain * _planck_signal(_to_kelvin(temperature, unit), calibration) + offset, dtype=np.float64)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return raw.astype(dtype)[()]
    if dtype.kind not in "iu":
        raise ValueError("dtype must be an integer or float dtype")
    if not np.all(np.isfinite(raw)):
        raise ValueError("temperatures outside the camera's Planck curve give no finite counts")
    info = np.iinfo(dtype)
    return np.clip(np.rint(raw), info.min, info.max).astype(dtype)[()]


# ----------------------------------------------------------------------------- reading


def _read_bytes(source) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), "data"
    if hasattr(source, "read"):
        return source.read(), getattr(source, "name", "file")
    path = os.fspath(source)
    with open(path, "rb") as handle:
        return handle.read(), os.path.basename(path)


def _jpeg_segments(data: bytes):
    """(marker, payload) of the JPEG segments before the image data."""
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            return
        marker = data[position + 1]
        if marker == 0xFF:  # fill byte
            position += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # markers without a length
            position += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image / start of scan: no metadata follows
            return
        length = struct.unpack_from(">H", data, position + 2)[0]
        if length < 2:
            return
        yield marker, data[position + 4:position + 2 + length]
        position += 2 + length


def _fff_from_jpeg(data: bytes, name: str) -> bytes:
    chunks, last = {}, None
    for marker, payload in _jpeg_segments(data):
        if marker == 0xE1 and payload.startswith(_FLIR_APP1) and len(payload) >= 8:
            chunks.setdefault(payload[6], payload[8:])
            last = payload[7] if last is None else last
    if not chunks:
        hint = ""
        if b"DJI" in data[:65536]:
            hint = "; DJI-format radiometric JPEGs (H20T, M30T, Mavic Enterprise, ...) need DJI's Thermal SDK"
        raise ValueError(f"{name}: no FLIR radiometric data (no FLIR APP1 segments){hint}")
    missing = [index for index in range(last + 1) if index not in chunks]
    if missing:
        raise ValueError(f"{name}: the FLIR data is incomplete (APP1 segments {missing} of 0-{last} missing)")
    return b"".join(chunks[index] for index in range(last + 1))


def _record_order(record: bytes) -> str:
    """Byte order of a record: its first uint16 is 2 in the record's own order."""
    return ">" if struct.unpack_from(">H", record)[0] < 0x100 else "<"


def _decode_png16(payload: bytes, name: str) -> np.ndarray:
    try:
        import cv2

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except ImportError:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("reading PNG-compressed FLIR raw data needs OpenCV or Pillow "
                              '(pip install "arraydataengine[image]")') from None
        with Image.open(io.BytesIO(payload)) as png:
            image = np.array(png)
    if image is None or image.ndim != 2:
        raise ValueError(f"{name}: the FLIR raw thermal image is not a single-channel PNG")
    return image.astype(np.uint16, copy=False)


def _roughness(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    rows = np.abs(np.diff(image, axis=1)).mean() if image.shape[1] > 1 else 0.0
    cols = np.abs(np.diff(image, axis=0)).mean() if image.shape[0] > 1 else 0.0
    return float(rows + cols)


def _parse_raw_data(record: bytes, name: str) -> np.ndarray:
    if len(record) < _RAW_HEADER_SIZE:
        raise ValueError(f"{name}: truncated FLIR raw data record")
    order = _record_order(record)
    width, height = struct.unpack_from(order + "HH", record, 2)
    payload = record[_RAW_HEADER_SIZE:]
    if payload.startswith(_PNG_MAGIC):
        image = _decode_png16(payload, name)
        # Most FLIR cameras write the PNG's 16-bit samples little-endian (against the PNG
        # standard), some correctly: the right order gives the smooth image.
        swapped = image.byteswap()
        return min((swapped, image), key=_roughness)
    count = width * height
    if count == 0 or len(payload) < 2 * count:
        raise ValueError(f"{name}: FLIR raw data record too short for a {width} x {height} image")
    return np.frombuffer(payload, dtype=order + "u2", count=count).reshape(height, width).astype(np.uint16)


def _parse_camera_info(record: bytes, name: str) -> dict:
    order = _record_order(record)

    def number(offset, code="f"):
        size = struct.calcsize(code)
        if offset + size > len(record):
            return None
        value = float(struct.unpack_from(order + code, record, offset)[0])
        return value if math.isfinite(value) else None

    values = {key: number(offset) for key, offset in _CAMERA_FLOATS.items()}
    values["planck_o"] = number(_CAMERA_PLANCK_O, "i")
    if not values["planck_r1"] or not values["planck_b"]:
        raise ValueError(f"{name}: the FLIR camera information has no Planck constants")
    calibration = {key: values[key] for key in (
        "planck_r1", "planck_b", "planck_f", "planck_o", "planck_r2", "emissivity", "object_distance",
        "window_transmission", "atmospheric_trans_alpha1", "atmospheric_trans_alpha2",
        "atmospheric_trans_beta1", "atmospheric_trans_beta2", "atmospheric_trans_x",
    ) if values[key] is not None}
    for key in ("reflected_temperature", "atmospheric_temperature", "window_temperature"):
        if values[key] is not None:
            calibration[key + "_c"] = values[key] - _ZERO_CELSIUS
    humidity = values["relative_humidity"]
    if humidity is not None:  # some cameras store percent (ExifTool: values above 2 are percent)
        calibration["relative_humidity"] = min(max(humidity / 100.0 if humidity > 2 else humidity, 0.0), 1.0)
    try:
        calibration = FlirCalibration(**calibration)
    except ValueError as exc:
        raise ValueError(f"{name}: invalid FLIR calibration: {exc}") from None

    info = {"calibration": calibration}
    for key, (offset, size) in _CAMERA_STRINGS.items():
        text = record[offset:offset + size].split(b"\x00", 1)[0]
        info[key] = text.decode("latin-1").strip()
    fov = values["field_of_view"]
    info["field_of_view"] = fov if fov is not None and fov > 0 else None
    if _CAMERA_DATETIME + 8 <= len(record):
        seconds, milliseconds = struct.unpack_from(order + "II", record, _CAMERA_DATETIME)
        info["capture_time"] = seconds + (milliseconds & 0xFFFF) / 1000.0 if seconds else None
    return info


def _parse_fff(fff: bytes, name: str) -> FlirRadiometricImage:
    if len(fff) < 0x40 or fff[:4] not in _FFF_MAGICS:
        raise ValueError(f"{name}: not FLIR FFF data")
    for order in (">", "<"):
        if 100 <= struct.unpack_from(order + "I", fff, 0x14)[0] < 200:
            break
    else:
        raise ValueError(f"{name}: unsupported FLIR FFF version")
    directory, count = struct.unpack_from(order + "II", fff, 0x18)
    records = {}
    for index in range(count):
        entry = directory + 0x20 * index
        if entry + 0x20 > len(fff):
            raise ValueError(f"{name}: truncated FLIR record directory")
        kind = struct.unpack_from(order + "H", fff, entry)[0]
        if kind == 0:  # free entry
            continue
        offset, length = struct.unpack_from(order + "II", fff, entry + 0x0C)
        if offset + length > len(fff):
            raise ValueError(f"{name}: FLIR record 0x{kind:02x} runs past the end of the data")
        records.setdefault(kind, fff[offset:offset + length])
    if _RECORD_RAW_DATA not in records:
        raise ValueError(f"{name}: no raw thermal image in the FLIR data")
    if _RECORD_CAMERA_INFO not in records:
        raise ValueError(f"{name}: no camera calibration in the FLIR data")
    raw = _parse_raw_data(records[_RECORD_RAW_DATA], name)
    return FlirRadiometricImage(raw=raw, **_parse_camera_info(records[_RECORD_CAMERA_INFO], name))


def read_flir_radiometric(source) -> FlirRadiometricImage:
    """Raw counts and calibration of a FLIR radiometric JPEG or ``.fff`` file.

    `source` is a path, the file's bytes, or a binary file object. Raises
    ValueError for files without FLIR radiometric data (an ordinary JPEG, or a
    DJI-format R-JPEG).
    """

    data, name = _read_bytes(source)
    if data[:4] in _FFF_MAGICS:
        return _parse_fff(data, name)
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{name}: not a JPEG or FLIR FFF file")
    return _parse_fff(_fff_from_jpeg(data, name), name)


def is_flir_radiometric(source) -> bool:
    """True if `source` (path, bytes or file object) is a FLIR R-JPEG or FFF file."""

    data, _ = _read_bytes(source)
    if data[:4] in _FFF_MAGICS:
        return True
    if data[:2] != b"\xff\xd8":
        return False
    return any(marker == 0xE1 and payload.startswith(_FLIR_APP1) for marker, payload in _jpeg_segments(data))
