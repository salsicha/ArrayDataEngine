"""FLIR radiometric JPEGs (R-JPEG) and FFF files: container parsing and the Planck conversion."""

import importlib.util
import json
import math
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from arraydataengine.ops import (
    FlirCalibration,
    flir_raw_to_temperature,
    flir_temperature_to_raw,
    is_flir_radiometric,
    radiometric_to_temperature,
    read_flir_radiometric,
)

ROOT = Path(__file__).resolve().parents[1]

# Planck constants of two kinds of FLIR cameras (Thermimage's defaults, and an E-series-like set).
CAMERA_A = dict(planck_r1=21106.77, planck_b=1501.0, planck_f=1.0, planck_o=-7340.0, planck_r2=0.012545258)
CAMERA_B = dict(planck_r1=14364.633, planck_b=1385.4, planck_f=1.0, planck_o=-5753.0, planck_r2=0.010603162)
FIELD_CONDITIONS = dict(emissivity=0.62, object_distance=25.0, reflected_temperature_c=-5.0,
                        atmospheric_temperature_c=31.0, window_temperature_c=12.0, window_transmission=0.83,
                        relative_humidity=0.85)


def _maker():
    pytest.importorskip("cv2")
    spec = importlib.util.spec_from_file_location("make_flir_rjpeg", ROOT / "scripts" / "datasets" / "make_flir_rjpeg.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(shape=(48, 64)):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    scene = 15.0 + 10.0 * cols / shape[1]
    scene[10:20, 5:30] = 80.0
    scene[30:40, 40:60] = -10.0
    return scene


# --- the conversion ---------------------------------------------------------------------------------


def _blackbody(kelvin, c):
    return c.planck_r1 / (c.planck_r2 * (np.exp(c.planck_b / kelvin) - c.planck_f)) - c.planck_o


def _half_path_transmission(c):
    t = c.atmospheric_temperature_c
    h2o = c.relative_humidity * math.exp(1.5587 + 0.06939 * t - 0.00027816 * t ** 2 + 0.00000068455 * t ** 3)
    d = math.sqrt(c.object_distance / 2)
    return (c.atmospheric_trans_x * math.exp(-d * (c.atmospheric_trans_alpha1 + c.atmospheric_trans_beta1 * math.sqrt(h2o)))
            + (1 - c.atmospheric_trans_x) * math.exp(-d * (c.atmospheric_trans_alpha2 + c.atmospheric_trans_beta2 * math.sqrt(h2o))))


def _sensor_counts(object_c, c):
    """Radiance at the sensor followed along its path: object, air, window, air, camera."""
    k, tau = 273.15, _half_path_transmission(c)
    leaving = c.emissivity * _blackbody(object_c + k, c) + (1 - c.emissivity) * _blackbody(c.reflected_temperature_c + k, c)
    at_window = tau * leaving + (1 - tau) * _blackbody(c.atmospheric_temperature_c + k, c)
    behind = c.window_transmission * at_window + (1 - c.window_transmission) * _blackbody(c.window_temperature_c + k, c)
    return tau * behind + (1 - tau) * _blackbody(c.atmospheric_temperature_c + k, c)


def _thermimage_raw2temp(raw, c):
    """Thermimage's raw2temp (also used by flirimageextractor), transcribed; RH in percent there."""
    E, OD, IRT = c.emissivity, c.object_distance, c.window_transmission
    RTemp, ATemp, IRWTemp, RH = c.reflected_temperature_c, c.atmospheric_temperature_c, c.window_temperature_c, 100 * c.relative_humidity
    PR1, PB, PF, PO, PR2 = c.planck_r1, c.planck_b, c.planck_f, c.planck_o, c.planck_r2
    ATA1, ATA2, ATB1, ATB2, ATX = (c.atmospheric_trans_alpha1, c.atmospheric_trans_alpha2, c.atmospheric_trans_beta1,
                                   c.atmospheric_trans_beta2, c.atmospheric_trans_x)
    emiss_wind = 1 - IRT
    refl_wind = 0
    h2o = (RH / 100) * np.exp(1.5587 + 0.06939 * ATemp - 0.00027816 * ATemp ** 2 + 0.00000068455 * ATemp ** 3)
    tau1 = ATX * np.exp(-np.sqrt(OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(-np.sqrt(OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))
    tau2 = tau1
    raw_refl1 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl1_attn = (1 - E) / E * raw_refl1
    raw_atm1 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
    raw_wind = PR1 / (PR2 * (np.exp(PB / (IRWTemp + 273.15)) - PF)) - PO
    raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
    raw_refl2 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
    raw_atm2 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
    raw_obj = raw / E / tau1 / IRT / tau2 - raw_atm1_attn - raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn
    return PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15


@pytest.mark.parametrize("camera", [CAMERA_A, CAMERA_B])
@pytest.mark.parametrize("conditions", [{}, FIELD_CONDITIONS], ids=["defaults", "field"])
def test_conversion_recovers_the_object_temperature_from_the_radiance_at_the_sensor(camera, conditions):
    c = FlirCalibration(**camera, **conditions)
    truth = _scene()
    raw = _sensor_counts(truth, c)
    np.testing.assert_allclose(flir_raw_to_temperature(raw, c), truth, atol=1e-6)
    np.testing.assert_allclose(flir_raw_to_temperature(raw, c), _thermimage_raw2temp(raw, c), atol=1e-6)
    np.testing.assert_allclose(flir_temperature_to_raw(truth, c, dtype=np.float64), raw, rtol=1e-12)


def test_pure_planck_inversion_without_emissivity_or_atmosphere():
    c = FlirCalibration(**CAMERA_A, emissivity=1.0, object_distance=0.0, window_transmission=1.0)
    raw = np.array([12000.0, 18000.0, 30000.0])
    expected = c.planck_b / np.log(c.planck_r1 / (c.planck_r2 * (raw + c.planck_o)) + c.planck_f) - 273.15
    np.testing.assert_allclose(flir_raw_to_temperature(raw, c), expected, rtol=1e-12)
    assert float(flir_raw_to_temperature(18000, c)) == pytest.approx(23.03, abs=0.01)


def test_in_thermal_equilibrium_every_parameter_cancels():
    # Object, surroundings, air and window all at 30 °C: the sensor sees a 30 °C blackbody,
    # and any emissivity, distance, humidity or window gives back 30 °C.
    raw = _blackbody(30.0 + 273.15, FlirCalibration(**CAMERA_A))
    for emissivity, distance, humidity, window in [(1.0, 0, 0.5, 1.0), (0.3, 100, 0.95, 0.5), (0.9, 7, 0.1, 0.9)]:
        c = FlirCalibration(**CAMERA_A, emissivity=emissivity, object_distance=distance, relative_humidity=humidity,
                            window_transmission=window, reflected_temperature_c=30.0, atmospheric_temperature_c=30.0,
                            window_temperature_c=30.0)
        assert float(flir_raw_to_temperature(raw, c)) == pytest.approx(30.0, abs=1e-9)


def test_emissivity_and_atmosphere_change_the_result_as_physics_says():
    base = FlirCalibration(**CAMERA_A, emissivity=1.0, object_distance=0.0, reflected_temperature_c=20.0)
    raw = flir_temperature_to_raw(60.0, base, dtype=np.float64)  # 60 °C apparent (blackbody) temperature
    # A grey surface emits less than a blackbody, so the same signal means a hotter object.
    assert float(flir_raw_to_temperature(raw, base.replace(emissivity=0.8))) > 62.0
    # Air absorbs part of the object's radiance on the way: hotter again, more with distance and humidity.
    near = float(flir_raw_to_temperature(raw, base.replace(object_distance=10.0, atmospheric_temperature_c=20.0)))
    far = float(flir_raw_to_temperature(raw, base.replace(object_distance=200.0, atmospheric_temperature_c=20.0)))
    humid = float(flir_raw_to_temperature(raw, base.replace(object_distance=200.0, atmospheric_temperature_c=20.0,
                                                           relative_humidity=0.95)))
    assert 60.0 < near < far < humid


def test_the_linear_tlinear_model_is_wrong_for_rjpeg_counts():
    c = FlirCalibration(**CAMERA_A, emissivity=1.0, object_distance=0.0)
    truth = np.linspace(-20.0, 150.0, 50)
    raw = flir_temperature_to_raw(truth, c, dtype=np.float64)
    assert np.abs(radiometric_to_temperature(raw, scale=0.01) - truth).max() > 100  # Boson/Lepton TLinear scale
    # Counts follow radiance, not temperature: even the best-fitting linear scale and offset is degrees off.
    scale, offset = np.polyfit(raw, truth + 273.15, 1)
    assert np.abs(radiometric_to_temperature(raw, scale=scale, offset=offset) - truth).max() > 3
    np.testing.assert_allclose(flir_raw_to_temperature(raw, c), truth, atol=1e-9)


def test_units_rounding_and_impossible_counts():
    c = FlirCalibration(**CAMERA_A, **FIELD_CONDITIONS)
    truth = _scene()
    counts = flir_temperature_to_raw(truth, c)
    assert counts.dtype == np.uint16
    assert np.abs(flir_raw_to_temperature(counts, c) - truth).max() < 0.02  # one count is ~0.01-0.02 K
    np.testing.assert_allclose(flir_raw_to_temperature(counts, c, unit="kelvin"), flir_raw_to_temperature(counts, c) + 273.15)
    np.testing.assert_allclose(flir_raw_to_temperature(counts, c, unit="fahrenheit"), flir_raw_to_temperature(counts, c) * 1.8 + 32)
    np.testing.assert_allclose(flir_temperature_to_raw(truth + 273.15, c, unit="kelvin"), counts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = flir_raw_to_temperature(np.array([0, 1, 65535]), c.replace(emissivity=0.1, reflected_temperature_c=40.0))
    assert np.isnan(result[0]) and np.isnan(result[1]) and np.isfinite(result[2])
    assert isinstance(flir_raw_to_temperature(20000, c), float)
    with pytest.raises(ValueError, match="unit"):
        flir_raw_to_temperature(counts, c, unit="rankine")


def test_calibration_validation_and_round_trip():
    c = FlirCalibration(**CAMERA_A)
    assert FlirCalibration.from_dict({**c.to_dict(), "camera_model": "ignored"}) == c
    assert c.replace(emissivity=0.9).emissivity == 0.9 and c.emissivity == 1.0
    with pytest.raises(TypeError, match="unknown"):
        c.replace(emisivity=0.9)
    with pytest.raises(ValueError, match="fraction"):
        FlirCalibration(**CAMERA_A, relative_humidity=50)
    with pytest.raises(ValueError, match="emissivity"):
        FlirCalibration(**CAMERA_A, emissivity=0)
    with pytest.raises(ValueError, match="window_transmission"):
        FlirCalibration(**CAMERA_A, window_transmission=0)
    with pytest.raises(ValueError, match="Celsius"):
        FlirCalibration(**CAMERA_A, reflected_temperature_c=-300)
    with pytest.raises(ValueError, match="positive"):
        FlirCalibration(planck_r1=0, planck_b=1500)
    with pytest.raises(ValueError, match="finite"):
        FlirCalibration(**CAMERA_A, object_distance=float("nan"))


# --- the file format --------------------------------------------------------------------------------


@pytest.mark.parametrize("raw_format, png_little_endian, record_order, directory_order", [
    ("png", True, "<", ">"),   # what most cameras write
    ("png", False, "<", ">"),  # PNG samples in the standard byte order
    ("raw", None, "<", ">"),
    ("raw", None, ">", ">"),
    ("raw", None, "<", "<"),
])
def test_rjpeg_raw_counts_and_calibration_are_read_back(tmp_path, raw_format, png_little_endian, record_order, directory_order):
    maker = _maker()
    c = FlirCalibration(**CAMERA_B, **FIELD_CONDITIONS)
    truth = _scene((120, 160))
    raw = flir_temperature_to_raw(truth, c)
    path = maker.write_flir_rjpeg(tmp_path / "IR_0001.jpg", raw, c, raw_format=raw_format,
                                  png_little_endian=bool(png_little_endian), record_order=record_order,
                                  directory_order=directory_order, chunk_size=20000,  # several APP1 segments
                                  camera_model="FLIR E8", serial="63912345", lens_model="FOL7",
                                  field_of_view=45.0, capture_time=1_700_000_123.25)
    assert path.read_bytes().count(b"\xff\xe1") >= 2 or raw_format == "png"
    assert is_flir_radiometric(path)

    image = read_flir_radiometric(path)
    assert image.raw.dtype == np.uint16
    np.testing.assert_array_equal(image.raw, raw)
    for key, value in c.to_dict().items():  # stored as float32 (kelvin for temperatures)
        assert getattr(image.calibration, key) == pytest.approx(value, rel=1e-6, abs=1e-4), key
    assert (image.camera_model, image.camera_serial_number, image.lens_model) == ("FLIR E8", "63912345", "FOL7")
    assert image.field_of_view == pytest.approx(45.0) and image.capture_time == pytest.approx(1_700_000_123.25)
    assert np.abs(image.temperature() - truth).max() < 0.02
    # The operator's emissivity can be corrected afterwards.
    corrected = image.temperature(emissivity=0.95)
    assert np.all(corrected[10:20, 5:30] < image.temperature()[10:20, 5:30])
    # The JPEG itself still decodes as the palette image.
    import cv2

    assert cv2.imread(str(path)).shape == (120, 160, 3)
    json.dumps(image.metadata())


def test_fff_files_bytes_and_file_objects_are_read(tmp_path):
    maker = _maker()
    c = FlirCalibration(**CAMERA_A)
    raw = flir_temperature_to_raw(_scene(), c)
    fff = maker.fff_block([(0x01, maker.raw_data_record(raw, raw_format="raw")),
                           (0x20, maker.camera_info_record(c, humidity_percent=True))], order="<")
    (tmp_path / "frame.fff").write_bytes(fff)
    for source in (tmp_path / "frame.fff", str(tmp_path / "frame.fff"), fff, open(tmp_path / "frame.fff", "rb")):
        image = read_flir_radiometric(source)
        np.testing.assert_array_equal(image.raw, raw)
        assert image.calibration.relative_humidity == pytest.approx(0.5)  # stored as 50 (%)
    assert is_flir_radiometric(fff)


def test_files_without_flir_data_are_rejected_clearly(tmp_path):
    maker = _maker()
    import cv2

    plain = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    assert not is_flir_radiometric(plain)
    with pytest.raises(ValueError, match="no FLIR APP1"):
        read_flir_radiometric(plain)
    exif = b"Exif\x00\x00" + b"\x00" * 20 + b"DJI\x00ZH20T\x00"
    dji = plain[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + plain[2:]
    with pytest.raises(ValueError, match="DJI"):
        read_flir_radiometric(dji)
    with pytest.raises(ValueError, match="not a JPEG"):
        read_flir_radiometric(b"\x89PNG\r\n\x1a\n" + bytes(32))

    c = FlirCalibration(**CAMERA_A)
    raw = flir_temperature_to_raw(_scene(), c)
    fff = maker.fff_block([(0x01, maker.raw_data_record(raw, raw_format="raw")), (0x20, maker.camera_info_record(c))])
    third = len(fff) // 3 + 1
    segments = b""
    for index in (0, 2):  # segment 1 lost
        payload = b"FLIR\x00\x01" + bytes([index, 2]) + fff[index * third:(index + 1) * third]
        segments += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    with pytest.raises(ValueError, match="incomplete"):
        read_flir_radiometric(plain[:2] + segments + plain[2:])
    with pytest.raises(ValueError, match="past the end"):
        read_flir_radiometric(fff[:-100])
    no_camera = maker.fff_block([(0x01, maker.raw_data_record(raw, raw_format="raw"))])
    with pytest.raises(ValueError, match="calibration"):
        read_flir_radiometric(no_camera)


# --- image sources and buffers ----------------------------------------------------------------------


def _write_sequence(tmp_path, emissivities=(0.95, 0.6, 0.95)):
    maker = _maker()
    truths = []
    for index, emissivity in enumerate(emissivities):
        c = FlirCalibration(**CAMERA_A, **{**FIELD_CONDITIONS, "emissivity": emissivity})
        truth = _scene() + index
        maker.write_flir_rjpeg(tmp_path / f"FLIR{index:04d}.jpg", flir_temperature_to_raw(truth, c), c)
        truths.append(truth)
    return truths


def test_data_sources_yield_flir_raw_counts_or_temperatures(tmp_path):
    from arraydataengine import DataSources

    truths = _write_sequence(tmp_path)  # per-file calibration: the second frame has another emissivity
    pattern = str(tmp_path / "*.jpg")

    messages = list(DataSources(pattern, flir="celsius").get_message())
    assert [m["name"] for m in messages] == ["FLIR0000.jpg", "FLIR0001.jpg", "FLIR0002.jpg"]
    for message, truth in zip(messages, truths):
        assert message["data"].dtype == np.float32 and message["encoding"] == "32FC1"
        assert np.abs(message["data"] - truth).max() < 0.02
        assert FlirCalibration.from_dict(message["flir"]).emissivity == pytest.approx(
            0.6 if message["name"] == "FLIR0001.jpg" else 0.95)
        json.dumps(message["flir"])

    raw = next(DataSources(pattern, flir="raw").get_message())
    assert raw["data"].dtype == np.uint16 and raw["encoding"] == "mono16"
    kelvin = next(DataSources(pattern, flir="kelvin").get_message())["data"]
    np.testing.assert_allclose(kelvin, messages[0]["data"] + 273.15, atol=1e-3)
    palette = next(DataSources(pattern).get_message())  # default: the JPEG's palette image
    assert palette["encoding"] == "rgb8" and palette["data"].shape == (48, 64, 3)

    with pytest.raises(ValueError, match="flir must be one of"):
        DataSources(pattern, flir="radiance")
    with pytest.raises(ValueError, match="applies to image files"):
        DataSources(str(ROOT / "example" / "test.bag"), flir="raw")
    plain = tmp_path / "plain"
    plain.mkdir()
    import cv2

    cv2.imwrite(str(plain / "visual.jpg"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="visual.jpg: no FLIR"):
        list(DataSources(str(plain / "*.jpg"), flir="celsius").get_message())


@pytest.mark.parametrize("backend", ["memory", "mcap", "arrow", "tiledb"])
def test_flir_temperatures_round_trip_through_every_buffer(tmp_path, backend):
    from arraydataengine import DataBuffer, DataSources

    truths = _write_sequence(tmp_path)
    source = DataSources(str(tmp_path / "*.jpg"), flir="celsius")
    if backend == "memory":
        buffer = DataBuffer(source, buffer_depth=3, axis="images", preload=True)
    else:
        pytest.importorskip({"mcap": "mcap", "arrow": "pyarrow", "tiledb": "tiledb"}[backend])
        uri = str(tmp_path / f"store_{backend}")
        with DataBuffer(source, data_uri=uri, backend=backend, axis="images", preload=0) as writer:
            writer.load_data_db("images")
        buffer = DataBuffer(None, data_uri=uri, backend=backend, axis="images")
    stored = buffer.get_index_range("images")["data"]
    assert stored.dtype == np.float32
    assert np.abs(stored - np.stack(truths)).max() < 0.02


def test_flir_messages_record_to_mcap_with_their_calibration(tmp_path):
    pytest.importorskip("mcap")
    from arraydataengine import DataSources
    from arraydataengine.recording import McapRecorder, decode_message

    _write_sequence(tmp_path)
    messages = list(DataSources(str(tmp_path / "*.jpg"), flir="raw").get_message())
    with McapRecorder(tmp_path / "flir.mcap") as recorder:
        for message in messages:
            recorder.write(message)
    from mcap.reader import make_reader

    with open(tmp_path / "flir.mcap", "rb") as handle:
        decoded = [decode_message(record.data) for _, _, record in make_reader(handle).iter_messages()]
    assert [FlirCalibration.from_dict(m["flir"]) for m in decoded] == [FlirCalibration.from_dict(m["flir"]) for m in messages]
    np.testing.assert_array_equal(decoded[1]["data"], messages[1]["data"])


def test_make_flir_rjpeg_cli_writes_files_with_ground_truth(tmp_path):
    maker = _maker()
    assert maker.main(["--out", str(tmp_path), "--frames", "2", "--size", "80x60", "--raw-format", "raw"]) == 0
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    for name in truth["files"]:
        image = read_flir_radiometric(tmp_path / name)
        assert image.raw.shape == (60, 80)
        assert np.abs(image.temperature() - np.load(tmp_path / "truth" / name.replace(".jpg", ".npy"))).max() < 0.02
