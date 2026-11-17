"""16-bit thermal imagery: sources, storage, AGC, radiometry, and display."""

import struct

import numpy as np
import pytest

from arraydataengine.ops import (
    colorize_thermal,
    convert_image_dtype,
    radiometric_to_temperature,
    temperature_to_radiometric,
    thermal_agc,
)


def _scene(seed=0, shape=(64, 80)):
    """A thermal frame: ~22000 counts with +/-40 counts of structure, one hot and one dead pixel."""
    rng = np.random.default_rng(seed)
    scene = 22000 + 40 * np.sin(np.linspace(0, 6, shape[1]))[None, :] + rng.normal(0, 3, shape)
    scene = scene.astype(np.uint16)
    if min(shape) > 6:
        scene[5, 5], scene[6, 6] = 65000, 200
    return scene


# --- sources ---------------------------------------------------------------------------

def test_image_files_keep_16_bits_and_multi_page_tiffs_yield_every_page(tmp_path):
    pytest.importorskip("cv2")
    from PIL import Image
    from arraydataengine.source import DataSources

    pages = [_scene(seed) for seed in range(3)]
    Image.fromarray(pages[0]).save(tmp_path / "a_stack.tiff", save_all=True,
                                   append_images=[Image.fromarray(p) for p in pages[1:]])
    single = _scene(9)
    Image.fromarray(single).save(tmp_path / "b_single.png")
    Image.fromarray(single).save(tmp_path / "c_single.tiff")

    tiffs = DataSources(str(tmp_path / "*.tiff"), period=0.5)
    messages = list(tiffs.get_message())
    assert tiffs.get_count("images") == 4
    assert [m["name"] for m in messages] == ["a_stack.tiff#0", "a_stack.tiff#1", "a_stack.tiff#2", "c_single.tiff"]
    assert [m["timestamp"] for m in messages] == [0.0, 0.5, 1.0, 1.5]
    for message, expected in zip(messages, [*pages, single]):
        assert message["data"].dtype == np.uint16 and message["encoding"] == "mono16"
        np.testing.assert_array_equal(message["data"], expected)
    png = next(DataSources(str(tmp_path / "b_single.png")).get_message())
    assert png["data"].dtype == np.uint16 and np.array_equal(png["data"], single)


def test_multi_page_tiff_pages_read_without_the_opencv_ranged_reader(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from PIL import Image
    import arraydataengine.sources.img_source as img_source
    from arraydataengine.source import DataSources

    pages = [_scene(seed) for seed in range(2)]
    Image.fromarray(pages[0]).save(tmp_path / "s.tiff", save_all=True, append_images=[Image.fromarray(pages[1])])
    colour = [np.dstack([np.full((4, 5), v, np.uint8), np.zeros((4, 5), np.uint8), np.full((4, 5), 255 - v, np.uint8)])
              for v in (10, 20)]
    Image.fromarray(colour[0]).save(tmp_path / "c.tiff", save_all=True, append_images=[Image.fromarray(colour[1])])

    def no_ranged_read(*args, **kwargs):
        raise TypeError("ranged imreadmulti unavailable")

    monkeypatch.setattr(img_source.cv2, "imreadmulti", no_ranged_read)
    thermal = [m["data"] for m in DataSources(str(tmp_path / "s.tiff")).get_message()]
    assert [f.dtype for f in thermal] == [np.uint16, np.uint16]
    np.testing.assert_array_equal(np.stack(thermal), np.stack(pages))
    rgb = [m["data"] for m in DataSources(str(tmp_path / "c.tiff")).get_message()]  # RGB like every source
    np.testing.assert_array_equal(np.stack(rgb), np.stack(colour))
    assert cv2 is not None


def test_ros_16_bit_images_decode_on_every_encoding(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    from arraydataengine.source import DataSources

    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    header = types["std_msgs/msg/Header"](stamp=types["builtin_interfaces/msg/Time"](sec=1, nanosec=0),
                                          frame_id="thermal")
    frame = _scene(1, shape=(6, 8))
    signed = (frame.astype(np.int32) - 22000).astype(np.int16)
    image_type, compressed_type = types["sensor_msgs/msg/Image"], types["sensor_msgs/msg/CompressedImage"]
    ok, png = cv2.imencode(".png", frame)
    messages = []
    for topic, encoding, array, big_endian in (("/mono16", "mono16", frame, False), ("/mono16_be", "mono16", frame, True),
                                               ("/u16", "16UC1", frame, False), ("/s16", "16SC1", signed, False)):
        raw = array.astype(array.dtype.newbyteorder(">" if big_endian else "<")).tobytes()
        messages.append((topic, "sensor_msgs/msg/Image", image_type(
            header=header, height=6, width=8, encoding=encoding, is_bigendian=int(big_endian), step=16,
            data=np.frombuffer(raw, np.uint8))))
    messages.append(("/png16", "sensor_msgs/msg/CompressedImage",
                     compressed_type(header=header, format="mono16; png compressed", data=np.frombuffer(png, np.uint8))))
    depth_config = struct.pack("<iff", 0, 0.0, 0.0)  # compressed_depth_image_transport header
    messages.append(("/depth16", "sensor_msgs/msg/CompressedImage", compressed_type(
        header=header, format="16UC1; compressedDepth png", data=np.frombuffer(depth_config + png.tobytes(), np.uint8))))
    with Writer(tmp_path / "bag", version=9) as writer:
        for topic, msgtype, message in messages:
            connection = writer.add_connection(topic, msgtype, typestore=store)
            writer.write(connection, 1_000_000_000, store.serialize_cdr(message, msgtype))

    decoded = {m["topic"]: m for m in DataSources(str(tmp_path / "bag")).get_message()}
    assert set(decoded) == {topic for topic, _, _ in messages}
    for topic, message in decoded.items():
        expected = signed if topic == "/s16" else frame
        assert message["data"].dtype == expected.dtype, topic
        np.testing.assert_array_equal(message["data"], expected, err_msg=topic)


@pytest.mark.parametrize("backend", ["memory", "arrow", "tiledb"])
def test_16_bit_frames_round_trip_through_every_buffer(tmp_path, backend):
    from arraydataengine import DataBuffer

    frames = [_scene(seed) for seed in range(3)]

    class Source:
        def get_topics(self):
            return ["/thermal"]

        def get_count(self, topic):
            return len(frames)

        def get_message(self):
            for index, frame in enumerate(frames):
                yield {"topic": "/thermal", "timestamp": float(index), "name": f"f{index}", "data": frame}

    if backend == "memory":
        buffer = DataBuffer(Source(), buffer_depth=3, axis="/thermal", preload=True)
    else:
        pytest.importorskip("pyarrow" if backend == "arrow" else "tiledb")
        with DataBuffer(Source(), data_uri=str(tmp_path / backend), backend=backend, axis="/thermal", preload=0) as writer:
            writer.load_data_db("/thermal")
        buffer = DataBuffer(None, data_uri=str(tmp_path / backend), backend=backend, axis="/thermal")
    stored = buffer.get_index_range("/thermal")["data"]
    assert stored.dtype == np.uint16
    np.testing.assert_array_equal(stored, np.stack(frames))


# --- AGC -------------------------------------------------------------------------------

def test_agc_keeps_thermal_detail_that_full_range_conversion_crushes():
    scene = _scene()
    interior = (slice(10, 60), slice(10, 70))
    assert np.unique(convert_image_dtype(scene, np.uint8)[interior]).size <= 3
    for method in ("linear", "plateau"):
        display = thermal_agc(scene, method=method)
        assert display.dtype == np.uint8 and display.shape == scene.shape
        assert np.unique(display[interior]).size > 60, method
        assert display[5, 5] == 255 and display[6, 6] == 0  # outliers saturate instead of setting the range


def test_linear_agc_maps_percentiles_and_handles_nan_and_flat_frames():
    ramp = np.linspace(0.0, 1000.0, 1001).reshape(1, -1)
    unit = thermal_agc(ramp, low=10, high=90, dtype=np.float32)
    assert unit[0, 100] == pytest.approx(0.0) and unit[0, 900] == pytest.approx(1.0)
    assert unit[0, 500] == pytest.approx(0.5)
    with_nan = ramp.copy()
    with_nan[0, 0] = np.nan
    assert thermal_agc(with_nan)[0, 0] == 0
    assert np.all(thermal_agc(np.full((4, 4), 22000, np.uint16), dtype=np.float32) == 0.5)
    assert thermal_agc(ramp, dtype=np.uint16).max() == 65535


def test_plateau_agc_is_monotonic_and_limits_large_uniform_areas():
    rng = np.random.default_rng(3)
    frame = 22000 + rng.normal(0, 40, (60, 80))  # detail
    frame[:30] = 22000.0  # half the frame is one uniform temperature inside the detail's range

    def detail_contrast(display):  # output change across 50 counts of detail, away from the uniform value
        counts, values = frame.ravel(), display.ravel()
        near = lambda target: values[np.argmin(np.abs(counts - target))]  # noqa: E731
        return near(21970) - near(21920)

    capped = thermal_agc(frame, method="plateau", dtype=np.float64)
    plain = thermal_agc(frame, method="plateau", plateau=1.0, dtype=np.float64)  # no cap = plain equalization
    order = np.argsort(frame.ravel(), kind="stable")
    assert np.all(np.diff(capped.ravel()[order]) >= -1e-12)  # monotonic in the counts
    # Plain equalization spends half the output range on the uniform value; the cap returns it to the detail.
    assert detail_contrast(capped) > 1.6 * detail_contrast(plain)


def test_agc_shares_one_mapping_across_a_sequence_unless_per_frame():
    dim, bright = _scene(1), _scene(2) + 500
    shared = thermal_agc(np.stack([dim, bright]))
    assert shared[1].mean() > shared[0].mean() + 50  # brightness change stays visible
    separate = thermal_agc(np.stack([dim, bright]), per_frame=True)
    assert abs(float(separate[1].mean()) - float(separate[0].mean())) < 10  # each frame stretched alone
    np.testing.assert_array_equal(thermal_agc(dim), thermal_agc(np.stack([dim]))[0])


def test_agc_rejects_bad_arguments():
    with pytest.raises(ValueError, match="method"):
        thermal_agc(_scene(), method="histogram")
    with pytest.raises(ValueError, match="percentiles"):
        thermal_agc(_scene(), low=60, high=40)
    with pytest.raises(ValueError, match="shape"):
        thermal_agc(np.zeros(5))
    with pytest.raises(ValueError, match="dtype"):
        thermal_agc(_scene(), dtype=np.int16)


# --- radiometry and colour ---------------------------------------------------------------

def test_radiometric_tlinear_counts_convert_to_temperature_and_back():
    counts = np.array([29815, 37315, 27315], np.uint16)  # centikelvin (FLIR high-resolution TLinear)
    np.testing.assert_allclose(radiometric_to_temperature(counts), [25.0, 100.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(radiometric_to_temperature(counts, unit="kelvin"), [298.15, 373.15, 273.15])
    np.testing.assert_allclose(radiometric_to_temperature(counts, unit="fahrenheit"), [77.0, 212.0, 32.0])
    np.testing.assert_allclose(radiometric_to_temperature(np.array([2982]), scale=0.1), [25.05])  # low resolution
    back = temperature_to_radiometric(radiometric_to_temperature(counts))
    assert back.dtype == np.uint16 and back.tolist() == counts.tolist()
    assert temperature_to_radiometric([-300.0, 1000.0]).tolist() == [0, 65535]  # clipped to uint16
    with pytest.raises(ValueError, match="unit"):
        radiometric_to_temperature(counts, unit="rankine")


def test_colorize_thermal_returns_rgb_frames():
    frames = np.stack([_scene(1), _scene(2)])
    grey = colorize_thermal(frames, cmap="gray")
    assert grey.shape == (*frames.shape, 3) and grey.dtype == np.uint8
    assert np.array_equal(grey[..., 0], grey[..., 2])
    pytest.importorskip("matplotlib")
    coloured = colorize_thermal(frames[0], cmap="inferno", method="plateau")
    assert coloured.shape == (*frames.shape[1:], 3) and not np.array_equal(coloured[..., 0], coloured[..., 2])


# --- display ------------------------------------------------------------------------------

def test_vistool_scales_16_bit_frames_by_percentiles():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("cv2")
    import matplotlib.pyplot as plt
    from arraydataengine.visualizers.video_segment import VisTool

    scene = _scene()
    tool = VisTool(embed=False, cmap="gray")
    tool.append_img(scene)
    tool.append_img((scene.astype(np.int32) + 300).clip(0, 65535).astype(np.uint16))
    first, last = tool.ax.images[-2], tool.ax.images[-1]
    lo, hi = first.get_clim()
    assert 21900 < lo < hi < 22100  # not the 200..65000 the outliers span
    assert last.get_clim() == first.get_clim()  # one scaling for the whole animation
    assert last.get_cmap().name == "gray"
    rgb = VisTool(embed=False)
    rgb.append_img(np.zeros((4, 4, 3), np.uint8))
    assert "cmap" not in rgb._imshow_options(np.zeros((4, 4, 3), np.uint8))
    plt.close("all")
