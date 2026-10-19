"""Live append, generator replay, and compatibility of ADE MCAP recordings."""

import builtins
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from arraydataengine import DataBuffer, DataSources, McapRecorder
from arraydataengine.recording import ENCODING, PROFILE, decode_message, is_ade_mcap

pytest.importorskip("mcap")
from mcap.reader import NonSeekingReader, make_reader


def message(index=0, **fields):
    return {"topic": "/sensor", "timestamp": float(index), "name": "Sensor",
            "data": np.array([index, index + 1], dtype=np.int16), **fields}


@pytest.mark.parametrize("compression", ["zstd", "lz4", "none"])
def test_roundtrip_ragged_metadata_and_file_order(tmp_path, compression):
    path = tmp_path / "session.MCAP"
    frames = [
        message(3, data=np.arange(12, dtype=">f4").reshape(4, 3), point_count=4,
                frame_id="lidar", position_covariance=np.eye(3), status=np.int32(1)),
        message(1, data=np.ones((2, 3), dtype=np.float64), point_count=2),
        message(2, topic="/image", data=np.zeros((5, 6, 3), dtype=np.uint8), encoding="rgb8"),
    ]
    with McapRecorder(path, compression=compression, chunk_size=128) as writer:
        assert writer.record(iter(frames)) == 3
    assert is_ade_mcap(path)
    source = DataSources(path)
    assert source.get_topics() == ["/sensor", "/image"]
    assert source.get_count("/sensor") == 2
    replayed = list(source.get_message())
    assert [row["timestamp"] for row in replayed] == [3, 1, 2]
    for expected, actual in zip(frames, replayed):
        np.testing.assert_array_equal(actual["data"], expected["data"])
        assert actual["data"].dtype == expected["data"].dtype
        assert actual["data"].flags.writeable
    assert replayed[0]["position_covariance"] == np.eye(3).tolist()
    assert replayed[0]["frame_id"] == "lidar"
    assert replayed[2]["encoding"] == "rgb8"
    # Reading again starts at the beginning; topic and point limits work.
    assert len(list(source.get_message())) == 3
    filtered = DataSources(path, topics="/sensor", max_points=2)
    assert filtered.get_count("/sensor") == 1
    assert [m["point_count"] for m in filtered.get_message()] == [2]
    assert DataSources(path, max_points=1).get_count("/sensor") == 0
    with pytest.raises(ValueError, match="not found"):
        DataSources(path, topics="/typo")


def test_exact_timestamps_and_standard_mcap_indexes(tmp_path):
    path = tmp_path / "indexed.mcap"
    ns = 1_800_000_000_123_456_789
    with McapRecorder(path) as writer:
        writer.write(message(timestamp=ns / 1e9, timestamp_ns=ns, publish_time_ns=ns - 9, sequence=42))
    with path.open("rb") as stream:
        reader = make_reader(stream)
        assert reader.get_header().profile == PROFILE
        summary = reader.get_summary()
        assert summary.statistics.message_count == 1
        assert len(summary.chunk_indexes) == 1
        _, channel, record = next(reader.iter_messages())
        assert channel.message_encoding == ENCODING
        assert (record.log_time, record.publish_time, record.sequence) == (ns, ns - 9, 42)
    actual = next(DataSources(path).get_message())
    assert actual["timestamp_ns"] == ns


def test_generator_lazy_and_usable_by_buffer(tmp_path, monkeypatch):
    from arraydataengine.sources import mcap_source

    path = tmp_path / "generator.mcap"
    with McapRecorder(path) as writer:
        assert writer.record(message(i) for i in range(10)) == 10
    source = DataSources(path)
    calls = []
    original = mcap_source.decode_message

    def decode(payload):
        calls.append(1)
        return original(payload)

    monkeypatch.setattr(mcap_source, "decode_message", decode)
    assert source.get_count("/sensor") == 10
    assert calls == []  # Counts come from indexes, not payload decoding.
    generator = source.get_message()
    assert calls == []
    assert next(generator)["timestamp"] == 0
    assert len(calls) == 1
    generator.close()
    with DataBuffer(source, axis="/sensor", buffer_depth=3, preload=0) as buffer:
        for _ in range(3):
            buffer.roll_buffer("/sensor")
        np.testing.assert_array_equal(buffer.get_buffer()["/sensor"]["data"], [[0, 1], [1, 2], [2, 3]])


def test_flush_before_close_and_continue_recording(tmp_path, monkeypatch):
    path = tmp_path / "flush.mcap"
    syncs = []
    monkeypatch.setattr("arraydataengine.recording.os.fsync", lambda fd: syncs.append(fd))
    with McapRecorder(path) as writer:
        writer.write(message())
        writer.flush(durable=True)
        assert len(syncs) == 1
        # A completed chunk is already readable even before final indexes exist.
        with path.open("rb") as stream:
            _, _, raw = next(NonSeekingReader(stream).iter_messages(log_time_order=False))
            assert decode_message(raw.data)["timestamp"] == 0
        writer.write(message(1))
    writer.close()  # Idempotent.
    assert DataSources(path).get_count("/sensor") == 2
    with pytest.raises(ValueError, match="closed"):
        writer.write(message(2))


def test_interrupted_generator_finalizes_already_written_messages(tmp_path):
    path = tmp_path / "interrupt.mcap"

    def incoming():
        yield message()
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt), McapRecorder(path) as writer:
        writer.record(incoming())
    assert len(list(DataSources(path).get_message())) == 1


@pytest.mark.parametrize("fields", [
    {"data": np.array([object()], dtype=object)}, {"timestamp": float("nan")},
    {"timestamp": -1}, {"timestamp": float("inf")}, {"timestamp": 1e100},
    {"timestamp_ns": -1}, {"timestamp_ns": 2**64}, {"sequence": 2**32},
    {"sequence": 1.5}, {"publish_time_ns": True}, {"topic": ""},
    {"custom": object()},
])
def test_rejected_messages_do_not_poison_recording(tmp_path, fields):
    path = tmp_path / "validate.mcap"
    with McapRecorder(path) as writer:
        with pytest.raises((ValueError, TypeError)):
            writer.write(message(**fields))
        writer.write(message(1))
    assert DataSources(path).get_count("/sensor") == 1


def test_overwrite_empty_file_and_callbacks(tmp_path):
    path = tmp_path / "callbacks.mcap"
    with McapRecorder(path):
        pass
    assert DataSources(path).get_topics() == []
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        McapRecorder(path)
    assert path.read_bytes() == before
    with McapRecorder(path, overwrite=True) as writer:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(writer.write, (message(i) for i in range(50))))
    assert sorted(m["timestamp"] for m in DataSources(path).get_message()) == list(range(50))


def test_missing_extra_and_header_probe(tmp_path, monkeypatch):
    path = tmp_path / "optional.mcap"
    with McapRecorder(path) as writer:
        writer.write(message())
    original = builtins.__import__

    def no_mcap(name, *args, **kwargs):
        if name == "mcap" or name.startswith("mcap."):
            raise ImportError("blocked for test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mcap)
    assert is_ade_mcap(path)
    with pytest.raises(ImportError, match=r"arraydataengine\[mcap\]"):
        DataSources(path)
    with pytest.raises(ImportError, match=r"arraydataengine\[mcap\]"):
        McapRecorder(tmp_path / "missing.mcap")
    assert not (tmp_path / "missing.mcap").exists()


@pytest.mark.parametrize("payload", [b"", struct.pack("<I", 100), struct.pack("<I", 2) + b"[]bad", struct.pack("<I", 2) + b"{}bad"])
def test_malformed_payload_rejected(payload):
    with pytest.raises(ValueError):
        decode_message(payload)


def test_mixed_ros_cdr_and_numpy_without_conversion(tmp_path):
    pytest.importorskip("rosbags")
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.LATEST)
    types = store.types
    type_name = "geometry_msgs/msg/PoseStamped"
    pose = types[type_name](
        header=types["std_msgs/msg/Header"](
            stamp=types["builtin_interfaces/msg/Time"](sec=12, nanosec=500), frame_id="map"),
        pose=types["geometry_msgs/msg/Pose"](
            position=types["geometry_msgs/msg/Point"](x=1.0, y=2.0, z=3.0),
            orientation=types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)),
    )
    payload = bytes(store.serialize_cdr(pose, type_name))
    definition, _ = store.generate_msgdef(type_name, ros_version=2)
    path = tmp_path / "mixed.mcap"
    with McapRecorder(path) as writer:
        writer.write(message(0, data=np.ones((2, 3), dtype=np.uint8), encoding="mono8"))
        writer.write_encoded(
            "/pose", payload, timestamp_ns=20_000_000_001, message_encoding="cdr",
            schema_name=type_name, schema_encoding="ros2msg", schema_data=definition.encode(),
        )
        writer.write(message(2, data=np.ones((7, 3), dtype=np.float32), point_count=7))
    source = DataSources(path)
    assert source.get_topics() == ["/sensor", "/pose"]
    assert source.get_count("/pose") == 1
    assert source.get_topic_types()["/pose"] == type_name
    rows = list(source.get_message())
    assert [row["topic"] for row in rows] == ["/sensor", "/pose", "/sensor"]
    assert rows[1]["frame_id"] == "map"
    assert rows[1]["timestamp"] == pytest.approx(12.0000005)
    assert rows[1]["timestamp_ns"] == 20_000_000_001
    assert rows[1]["name"] == "PoseStamped"
    assert rows[0]["data"].shape == (2, 3)
    assert rows[2]["data"].shape == (7, 3)
    with path.open("rb") as stream:
        _, schema_channel, raw = list(make_reader(stream).iter_messages(topics="/pose"))[0]
        assert schema_channel.message_encoding == "cdr"
        assert raw.data == payload  # Original wire representation is retained.

    # Explicit generic reader also handles an external ROS-profile MCAP.
    from mcap.writer import Writer

    external = tmp_path / "ros.mcap"
    with external.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="ros2")
        schema = writer.register_schema(type_name, "ros2msg", definition.encode())
        channel = writer.register_channel("/pose", "cdr", schema)
        writer.add_message(channel, 1, payload, 1)
        writer.finish()
    standard = next(DataSources(external).get_message())
    generic = next(DataSources(external, mcap_decoders={}).get_message())
    np.testing.assert_array_equal(standard["data"], generic["data"])

    pytest.importorskip("pyarrow")
    from arraydataengine import mcap_to_parquet, parquet_to_mcap
    cache = mcap_to_parquet(path, tmp_path / "analysis", trajectory_topics={"/pose": "pose"}, fidelity="copy")
    pose_row = next(cache.rows("/pose", columns=["x", "y", "z", "log_time_ns", "ts"]))
    assert [pose_row[k] for k in ("x", "y", "z")] == [1., 2., 3.]
    assert pose_row["log_time_ns"] == 20_000_000_001
    assert pose_row["ts"] == pytest.approx(12.0000005)
    assert parquet_to_mcap(cache, tmp_path / "restored.mcap").read_bytes() == path.read_bytes()


def test_mixed_encoding_hooks_receive_schema_and_only_decode_selected_topics(tmp_path):
    import json

    path = tmp_path / "heterogeneous.mcap"
    with McapRecorder(path) as writer:
        writer.write(message())
        for name, values in [("Vector", [1, 2]), ("Matrix", [[3, 4], [5, 6]])]:
            writer.write_encoded(
                "/json", json.dumps({"values": values}).encode(), timestamp_ns=42,
                message_encoding="json", schema_name=name, schema_encoding="jsonschema",
                schema_data=b'{"type":"object"}',
            )
    calls = []

    def json_decoder(schema, channel, record):
        calls.append((schema.name, channel.topic, record.log_time))
        return {"data": json.loads(record.data)["values"], "frame_id": "world"}

    def vector_decoder(schema, channel, record):
        row = json_decoder(schema, channel, record)
        row["data"] = np.asarray(row["data"], dtype=np.int16)
        return row

    source = DataSources(path, mcap_decoders={"json": json_decoder, ("json", "Vector"): vector_decoder})
    assert source.get_count("/json") == 2  # Two schemas/channels for the same topic.
    assert calls == []
    rows = list(source.get_message())
    assert calls == [("Vector", "/json", 42), ("Matrix", "/json", 42)]
    assert rows[1]["data"].dtype == np.int16
    assert rows[2]["data"].shape == (2, 2)
    assert rows[2]["frame_id"] == "world"
    assert len(list(DataSources(path, topics="/sensor").get_message())) == 1
    with pytest.raises(ValueError, match="No MCAP decoder"):
        list(DataSources(path).get_message())


def test_encoded_validation_and_mutable_input_ownership(tmp_path):
    path = tmp_path / "owned.mcap"
    with McapRecorder(path) as writer:
        array = np.ones(4)
        writer.write(message(data=array))
        array[:] = 10
        with pytest.raises(ValueError, match="schema_name"):
            writer.write_encoded("/cdr", b"abc", message_encoding="cdr", timestamp_ns=1,
                                 schema_encoding="ros2msg", schema_data=b"float32 value")
    np.testing.assert_array_equal(next(DataSources(path).get_message())["data"], np.ones(4))


def test_source_pipeline_can_persist_mixed_topic_arrays(tmp_path):
    pytest.importorskip("pyarrow")
    from arraydataengine.ops import source_pipeline

    path = tmp_path / "input.mcap"
    with McapRecorder(path) as writer:
        for index in range(3):
            writer.write(message(index))
            writer.write(message(index, topic="/images", data=np.full((2, 3), index, dtype=np.uint8)))
    pipeline = source_pipeline(DataSources(path))
    with pipeline.to_buffer(data_uri=str(tmp_path / "parquet"), use_db=True, axis="/sensor") as buffer:
        assert buffer.msg_len["/sensor"] == 3
        assert buffer.msg_len["/images"] == 3
