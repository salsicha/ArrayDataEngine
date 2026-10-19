"""Schema-partitioned analytical Parquet caches backed by original MCAP files.

Payloads remain pickle-free NPY envelopes, while record identity, exact MCAP
clock values and optional trajectory coordinates are queryable columns. A
single staging budget covers all topic/schema partitions. The budget limits
staged bytes, not Python/Arrow/compressor RSS or the size of decoder temporaries.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .recording import _encode_message, decode_message
from .buffers.mcap_buffer import read_manifest

FORMAT = "ade-parquet-recording-v1"


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError('Parquet analysis requires: pip install "arraydataengine[arrow]"') from exc
    return pa, ds, pq


def _digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def _inputs(recording):
    path = Path(recording).resolve()
    if path.is_file():
        return [path], None
    manifest = read_manifest(path)
    if manifest is None:
        raise ValueError("recording must be an MCAP file or ADE MCAP store")
    return [path / name for name in manifest["segments"]], manifest


def _fingerprints(paths):
    return [{"path": str(path), "sha256": _digest(path), "size": path.stat().st_size} for path in paths]


def _schema():
    pa, _, _ = _arrow()
    fields = [("record_index", pa.int64()), ("topic_index", pa.int64()), ("topic", pa.string()),
              ("schema_key", pa.string()), ("ts", pa.float64()), ("log_time_ns", pa.uint64()),
              ("publish_time_ns", pa.uint64()), ("sequence", pa.uint32()),
              ("name", pa.string()), ("frame_id", pa.string()), ("payload", pa.large_binary())]
    fields += [(key, pa.float64()) for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    return pa.schema(fields)


def mcap_to_parquet(recording, destination, *, fidelity="reference", topics=None,
                    trajectory_topics=None, memory_budget_bytes=64*1024**2,
                    row_group_bytes=8*1024**2, mcap_decoders=None):
    """Build/reuse a cache; no existing non-cache directory is overwritten.

    ``fidelity='reference'`` retains SHA-256 checked references to the original
    MCAPs. ``'copy'`` copies them into the cache so byte-for-byte restoration is
    independent of the source. Both retain attachments, schemas, order and raw
    CDR through the original, including channels not selected for analysis.
    ``trajectory_topics`` maps topic names to 'pose' or 'odometry' converters.
    Shape/dtype/schema changes create separate Parquet partitions automatically.
    """
    pa, _, pq = _arrow()
    if fidelity not in ("reference", "copy"):
        raise ValueError("fidelity must be 'reference' or 'copy'")
    for value in (memory_budget_bytes, row_group_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("memory and row-group budgets must be positive integers")
    selected = None if topics is None else sorted(set([topics] if isinstance(topics, str) else topics))
    trajectory_topics = dict(trajectory_topics or {})
    paths, source_manifest = _inputs(recording)
    fingerprints = _fingerprints(paths)
    config = {"topics": selected, "trajectory_topics": trajectory_topics, "fidelity": fidelity}
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        cached = ParquetRecording(destination)
        if (cached.manifest["sources"] != fingerprints or cached.manifest["config"] != config or
                cached.manifest["source_manifest"] != source_manifest):
            raise ValueError("Parquet cache is stale or uses different options; choose a new destination")
        if mcap_decoders:
            raise ValueError("Custom decoder caches require a new destination (decoder identity is not persisted)")
        return cached
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".parquet-build-", dir=destination.parent))
    stages, sizes, partitions, counts = {}, {}, {}, {}
    peak_staged = 0
    schema = _schema()

    def flush(key):
        rows = stages.pop(key)
        size = sizes.pop(key)
        partition = partitions[key]
        name = f"schema-{key}/part-{len(partition['files']):06d}.parquet"
        file = temporary / name
        file.parent.mkdir(exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=schema)
        rows_per_group = max(1, int(min(row_group_bytes, memory_budget_bytes) / max(1, size / len(rows))))
        pq.write_table(table, file, row_group_size=rows_per_group, compression="zstd")
        partition["files"].append(name)
        partition["count"] += len(rows)

    try:
        from .sources.mcap_source import McapSource
        index = 0
        for path in paths:
            source = McapSource(path, topics=selected, decoders=mcap_decoders)
            for msg in source.messages(with_record_index=True):
                # Keep an original-record ordinal even when analysis selects only some topics.
                record_index = index + msg.pop("_mcap_record_index")
                topic = msg["topic"]
                if selected is not None and topic not in selected:
                    continue
                for field in ("name", "frame_id"):
                    if isinstance(msg.get(field), bytes):
                        msg[field] = msg[field].decode("utf-8")
                array = np.asarray(msg["data"])
                identity = {"topic": topic, "dtype": array.dtype.descr, "shape": array.shape,
                            "schema": msg.get("schema_sha256"), "name": msg.get("name")}
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                topic_index = counts.get(topic, 0)
                _, log_ns, publish_ns, sequence, payload = _encode_message(msg)
                size = len(payload) + 1024
                if size > memory_budget_bytes:
                    raise ValueError(f"message on {topic} exceeds memory_budget_bytes")
                while sizes and sum(sizes.values()) + size > memory_budget_bytes:
                    flush(max(sizes, key=sizes.get))
                row = {"record_index": record_index, "topic_index": topic_index, "topic": topic,
                       "schema_key": key, "ts": float(msg["timestamp"]), "log_time_ns": log_ns,
                       "publish_time_ns": publish_ns, "sequence": sequence, "payload": payload,
                       "name": str(msg.get("name", topic)), "frame_id": msg.get("frame_id")}
                if topic in trajectory_topics:
                    from .ops.nav import sensor_to_trajectory
                    trajectory = sensor_to_trajectory({"data": array[None], "ts": np.array([row["ts"]])},
                                                      kind=trajectory_topics[topic])
                    pose = np.r_[trajectory["position"][0], trajectory["orientation"][0]]
                    if not np.isfinite(pose).all():
                        raise ValueError(f"non-finite trajectory pose on {topic}")
                    row.update(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose.tolist()))
                partitions.setdefault(key, {"topic": topic, "identity": identity, "files": [], "count": 0})
                stages.setdefault(key, []).append(row)
                sizes[key] = sizes.get(key, 0) + size
                peak_staged = max(peak_staged, sum(sizes.values()))
                counts[topic] = topic_index + 1
                if sizes[key] >= min(row_group_bytes, memory_budget_bytes):
                    flush(key)
            index += source._records_read
        for key in list(stages):
            flush(key)
        # Fail rather than publish a cache built across a changing source.
        if _fingerprints(paths) != fingerprints or _inputs(recording)[1] != source_manifest:
            raise ValueError("MCAP recording changed during conversion; close the writer and retry")
        if fidelity == "copy":
            (temporary / "original").mkdir()
            for i, path in enumerate(paths):
                copied = temporary / "original" / f"{i:06d}.mcap"
                shutil.copyfile(path, copied)
                if _digest(copied) != fingerprints[i]["sha256"]:
                    raise ValueError("MCAP source changed during archival copy")
        manifest = {"format": FORMAT, "config": config, "sources": fingerprints,
                    "source_manifest": source_manifest, "counts": counts,
                    "partitions": partitions, "peak_staged_bytes": peak_staged,
                    "memory_budget_bytes": memory_budget_bytes}
        (temporary / "recording.json").write_text(json.dumps(manifest, indent=2))
        os.replace(temporary, destination)
        return ParquetRecording(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


class ParquetRecording:
    """An analytical cache with schema-separated files and lazy payload access."""
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.manifest = json.loads((self.path / "recording.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Not an ADE Parquet recording: {path}") from exc
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Unsupported Parquet recording format")

    def dataset(self, topic):
        _, ds, _ = _arrow()
        files = [str(self.path / file) for partition in self.manifest["partitions"].values()
                 if partition["topic"] == topic for file in partition["files"]]
        if not files:
            raise ValueError(f"No cached topic {topic!r}")
        return ds.dataset(files, format="parquet", schema=_schema())

    def rows(self, topic, *, columns=None, filter=None, batch_size=64):
        for batch in self.dataset(topic).scanner(columns=columns, filter=filter, batch_size=batch_size,
                                                batch_readahead=1, fragment_readahead=1,
                                                use_threads=False).to_batches():
            yield from batch.to_pylist()

    def message(self, topic, index):
        _, ds, _ = _arrow()
        rows = self.rows(topic, columns=["payload"], filter=ds.field("topic_index") == int(index), batch_size=1)
        row = next(rows, None)
        if row is None:
            raise IndexError(index)
        return decode_message(row["payload"])

    def nearest(self, topic, timestamp, tolerance):
        _, ds, _ = _arrow()
        predicate = (ds.field("ts") >= timestamp-tolerance) & (ds.field("ts") <= timestamp+tolerance)
        rows = self.rows(topic, columns=["ts", "topic_index"], filter=predicate)
        best = min(rows, key=lambda row: (abs(row["ts"]-timestamp), row["topic_index"]), default=None)
        if best is None:
            raise ValueError(f"No {topic} message within {tolerance}s of {timestamp}")
        return best["topic_index"]


def parquet_to_mcap(cache, destination):
    """Restore the original MCAP byte-for-byte, or an original MCAP store.

    This restores the original recording, not modifications to derived columns.
    Reference caches require unchanged source files. Copy caches are standalone.
    """
    cache = cache if isinstance(cache, ParquetRecording) else ParquetRecording(cache)
    manifest = cache.manifest
    sources = []
    for i, entry in enumerate(manifest["sources"]):
        path = cache.path / "original" / f"{i:06d}.mcap" if manifest["config"]["fidelity"] == "copy" else Path(entry["path"])
        if not path.is_file() or _digest(path) != entry["sha256"]:
            raise ValueError("Original MCAP is missing or changed; cannot restore faithfully")
        sources.append(path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".mcap-restore-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "recording"
        if manifest["source_manifest"] is None:
            shutil.copyfile(sources[0], staged)
            if _digest(staged) != manifest["sources"][0]["sha256"]:
                raise ValueError("Original MCAP changed during restoration")
            # A hard link publishes a complete file and refuses an existing destination.
            os.link(staged, destination)
        else:
            staged.mkdir()
            from .buffers.mcap_buffer import MANIFEST
            for path, name, entry in zip(sources, manifest["source_manifest"]["segments"], manifest["sources"]):
                shutil.copyfile(path, staged / name)
                if _digest(staged / name) != entry["sha256"]:
                    raise ValueError("Original MCAP changed during restoration")
            (staged / MANIFEST).write_text(json.dumps(manifest["source_manifest"]))
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(staged, destination)
    return destination
