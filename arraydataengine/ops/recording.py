"""Loop closure on recordings using a derived Parquet cache and scan pairs."""
from __future__ import annotations

import heapq
import math
from pathlib import Path
import tempfile

import numpy as np


def iter_recording_loop_closures(recording, *, trajectory_topic, points_topic,
                                 radius, min_separation=30, max_candidates_per_pose=1,
                                 trajectory_kind="pose", time_tolerance=0.05,
                                 cache_dir=None, memory_budget_bytes=64*1024**2,
                                 fidelity="reference", **verification_options):
    """Automatically build a Parquet cache and yield verified loop closures.

    Trajectory coordinates are scalar Parquet columns. Each spatial search
    scans only trajectory metadata; at most max_candidates_per_pose rows and
    two point clouds are retained for verification. Candidate queries can be
    expensive on poorly clustered trajectories, but no all-trajectory index or
    all-cloud list is materialized. Pose indices refer to trajectory topic order;
    point clouds are matched by nearest timestamp within time_tolerance.

    With no cache_dir, derived files are removed on exhaustion/close of this
    generator. An explicit cache_dir can be reused if source hashes/options
    match. memory_budget_bytes controls conversion staging, not ICP workspace.
    """
    from ..parquet_recording import mcap_to_parquet, _arrow
    from .point_cloud import verify_loop_closures

    if not math.isfinite(radius) or radius < 0 or not math.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("radius and time_tolerance must be finite and non-negative")
    if isinstance(min_separation, bool) or int(min_separation) != min_separation or min_separation < 1:
        raise ValueError("min_separation must be a positive integer")
    if isinstance(max_candidates_per_pose, bool) or int(max_candidates_per_pose) != max_candidates_per_pose or max_candidates_per_pose < 1:
        raise ValueError("max_candidates_per_pose must be a positive integer")
    if trajectory_topic == points_topic:
        raise ValueError("trajectory and points must be separate topics")
    _, ds, _ = _arrow()
    temporary = tempfile.TemporaryDirectory(prefix="ade-loop-closure-") if cache_dir is None else None
    try:
        destination = Path(temporary.name) / "cache" if temporary else Path(cache_dir)
        cache = mcap_to_parquet(recording, destination, fidelity=fidelity,
                                topics=[trajectory_topic, points_topic],
                                trajectory_topics={trajectory_topic: trajectory_kind},
                                memory_budget_bytes=memory_budget_bytes)
        columns = ["topic_index", "ts", "x", "y", "z", "qx", "qy", "qz", "qw"]
        xyz = ("x", "y", "z")
        pose_keys = (*xyz, "qx", "qy", "qz", "qw")
        for source in cache.rows(trajectory_topic, columns=columns):
            source_index = source["topic_index"]
            if source_index < min_separation:
                continue
            predicate = ds.field("topic_index") <= source_index - min_separation
            for axis in xyz:
                predicate = predicate & (ds.field(axis) >= source[axis]-radius) & (ds.field(axis) <= source[axis]+radius)

            def candidates():
                for target in cache.rows(trajectory_topic, columns=columns, filter=predicate):
                    distance = math.sqrt(sum((source[k]-target[k])**2 for k in xyz))
                    if distance <= radius:
                        yield distance, target["topic_index"], target

            nearest = heapq.nsmallest(int(max_candidates_per_pose), candidates(), key=lambda item: (item[0], item[1]))
            if not nearest:
                continue
            source_cloud_index = cache.nearest(points_topic, source["ts"], time_tolerance)
            source_cloud = cache.message(points_topic, source_cloud_index)["data"]
            for distance, target_index, target in nearest:
                target_cloud_index = cache.nearest(points_topic, target["ts"], time_tolerance)
                target_cloud = cache.message(points_topic, target_cloud_index)["data"]
                poses = np.array([[row[key] for key in pose_keys] for row in (target, source)])
                result = verify_loop_closures(
                    [target_cloud, source_cloud], poses,
                    candidates=[{"source_index": 1, "target_index": 0, "pose_distance": distance}],
                    **verification_options,
                )
                if len(result["source_index"]):
                    row = {key: value[0] for key, value in result.items()}
                    row.update(source_index=source_index, target_index=target_index,
                               source_cloud_index=source_cloud_index, target_cloud_index=target_cloud_index)
                    yield row
    finally:
        if temporary is not None:
            temporary.cleanup()


def verify_recording_loop_closures(recording, **options):
    """Collect loop-closure results; use the iterator when results exceed RAM."""
    from .point_cloud import _loop_closure_records_to_arrays
    rows = list(iter_recording_loop_closures(recording, **options))
    result = _loop_closure_records_to_arrays(rows, include_verification=True)
    for key in ("source_cloud_index", "target_cloud_index"):
        result[key] = np.array([row[key] for row in rows], dtype=np.int64)
    return result
