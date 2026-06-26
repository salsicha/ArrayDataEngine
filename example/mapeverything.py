from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ArrayDataEngine as ADE


BAG_FILE = "mapeverything_0.bag"
POINTCLOUD_TOPIC = "/mapping/pointcloud/depth_anything"
POSE_TOPIC = "/mapping/pose"
CALIBRATION_TOPIC = "/mapping/depth_anything/calibration"
REFERENCE_OBJ = "final_overlay_mesh.obj"


def main() -> None:
    bag_path = Path(__file__).with_name(BAG_FILE)
    source = ADE.DataSources(str(bag_path))

    topics = set(source.get_topics())
    missing = [
        topic
        for topic in (POINTCLOUD_TOPIC, POSE_TOPIC, CALIBRATION_TOPIC)
        if topic not in topics
    ]
    if missing:
        available = "\n".join(f"  - {topic}" for topic in sorted(topics))
        raise RuntimeError(
            "The bag does not contain the required topic(s): "
            f"{', '.join(missing)}\nAvailable topics:\n{available}"
        )

    visualizer = ADE.Visualizer(
        "pointcloud",
        embed=False,
        backend="html",
        output_path=bag_path.with_name("mapeverything_stitched_pointcloud.html"),
    )

    stitched = stitch_depth_anything_pointclouds(source, visualizer=visualizer)
    compare_to_reference_mesh(stitched, bag_path.with_name(REFERENCE_OBJ))
    visualizer.show()


def stitch_depth_anything_pointclouds(source, visualizer=None) -> np.ndarray:
    latest_pose = None
    latest_calibration = None
    stitched_clouds = []

    stream = ADE.source_pipeline(source).select_topics(
        POSE_TOPIC,
        CALIBRATION_TOPIC,
        POINTCLOUD_TOPIC,
    )
    for message in stream.iter_messages():
        topic = message["topic"]
        if topic == POSE_TOPIC:
            latest_pose = ADE.pose_to_matrix(message["data"])
            continue
        if topic == CALIBRATION_TOPIC:
            latest_calibration = message
            continue
        if topic != POINTCLOUD_TOPIC:
            continue
        if latest_pose is None or latest_calibration is None:
            continue

        relative_points = ADE.valid_point_cloud_points(message["data"])
        if relative_points.size == 0:
            continue

        metric_points = ADE.calibrate_depth_anything_point_cloud(
            relative_points,
            latest_calibration,
        )
        if metric_points.size == 0:
            continue

        points_in_map = ADE.apply_transform(metric_points, latest_pose)
        stitched_clouds.append(points_in_map)
        if visualizer is not None:
            visualizer.add_point_cloud(points_in_map)
            visualizer.add_pose_arrow(latest_pose)

    if not stitched_clouds:
        raise RuntimeError("No calibrated pointcloud/pose samples were available to visualize.")
    return np.vstack(stitched_clouds)


def compare_to_reference_mesh(points: np.ndarray, obj_path: Path) -> None:
    if not obj_path.exists():
        return
    reference = load_obj_vertices(obj_path)
    if reference.size == 0:
        return
    distances = sampled_nearest_distances(points[:, :3], reference)
    p50, p75, p90, p95, p99 = np.percentile(distances, [50, 75, 90, 95, 99])
    print(
        "Reference mesh nearest-distance metrics "
        f"count={points.shape[0]} "
        f"median={p50:.3f}m p75={p75:.3f}m p90={p90:.3f}m "
        f"p95={p95:.3f}m p99={p99:.3f}m mean={distances.mean():.3f}m"
    )


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            _, x, y, z, *_ = line.split()
            vertices.append((float(x), float(y), float(z)))
    return np.asarray(vertices, dtype=np.float64)


def sampled_nearest_distances(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    max_points = 3000
    max_reference = 30000
    if points.shape[0] > max_points:
        points = points[np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)]
    if reference.shape[0] > max_reference:
        reference = reference[np.linspace(0, reference.shape[0] - 1, max_reference).astype(np.int64)]

    chunks = []
    for start in range(0, points.shape[0], 256):
        chunk = points[start:start + 256]
        squared = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(chunks)


if __name__ == "__main__":
    main()
