from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ArrayDataEngine as ADE


BAG_FILE = "mapeverything.db3"
POINTCLOUD_TOPIC = "/mapping/pointcloud/depth_anything"
POSE_TOPIC = "/mapping/pose"


def main() -> None:
    bag_path = Path(__file__).with_name(BAG_FILE)
    source = ADE.DataSources(str(bag_path))

    topics = set(source.get_topics())
    missing = [topic for topic in (POINTCLOUD_TOPIC, POSE_TOPIC) if topic not in topics]
    if missing:
        available = "\n".join(f"  - {topic}" for topic in sorted(topics))
        raise RuntimeError(
            "The bag does not contain the required topic(s): "
            f"{', '.join(missing)}\nAvailable topics:\n{available}"
        )

    visualizer = ADE.Visualizer(
        "pointcloud",
        embed=False,
        output_path=bag_path.with_name("mapeverything_stitched_pointcloud.html"),
    )
    pairs = ADE.source_pipeline(source).nearest_topic_pairs(
        reference_topic=POINTCLOUD_TOPIC,
        target_topic=POSE_TOPIC,
    )

    stitched_count = 0
    for pointcloud_message, pose_message in pairs:
        points = ADE.valid_point_cloud_points(pointcloud_message["data"])
        if points.size == 0:
            continue

        pose_in_map = ADE.pose_to_matrix(pose_message["data"])
        points_in_map = ADE.apply_transform(points, pose_in_map)
        visualizer.add_point_cloud(points_in_map)
        visualizer.add_pose_arrow(pose_in_map)
        stitched_count += 1

    if stitched_count == 0:
        raise RuntimeError("No pointcloud/pose pairs were available to visualize.")

    visualizer.show()


if __name__ == "__main__":
    main()
