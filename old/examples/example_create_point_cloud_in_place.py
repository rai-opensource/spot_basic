#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import open3d as o3d
import rclpy
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_vision import CameraLocation, create_point_cloud_in_place

"""
This example makes spot grab point clouds from multiple cameras and
stitch them together, and display the result in open3d visualizer.
The spot possibly would move in place, to capture a more comprehensive
point cloud.

Example command:

    python example_create_point_cloud_in_place.py -C frontleft frontright back

If -C is not provided, pointcloud from all cameras will be grabbed.
"""

AVAILABLE_CAMERA_NAMES = [loc.value for loc in CameraLocation]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace.")
    parser.add_argument(
        "-C",
        "--cameras",
        nargs="+",
        help=f"Choose a camera to get the image {AVAILABLE_CAMERA_NAMES}.",
        choices=AVAILABLE_CAMERA_NAMES,
        default=AVAILABLE_CAMERA_NAMES,
    )
    options = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("create_point_cloud_in_place_example")

    spot = SpotBasic(options.robot, parent_node_name=node.get_name())
    spot.initialize_robot()

    tf = TFListenerWrapper(
        spot.robot_name + "_tf",
        cache_time_s=30,
        wait_for_transform=[spot.body_frame_name, spot.vision_frame_name],
    )
    spot.set_tf_listener(tf)

    camera_locations = [CameraLocation(loc) for loc in options.cameras]
    pcd = create_point_cloud_in_place(spot, camera_locations, timeout_sec=15, down_voxel_size=0.02)
    o3d.visualization.draw_geometries([pcd])

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
