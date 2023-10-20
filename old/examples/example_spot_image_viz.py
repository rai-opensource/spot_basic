# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import time

import bdai_ros2_wrappers.process as ros_process
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_vision import CameraLocation, SpotImages

""" This is a test script to demo how spot_ros2 publishes image and camera data on spot.
It visualizes a point cloud, rgb, and depth images from a camera.

Use:
    1) start spot driver with a config file that makes spot_ros2 publish the depth registered in the rgb camera.
        for example, 'bdai spot driver Autumn --config-file projects/_experimental/spot_ros.yaml --no-build'
        where the yaml has the line 'publish_depth_registered: True'

    2) execute this script, pass in the robot name and camera location (see --help for valid locations)
    3) three GUI's will pop up in succession: a pointcloud, am rgb image, and a grayscale image representing depth
        (might be hard to make out any detail in this one)
"""


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace.")
    parser.add_argument(
        "-c",
        "--camera-location",
        help=f"Choose a camera to get the image:\n{[e.value for e in CameraLocation]}.",
        default="frontleft",
    )
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    ###############################################
    # Boiler plate code
    spot = SpotBasic(args.robot)
    tf = TFListenerWrapper(cache_time_s=30)
    tf.wait_for_a_tform_b(spot.body_frame_name, spot.vision_frame_name)
    spot.set_tf_listener(tf)
    cam_loc = CameraLocation(args.camera_location)
    time.sleep(1.0)
    ###############################################

    spot_images = SpotImages(use_registered_depth=True)

    spot_images.display_pointcloud(cam_loc, spot, 10.0)

    rotation_angle = 0
    if cam_loc == CameraLocation.FRONT_LEFT or cam_loc == CameraLocation.FRONT_RIGHT:
        rotation_angle = -90
    if cam_loc == CameraLocation.RIGHT:
        rotation_angle = -180
    [rgb, depth] = spot_images.rbg_and_depth_camera_from_location(cam_loc)
    spot_images.display_image(rgb, spot, rotation_angle=rotation_angle, timeout_sec=5.0)
    spot_images.display_image(depth, spot, rotation_angle=rotation_angle, timeout_sec=5.0)


if __name__ == "__main__":
    main()
