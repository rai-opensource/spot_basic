# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""
Super basic examples of using utility functions in spot_basic_util
"""
import argparse

import rclpy
from spot_utilities.spot_basic_util import create_spot_basic_with_tf


def main() -> None:
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace.")
    options = parser.parse_args()

    node = rclpy.create_node("example_spot_basic_util")

    spot = create_spot_basic_with_tf(options.robot, node)
    spot.initialize_robot()
    spot.ready_arm()
    spot.stow_arm()

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
