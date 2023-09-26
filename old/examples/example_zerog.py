# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse

import rclpy
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo starting ZeroG mode.

Use:
    0) have a spot with an arm
    1) start spot driver
    2) make sure there isn't anything within a meter of spot
    3) python relative/path/to/this/script/here robot-name-here
    4) spot will unstow the arm, then be in zero-g mode (low joint stiffness)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    options = parser.parse_args()

    robot_name = options.robot

    rclpy.init()
    node = rclpy.create_node("spot_zerog_example")
    spot = SpotBasic(robot_name, parent_node_name=node.get_name())

    spot.initialize_robot()

    spot.stand()

    spot.ready_arm()

    spot.zero_g()

    rclpy.spin(node)

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
