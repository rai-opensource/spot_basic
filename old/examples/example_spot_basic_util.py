# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""
Super basic examples of using utility functions in spot_basic_util
"""
import argparse

import bdai_ros2_wrappers.process as ros_process
from spot_utilities.spot_basic_util import create_spot_basic_with_tf


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace.")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    spot = create_spot_basic_with_tf(args.robot)
    if spot.initialize_robot():
        spot.ready_arm()
        spot.stow_arm()


if __name__ == "__main__":
    main()
