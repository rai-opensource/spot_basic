# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse

import bdai_ros2_wrappers.process as ros_process
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo starting ZeroG mode.

Use:
    0) have a spot with an arm
    1) start spot driver
    2) make sure there isn't anything within a meter of spot
    3) python relative/path/to/this/script/here robot-name-here
    4) spot will unstow the arm, then be in zero-g mode (low joint stiffness)
"""


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    spot = SpotBasic(args.robot)

    spot.initialize_robot()

    spot.stand()

    spot.ready_arm()

    spot.zero_g()

    main.wait_for_shutdown()


if __name__ == "__main__":
    main()
