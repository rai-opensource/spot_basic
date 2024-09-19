#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Uses utility in `spot_basic_util`

    Use the following command to run the example:
    `python example_spot_basic_util.py <robot_name>`
    """

    def __init__(self, args: argparse.Namespace) -> None:
        robot_name: str = args.robot_name
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(robot_name=robot_name)

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")
            self._logger.info("Attempting to ready arm")
            self._robot.ready_arm.synchronously()
            self._logger.info("Arm is ready")
            self._logger.info("Attempting to stow arm")
            self._robot.stow_arm.synchronously()
            self._logger.info("Arm is stowed")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
