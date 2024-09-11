#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

from spot_utilities.spot_basic_util import create_spot_basic_with_tf

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Uses utility in `spot_basic_util`

    Use the following command to run the example:
    `python example_spot_basic_util.py <robot_name>`
    """

    def __init__(self, robot_name: str) -> None:
        self._robot = create_spot_basic_with_tf(robot_name=robot_name)
        self._logger = logging.getLogger(fqn(self.__class__))

    def start(self) -> int:
        self._logger.info("Attempting to initialize robot")
        self._robot.initialize_robot()
        self._logger.info("Robot is initialized")

        self._logger.info("Attempting to stand robot up")
        self._robot.stand.synchronously()
        self._logger.info("Robot is standing up")

        self._logger.info("Attempting to ready arm")
        self._robot.ready_arm()
        self._logger.info("Arm is ready")

        self._logger.info("Attempting to stow arm")
        self._robot.stow_arm()
        self._logger.info("Arm is stowed")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    example = Example(args.robot_name)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
