# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import logging
import sys

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn
from spot_utilities.spot_basic import SpotBasic


class Example:
    """
    Spot unstows arm and is set to zero-g mode (low joint stiffness)

    Use the following command to run the example:
    `python example_zerog.py <robot_name>`
    """

    def __init__(self, robot_name: str) -> None:
        self._robot = SpotBasic(robot_name=robot_name)
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

        self._logger.info("Attempting to set arm in zero gravity mode")
        self._robot.zero_g()
        self._logger.info("Arm is in zero gravity mode")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args.robot_name)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
