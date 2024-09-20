#  Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import math
import sys

from spatialmath import Twist2
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot walks forward at a specified velocity.

    Use the following command to run the example:
    `python example_walk_velocity_with_duration.py <robot_name> <vx> <vy> <vyaw>`
    """

    def __init__(self, args: argparse.Namespace) -> None:
        robot_name: str = args.robot_name
        self._duration: float = args.duration
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(robot_name)
        self._vx: float = args.vx
        self._vy: float = args.vy
        self._vyaw: float = args.vyaw

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")
            body_velocity = Twist2([self._vx, self._vy], math.pi * self._vyaw / 180)
            self._logger.info("Attempting to walk robot forward")
            self._robot.walk_velocity(body_velocity=body_velocity, duration=self._duration)
            self._logger.info("Robot walked forward")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    parser.add_argument("vx", help="Target SE2 x-axis velocity of body", type=float)
    parser.add_argument("vy", help="Target SE2 y-axis velocity of body", type=float)
    parser.add_argument("vyaw", help="Target SE2 yaw velocity of body in degrees", type=float)
    parser.add_argument("--duration", default=3.0, help="Duration (s) to walk with given velocity", type=float)

    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    example = Example(args)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
