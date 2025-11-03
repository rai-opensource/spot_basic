#!/usr/bin/env python
# Copyright (c) 2024-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import math
import sys

import synchros2.process as ros_process
from spatialmath import Twist2
from spot_utilities.spot_basic import SpotBasic


class WalkWithVelocityExample:
    """
    Make Spot walk forward at a specified velocity.

    The Spot driver must be running before attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._duration: float = args.duration
        self._vx: float = args.vx
        self._vy: float = args.vy
        self._vyaw: float = args.vyaw

    def start(self) -> int:
        with self._robot.leash():
            body_velocity = Twist2([self._vx, self._vy], math.pi * self._vyaw / 180)
            self._logger.info("Attempting to walk robot forward")
            self._robot.walk_velocity(body_velocity=body_velocity, duration=self._duration)
            self._logger.info("Robot walked forward")

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=WalkWithVelocityExample.__doc__)
    parser.add_argument("--duration", default=3.0, help="Duration (s) to walk with given velocity", type=float)
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    parser.add_argument("vx", help="Target SE2 x-axis velocity of body", type=float)
    parser.add_argument("vy", help="Target SE2 y-axis velocity of body", type=float)
    parser.add_argument("vyaw", help="Target SE2 yaw velocity of body in degrees", type=float)
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    example = WalkWithVelocityExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
