#!/usr/bin/env python
# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from synchros2.futures import unwrap_future

from spot_basic import FrameHint, SpotBasic


class ArmGazeAtExample:
    """
    Spot gazes at a point in space with its arm.

    Make sure you have a Spot with an arm and that there is enough free space (about ~2 m)
    around it. The Spot driver must be started before executing this example, and the robot
    teleoperated so that the origin of the vision frame won't be obstructed by its own body.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._asynchronous = args.asynchronous

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")
            self._logger.info("Attempting to stow arm")
            self._robot.stow_arm.synchronously()
            self._logger.info("Arm is stowed")
            point = [0, 0, 0]
            frame = FrameHint.VISION
            self._logger.info("Attempting to gaze with arm")

            if self._asynchronous:
                future = self._robot.arm_gaze.asynchronously(point=point, frame=frame)
                print("Gazed with arm asynchronously")
                unwrap_future(future)
            else:
                self._robot.arm_gaze.synchronously(point=point, frame=frame)
                print("Gazed with arm synchronously")

            self._logger.info("Gazed with arm")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ArmGazeAtExample.__doc__)
    parser.add_argument(
        "--asynchronous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to exercise the async code",
    )
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ArmGazeAtExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
