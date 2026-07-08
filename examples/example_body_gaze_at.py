#!/usr/bin/env python
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from synchros2.futures import unwrap_future

from spot_basic import FrameHint, SpotBasic


class BodyGazeAtExample:
    """
    Spot gazes at a point in space with its body.

    Make sure to teleop Spot so that the origin of the vision frame is outside
    the support polygon of Spot feet on power on. The Spot driver must be started
    before executing this example too.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._asynchronous = args.asynchronous

    def start(self) -> int:
        with self._robot.leash():
            target = [1, 1, 1]
            frame = FrameHint.VISION
            self._logger.info("Attempting to gaze with body")

            if self._asynchronous:
                future = self._robot.body_gaze.asynchronously(target=target, frame=frame)
                print("Gazed with body asynchronously")
                unwrap_future(future)
            else:
                self._robot.body_gaze.synchronously(target=target, frame=frame)
                print("Gazed with body synchronously")

            self._logger.info("Gazed with body")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=BodyGazeAtExample.__doc__)
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
    example = BodyGazeAtExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
