#!/usr/bin/env python
# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from spatialmath import SE2
from spot_utilities.spot_basic import SpotBasic
from synchros2.futures import unwrap_future


class AsyncWalkForwardExample:
    """
    Have Spot walk forward synchronously.

    The Spot driver must be running before attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            world_t_robot = self._robot.lookup_a_tform_b(
                self._robot.vision_frame_name, self._robot.body_frame_name
            ).yaw_SE2()

            robot_t_goal = SE2(1.0, 0.0, 0.0)
            world_t_goal = world_t_robot * robot_t_goal
            self._logger.info("Attempting to walk robot forward")
            future = self._robot.walk.asynchronously(frame_t_goal=world_t_goal)
            print("Walking asynchronously")
            unwrap_future(future)
            self._logger.info("Robot walked forward")

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=AsyncWalkForwardExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = AsyncWalkForwardExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
