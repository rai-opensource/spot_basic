#!/usr/bin/env python
# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from spatialmath import SE3, SO3
from spot_utilities.spot_basic import SpotBasic


class StandWithOrientationExample:
    """
    Have Spot rotate its body in place.

    The Spot driver must be running before attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            body_t_goals = [
                SE3.Rt(R=SO3.RPY(0, 0, 0), t=[0, 0, 0]),
                SE3.Rt(R=SO3.RPY(0, 0, 45), t=[0, 0, 0]),
                SE3.Rt(R=SO3.RPY(0, 0, -90), t=[0, 0, 0]),
                SE3.Rt(R=SO3.RPY(0, 45, 0), t=[0, 0, 0]),
                SE3.Rt(R=SO3.RPY(0, -90, 0), t=[0, 0, 0]),
                SE3.Rt(R=SO3.RPY(45, 45, 0), t=[0, 0, 0]),
                SE3.Rt(R=SO3.RPY(-90, -45, 0), t=[0, 0, 0]),
            ]

            gravbody_t_body = self._robot.lookup_a_tform_b(
                self._robot.FrameHint.GRAV_ALIGNED_BODY.value, self._robot.FrameHint.BODY.value, wait_for_frames=True
            )

            gravbody_r_routine = [SO3((gravbody_t_body * body_t_pose).R) for body_t_pose in body_t_goals]

            self._logger.info("Attempting to run poses")
            self._robot.stand_with_orientation.vectorized.synchronously(gravbody_r_routine)

            self._logger.info("All poses finished")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=StandWithOrientationExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = StandWithOrientationExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
