#!/usr/bin/env python
# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from spot_utilities.spot_basic import FrameHint, SpotBasic


class WalkToObjectRayInWorldExample:
    """
    Have Spot walk forward about 1.25 m.

    The Spot driver must be running before attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            body_t_ground = self._robot.lookup_a_tform_b(
                FrameHint.BODY, FrameHint.GROUND_PLANE, wait_for_frames=True, timeout=5.0
            )
            ray_start_rt_frame = [1.25, 0, body_t_ground.z + 1]
            ray_end_rt_frame = [1.25, 0, body_t_ground.z - 1]
            self._logger.info("Attempting to walk robot forward")

            self._robot.walk_to_object_ray_in_world.synchronously(
                ray_start_rt_frame=ray_start_rt_frame,
                ray_end_rt_frame=ray_end_rt_frame,
                frame=FrameHint.BODY.value,
                offset_distance=0.25,
            )

            self._logger.info("Robot walked forward")
        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=WalkToObjectRayInWorldExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = WalkToObjectRayInWorldExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
