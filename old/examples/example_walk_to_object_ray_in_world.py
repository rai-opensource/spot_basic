#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot will walk forward about 0.5 meters.

    Uses `WalkToObjectRayInWorld` API to request from the spot-sdk.

    Use the following command to run the example:
    `python example_walk_to_object_ray_in_world.py <robot_name>`
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
            body_t_ground = self._robot.lookup_a_tform_b(SpotBasic.FrameHint.BODY, SpotBasic.FrameHint.GROUND_PLANE)
            ray_start_rt_frame = [1.25, 0, body_t_ground.z + 1]
            ray_end_rt_frame = [1.25, 0, body_t_ground.z - 1]
            self._logger.info("Attempting to walk robot forward")

            self._robot.walk_to_object_ray_in_world.synchronously(
                ray_start_rt_frame=ray_start_rt_frame,
                ray_end_rt_frame=ray_end_rt_frame,
                frame=SpotBasic.FrameHint.BODY.value,
                offset_distance=0.25,
            )

            self._logger.info("Robot walked forward")

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
