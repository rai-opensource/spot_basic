# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
#
import argparse
import logging
import sys

from spatialmath import SE3, SO3
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot makes a sequence of poses while standing in place

    Use the following command to run the example:
    `python example_stand_with_orientation.py <robot_name>`
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

        # We only care about in-place rotations
        gravbody_r_routine = [SO3((gravbody_t_body * body_t_pose).R) for body_t_pose in body_t_goals]

        # This is the height of Spot's body relative to its normal height (0.0 means at normal height)
        body_height = 0.0

        self._logger.info("Attempting to run poses")
        for gravbody_r_orien in gravbody_r_routine:
            self._robot.stand_with_orientation(gravbody_r_orien, body_height)
        self._logger.info("All poses finished")

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
