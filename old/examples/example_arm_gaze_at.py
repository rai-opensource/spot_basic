# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.futures import unwrap_future
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot gazes at a point in space with its arm

    Teleop the robot so that the origin of the vision frame won't be obstructed by spot's own body

    Use the following command to run the example:
    `python example_arm_gaze_at.py <robot_name>`
    """

    def __init__(self, args: argparse.Namespace) -> None:
        robot_name: str = args.robot_name
        self._asynchronous: bool = args.asynchronous
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(robot_name=robot_name)

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")
            self._logger.info("Attempting to stow arm")
            self._robot.stow_arm.synchronously()
            self._logger.info("Arm is stowed")
            point = [0, 0, 0]
            frame = self._robot.FrameHint.VISION
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
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    parser.add_argument(
        "--asynchronous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to exercise the async code",
    )

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
