#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys
from threading import Event

from bosdyn_api_msgs.msg import RobotCommandFeedback
from spatialmath import SE2
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot walks forward asynchronously

    Use the following command to run the example:
    `python example_walk_forward_async.py <robot_name>`
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

        world_t_robot = self._robot.lookup_a_tform_b(
            self._robot.vision_frame_name, self._robot.body_frame_name
        ).yaw_SE2()

        robot_t_goal = SE2(1.0, 0.0, 0.0)
        world_t_goal = world_t_robot * robot_t_goal
        event = Event()

        def result_callback(_: RobotCommandFeedback) -> None:
            nonlocal event
            event.set()

        self._logger.info("Attempting to walk robot forward")
        self._robot.walk_to_async(frame_t_goal=world_t_goal, result_callback=result_callback)
        event.wait()
        self._logger.info("Robot walked forward")

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
