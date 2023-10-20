#  Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import bdai_ros2_wrappers.process as ros_process
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.utilities import fqn
from spatialmath import SE2
from spot_utilities.spot_basic import SpotBasic

# Where we want the robot to walk to relative to itself
ROBOT_T_GOAL = SE2(1.0, 0.0, 0.0)


class WalkForward:
    def __init__(self, robot_name: str) -> None:
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(robot_name)
        self._tf_listener = TFListenerWrapper()
        self._tf_listener.wait_for_a_tform_b(self._robot.body_frame_name, self._robot.vision_frame_name)

    def initialize_robot(self) -> bool:
        self._logger.info("Claiming robot")
        success = self._robot.claim()
        if not success:
            return False
        self._logger.info("Claimed robot")

        # Stand the robot up.
        self._logger.info("Powering robot on")
        success = self._robot.power_on()
        if not success:
            return False
        self._logger.info("Standing robot up")
        success = self._robot.stand()
        if not success:
            return False
        self._logger.info("Successfully stood up.")
        return True

    def walk_forward_with_world_frame_goal(self) -> None:
        self._logger.info("Walking forward")
        world_t_robot = self._tf_listener.lookup_a_tform_b_se2(
            self._robot.vision_frame_name, self._robot.body_frame_name
        )
        world_t_goal = world_t_robot * ROBOT_T_GOAL
        print(f"Going to \n{world_t_goal}.\nx={world_t_goal.x}, y={world_t_goal.y}, theta={world_t_goal.theta()}")

        self._robot.walk_to(world_t_goal)
        self._logger.info("Successfully walked forward")


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    goto = WalkForward(args.robot)
    if goto.initialize_robot():
        goto.walk_forward_with_world_frame_goal()
    return 0


if __name__ == "__main__":
    sys.exit(main())
