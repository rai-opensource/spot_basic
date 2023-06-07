#  Copyright (c) 2023 Boston Dynamics AI Institute, Inc. All rights reserved.
import argparse
import sys

import rclpy
from bdai.utilities.math_helpers import SE2Pose
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from rclpy.node import Node
from spot_utilities.spot_basic import SpotBasic

# Where we want the robot to walk to relative to itself
ROBOT_T_GOAL = SE2Pose(1.0, 0.0, 0.0)


class WalkForward(Node):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__("walk_forward", namespace=options.robot)

        self._body_frame_name = options.robot + "/" + BODY_FRAME_NAME
        self._vision_frame_name = options.robot + "/" + VISION_FRAME_NAME
        self._tf_listener = TFListenerWrapper(
            "walk_forward_tf", wait_for_transform=[self._body_frame_name, self._vision_frame_name]
        )
        self._robot = SpotBasic(robot_name=options.robot, parent_node_name=self.get_name())

    def initialize_robot(self) -> bool:
        self.get_logger().info("Claiming robot")
        success = self._robot.claim()
        if not success:
            return False
        self.get_logger().info("Claimed robot")

        # Stand the robot up.
        self.get_logger().info("Powering robot on")
        success = self._robot.power_on()
        if not success:
            return False
        self.get_logger().info("Standing robot up")
        success = self._robot.stand()
        if not success:
            return False
        self.get_logger().info("Successfully stood up.")
        return True

    def walk_forward_with_world_frame_goal(self) -> None:
        self.get_logger().info("Walking forward")
        world_t_robot = self._tf_listener.lookup_a_tform_b_se2(self._vision_frame_name, self._body_frame_name)
        world_t_goal = world_t_robot * ROBOT_T_GOAL
        self._robot.walk_to(world_t_goal)
        self.get_logger().info("Successfully walked forward")

    def shutdown(self) -> None:
        self._tf_listener.shutdown()


def main() -> int:
    rclpy.init()
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace")
    options = parser.parse_args()

    goto = WalkForward(options)
    goto.initialize_robot()
    goto.walk_forward_with_world_frame_goal()
    goto.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
