#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse

import rclpy
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn_msgs.msg import RobotCommandFeedback
from rclpy.node import Node
from spatialmath import SE3, UnitQuaternion
from spot_utilities.spot_basic import SpotBasic


class Test6DPoseCommand(Node):
    """A node that tests Spot 6D walk to pose commands"""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("Test6DPoseCommandNode")
        self._robot_name = args.robot
        self._relative = args.relative
        self._goal = args.goal

        self._spot = SpotBasic(robot_name=self._robot_name, parent_node_name=self.get_name())
        self._spot.initialize_robot()

        self._tf = TFListenerWrapper(
            self.get_name() + "_tf",
            cache_time_s=30,
            wait_for_transform=[self._spot.body_frame_name, self._spot.vision_frame_name],
        )
        self._spot.set_tf_listener(self._tf)

        self.test_walk_6d()

    def test_walk_6d(self) -> None:
        if len(self._goal) == 3:
            x, y, z = self._goal
            quaternion = UnitQuaternion(s=1.0, v=[0.0, 0.0, 0.0])
        elif len(self._goal) == 6:
            x, y, z, thx, thy, thz = self._goal
            quaternion = UnitQuaternion.RPY([thx, thy, thz], unit="deg")
        elif len(self._goal) == 7:
            x, y, z, qx, qy, qz, qw = self._goal
            quaternion = UnitQuaternion(s=qw, v=[qx, qy, qz])
        else:
            raise ValueError(f"invalid goal: {self._goal}")

        thx, thy, thz = quaternion.SO3().rpy()
        self.get_logger().info(
            f"\n Goal:\n- x: {x}\n- y: {y}\n- z: {z}\n- thx (roll): {thx}\n- thy (pitch): {thy}\n- thz (yaw): {thz}"
        )

        if self._relative:
            body_t_goal = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])
            self.get_logger().info(f"body_t_goal z: {str(body_t_goal.z)}")
            vision_t_body = self._tf.lookup_a_tform_b(self._spot.vision_frame_name, self._spot.body_frame_name)
            vision_t_goal = vision_t_body * body_t_goal
        else:
            vision_t_goal = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])

        try:
            # uncomment lines below to test synchronous walk_and_orient()
            self._spot.walk_and_orient(frame_t_goal=vision_t_goal)
            self.get_logger().info("Move to 6D pose test done.")
            # self._spot.walk_and_orient_async(
            #     frame_t_goal=vision_t_goal, result_callback=Test6DPoseCommand.test_result_callback
            # )
        except KeyboardInterrupt:
            self._spot.stop()

    @staticmethod
    def test_result_callback(feedback: RobotCommandFeedback) -> None:
        print("Test result_callback: Move to 6D pose test done.")

    def shutdown(self) -> None:
        self._spot.shutdown()
        self._tf.shutdown()
        self.destroy_node()


def test() -> None:
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    parser.add_argument(
        "goal",
        type=float,
        nargs="+",
        help=(
            "target SE3 pose in vision frame by default, unless '--relative'."
            "Could be either of length 3 (x,y,z), 6 (x,y,z,roll,pitch,yaw), or 7 (x,y,z,qx,qy,qz,qw)."
            "Note that roll, pitch, yaw are in degrees."
        ),
    )
    parser.add_argument(
        "--relative", action="store_true", help="goal will be in body frame; Otherwise, goal is in vision frame"
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = Test6DPoseCommand(args)
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    test()
