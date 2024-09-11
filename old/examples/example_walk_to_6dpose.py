#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import logging
import typing

from bosdyn_msgs.msg import RobotCommandFeedback
from spatialmath import SE3, UnitQuaternion
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.utilities import fqn


class Test6DPoseCommand:
    """A helper that tests Spot 6D walk to pose commands"""

    def __init__(self, robot_name: str) -> None:
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot_name = robot_name

        self._spot = SpotBasic(self._robot_name)
        self._spot.initialize_robot()

        self._tf = TFListenerWrapper(cache_time_s=30)
        self._tf.wait_for_a_tform_b(self._spot.body_frame_name, self._spot.vision_frame_name)
        self._spot.set_tf_listener(self._tf)

    def test_walk_6d(self, goal: typing.List[float], relative: bool) -> None:
        if len(goal) == 3:
            x, y, z = goal
            quaternion = UnitQuaternion(s=1.0, v=[0.0, 0.0, 0.0])
        elif len(goal) == 6:
            x, y, z, thx, thy, thz = goal
            quaternion = UnitQuaternion.RPY([thx, thy, thz], unit="deg")
        elif len(goal) == 7:
            x, y, z, qx, qy, qz, qw = goal
            quaternion = UnitQuaternion(s=qw, v=[qx, qy, qz])
        else:
            raise ValueError(f"invalid goal: {goal}")

        thx, thy, thz = quaternion.SO3().rpy()
        self._logger.info(
            f"\n Goal:\n- x: {x}\n- y: {y}\n- z: {z}\n- thx (roll): {thx}\n- thy (pitch): {thy}\n- thz (yaw): {thz}"
        )

        if relative:
            body_t_goal = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])
            self._logger.info(f"body_t_goal z: {str(body_t_goal.z)}")
            vision_t_body = self._tf.lookup_a_tform_b(self._spot.vision_frame_name, self._spot.body_frame_name)
            vision_t_goal = vision_t_body * body_t_goal
        else:
            vision_t_goal = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])

        try:
            # uncomment lines below to test synchronous walk_and_orient()
            self._spot.walk_and_orient(frame_t_goal=vision_t_goal)
            self._logger.info("Move to 6D pose test done.")
            # self._spot.walk_and_orient_async(
            #     frame_t_goal=vision_t_goal, result_callback=Test6DPoseCommand.test_result_callback
            # )
        except KeyboardInterrupt:
            self._spot.stop()

    @staticmethod
    def test_result_callback(feedback: RobotCommandFeedback) -> None:
        print("Test result_callback: Move to 6D pose test done.")


def cli() -> argparse.ArgumentParser:
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
    return parser


@ros_process.main(cli())
def test(args: argparse.Namespace) -> None:
    tester = Test6DPoseCommand(args.robot)
    tester.test_walk_6d(args.goal, args.relative)


if __name__ == "__main__":
    test()
