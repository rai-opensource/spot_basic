#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
import sys
import time
from threading import Event
from typing import Any, Optional, Sequence

import numpy as np
import rclpy
from bdai_ros.utilities.manual_conversions import list_to_arm_joint_map
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from rclpy.node import Node
from spatialmath import SE3
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_dataclasses import ArmJointMap


class SpotRunner(Node):
    """Class to run the Spot robot."""

    def __init__(
        self,
        options: argparse.Namespace,
        root_frame_name: Optional[str] = None,
    ):
        super().__init__("spot_runner", namespace=options.robot)

        self.root_frame_name = root_frame_name

        self._max_acc = 100.0  # rad/sec^2
        self._max_vel = 10.0  # rad/sec
        self._practice_config_list = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, 0.0, -np.pi / 2]
        self._practice_config = list_to_arm_joint_map(self._practice_config_list)
        self.num_arm_joints = 6

        self._robot = SpotBasic(robot_name=options.robot, parent_node_name=self.get_name())
        self._tf_listener = TFListenerWrapper(
            "joint_trajectory_tf", wait_for_transform=[self._robot.body_frame_name, self._robot.hand_frame_name]
        )

    def initialize_robot(self) -> None:
        self._robot.initialize_robot()
        self.get_logger().info("Standing robot up")
        self._robot.stand()
        self.get_logger().info("Successfully stood up.")

        # Move the arm to a specific location
        self.get_logger().info("Moving arm to start config")
        self._robot.stow_arm()
        self.get_logger().info("Successfully moved arm to starting config")

    def get_odom_to_hand_tf(self) -> SE3:
        return self._tf_listener.lookup_a_tform_b(self._robot.odom_frame_name, self._robot.hand_frame_name)

    def get_body_to_hand_tf(self) -> SE3:
        return self._tf_listener.lookup_a_tform_b(self._robot.body_frame_name, self._robot.hand_frame_name)

    def stow_arm(self) -> None:
        self._robot.stow_arm()

    def move_arm_along_joint_trajectory(
        self,
        joint_angles: Sequence[ArmJointMap],
        times: Sequence[float],
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        ref_time: Optional[float] = None,
    ) -> None:
        self.get_logger().info("Moving arm along trajectory")
        """Invokes the arm follow some joint trajectory.

        Args:
            joint_positions: List of joint angles to command the arm to move to. Each set of joint
                            angles must be size 6, as there are 6 joints.
            times: List of times we want to reach each joint position relative to the reference time
            joint_velocities: List of joint velocities for the arm to achieve along the trajectory.
            ref_time: reference time to begin trajectory, relative to the robot clock.
            max_acc: maximum acceleration allowed for any joint along the trajectory (rad/s^2)
            max_vel: maximum velocity allowed for any joint along the trajectory (rad/s)
            move_async: whether or not to move asynchronously. This is a risky move so be careful
        """
        self._robot.move_arm_joint_trajectory(
            joint_angles=joint_angles,
            times=times,
            joint_velocities=joint_velocities,
            ref_time=ref_time,
            max_acc=self._max_acc,
            max_vel=self._max_vel,
        )
        self.get_logger().info("Completed!")

    def move_arm_to_joint_angles(
        self,
        joint_angles: Optional[ArmJointMap],
        time_since_reference_secs: Optional[float] = None,
    ) -> None:
        self.get_logger().info("Moving arm to joint position")
        """Moves the arm to a specific set of joint angles.

        Args:
            joint_angles: the joint angles of the robot, in order from shoulder to wrist
            lock_joints: keep the joints in place
            timeout_sec: seconds between receive and execution until the command times out
            move_async: whether or not to execute the move asynchronously
        """
        self._robot.move_arm_to_joint_angles(
            joint_angles=joint_angles,
            time_since_reference_secs=time_since_reference_secs,
        )
        self.get_logger().info("Completed!")

    def test_run(self) -> None:
        # Test a variety of behaviours
        event = Event()

        # Do the async movements
        def _result_callback(_: Any) -> None:
            nonlocal event
            event.set()

        self.stow_arm()
        time.sleep(2.0)

        self.move_arm_to_start_config()
        time.sleep(1.0)

        signs = [-1, 1]

        # # Generate the testing suite
        joint_moves = []
        times = []
        suite_delay = 1.5
        for i in range(2 * self.num_arm_joints):
            individual_joint_move = np.array(self._practice_config_list)
            angle_ind = int(i / 2)
            # Perturb the joint in either direction
            new_angle = self._practice_config_list[angle_ind] + signs[i % 2] * 0.2
            individual_joint_move[angle_ind] = new_angle
            joint_moves.append(list_to_arm_joint_map(individual_joint_move))
            times.append((i + 1) * suite_delay)

        # Test each joint individually from a good starting position, synchronously
        for joint_move in joint_moves:
            self.move_arm_to_joint_angles(joint_angles=joint_move)
            time.sleep(suite_delay)

        # Test the arm movements asynch
        for joint_move in joint_moves:
            self._robot.move_arm_to_joint_angles_async(joint_angles=joint_move, result_callback=_result_callback)
            event.wait()
            time.sleep(suite_delay)

        # Test each trajectory in sections of 10 points
        self.move_arm_along_joint_trajectory(joint_angles=joint_moves[:10], times=times[:10])
        time.sleep(10.0)

        # Test freezing the arm in place
        self._robot.lock_arm()
        time.sleep(7.0)

        # Test each trajectory in sections of 10 points, async
        self._robot.move_arm_joint_trajectory_async(
            joint_angles=joint_moves[:10], times=times[:10], result_callback=_result_callback
        )
        event.wait()
        time.sleep(3.0)

        # Test exception catching
        try:
            self.move_arm_along_joint_trajectory(joint_angles=joint_moves, times=times)
        except Exception as e:
            self.get_logger().info(str(e))

        self.move_arm_to_start_config()
        time.sleep(1.0)

        # Test the arm trajectory with a velocity, sync and async
        wrist_1_amt = 0.15
        wrist_0_amt = 0.25
        wrist_1_move = np.array(self._practice_config_list)
        wrist_1_move[5] += wrist_1_amt
        wrist_0_move = np.copy(wrist_1_move)
        wrist_0_move[4] += wrist_0_amt

        wrist_1_vel = np.zeros((self.num_arm_joints,))
        wrist_1_vel[5] = wrist_1_amt
        wrist_0_vel = np.zeros((self.num_arm_joints,))
        wrist_0_vel[4] = wrist_0_amt

        arm_joint_position = [
            list_to_arm_joint_map(wrist_1_move),
            list_to_arm_joint_map(wrist_0_move),
        ]
        arm_joint_velocity = [
            list_to_arm_joint_map(wrist_1_vel),
            list_to_arm_joint_map(wrist_0_vel),
        ]

        self.move_arm_along_joint_trajectory(
            joint_angles=arm_joint_position,
            times=[1.0, 2.0],
            joint_velocities=arm_joint_velocity,
        )
        time.sleep(3.0)

        self.move_arm_to_start_config()
        self._robot.move_arm_joint_trajectory_async(
            joint_angles=arm_joint_position,
            times=[1.0, 2.0],
            joint_velocities=arm_joint_velocity,
            result_callback=_result_callback,
        )
        event.wait()
        time.sleep(3.0)

        # Test the locking of the arm async
        self._robot.lock_arm_async(result_callback=_result_callback)
        event.wait()
        time.sleep(8.0)

        self._robot.stow_arm()

    def move_arm_to_start_config(self) -> None:
        """Moves the arm to the preferred starting configuration."""
        self.move_arm_to_joint_angles(joint_angles=self._practice_config)

    def shutdown(self) -> None:
        self._tf_listener.shutdown()


def main() -> int:
    rclpy.init()
    """Command-line interface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace")
    options = parser.parse_args()

    spot_runner = SpotRunner(
        options,
        root_frame_name=BODY_FRAME_NAME,
    )
    spot_runner.initialize_robot()

    spot_runner.get_logger().info("Check")

    spot_runner.test_run()

    spot_runner.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
