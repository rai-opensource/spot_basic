#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

import bdai_ros2_wrappers.process as ros_process
import numpy as np
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.utilities import fqn
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from spatialmath import SE3, SpatialForce, Twist3
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_dataclasses import AxisModeMap


# Where we want the robot to walk to relative to itself
class SpotRunner:
    def __init__(
        self,
        robot_name: str,
        root_frame_name: Optional[str] = None,
        wrist_t_tool: Optional[geometry_pb2.SE3Pose] = None,
        root_t_task: Optional[geometry_pb2.SE3Pose] = None,
    ):
        self.logger = logging.getLogger(fqn(self.__class__))

        self.root_frame_name = root_frame_name
        self.wrist_t_tool = wrist_t_tool
        self.root_t_task = root_t_task

        start_pose = np.zeros((4, 4))
        start_pose[:3, :3] = np.eye(3)
        x_start = 0.45
        y_start = 0.0
        z_start = 0.5
        start_pose[0, 3] = x_start
        start_pose[1, 3] = y_start
        start_pose[2, 3] = z_start
        start_pose[3, 3] = 1
        self._cobra_pose = SE3(np.copy(start_pose))  # Baseline pose for this set of actions
        self._frequency = 65  # Hz
        self.max_acceleration = 10.0  # meters/sec^2
        self.max_linear_velocity = 2.0  # meters/sec
        self.max_angular_velocity = np.pi / 2  # radians/sec
        self.max_pos_tracking_error = 0.5  # meters
        self.max_rot_tracking_error = np.pi / 3  # Radians
        self.I = SE3()
        self._position_mode = arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_POSITION
        self._wrench_mode = arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE

        self._robot = SpotBasic(robot_name)
        self._tf_listener = TFListenerWrapper()
        self._tf_listener.wait_for_a_tform_b(self._robot.body_frame_name, self._robot.hand_frame_name)

    def initialize_robot(self) -> None:
        self._robot.initialize_robot()
        self.logger.info("Standing robot up")
        self._robot.stand()
        self.logger.info("Successfully stood up.")

        # Move the arm to a specific location
        self.logger.info("Moving arm to start config")
        self._robot.stow_arm()
        self.logger.info("Successfully moved arm to starting config")

    def get_odom_to_hand_tf(self) -> SE3:
        return self._tf_listener.lookup_a_tform_b(self._robot.odom_frame_name, self._robot.hand_frame_name)

    def get_body_to_hand_tf(self) -> SE3:
        return self._tf_listener.lookup_a_tform_b(self._robot.body_frame_name, self._robot.hand_frame_name)

    def move_arm_relative_to_body(
        self,
        timestamps_between_poses: Optional[np.typing.ArrayLike] = None,
        timestamps_between_wrenches: Optional[np.typing.ArrayLike] = None,
        pose_vecs: Optional[Sequence[SE3]] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        wrench_vecs: Optional[Sequence[SpatialForce]] = None,
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        max_acceleration: Optional[float] = None,
        max_linear_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        max_pos_tracking_error: Optional[float] = None,
        max_rot_tracking_error: Optional[float] = None,
        stay_near_current_joint_angles: Optional[bool] = None,
        reference_time: Optional[float] = None,
        move_async: bool = False,
        timeout_sec: Optional[float] = 5.0,
    ) -> None:
        """Calls the arm trajectory movements relative to the fixed body frame.

        Args:
            axis_mode_map: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
            timestamps_between_poses: array of timestamps of when each pose point is traversed (seconds)
            timestamps_between_wrenches: array of timestamps of when each wrench point is traversed (seconds)
            pose_vecs: vector of SE3 poses for the end effector to pass through described by the spatialmath library
            vel_vecs: vector of velocities to pass through at each point
            wrench_vecs: vector of SpatialForces for the end effector to output in the spatialmath library
            wrist_t_tool: tool pose relative to the parent link (wrist)
            root_t_task: transform between root and task frame
            max_acceleration: max acceleration of end effector (m/s^2)
            max_linear_velocity: max linear velocity of end effector (m/s)
            max_angular_velocity: max radial velocity of end effector (rad/s)
            max_pos_tracking_error: max deviation between inputted and tracked end effector position
            max_rot_tracking_error: max deviation between inputted and tracked end effector orientations
            stay_near_current_joint_angles: forces the system to look for solutions near the current joint angles
            reference_time: time with respect to the robot clock of when to reference when trajectories start and end
            timeout_sec: seconds it will attempt to run the command before timing out and returning
            move_async: make the arm move asynchronously. Not recommended
        """
        self.logger.info("Moving arm along trajectory")
        if not move_async:
            self._robot.move_end_effector_cartesian_trajectory(
                root_frame_name=BODY_FRAME_NAME,
                axis_mode_map=self.axis_mode_map,
                timestamps_between_poses=timestamps_between_poses,
                timestamps_between_wrenches=timestamps_between_wrenches,
                pose_vecs=pose_vecs,
                vel_vecs=vel_vecs,
                wrench_vecs=wrench_vecs,
                wrist_t_tool=wrist_t_tool,
                root_t_task=root_t_task,
                max_acceleration=self.max_acceleration,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                max_pos_tracking_error=self.max_pos_tracking_error,
                max_rot_tracking_error=self.max_rot_tracking_error,
                stay_near_current_joint_angles=stay_near_current_joint_angles,
                reference_time=reference_time,
                body_follow_arm=False,
                timeout_sec=timeout_sec,
            )
        else:
            self._robot.move_end_effector_cartesian_trajectory_async(
                root_frame_name=BODY_FRAME_NAME,
                axis_mode_map=self.axis_mode_map,
                timestamps_between_poses=timestamps_between_poses,
                timestamps_between_wrenches=timestamps_between_wrenches,
                pose_vecs=pose_vecs,
                vel_vecs=vel_vecs,
                wrench_vecs=wrench_vecs,
                wrist_t_tool=wrist_t_tool,
                root_t_task=root_t_task,
                max_acceleration=self.max_acceleration,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                max_pos_tracking_error=self.max_pos_tracking_error,
                max_rot_tracking_error=self.max_rot_tracking_error,
                stay_near_current_joint_angles=stay_near_current_joint_angles,
                reference_time=reference_time,
                body_follow_arm=False,
            )
        self.logger.info("Completed!")

    def test_run(self) -> None:
        # Test a variety of behaviours
        # Initial pose
        time.sleep(0.5)
        self.cobra()
        time.sleep(0.5)

        # Test a set force
        timestamps_between_wrenches = np.array([1.5])
        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._wrench_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._position_mode,
        )
        wrench_vecs = [SpatialForce([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
        self.move_arm_relative_to_body(
            timestamps_between_wrenches=timestamps_between_wrenches, wrench_vecs=wrench_vecs, timeout_sec=1.0
        )
        time.sleep(0.1)
        self.cobra()
        time.sleep(0.5)

        # Linearly move from 1 to 3 Newtons and back over 4 seconds
        timestamps_between_wrenches = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        self.axis_mode_map = AxisModeMap(
            x=self._wrench_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._position_mode,
        )

        wrench_vecs = []
        num_wrenches = len(timestamps_between_wrenches)
        # Push in the x direction
        fx = 0
        for i in range(num_wrenches):
            if i <= num_wrenches / 2:
                fx += 1
            else:
                fx -= 1
            wrench_vecs.append(SpatialForce([fx, 0, 0, 0, 0, 0]))
        self.logger.info("Force trajectory command")
        self.move_arm_relative_to_body(
            timestamps_between_wrenches=timestamps_between_wrenches,
            wrench_vecs=wrench_vecs,
            timeout_sec=max(timestamps_between_wrenches[-1], 5.0),
        )

        time.sleep(0.1)
        self.cobra()
        time.sleep(0.5)

        # Hybrid trajectory
        # Move the arm in the x-y plane relative to the body and apply a force in z
        # For this particular command, you want to set the other axis in position mode
        # or else the position of the end effector will drift
        self.logger.info("Hybrid force command")
        timestamps_between_poses = np.array([0.5, 1.0])
        timestamps_between_wrenches = np.array([2.0])
        pose_vecs = [SE3(x=0.45, y=0.0, z=0.55), SE3(x=0.65, y=0.0, z=0.55)]
        wrench_vecs = [SpatialForce([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])]
        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._wrench_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._position_mode,
        )

        self.move_arm_relative_to_body(
            timestamps_between_poses=timestamps_between_poses,
            # timestamps_between_wrenches=timestamps_between_wrenches,
            pose_vecs=pose_vecs,
            # wrench_vecs=wrench_vecs,
            timeout_sec=1.0,
        )
        time.sleep(0.1)
        self.cobra()
        time.sleep(0.5)

        # Test a torque
        self.logger.info("Torque command")
        timestamps_between_wrenches = np.array([1.5])
        wrench_vecs = [SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])]
        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._wrench_mode,
        )
        self.move_arm_relative_to_body(
            timestamps_between_wrenches=timestamps_between_wrenches,
            wrench_vecs=wrench_vecs,
            timeout_sec=1.0,
        )
        time.sleep(0.1)
        self.cobra()
        time.sleep(0.5)

        # Force-torque command
        self.logger.info("Force torque command")
        timestamps_between_wrenches = np.array([1.5])
        wrench_vecs = [SpatialForce([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])]
        self.axis_mode_map = AxisModeMap(
            x=self._wrench_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._wrench_mode,
            rz=self._position_mode,
        )
        self.move_arm_relative_to_body(
            timestamps_between_wrenches=timestamps_between_wrenches,
            wrench_vecs=wrench_vecs,
            timeout_sec=1.0,
        )
        time.sleep(0.1)
        self.cobra()
        time.sleep(0.5)

        # Hybrid torque trajectory
        # Move the arm in the x-y plane relative to the body and apply a force in z
        # For this particular command, you want to set the other axis in position mode
        # or else the position of the end effector will drift
        self.logger.info("Hybrid command")
        timestamps_between_poses = np.array([0.375, 0.75])
        timestamps_between_wrenches = np.array([2.0])
        pose_vecs = [SE3(x=0.45, y=0.0, z=0.55), SE3(x=0.45, y=0.2, z=0.55)]
        wrench_vecs = [SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])]
        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._wrench_mode,
        )

        self.move_arm_relative_to_body(
            timestamps_between_poses=timestamps_between_poses,
            timestamps_between_wrenches=timestamps_between_wrenches,
            pose_vecs=pose_vecs,
            wrench_vecs=wrench_vecs,
            timeout_sec=1.0,
        )
        time.sleep(0.1)
        self.cobra()
        time.sleep(0.5)

        self._robot.stow_arm()

    def cobra(self) -> None:
        """Move to a cobra-like position."""
        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._position_mode,
        )
        self.move_arm_relative_to_body(timestamps_between_poses=[0.5], pose_vecs=[self._cobra_pose], move_async=False)


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    spot_runner = SpotRunner(
        args.robot,
        root_frame_name=BODY_FRAME_NAME,
    )
    spot_runner.initialize_robot()

    spot_runner.logger.info("Check")
    spot_runner.cobra()
    time.sleep(1)

    spot_runner.test_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
