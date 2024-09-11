#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

import numpy as np
from bosdyn.api import arm_command_pb2, geometry_pb2, trajectory_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME
from spatialmath import SE3, Twist3
from spatialmath.base import trinterp
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_dataclasses import AxisModeMap

import bdai_ros2_wrappers.process as ros_process
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.utilities import fqn


class SpotRunner:
    """Class to run the Spot robot."""

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
        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._position_mode,
        )

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

    def _move_arm_relative_to_body(
        self,
        timestamps_between_points: np.typing.ArrayLike,
        pose_vecs: Sequence[SE3],
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_LINEAR,
        stay_near_current_joint_angles: Optional[bool] = None,
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        timeout_sec: Optional[float] = 5.0,
        reference_time: Optional[float] = None,
        move_async: bool = False,
    ) -> None:
        self.logger.info("Moving arm along trajectory")
        """Invokes the arm to move relative to body following some trajectory.

        Args:
            timestamps_between_points: vector of timestamps for each point along the trajectory
            pose_vecs: vector of SE3 poses forming the trajectory
            wrist_t_tool: optional transform from the wrist of the spot to a mounted tool
            root_t_task: optional transform from the root frame to the task frame of spot. Defaults to identity
            vel_vecs: optional velocity trajectory to track
            pose_interpolation: interpolation method for the positions along the trajectory
            ang_interpolation: interpolation method for the angles along the trajectory
            stay_near_current_joint_angles: ask the system to find solutions near the current arm config
            preferred_joint_configuration: preferred final joint configuration
            timeout_sec: time before the command times out, in seconds
            reference_time: reference time to begin the trajectory, referecing the robot clock
            move_async: whether or not to move asynchronously. This is a risky move so be careful
        """
        if not move_async:
            self._robot.move_end_effector_cartesian_trajectory(
                root_frame_name=BODY_FRAME_NAME,
                axis_mode_map=self.axis_mode_map,
                timestamps_between_poses=timestamps_between_points,
                pose_vecs=pose_vecs,
                vel_vecs=vel_vecs,
                pose_interpolation=pose_interpolation,
                ang_interpolation=ang_interpolation,
                wrist_t_tool=wrist_t_tool,
                root_t_task=root_t_task,
                max_acceleration=self.max_acceleration,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                max_pos_tracking_error=self.max_pos_tracking_error,
                max_rot_tracking_error=self.max_rot_tracking_error,
                stay_near_current_joint_angles=stay_near_current_joint_angles,
                preferred_joint_configuration=preferred_joint_configuration,
                timeout_sec=timeout_sec,
                body_follow_arm=False,
                reference_time=reference_time,
            )
        else:
            self._robot.move_end_effector_cartesian_trajectory_async(
                root_frame_name=BODY_FRAME_NAME,
                axis_mode_map=self.axis_mode_map,
                timestamps_between_poses=timestamps_between_points,
                pose_vecs=pose_vecs,
                vel_vecs=vel_vecs,
                pose_interpolation=pose_interpolation,
                ang_interpolation=ang_interpolation,
                wrist_t_tool=wrist_t_tool,
                root_t_task=root_t_task,
                max_acceleration=self.max_acceleration,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                max_pos_tracking_error=self.max_pos_tracking_error,
                max_rot_tracking_error=self.max_rot_tracking_error,
                stay_near_current_joint_angles=stay_near_current_joint_angles,
                preferred_joint_configuration=preferred_joint_configuration,
                body_follow_arm=False,
                reference_time=reference_time,
            )
        self.logger.info("Completed!")

    def _move_hand_body_follow_wrt_odom(
        self,
        timestamps_between_points: np.typing.ArrayLike,
        pose_vecs: Sequence[SE3],
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        stay_near_current_joint_angles: Optional[bool] = None,
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        timeout_sec: Optional[float] = 5.0,
        reference_time: Optional[float] = None,
    ) -> None:
        self.logger.info("Moving arm, body following along trajectory")
        """Calls arm trajectory movements with the body following the arm.

        Trajectories must be described in the odometry frame for this method.
        Specifying in the body frame will not work as these motions are intended
        to go outside the body frame.

        Args:
            timestamps_between_points: vector of timestamps for each point along the trajectory
            pose_vecs: vector of SE3 poses forming the trajectory
            wrist_t_tool: optional transform from the wrist of the spot to a mounted tool
            root_t_task: optional transform from the root frame to the task frame of spot. Defaults to identity
            vel_vecs: optional velocity trajectory to track
            pose_interpolation: interpolation method for the positions along the trajectory
            ang_interpolation: interpolation method for the angles along the trajectory
            stay_near_current_joint_angles: ask the system to find solutions near the current arm config
            preferred_joint_configuration: preferred final joint configuration
            timeout_sec: time before the command times out, in seconds
            reference_time: reference time to begin the trajectory, referecing the robot clock
        """
        self._robot.move_end_effector_cartesian_trajectory(
            root_frame_name=ODOM_FRAME_NAME,
            axis_mode_map=self.axis_mode_map,
            timestamps_between_poses=timestamps_between_points,
            pose_vecs=pose_vecs,
            vel_vecs=vel_vecs,
            pose_interpolation=pose_interpolation,
            ang_interpolation=ang_interpolation,
            wrist_t_tool=wrist_t_tool,
            root_t_task=root_t_task,
            max_acceleration=self.max_acceleration,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            max_pos_tracking_error=self.max_pos_tracking_error,
            max_rot_tracking_error=self.max_rot_tracking_error,
            stay_near_current_joint_angles=stay_near_current_joint_angles,
            preferred_joint_configuration=preferred_joint_configuration,
            timeout_sec=timeout_sec,
            body_follow_arm=True,
            reference_time=reference_time,
        )
        self.logger.info("Completed!")

    def move_arm_with_body_follow(
        self,
        end_pose_body_frame: SE3,
        num_points: int,
        end_time: float,
        vel_traj: Optional[Sequence[Twist3]] = None,
        reference_time: Optional[float] = None,
    ) -> None:
        """Generates an arm trajectory with the body following along.

        Args:
            end_pose_body_frame: the final pose of the robot, in the body frame
            num_points: number of points you wish to sample along the trajectory
            end_time: time you want to reach the goal
            vel_traj: optional velocity trajectory to follow along with the pose trajectory.
            reference_time: reference time with respect to the robot clock. Only set it if you know
                            what you're doing.
        """
        start_body_hand_pose = self.get_body_to_hand_tf()
        odom_to_body = self._tf_listener.lookup_a_tform_b(self._robot.odom_frame_name, self._robot.body_frame_name)

        timestamps, trajectory = generate_trajectory(
            start_pose=start_body_hand_pose, end_pose=end_pose_body_frame, num_points=num_points, end_time=end_time
        )

        odom_hand_traj = []

        for point in trajectory:
            odom_hand_traj.append(odom_to_body * point)

        self._move_hand_body_follow_wrt_odom(
            timestamps_between_points=timestamps,
            pose_vecs=odom_hand_traj,
            root_t_task=self.I,
            wrist_t_tool=self.I,
            vel_vecs=vel_traj,
            reference_time=reference_time,
        )

    def move_arm_without_body(
        self,
        end_pose_body_frame: SE3,
        num_points: int,
        end_time: float,
        vel_traj: Optional[Sequence[Twist3]] = None,
        reference_time: Optional[float] = None,
    ) -> None:
        """Commands the arm to follow a trajectory without the body moving.

        Args:
            end_pose_body_frame: the final pose of the robot, in the body frame
            num_points: number of points you wish to sample along the trajectory
            end_time: time you want to reach the goal
            vel_traj: optional velocity trajectory to follow along with the pose trajectory.
            reference_time: reference time with respect to the robot clock.
                            Only set it if you know what you're doing.
        """
        start_body_hand_pose = self.get_body_to_hand_tf()
        timestamps, trajectory = generate_trajectory(
            start_pose=start_body_hand_pose,
            end_pose=end_pose_body_frame,
            num_points=num_points,
            end_time=end_time,
        )

        self._move_arm_relative_to_body(
            timestamps_between_points=timestamps,
            pose_vecs=trajectory,
            root_t_task=self.I,
            vel_vecs=vel_traj,
            reference_time=reference_time,
        )

    def _get_points(self, duration: float) -> int:
        """Returns a number of points depending on the frequency you set."""
        return int(self._frequency * duration)

    def test_run(self) -> None:
        # Test a variety of behaviours
        time.sleep(0.5)
        self.cobra()
        time.sleep(0.5)

        # Move the hand relative to the body in a basic trajectory
        end_pose_body_hand = self.get_body_to_hand_tf() * SE3.Trans([0.2, 0.2, -0.1])
        no_vel_body_end_time = 1.5  # seconds
        self.move_arm_without_body(
            end_pose_body_frame=end_pose_body_hand,
            num_points=self._get_points(no_vel_body_end_time),
            end_time=no_vel_body_end_time,
        )
        time.sleep(0.5)

        # Rotate the hand to -pi/3 in the body frame
        end_pose_rotate_hand = self.get_body_to_hand_tf() * SE3.Rx(-np.pi / 3)
        rotate_body_end_time = 0.5  # seconds
        self.move_arm_without_body(
            end_pose_body_frame=end_pose_rotate_hand,
            num_points=self._get_points(rotate_body_end_time),
            end_time=rotate_body_end_time,
        )
        time.sleep(0.5)

        # Stow the arm
        self.cobra()
        time.sleep(1.0)

        # Translate the body and arm
        end_pose_translate_hand = self.get_body_to_hand_tf() * SE3.Trans([0.25, -0.3, 0.2])
        translate_body_move_time = 1.25  # seconds
        self.move_arm_with_body_follow(
            end_pose_body_frame=end_pose_translate_hand,
            num_points=self._get_points(translate_body_move_time),
            end_time=translate_body_move_time,
        )
        time.sleep(1.0)

        # Rotate the body and arm
        end_pose_rotate_hand = self.get_body_to_hand_tf() * SE3.Rz(np.pi / 3)
        rotate_body_move_time = 1.5  # seconds
        self.move_arm_with_body_follow(
            end_pose_body_frame=end_pose_rotate_hand,
            num_points=self._get_points(rotate_body_move_time),
            end_time=rotate_body_move_time,
        )
        time.sleep(0.5)

        # Go back to cobra
        self.cobra()
        time.sleep(0.5)

        # # # Move arm with a velocity profile
        translate_arm_vel_time = 1.0  # seconds
        translate_arm_vel_points = 100
        translate_arm_base_vel = np.array([0.3, 0.0, 0.0])
        end_pose_translate_arm = self.get_body_to_hand_tf() * SE3.Trans(translate_arm_base_vel)
        translate_arm_vel_traj = []

        for i in range(translate_arm_vel_points):
            if i < translate_arm_vel_points / 2:
                lin_vel = translate_arm_base_vel * 3 / 2
            else:
                lin_vel = translate_arm_base_vel * 1 / 2
            translate_arm_vel_traj.append(Twist3(lin_vel, np.zeros((3,))))
        self.move_arm_without_body(
            end_pose_body_frame=end_pose_translate_arm,
            num_points=translate_arm_vel_points,
            end_time=translate_arm_vel_time,
            vel_traj=translate_arm_vel_traj,
        )
        time.sleep(0.5)

        # Move arm in cosine wave
        self.cobra()
        time.sleep(0.5)

        # # Generate the cosine wave and follow it using the arm
        start_pose_cosine_arm = self.get_body_to_hand_tf()
        end_pose_cosine_arm = start_pose_cosine_arm * SE3.Trans([0.3, 0.0, 0.0])
        cosine_arm_traj = []
        cosine_arm_duration = 1.5  # seconds
        b = start_pose_cosine_arm.z
        A = 0.1
        for i in range(translate_arm_vel_points):
            # formula is Asin(x/(2*pi)) + b
            lin_coeff = i / translate_arm_vel_points
            x = end_pose_cosine_arm.x * lin_coeff + start_pose_cosine_arm.x * (1.0 - lin_coeff)
            z = A * np.sin(2 * np.pi * lin_coeff) + b
            cosine_arm_traj.append(SE3.Trans(np.array([x, start_pose_cosine_arm.y, z])))
        self._move_arm_relative_to_body(
            timestamps_between_points=np.linspace(0.0, cosine_arm_duration, translate_arm_vel_points),
            pose_vecs=cosine_arm_traj,
        )
        time.sleep(0.5)

        self.cobra()

        # Test the axis mode dictionary to see if it's catching the errors
        try:
            self.axis_mode_map = AxisModeMap(
                x=self._position_mode, y=self._position_mode, ry=self._position_mode, rz=self._position_mode
            )
            self.cobra()
        except Exception as e:
            print(e)

        self.axis_mode_map = AxisModeMap(
            x=self._position_mode,
            y=self._position_mode,
            z=self._position_mode,
            rx=self._position_mode,
            ry=self._position_mode,
            rz=self._position_mode,
        )
        self.cobra()
        time.sleep(0.5)
        self._robot.stow_arm()

    def cobra(self) -> None:
        self._move_arm_relative_to_body(timestamps_between_points=[0.5], pose_vecs=[self._cobra_pose], move_async=False)


def generate_trajectory(
    start_pose: SE3,
    end_pose: SE3,
    num_points: int,
    end_time: float,
) -> Sequence[SE3]:
    """Generates a SE3 straight line position trajectory given a start and end time / pose.

    Interpolates a start and end pose using the interpolation method in the SE3 file.
    """
    timestamps = np.linspace(0.0, end_time, num_points)
    trajectory = []
    for ts in timestamps:
        interp_amt = ts / end_time
        interp_traj_point = trinterp(start_pose.data[0], end_pose.data[0], interp_amt)
        trajectory.append(SE3(np.copy(interp_traj_point)))
    return (timestamps, trajectory)


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    spot_runner = SpotRunner(
        args.robot_name,
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
