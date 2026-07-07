#!/usr/bin/env python
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

import argparse
import logging
import sys
from typing import List

import numpy as np
import synchros2.process as ros_process
from spatialmath import SE3, Twist3
from spatialmath.base import trinterp

from spot_basic import FrameHint, SpotBasic
from spot_basic.command.arm import AxisModeMap


def se3_linspace(start: SE3, end: SE3, num_points: int) -> List[SE3]:
    """Generates a SE3 straight line position trajectory given a start and end time / pose.

    Interpolates a start and end pose using the interpolation method in the SE3 file.
    """
    return [SE3(trinterp(start.data[0], end.data[0], alpha)) for alpha in np.linspace(0.0, 1.0, num_points)]


class ArmCartesianTrajectoryExample:
    """
    Spot moves its hand around.

    When running this example, Spot will perform several motions with its arm end effector.

    Make sure you have a Spot with an arm and there is enough free space (about ~2 m) around it.
    The Spot driver must be started before executing this example too.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def _move_cosine_optional_body_follow(self, body_follow: bool) -> None:
        # Generate the cosine wave and follow it using the arm
        self._logger.info(f"Using hand to follow a wave body_follow is: {body_follow}...")
        world_t_body = self._robot.lookup_a_tform_b(FrameHint.ODOM, FrameHint.BODY)
        body_t_start_pose = self._robot.lookup_a_tform_b(FrameHint.BODY, FrameHint.HAND, timeout=5.0)
        body_t_end_pose = body_t_start_pose * SE3.Trans([0.3, 0.0, 0.0])

        num_points = 5
        pose_trajectory = []
        timestamps = np.linspace(0.0, 1.5, num_points)
        for alpha in np.linspace(0.0, 1.0, num_points):
            # formula is Asin(x/(2*pi)) + b
            x = body_t_end_pose.x * alpha + body_t_start_pose.x * (1.0 - alpha)
            z = 0.1 * np.sin(2.0 * np.pi * alpha) + body_t_start_pose.z
            pose_trajectory.append(world_t_body * SE3.Trans([x, body_t_start_pose.y, z]))

        self._robot.move_end_effector_along_trajectory.synchronously(
            pose_vecs=pose_trajectory,
            timestamps_between_poses=timestamps,
            axis_mode_map=AxisModeMap.position_mode(),
            root_frame=FrameHint.ODOM,
            max_acceleration=10.0,  # meters/sec^2
            max_linear_velocity=2.0,  # meters/sec
            max_angular_velocity=np.pi / 2,  # radians/sec
            max_pos_tracking_error=0.5,  # meters
            max_rot_tracking_error=np.pi / 3,  # radians
            body_follow_arm=body_follow,
        )

    def start(self) -> int:
        if not self._robot.has_arm():
            self._logger.error("Robot has no arm.")
            return 1

        cobra_pose = SE3.Trans([0.45, 0.0, 0.5])
        with self._robot.leash(timeout_sec=10.0):
            # Go to cobra
            self._logger.info("Going to cobra pose...")
            self._robot.stow_arm.synchronously(timeout_sec=5.0)
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            self._logger.info("Moving the robot hand...")
            pose = self._robot.lookup_a_tform_b(
                FrameHint.BODY, FrameHint.HAND, wait_for_frames=True, timeout=5.0
            ) * SE3.Trans([0.2, 0.2, -0.10])
            self._robot.move_end_effector.synchronously(pose, duration_sec=2.0, base_frame=FrameHint.BODY)

            # Go back to cobra
            self._logger.info("Going back to cobra pose...")
            self._robot.stow_arm.synchronously(timeout_sec=5.0)
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Rotate the hand to -pi/3 in the body frame
            self._logger.info("Rotating the robot hand...")
            pose = self._robot.lookup_a_tform_b(FrameHint.BODY, FrameHint.HAND, timeout=5.0) * SE3.Rx(-np.pi / 3)
            self._robot.move_end_effector.synchronously(pose, duration_sec=2.0, base_frame=FrameHint.BODY)

            # Go back to cobra
            self._logger.info("Going back to cobra pose...")
            self._robot.stow_arm.synchronously(timeout_sec=5.0)
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Translate body and arm
            self._logger.info("Moving arm and body...")
            pose = self._robot.lookup_a_tform_b(
                FrameHint.ODOM, FrameHint.HAND, wait_for_frames=True, timeout=5.0
            ) * SE3.Trans([0.25, -0.3, 0.2])
            self._robot.move_end_effector.synchronously(
                pose,
                base_frame=FrameHint.ODOM,
                duration_sec=2.5,
                body_follow=True,
            )

            # Go back to cobra
            self._logger.info("Going back to cobra pose...")
            self._robot.stow_arm.synchronously(timeout_sec=5.0)
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Rotate body and arm
            self._logger.info("Rotating arm and body...")
            pose = self._robot.lookup_a_tform_b(FrameHint.ODOM, FrameHint.HAND, timeout=5.0) * SE3.Rz(np.pi / 3)
            self._robot.move_end_effector.synchronously(
                pose,
                base_frame=FrameHint.ODOM,
                duration_sec=2.5,
                body_follow=True,
            )

            # Go back to cobra
            self._logger.info("Going back to cobra pose...")
            self._robot.stow_arm.synchronously(timeout_sec=5.0)
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Move arm with a velocity profile
            self._logger.info("Moving hand with an specific velocity profile...")
            start_pose = self._robot.lookup_a_tform_b(FrameHint.BODY, FrameHint.HAND, timeout=5.0)
            end_pose = start_pose * SE3.Trans([0.3, 0.0, 0.0])

            num_points = 100
            timestamps = np.linspace(0.0, 1.0, num_points)
            pose_trajectory = se3_linspace(start_pose, end_pose, num_points)
            twist_trajectory = [Twist3([1.5, 0.0, 0.0], np.zeros(3))] * 50
            twist_trajectory += [Twist3([0.5, 0.0, 0.0], np.zeros(3))] * 50
            self._robot.move_end_effector_along_trajectory.synchronously(
                pose_vecs=pose_trajectory,
                vel_vecs=twist_trajectory,
                timestamps_between_poses=timestamps,
                axis_mode_map=AxisModeMap.position_mode(),
                root_frame=FrameHint.BODY,
                max_acceleration=10.0,  # meters/sec^2
                max_linear_velocity=2.0,  # meters/sec
                max_angular_velocity=np.pi / 2,  # radians/sec
                max_pos_tracking_error=0.5,  # meters
                max_rot_tracking_error=np.pi / 3,  # radians
            )

            # Go back to cobra
            self._logger.info("Going back to cobra pose...")
            self._robot.stow_arm.synchronously(timeout_sec=5.0)
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            self._move_cosine_optional_body_follow(True)

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ArmCartesianTrajectoryExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ArmCartesianTrajectoryExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
