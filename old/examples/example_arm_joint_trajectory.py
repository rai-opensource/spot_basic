#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import numpy as np
import synchros2.process as ros_process
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.state import ArmJointMap


class ArmJointTrajectoryExample:
    """
    Spot moves its arm around.

    To run this example, make sure you have a Spot with an arm.
    The Spot driver must be started before executing it too.
    """

    def __init__(self, args: argparse.Namespace):
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        if not self._robot.has_arm():
            self._logger.error("Robot has no arm")
            return 1

        num_arm_joints = 6
        max_acceleration = 100.0  # rad/sec^2
        max_velocity = 10.0  # rad/sec
        start_joint_position = ArmJointMap.from_list([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, 0.0, -np.pi / 2])
        with self._robot.leash(timeout_sec=10.0):
            self._logger.info("Moving arm...")
            self._robot.move_arm.synchronously(
                start_joint_position, max_acceleration=max_acceleration, max_velocity=max_velocity, timeout_sec=10.0
            )

            joint_trajectory = []
            for i in range(2 * num_arm_joints):
                joint_position_coords = list(start_joint_position.to_list())
                joint_position_coords[i // 2] += 0.2 if i % 2 else -0.2
                joint_trajectory.append(ArmJointMap.from_list(joint_position_coords))
            times = [i + 1 for i in range(2 * num_arm_joints)]

            self._logger.info("Exercising each arm joint...")
            for joint_position in joint_trajectory:
                self._robot.move_arm.synchronously(
                    joint_position, max_acceleration=max_acceleration, max_velocity=max_velocity, timeout_sec=10.0
                )

            self._logger.info("Smooth out arm joint perturbations...")
            self._robot.move_arm_along_trajectory.synchronously(
                joint_trajectory[:10],
                times[:10],
                max_acceleration=max_acceleration,
                max_velocity=max_velocity,
                timeout_sec=times[-1],
            )

            self._logger.info("Go back to the initial arm configuration...")
            self._robot.move_arm.synchronously(
                start_joint_position, max_acceleration=max_acceleration, max_velocity=max_velocity, timeout_sec=10.0
            )

            wrist1_joint_reposition = dataclasses.replace(start_joint_position, wr1=start_joint_position.wr1 + 0.15)
            wrist1_joint_velocity = ArmJointMap(sh0=0.0, sh1=0.0, el0=0.0, el1=0.0, wr0=0.0, wr1=0.15)
            wrist0_joint_reposition = dataclasses.replace(
                wrist1_joint_reposition, wr0=wrist1_joint_reposition.wr0 + 0.25
            )
            wrist0_joint_velocity = ArmJointMap(sh0=0.0, sh1=0.0, el0=0.0, el1=0.0, wr0=0.25, wr1=0.0)

            self._logger.info("Make controlled wrist movements...")
            self._robot.move_arm_along_trajectory.synchronously(
                joint_positions=[wrist1_joint_reposition, wrist0_joint_reposition],
                joint_velocities=[wrist1_joint_velocity, wrist0_joint_velocity],
                max_acceleration=max_acceleration,
                max_velocity=max_velocity,
                times=[1.0, 2.0],
                timeout_sec=5.0,
            )

            self._logger.info("Done!")

            self._robot.stow_arm.synchronously(timeout_sec=10.0)

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ArmJointTrajectoryExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ArmJointTrajectoryExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
