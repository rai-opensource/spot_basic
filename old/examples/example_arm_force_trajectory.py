#!/usr/bin/env python
# Copyright (c) 2023-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import synchros2.process as ros_process
from bosdyn.api import arm_command_pb2
from spatialmath import SE3, SpatialForce
from spot_utilities.spot_basic import FrameHint, SpotBasic
from spot_utilities.spot_basic.command.arm import AxisModeMap

AXIS_MODE_FORCE = arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE


class ArmForceTrajectoryExample:
    """
    Spot moves its arm around with controlled forces.

    When running this example, Spot will perform several motions with its arm end effector.

    Make sure you have a Spot with an arm and there is enough free space (about ~2 m) around it.
    The Spot driver must be started before executing this example too.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        if not self._robot.has_arm():
            self._logger.error("Robot has no arm.")
            return 1

        cobra_pose = SE3.Trans([0.45, 0.0, 0.5])
        with self._robot.leash(timeout_sec=10.0):
            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            self._robot.move_end_effector_cartesian_trajectory.synchronously(
                timestamps_between_wrenches=[1.5],
                wrench_vecs=[SpatialForce([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])],
                axis_mode_map=dataclasses.replace(AxisModeMap.position_mode(), z=AXIS_MODE_FORCE),
                root_frame=FrameHint.BODY,
                timeout_sec=10.0,
            )

            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Linearly move from 1 to 3 Newtons and back over 4 seconds
            self._robot.move_end_effector_cartesian_trajectory.synchronously(
                wrench_vecs=[SpatialForce([fx, 0.0, 0.0, 0.0, 0.0, 0.0]) for fx in (1.0, 2.0, 3.0, 2.0, 1.0)],
                timestamps_between_wrenches=[0.0, 0.5, 1.0, 1.5, 2.0],
                axis_mode_map=dataclasses.replace(AxisModeMap.position_mode(), x=AXIS_MODE_FORCE),
                root_frame=FrameHint.BODY,
                timeout_sec=5.0,
            )

            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Hybrid trajectory
            # Move the arm in the x-y plane relative to the body and apply a force in z
            # For this particular command, you want to set the other axis in position mode
            # or else the position of the end effector will drift
            self._robot.move_end_effector_cartesian_trajectory.synchronously(
                pose_vecs=[SE3([0.45, 0.0, 0.55]), SE3([0.65, 0.0, 0.55])],
                wrench_vecs=[SpatialForce([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])],
                timestamps_between_poses=[0.5, 1.0],
                timestamps_between_wrenches=[1.0],
                axis_mode_map=dataclasses.replace(AxisModeMap.position_mode(), z=AXIS_MODE_FORCE),
                root_frame=FrameHint.BODY,
                timeout_sec=5.0,
            )

            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            # Test a torque
            self._robot.move_end_effector_cartesian_trajectory.synchronously(
                timestamps_between_wrenches=[1.5],
                wrench_vecs=[SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
                axis_mode_map=dataclasses.replace(AxisModeMap.position_mode(), rz=AXIS_MODE_FORCE),
                root_frame=FrameHint.BODY,
                timeout_sec=5.0,
            )

            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            self._robot.move_end_effector_cartesian_trajectory.synchronously(
                timestamps_between_wrenches=[1.5],
                wrench_vecs=[SpatialForce([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])],
                axis_mode_map=dataclasses.replace(AxisModeMap.position_mode(), x=AXIS_MODE_FORCE, ry=AXIS_MODE_FORCE),
                root_frame=FrameHint.BODY,
                timeout_sec=5.0,
            )

            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

            self._robot.move_end_effector_cartesian_trajectory.synchronously(
                pose_vecs=[SE3([0.45, 0.0, 0.55]), SE3([0.65, 0.0, 0.55])],
                wrench_vecs=[SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])],
                timestamps_between_poses=[0.5, 1.0],
                timestamps_between_wrenches=[1.0],
                axis_mode_map=dataclasses.replace(AxisModeMap.position_mode(), rz=AXIS_MODE_FORCE),
                root_frame=FrameHint.BODY,
                timeout_sec=5.0,
            )

            self._robot.move_end_effector.synchronously(cobra_pose, duration_sec=0.5, base_frame=FrameHint.BODY)

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ArmForceTrajectoryExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ArmForceTrajectoryExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
