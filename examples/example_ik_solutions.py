#! /usr/bin/env python3
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import logging
import sys

import numpy as np
import synchros2.process as ros_process
from bosdyn.api.spot import inverse_kinematics_pb2, robot_command_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_msgs.conversions import convert
from spot_basic import FrameHint, SpotBasic

from bdai_ros.utilities.manual_conversions import se3_to_se3pose  # FIXME


class IKSolutionsExample:
    """
    Query and apply IK solutions to Spot arm.

    This example shows how to solve IK for Spot arm and have it
    execute it. You need a Spot with an arm. The Spot driver must
    be running too before executing this.
    """

    def __init__(self, args: argparse.Namespace):
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._num_poses: int = args.num_poses

    def start(self) -> int:
        with self._robot.leash(timeout_sec=10.0):
            self._robot.ready_arm.synchronously(timeout_sec=10.0)

            # Look for known transforms published by the robot.
            odom_T_flat_body: SE3Pose = se3_to_se3pose(
                self._robot.lookup_a_tform_b(
                    FrameHint.ODOM, FrameHint.GRAV_ALIGNED_BODY, wait_for_frames=True, timeout=10.0
                )
            )
            odom_T_ground_plane: SE3Pose = se3_to_se3pose(
                self._robot.lookup_a_tform_b(FrameHint.ODOM, FrameHint.GROUND_PLANE, wait_for_frames=True, timeout=10.0)
            )

            # Construct the frame on the ground right underneath the center of the body.
            odom_T_ground_body = SE3Pose(
                odom_T_flat_body.x, odom_T_flat_body.y, odom_T_ground_plane.z, odom_T_flat_body.rot
            )

            # Now, construct a task frame slightly above the ground, in front of the robot.
            odom_T_task: SE3Pose = odom_T_ground_body * SE3Pose(0.4, 0.0, 0.05, Quat())

            # Now, let's set our tool frame to be the tip of the robot's bottom jaw. Flip the
            # orientation so that when the hand is pointed downwards, the tool's z-axis is
            # pointed upward.
            wr1_T_tool: SE3Pose = SE3Pose(0.23589, 0.0, -0.03943, Quat.from_pitch(-np.pi / 2))

            # Generate several random poses in front of the task frame where we want the tool to move to.
            # The desired tool poses are defined relative to thr task frame in front of the robot and slightly
            # above the ground. The task frame is aligned with the "gravity aligned body frame", such that
            # the positive-x direction is to the front of the robot, the positive-y direction is to the left
            # of the robot, and the positive-z direction is opposite to gravity.
            x_size = 0.6  # m
            y_size = 0.6  # m
            x_rt_task = x_size * np.random.random(self._num_poses)
            y_rt_task = -y_size / 2 + y_size * np.random.random(self._num_poses)
            task_T_desired_tools = [
                SE3Pose(xi_rt_task, yi_rt_task, 0.0, Quat())
                for (xi_rt_task, yi_rt_task) in zip(x_rt_task.flatten(), y_rt_task.flatten(), strict=False)
            ]

            # Define a stand command that we'll send if the IK service does not find a solution.
            body_control = robot_command_pb2.BodyControlParams(
                body_assist_for_manipulation=robot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                    enable_hip_height_assist=True, enable_body_yaw_assist=True
                )
            )
            body_assist_enabled_stand_command = RobotCommandBuilder.synchro_stand_command(
                params=robot_command_pb2.MobilityParams(body_control=body_control)
            )

            # Send IK requests.
            for task_T_desired_tool in task_T_desired_tools:
                self._logger.info(f"Taking tool to {task_T_desired_tool}")
                ik_request = inverse_kinematics_pb2.InverseKinematicsRequest(
                    root_frame_name=ODOM_FRAME_NAME,
                    scene_tform_task=odom_T_task.to_proto(),
                    wrist_mounted_tool=inverse_kinematics_pb2.InverseKinematicsRequest.WristMountedTool(
                        wrist_tform_tool=wr1_T_tool.to_proto()
                    ),
                    tool_pose_task=inverse_kinematics_pb2.InverseKinematicsRequest.ToolPoseTask(
                        task_tform_desired_tool=task_T_desired_tool.to_proto()
                    ),
                )
                response = self._robot.get_inverse_kinematic_solutions.synchronously(ik_request, timeout_sec=10.0)
                ik_response = inverse_kinematics_pb2.InverseKinematicsResponse()
                convert(response.response, ik_response)
                # Attempt to move to each of the desired tool pose to check the IK results.
                if ik_response.status == inverse_kinematics_pb2.InverseKinematicsResponse.Status.STATUS_OK:
                    self._logger.info("IK solution found")

                    odom_T_desired_body = get_a_tform_b(
                        ik_response.robot_configuration.transforms_snapshot,
                        ODOM_FRAME_NAME,
                        BODY_FRAME_NAME,
                    )
                    mobility_params = robot_command_pb2.MobilityParams(
                        body_control=robot_command_pb2.BodyControlParams(
                            body_pose=RobotCommandBuilder.body_pose(ODOM_FRAME_NAME, odom_T_desired_body.to_proto())
                        )
                    )
                    stand_command = RobotCommandBuilder.synchro_stand_command(params=mobility_params)
                elif (
                    ik_response.status
                    == inverse_kinematics_pb2.InverseKinematicsResponse.Status.STATUS_NO_SOLUTION_FOUND
                ):
                    self._logger.info("No IK solution found")
                    stand_command = body_assist_enabled_stand_command
                else:
                    self._logger.info("IK solver in unknown state, ignoring")
                    stand_command = body_assist_enabled_stand_command

                # Move the arm tool to the requested position.
                command = RobotCommandBuilder.arm_pose_command_from_pose(
                    hand_pose=(odom_T_task * task_T_desired_tool).to_proto(),
                    frame_name=ODOM_FRAME_NAME,
                    build_on_command=stand_command,
                    seconds=1,
                )
                arm_command = command.synchronized_command.arm_command
                arm_command.arm_cartesian_command.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())
                try:
                    self._robot.execute.synchronous(command, timeout_sec=5.0)
                except Exception as e:
                    self._logger.error(str(e))

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=IKSolutionsExample.__doc__)
    parser.add_argument("-n", "--num-poses", type=int, default=50, help="Number of desired tool poses to query")
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = IKSolutionsExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
