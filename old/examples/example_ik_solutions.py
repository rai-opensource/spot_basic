#!/usr/bin/python3

# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# We disable Pylint warnings for all Protobuf files which contain objects with
# dynamically added member attributes.
# pylint: disable=no-member

"""
This example shows how to query for IK solutions and move
the robot arm to that solution.
"""

import argparse
import logging

import bdai_ros2_wrappers.process as ros_process
import numpy as np
import spot_driver.conversions as conv
from bdai_ros.utilities.manual_conversions import se3_to_se3pose
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.utilities import fqn, namespace_with
from bosdyn.api.spot import inverse_kinematics_pb2, robot_command_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_command import RobotCommandBuilder
from spot_msgs.action import RobotCommand  # type: ignore
from spot_utilities.spot_basic import SpotBasic


class SpotRunner:
    """
    This example shows how to query for IK solutions and move
    the robot arm to that solution.
    """

    def __init__(self, args: argparse.Namespace):
        self._robot_name: str = args.robot
        self._poses: int = args.poses
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(self._robot_name)
        self._robot_command_client = ActionClientWrapper(RobotCommand, namespace_with(args.robot, "robot_command"))
        self._tf_listener = TFListenerWrapper()

    def _send_ik_request(
        self, odom_T_task: SE3Pose, wr1_T_tool: SE3Pose, task_T_desired_tool: SE3Pose
    ) -> inverse_kinematics_pb2.InverseKinematicsResponse:
        """
        Send an IK request for a given task frame, tool and tool desired pose.
        Returns:
            An IK solution, if any.
        """
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
        ik_reponse = self._robot.get_inverse_kinematic_solutions(ik_request)
        return ik_reponse

    def test_run(self) -> bool:
        """
        Send one or more IK requests, evaluate them and move Spot accordingly.
        Returns:
            True the process runs without errors, False otherwise.
        """

        # Frame names.
        odom_frame_name = namespace_with(self._robot_name, ODOM_FRAME_NAME)
        flat_body_frame_name = namespace_with(self._robot_name, GRAV_ALIGNED_BODY_FRAME_NAME)
        ground_plane_frame_name = namespace_with(self._robot_name, GROUND_PLANE_FRAME_NAME)

        # Wait for the robot to publish the TF state.
        self._tf_listener.wait_for_a_tform_b(odom_frame_name, flat_body_frame_name)

        # Claim robot.
        self._logger.info("Claiming robot")
        result = self._robot.claim()
        if not result:
            self._logger.error("Unable to claim robot")
            return False
        self._logger.info("Claimed robot")

        # Power on robot.
        self._logger.info("Powering robot on")
        result = self._robot.power_on()
        if not result:
            self._logger.error("Unable to power on robot")
            return False

        # Stand up robot.
        self._logger.info("Standing robot up")
        result = self._robot.stand()
        if not result:
            self._logger.error("Robot did not stand")
            return False
        self._logger.info("Successfully stood up.")

        # Unstow the arm.
        self._logger.info("Unstow the arm")
        result = self._robot.ready_arm()
        if not result:
            self._logger.error("Failed to unstow the arm.")
            return False
        self._logger.info("Arm ready.")

        # Look for known transforms published by the robot.
        odom_T_flat_body: SE3Pose = se3_to_se3pose(
            self._tf_listener.lookup_a_tform_b(odom_frame_name, flat_body_frame_name)
        )
        odom_T_gpe: SE3Pose = se3_to_se3pose(
            self._tf_listener.lookup_a_tform_b(odom_frame_name, ground_plane_frame_name)
        )

        # Construct the frame on the ground right underneath the center of the body.
        odom_T_ground_body: SE3Pose = odom_T_flat_body
        odom_T_ground_body.z = odom_T_gpe.z

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
        x_size = 0.8  # m
        y_size = 0.8  # m
        x_rt_task = x_size * np.random.random(self._poses)
        y_rt_task = -y_size / 2 + y_size * np.random.random(self._poses)
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
        for task_T_desired_tool in enumerate(task_T_desired_tools):
            ik_response = self._send_ik_request(
                odom_T_task=odom_T_task, wr1_T_tool=wr1_T_tool, task_T_desired_tool=task_T_desired_tool
            )

            # Attempt to move to each of the desired tool pose to check the IK results.
            stand_command = None
            if ik_response.status == inverse_kinematics_pb2.InverseKinematicsResponse.Status.STATUS_OK:
                self._logger.info("Solution found")

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
            elif ik_response.status == inverse_kinematics_pb2.InverseKinematicsResponse.Status.STATUS_NO_SOLUTION_FOUND:
                self._logger.info("No solution found")
                stand_command = body_assist_enabled_stand_command
            else:
                self._logger.info("Status unknown")
                stand_command = body_assist_enabled_stand_command

            # Move the arm tool to the requested position.
            arm_command = RobotCommandBuilder.arm_pose_command_from_pose(
                hand_pose=(odom_T_task * task_T_desired_tool).to_proto(),
                frame_name=ODOM_FRAME_NAME,
                seconds=1,
                build_on_command=stand_command,
            )
            arm_command.synchronized_command.arm_command.arm_cartesian_command.wrist_tform_tool.CopyFrom(
                wr1_T_tool.to_proto()
            )

            arm_command_goal = RobotCommand.Goal()
            conv.convert_proto_to_bosdyn_msgs_robot_command(arm_command, arm_command_goal.command)
            result = self._robot_command_client.send_goal_and_wait(
                action_name="arm_move_one", goal=arm_command_goal, timeout_sec=5
            )

        # Stow the arm.
        self._logger.info("Stow the arm")
        result = self._robot.stow_arm()
        if not result:
            self._logger.error("Failed to stow the arm.")
            return False
        self._logger.info("Arm stowed.")

        return True


def cli() -> argparse.ArgumentParser:
    """
    Parse all arguments.
    --robot [string]
        The robot name e.g. Opal.
    -n --poses [int]
        Number of desired tool poses to query.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The robot name.")
    parser.add_argument("-n", "--poses", type=int, default=50, help="Number of desired tool poses to query.")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    """
    Execute the example.
    """
    spot_runner = SpotRunner(args)
    spot_runner.test_run()


if __name__ == "__main__":
    main()
