#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from bosdyn.api import (
    geometry_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
)
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
)
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import (
    RobotCommandFeedback,
)
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_utilities.spot_basic.base import SpotBase
from spot_utilities.spot_basic.cam.audio import SpotCAMAudioMixin
from spot_utilities.spot_basic.choreography import SpotChoreographyMixin
from spot_utilities.spot_basic.command import SpotCommandMixin
from spot_utilities.spot_basic.command.arm import SpotArmCommandMixin
from spot_utilities.spot_basic.command.full_body import SpotFullBodyCommandMixin
from spot_utilities.spot_basic.command.gripper import SpotGripperCommandMixin
from spot_utilities.spot_basic.command.mobility import SpotMobilityCommandMixin
from spot_utilities.spot_basic.docking import SpotDockingMixin
from spot_utilities.spot_basic.estop import SpotEStopMixin
from spot_utilities.spot_basic.graph_nav import SpotGraphNavMixin
from spot_utilities.spot_basic.gripper_camera import SpotGripperCameraMixin
from spot_utilities.spot_basic.ik import SpotIKMixin
from spot_utilities.spot_basic.image import SpotImageMixin
from spot_utilities.spot_basic.lease import SpotLeaseMixin
from spot_utilities.spot_basic.manipulation import SpotManipulationMixin
from spot_utilities.spot_basic.power import SpotPowerMixin
from spot_utilities.spot_basic.state import FrameHint, SpotStateMixin
from spot_utilities.spot_basic.utilities import SpotActionHandle
from spot_utilities.spot_basic.world import SpotWorldMixin
from spot_utilities.spot_util import spot_namespaced


class SpotBasic(
    SpotBase,
    SpotLeaseMixin,
    SpotEStopMixin,
    SpotPowerMixin,
    SpotDockingMixin,
    SpotStateMixin,
    SpotCAMAudioMixin,
    SpotGraphNavMixin,
    SpotIKMixin,
    SpotWorldMixin,
    SpotArmCommandMixin,
    SpotGripperCommandMixin,
    SpotGripperCameraMixin,
    SpotMobilityCommandMixin,
    SpotFullBodyCommandMixin,
    SpotChoreographyMixin,
    SpotManipulationMixin,
    SpotCommandMixin,
    SpotImageMixin,
):
    """Base ROS2 Node that provides interfaces to calling various commands"""

    def __init__(
        self,
        robot_name: str,
        *,
        node: Optional[Node] = None,
        disable_collisions: bool = False,
        dry: bool = False,
        rviz: bool = False,
    ):
        """Constructor

        Args:
            robot_name: name of the robot
            node: optional parent node, defaults to the current process node
            disable_collisions: True if collisions should be ignored, False otherwise
            dry: True if this is a dry run and commands should not actually be sent, False otherwise
            rviz: True if visuals should be published, False otherwise
        """
        super().__init__(robot_name, dry=dry, disable_collisions=disable_collisions, node=node)
        self._dry = dry
        self._rviz = rviz

        self._get_parameters_client = self.node.create_client(
            GetParameters, spot_namespaced(self.robot_name, "spot_ros2/get_parameters")
        )

        self.node.get_logger().info(f"Waiting for the 'RobotCommand' action server for {self.robot_name} ...")
        self.execute.wait_for_server()
        self.node.get_logger().info(f"Found the 'RobotCommand' action server for {self.robot_name}.")
        # Check for arm and manipulation client
        if self.has_arm():
            self.node.get_logger().info(f"Waiting for the 'Manipulation' action server for {self.robot_name} ...")
            if self.perform.wait_for_server(1.0):
                self.node.get_logger().info(f"Found the 'Manipulation' action server for {self.robot_name}.")
            else:
                self.node.get_logger().info(
                    f"No 'Manipulation' action server for {self.robot_name} found. Assuming it has no arm."
                )

    def get_set_base_frame(self, base_frame: Optional[FrameHint]) -> FrameHint:
        """Function which returns the provided frame, or the VISION frame if base_frame is not provided"""
        if base_frame is not None:
            return base_frame
        return FrameHint.VISION

    def initialize_robot(self) -> bool:
        """Returns true/false based on success of claiming and powering up Spot."""
        # Claim robot.
        if not self.claim():
            return False

        # Power the robot.
        if not self.power_motors_on():
            return False
        return True

    @staticmethod
    def convert_proto_to_robot_cmd_action_goal(proto: robot_command_pb2.RobotCommand) -> RobotCommandAction.Goal:
        return SpotBasic._convert_proto_to_robot_cmd_action_goal(proto)

    @staticmethod
    def _convert_proto_to_robot_cmd_action_goal(proto: robot_command_pb2.RobotCommand) -> RobotCommandAction.Goal:
        action_goal = RobotCommandAction.Goal()
        convert(proto, action_goal.command)
        return action_goal

    def move_end_effector_force(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        raise NotImplementedError()

    def move_end_effector_force_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """TODO

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        raise NotImplementedError()

    @staticmethod
    def set_tight_top_down_grasp(grasp: manipulation_api_pb2.PickObjectInImage) -> None:
        """A helper method to add constraints to the grasp such that the object
        is grasped from the top, moving down along the vertical axis, with high accuracy
        relative to the user requested position and constraints.
        """

        # Specify the frame we're using.
        grasp.grasp_params.grasp_params_frame_name = BODY_FRAME_NAME

        # The axis on the gripper is the x-axis.
        axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

        # The axis in the frame is the negative z-axis
        axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vision
        )

        # We'll take anything within about 1.0 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = np.deg2rad(1.0)

        grasp.grasp_params.position_constraint.CopyFrom(
            manipulation_api_pb2.GRASP_POSITION_CONSTRAINT_FIXED_AT_USER_POSITION
        )
