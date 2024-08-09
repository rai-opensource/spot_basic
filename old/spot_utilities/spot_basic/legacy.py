#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
from typing import Callable, Optional

import bdai_ros2_wrappers.process as ros_process
import numpy as np
import rclpy
from bdai_ros.utilities.manual_conversions import (
    to_ros_pose,
)
from bdai_ros2_wrappers.type_hints import Action, SrvTypeRequest
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
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future
from spatialmath import SE2, SE3
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_msgs.srv import (
    OverrideGraspOrCarry,
)
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
from spot_utilities.spot_basic.utilities import SpotActionHandle, SpotServiceHandle
from spot_utilities.spot_basic.world import SpotWorldMixin
from spot_utilities.spot_util import spot_namespaced
from std_msgs.msg import ColorRGBA as Color
from visualization_msgs.msg import Marker


class SpotBasic(
    SpotBase,
    SpotLeaseMixin,
    SpotEStopMixin,
    SpotImageMixin,
    SpotPowerMixin,
    SpotDockingMixin,
    SpotStateMixin,
    SpotArmCommandMixin,
    SpotGripperCommandMixin,
    SpotGripperCameraMixin,
    SpotMobilityCommandMixin,
    SpotFullBodyCommandMixin,
    SpotCommandMixin,
    SpotGraphNavMixin,
    SpotChoreographyMixin,
    SpotCAMAudioMixin,
    SpotManipulationMixin,
    SpotIKMixin,
    SpotWorldMixin,
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

        if self._rviz:
            self._vis_pub = self.node.create_publisher(Marker, "/visualization_marker", 10)

        self._get_parameters_client = self.node.create_client(
            GetParameters, spot_namespaced(self.robot_name, "spot_ros2/get_parameters")
        )
        self._override_grasp_carry_client = self.node.create_client(
            OverrideGraspOrCarry, spot_namespaced(self.robot_name, "override_grasp_or_carry")
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

    def _send_action_goal_async(
        self,
        command_name: str,
        goal: Action.Goal,
        action_client: ActionClient,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        if goal is None:
            # Is there something better that we can return with information about the failure?
            return None
        if self._dry:
            # TODO: What to return on a dry run? Just success?
            return None

        handle = SpotActionHandle(self._robot_name, command_name, context=self.node.context)
        if result_callback is not None:
            handle.set_result_callback(result_callback)
        if on_failure_callback is not None:
            handle.set_on_failure_callback(on_failure_callback)

        if feedback_callback is None:
            send_goal_future = action_client.send_goal_async(goal)
        else:
            handle.set_feedback_callback(feedback_callback)
            send_goal_future = action_client.send_goal_async(goal, feedback_callback=handle.get_feedback_callback)
        handle.set_send_goal_future(send_goal_future)

        return handle

    def _send_service_request_async(
        self,
        service_name: str,
        request: SrvTypeRequest,
        service_client: Client,
        *,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> SpotServiceHandle:
        """Sends an asynchronous service request to the service client and handles the response.

        Args:
            service_name: name of the service we are requesting from
            request: the request we are sending to the service
            service_client: the client used to request the service
            result_callback: callback executing on the future returned from a service call
            on_failure_callback: callback executing if a service fails

        Raises:
            ValueError if a request is None
        """
        if request is None:
            raise ValueError("Request must not be None!")

        handle = SpotServiceHandle(robot_name=self._robot_name, service_name=service_name, context=self.node.context)

        if result_callback is not None:
            handle.set_result_callback(result_callback)

        if on_failure_callback is not None:
            handle.set_on_failure_callback(on_failure_callback=on_failure_callback)

        service_future = service_client.call_async(request)
        handle.set_send_service_future(service_future)
        return handle

    def _call_service_async(
        self,
        service_name: str,
        request: SrvTypeRequest,
        service_client: Client,
        *,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> SpotServiceHandle:
        """Triggers a service asynchronous

        Args:
            service_name: A representative name of the service for logging
            request: The request we are sending to the service server
            service_client: the client we want to trigger the request for
            on_failure_callback: A callback to process/handle when the request fails for various reasons

        Returns:
            A handle for the asynchronous lifecycle of the service
        """
        self.node.get_logger().info(f"Trigger command '{service_name}' for {self.robot_name}")
        return self._send_service_request_async(
            service_name=service_name,
            request=request,
            service_client=service_client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
        )

    def _draw_arrow(self, frame_id: str, ns: str, frame_t_arrow: SE2 | SE3, color: Color) -> None:
        """Draw an arrow in RVIZ"""
        marker = Marker()
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.id = 0
        marker.ns = ns
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = to_ros_pose(frame_t_arrow)
        marker.color = color
        marker.scale.x = 0.3
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        self._vis_pub.publish(marker)

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


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    spot_basic = SpotBasic(args.robot_name)

    while rclpy.ok():
        cmd = input("Please enter a command:\n" + " ".join(SpotBasic.TRIGGER_SERVICES) + " exit\n> ")
        if cmd == "exit":
            break
        success = spot_basic.trigger_service(cmd)
        if not success:
            print(f"Error with command: {cmd}")
        else:
            print(f"Successfully executed command: {cmd}")


if __name__ == "__main__":
    main()
