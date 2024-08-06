#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
from typing import Callable, List, Optional, Union

import bdai_ros2_wrappers.process as ros_process
import numpy as np
import rclpy
from bdai_ros.utilities.manual_conversions import (
    to_ros_pose,
)
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.type_hints import Action, SrvTypeRequest
from bosdyn.api import (
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
)
from bosdyn.api.manipulation_api_pb2 import ApiGraspOverride
from bosdyn.api.robot_state_pb2 import ManipulatorState
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.spot.inverse_kinematics_pb2 import InverseKinematicsRequest, InverseKinematicsResponse
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
)
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
)
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import (
    GripperCameraParams,
    ListWorldObjectRequest,
    ListWorldObjectResponse,
    ManipulationApiFeedbackResponse,
    RobotCommandFeedback,
)
from geometry_msgs.msg import Twist
from google.protobuf import wrappers_pb2
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sklearn.preprocessing import normalize
from spatialmath import SE2, SE3, SpatialForce, Twist2
from spatialmath.base.types import R3
from spatialmath.spatialvector import SpatialVector
from spot_msgs.action import Manipulation
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_msgs.srv import (
    GetGripperCameraParameters,
    GetInverseKinematicSolutions,
    ListWorldObjects,
    OverrideGraspOrCarry,
    SetGripperCameraParameters,
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
from spot_utilities.spot_basic.image import SpotImageMixin
from spot_utilities.spot_basic.lease import SpotLeaseMixin
from spot_utilities.spot_basic.power import SpotPowerMixin
from spot_utilities.spot_basic.state import FrameHint, SpotStateMixin
from spot_utilities.spot_basic.utilities import RobotComposeCommandHandle, SpotActionHandle, SpotServiceHandle
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
    SpotMobilityCommandMixin,
    SpotFullBodyCommandMixin,
    SpotCommandMixin,
    SpotGraphNavMixin,
    SpotChoreographyMixin,
    SpotCAMAudioMixin,
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

        self._velocity_cmd_publisher = self.node.create_publisher(
            Twist, spot_namespaced(self.robot_name, "cmd_vel"), 10
        )

        self._ik_client = self.node.create_client(
            GetInverseKinematicSolutions, spot_namespaced(self.robot_name, "get_inverse_kinematic_solutions")
        )
        self._list_world_objects_client = self.node.create_client(
            ListWorldObjects, spot_namespaced(self.robot_name, "list_world_objects")
        )
        self._get_parameters_client = self.node.create_client(
            GetParameters, spot_namespaced(self.robot_name, "spot_ros2/get_parameters")
        )
        self.get_gripper_camera_parameters_client = self.node.create_client(
            GetGripperCameraParameters, spot_namespaced(self.robot_name, "get_gripper_camera_parameters")
        )
        self.set_gripper_camera_parameters_client = self.node.create_client(
            SetGripperCameraParameters, spot_namespaced(self.robot_name, "set_gripper_camera_parameters")
        )
        self._override_grasp_carry_client = self.node.create_client(
            OverrideGraspOrCarry, spot_namespaced(self.robot_name, "override_grasp_or_carry")
        )

        self.node.get_logger().info(f"Waiting for the 'RobotCommand' action server for {self.robot_name} ...")
        self.execute.wait_for_server()
        self.node.get_logger().info(f"Found the 'RobotCommand' action server for {self.robot_name}.")
        # Check for arm and manipulation client
        if self.has_arm():
            self._manipulation_action_client = ActionClientWrapper(
                Manipulation, spot_namespaced(self.robot_name, "manipulation"), self.node
            )
            self.node.get_logger().info(f"Waiting for the 'Manipulation' action server for {self.robot_name} ...")
            if self._manipulation_action_client.wait_for_server(1.0):
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

    def get_inverse_kinematic_solutions(self, ik_request: InverseKinematicsRequest) -> InverseKinematicsResponse:
        """Query an IK solution for a given pose.
        Returns:
            A solution to the IK query, if any.
        """
        self.node.get_logger().info(f"Query IK solutions on {self.robot_name}")
        request = GetInverseKinematicSolutions.Request()
        convert(ik_request, request.request)
        future = self._ik_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return None
        response = future.result()

        proto = InverseKinematicsResponse()
        convert(response.response, proto)
        return proto

    def list_world_objects(self, p_request: ListWorldObjectRequest) -> Optional[ListWorldObjectResponse]:
        """Requests a list of world objects

        When constructing the calling ListWorldObjectRequest, you must follow the correct typing for
        requested objects, which is somewhat non-intuitive.

        # imports required for below
        from bosdyn_msgs.msg import ListWorldObjectRequest, WorldObjectType
        from bosdyn.api import world_object_pb2

        # must be constructed explicitly, WorldObjectType.WORLD_OBJECT_DOCK is also not the same
        world_object_dock_type = WorldObjectType(value=world_object_pb2.WORLD_OBJECT_DOCK)

        # add any requested objects to the list of object types (or leave it blank for all)
        request_fiducials = ListWorldObjectRequest()
        request_fiducials.object_type.append(world_object_dock_type)

        TODO: make this more robust and user intuitive
        """
        self.node.get_logger().info(f"List world objects for {self.robot_name}")
        request = ListWorldObjects.Request()
        request.request = p_request
        future = self._list_world_objects_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return None
        response = future.result()
        return response.response

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

    def manipulation(
        self, command_name: str, goal: Manipulation.Goal, timeout_sec: Optional[float] = None
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Synchronously run a manipulation request

        Args:
            command_name (str): A representative name of the command for logging
            goal (Manipulation.Goal): The Action Goal sent to the action server
            timeout_sec (Optional[float]): A timeout for waiting on a response/result from the action server. Setting to
                None creates no timeout

        Returns:
            Optional[ManipulationApiFeedbackResponse]: The feedback from the SpotDriver about the result of the command
                if it was accepted and a result was successfully received, None otherwise

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if not self.has_arm():
            self.node.get_logger().error("Cannot send Manipulation api request, robot has no arm.")
            return None

        failed = False

        def _on_failure() -> None:
            nonlocal failed
            failed = True

        handle = self._send_action_goal_async(
            command_name=command_name,
            goal=goal,
            action_client=self._manipulation_action_client,
            on_failure_callback=_on_failure,
        )
        if handle is None:
            return None
        if not handle.wait_for_result(timeout_sec=timeout_sec):
            return None
        if failed:
            return None
        return handle.result

    def manipulation_async(
        self,
        command_name: str,
        goal: Manipulation.Goal,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        if not self.has_arm():
            self.node.get_logger().error("Cannot send Manipulation api request, robot has no arm.")
            return None
        return self._send_action_goal_async(
            command_name,
            goal,
            self._manipulation_action_client,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
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

    @staticmethod
    def _convert_proto_to_manip_cmd_action_goal(
        proto: manipulation_api_pb2.ManipulationApiRequest,
    ) -> Manipulation.Goal:
        action_goal = Manipulation.Goal()
        convert(proto, action_goal.command)
        return action_goal

    @staticmethod
    def _activate_body_assist(hip_assist: bool, yaw_assist: bool) -> spot_command_pb2.RobotCommand:
        """Creates a command that allows the body to assist the movements.

        Args:
            hip_assist: whether or not the hips can raise and lower to assist whole-body motions
            yaw_assist: whether or not the body can twist in the yaw to assist whole-body motions

        Returns:
            A robot stand command that can be composed with another command to activate the body assist.
        """
        body_control_params = spot_command_pb2.MobilityParams(
            body_control=spot_command_pb2.BodyControlParams(
                body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                    enable_hip_height_assist=hip_assist, enable_body_yaw_assist=yaw_assist
                )
            )
        )
        return RobotCommandBuilder.synchro_stand_command(params=body_control_params)

    def _body_gaze_at_frame_t_body_desired(
        self,
        base_frame: FrameHint,
        body_t_target: SE3,
        frame_t_body: Optional[SE3] = None,
    ) -> Optional[RobotCommandFeedback]:
        """
        Args:
            base_frame: Frame the goal is specified in
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            frame_t_body: SE3 transform from the frame from to the current body position

        Return:
            frame_t_body_desired: body_t_target: SE3 transform to the target you want to gaze at
        """
        x_axis = np.array([1, 0, 0])
        new_vec_norm = normalize([body_t_target.t])[0]

        # actual interesting math to compute angle and axis
        normal_axis = np.cross(x_axis, new_vec_norm)
        angle = np.arccos(np.dot(x_axis, new_vec_norm))
        body_current_t_body_pointed_at_target = SE3.AngleAxis(angle, normal_axis)

        if frame_t_body is None:
            frame_t_body = self.lookup_a_tform_b(base_frame.value, FrameHint.BODY.value, wait_for_frames=True)

        frame_t_body_desired = frame_t_body * body_current_t_body_pointed_at_target
        return frame_t_body_desired

    def body_gaze_at(
        self,
        body_t_target: SE3,
        frame_t_body: Optional[SE3] = None,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Aligns the x-axis of the body frame along the vector from the current body position
        to the target. If you don't provide vision_t_body, you must set the tf listener for spot.

        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            frame_t_body: SE3 transform from the base frame from to the current body position
            timeout_sec: A timeout for waiting on a response/result from the action server. The timeout arg is used
                on each command, so max duration is 2x timeout. Setting to None creates no timeout.
            base_frame: The frame in which the goal is specified. Defaults to VISION frame in the get_set_base_frame

        Returns:
            The feedback from the SpotDriver about the result of the commands if it was accepted, None otherwise
        """
        base_frame = self.get_set_base_frame(base_frame)
        frame_t_body_desired = self._body_gaze_at_frame_t_body_desired(base_frame, body_t_target, frame_t_body)
        return self.walk_and_orient(frame_t_body_desired, base_frame=base_frame, timeout_sec=timeout_sec)

    def body_gaze_at_async(
        self,
        body_t_target: SE3,
        frame_t_body: Optional[SE3] = None,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[RobotComposeCommandHandle]:
        """Aligns the x-axis of the body frame along the vector from the current body position
        to the target. If you don't provide vision_t_body, you must set the tf listener for spot.

        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            frame_t_body: SE3 transform from the base_frame from to the current body position
            base_frame: Frame the goal is specified in. Defaults to VISION frame in get_set_base_frame
            result_callback: A callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: A callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: A callback to process/handle when any of actions fail for various reasons

        """
        base_frame = self.get_set_base_frame(base_frame)
        frame_t_body_desired = self._body_gaze_at_frame_t_body_desired(base_frame, body_t_target, frame_t_body)
        return self.walk_and_orient_async(
            frame_t_body_desired,
            base_frame=base_frame,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def walk_velocity(
        self, body_velocity: Twist2, duration: Optional[float] = None, rate: Optional[float] = 50.0
    ) -> None:
        """Send the command to walk/turn in a specific body-frame direction at a specific speed

        Args:
            body_velocity: the desired body velocity
            (twist2 is a member of se2, which can be viewed as SE2 velocities, see Twist2.skewa)
            duration: the time (in seconds) to maintain the desired velocity
                setting to None (default) means the robot would walk for
                a minimum duration set by the "cmd_duration" parameter for in spot_ros2.
            rate: Rate (hz) at which velocity command is sent for the given duration.
                No effect if duration is None.
        """
        if duration is None:
            self._publish_velocity_cmd(body_velocity)
        else:
            rate_timer = self.node.create_rate(rate)
            start_time = self.node.get_clock().now()
            dt = self.node.get_clock().now() - start_time
            while dt.nanoseconds * 1e-9 < duration:
                self._publish_velocity_cmd(body_velocity)
                rate_timer.sleep()
                dt = self.node.get_clock().now() - start_time

    def _publish_velocity_cmd(self, body_velocity: Twist2) -> None:
        velocity_cmd_msg = Twist()
        velocity_cmd_msg.linear.x = body_velocity.v[0]
        velocity_cmd_msg.linear.y = body_velocity.v[1]
        velocity_cmd_msg.angular.z = body_velocity.w
        self._velocity_cmd_publisher.publish(velocity_cmd_msg)

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

    def get_gripper_camera_parameters(self) -> tuple[bool, str, GripperCameraParams]:
        """Get the gripper camera parameters on the SpotDriver

        Returns: A tuple containing [success, message, response]
        """
        self.node.get_logger().info(f"Getting gripper camera parameters for {self.robot_name}")
        request = GetGripperCameraParameters.Request()
        future = self.get_gripper_camera_parameters_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out", None
        response = future.result()
        if not response.success:
            self.node.get_logger().warning(
                f"Getting gripper camera parameters for {self.robot_name} failed. Msg: {response.message}."
            )
        return response.success, response.message, response.response.params

    def set_gripper_camera_parameters(self, gripper_camera_params: GripperCameraParams) -> bool:
        """Set the gripper camera parameters on the SpotDriver

        Args:
            gripper_camera_params: the gripper camera parameters and their desired values.
                                    Of type bosdyn_msgs/GripperCameraParams

        Returns: a boolean indicating if the service was successful or not
        """
        self.node.get_logger().info(f"Setting gripper camera parameters for {self.robot_name}")
        request = SetGripperCameraParameters.Request()
        request.request.params = gripper_camera_params
        future = self.set_gripper_camera_parameters_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False
        response = future.result()
        if not response.success:
            self.node.get_logger().warning(
                f"Setting gripper camera parameters for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

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

    def walk_to_object_image(
        self,
        pixel_xy: List[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        offset: Optional[float] = None,
        timeout_sec: Optional[float] = 10.0,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Walk to an object in an image taken with a camera on spot.

        Args:
            pixel_xy: The point in the image where spot will detect the object.
            camera_data: Camera data associated with the image spot took.
            image_timestamp: Uses the timestamp to get the associated transformation tree snapshot.
            offset: Target distance between spot's CoM and the object.
            timeout_sec: Timeout for response from action server

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """

        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_data.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_data.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_data.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_data.k[5]

        args = {
            "pixel_xy": geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1]),
            "transforms_snapshot_for_camera": self._get_basic_tform_tree_snapshot(image_timestamp),
            "frame_name_image_sensor": camera_data.header.frame_id.split("/")[-1],
            "camera_model": pinhole_model,
        }
        if offset is not None:
            args["offset_distance"] = wrappers_pb2.FloatValue(value=offset)
        move_targ = manipulation_api_pb2.WalkToObjectInImage(**args)

        robot_mani_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=move_targ)
        action_goal = self._convert_proto_to_manip_cmd_action_goal(robot_mani_request)
        return self.manipulation("walk_to_object_image", action_goal, timeout_sec)

    def walk_to_object_image_async(
        self,
        pixel_xy: List[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        offset: Optional[float] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Walk to an object in an image taken with a camera on spot asynchronously.

        Args:
            pixel_xy: The point in the image where spot will detect the object.
            camera_data: Camera data associated with the image spot took.
            image_timestamp: Uses the timestamp to get the associated transformation tree snapshot.
            offset: Target distance between spot's CoM and the object.
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """

        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_data.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_data.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_data.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_data.k[5]

        args = {
            "pixel_xy": geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1]),
            "transforms_snapshot_for_camera": self._get_basic_tform_tree_snapshot(image_timestamp),
            "frame_name_image_sensor": camera_data.header.frame_id.split("/")[-1],
            "camera_model": pinhole_model,
        }
        if offset is not None:
            args["offset_distance"] = wrappers_pb2.FloatValue(value=offset)
        move_targ = manipulation_api_pb2.WalkToObjectInImage(**args)

        robot_mani_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=move_targ)
        action_goal = self._convert_proto_to_manip_cmd_action_goal(robot_mani_request)
        return self.manipulation_async(
            "walk_to_object_image",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _grasp_in_image(
        self,
        pixel_xy: List[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable],
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Create protobug for grasping an item in an image taken with a camera on spot.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            camera_data: Camera data associate with the image spot took.
            image_timestamp: Uses the timestamp to get retrieve the associated transformation tree snapshot.
            palm_placement_ratio: a float [0.0, 1.0] that indicates how far
                in the edge of the object is into the palm cavity.
                (0.0 will place the base of palm with camera against the object,
                1.0 will be a 'finger pinch')
            apply_grasp_params_func: A helper function that will apply grasp parameters
                to modify default grasping behavior.
                (see https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=pickobjectinimage#graspparams)

        Returns:
            The protobuf
        """
        # Build manipulation command.
        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_data.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_data.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_data.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_data.k[5]

        pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])

        # Strip away robot name from frame name
        frame_name_image_sensor = camera_data.header.frame_id.split("/")[-1]

        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=self._get_basic_tform_tree_snapshot(image_timestamp),
            frame_name_image_sensor=frame_name_image_sensor,
            camera_model=pinhole_model,
        )

        # We can specify where in the gripper we want to grasp.
        grasp.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio

        if apply_grasp_params_func is not None:
            apply_grasp_params_func(grasp)

        # Build the proto.
        return manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

    def grasp_in_image(
        self,
        pixel_xy: List[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Grasp an item in an image taken with a camera on spot.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            camera_data: Camera data associate with the image spot took.
            image_timestamp: Uses the timestamp to get retrieve the associated transformation tree snapshot.
            palm_placement_ratio: a float [0.0, 1.0] that indicates how far
                in the edge of the object is into the palm cavity.
                (0.0 will place the base of palm with camera against the object,
                1.0 will be a 'finger pinch')
            apply_grasp_params_func: A helper function that will apply grasp parameters
                to modify default grasping behavior.
                (see https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=pickobjectinimage#graspparams)

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """
        grasp_request = self._grasp_in_image(
            pixel_xy, image_timestamp, camera_data, palm_placement_ratio, apply_grasp_params_func
        )
        action_goal = self._convert_proto_to_manip_cmd_action_goal(grasp_request)
        return self.manipulation("grasp_in_image", action_goal, 10.0)

    def grasp_in_image_async(
        self,
        pixel_xy: List[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Grasp an item in an image taken with a camera on spot asynchronously.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            camera_data: Camera data associate with the image spot took.
            image_timestamp: Uses the timestamp to get retrieve the associated transformation tree snapshot.
            palm_placement_ratio: a float [0.0, 1.0] that indicates how far
                in the edge of the object is into the palm cavity.
                (0.0 will place the base of palm with camera against the object,
                1.0 will be a 'finger pinch')
            apply_grasp_params_func: A helper function that will apply grasp parameters
                to modify default grasping behavior.
                (see https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=pickobjectinimage#graspparams)
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """
        grasp_request = self._grasp_in_image(
            pixel_xy, image_timestamp, camera_data, palm_placement_ratio, apply_grasp_params_func
        )

        action_goal = self._convert_proto_to_manip_cmd_action_goal(grasp_request)
        return self.manipulation_async(
            "grasp_in_image",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @staticmethod
    def _walk_to_object_ray_in_world(
        ray_start_rt_frame: R3, ray_end_rt_frame: R3, frame_name: str, offset_distance: float
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Walks the robot up to an object.  Useful to prepare to grasp or manipulate something.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto#L20
        """
        proto = manipulation_api_pb2.WalkToObjectRayInWorld(
            ray_start_rt_frame=geometry_pb2.Vec3(
                x=ray_start_rt_frame[0], y=ray_start_rt_frame[1], z=ray_start_rt_frame[2]
            ),
            ray_end_rt_frame=geometry_pb2.Vec3(x=ray_end_rt_frame[0], y=ray_end_rt_frame[1], z=ray_end_rt_frame[2]),
            frame_name=frame_name,
            offset_distance=wrappers_pb2.FloatValue(value=offset_distance),
        )
        return manipulation_api_pb2.ManipulationApiRequest(walk_to_object_ray_in_world=proto)

    def walk_to_object_ray_in_world(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        offset_distance: float,
        timeout_sec: Optional[float] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Walks the robot up to an object.  Useful to prepare to grasp or manipulate something.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto#L20
        """
        proto = self._walk_to_object_ray_in_world(ray_start_rt_frame, ray_end_rt_frame, frame_name, offset_distance)
        action_goal = self._convert_proto_to_manip_cmd_action_goal(proto)
        return self.manipulation("walk_to_object_ray_in_world", action_goal, timeout_sec)

    def walk_to_object_ray_in_world_async(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        offset_distance: float,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Walks the robot up to an object.  Useful to prepare to grasp or manipulate something.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto#L20
        """
        proto = self._walk_to_object_ray_in_world(ray_start_rt_frame, ray_end_rt_frame, frame_name, offset_distance)
        action_goal = self._convert_proto_to_manip_cmd_action_goal(proto)
        return self.manipulation_async(
            "walk_to_object_ray_in_world",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @staticmethod
    def _pick_object_ray_in_world(
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Cast a ray in the world and pick the first object found along the ray.

        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto
        """
        proto = manipulation_api_pb2.PickObjectRayInWorld(
            ray_start_rt_frame=geometry_pb2.Vec3(
                x=ray_start_rt_frame[0], y=ray_start_rt_frame[1], z=ray_start_rt_frame[2]
            ),
            ray_end_rt_frame=geometry_pb2.Vec3(x=ray_end_rt_frame[0], y=ray_end_rt_frame[1], z=ray_end_rt_frame[2]),
            frame_name=frame_name,
            grasp_params=grasp_params,
            walk_gaze_mode=walk_gaze_mode,
        )
        return manipulation_api_pb2.ManipulationApiRequest(pick_object_ray_in_world=proto)

    def pick_object_ray_in_world(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
        timeout_sec: Optional[float] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Cast a ray in the world and pick the first object found along the ray.

        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto
        """
        proto = self._pick_object_ray_in_world(
            ray_start_rt_frame, ray_end_rt_frame, frame_name, grasp_params, walk_gaze_mode
        )
        action_goal = self._convert_proto_to_manip_cmd_action_goal(proto)
        return self.manipulation("grasp_along_ray", action_goal, timeout_sec)

    def pick_object_ray_in_world_async(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Cast a ray in the world and pick the first object found along the ray.

        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto
        """
        proto = self._pick_object_ray_in_world(
            ray_start_rt_frame, ray_end_rt_frame, frame_name, grasp_params, walk_gaze_mode
        )
        action_goal = self._convert_proto_to_manip_cmd_action_goal(proto)
        return self.manipulation_async(
            "grasp_along_ray",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def zero_g(self) -> Optional[RobotCommandFeedback]:
        """Sets spot arm into gravity compensated mode such that the arm 'renders'
        no apparent gravitational acceleration and low damping.

        Useful for allowing a person to move the arm around freely. Does not compensate
        for additional unmodeled mass (like putting something in the gripper.)

        """
        root_t_task = SE3.Tz(0.3)
        stationary_traj = [(SE3(), 1.0)]

        return self.move_end_effector_impedance(
            root_frame_name=FrameHint.BODY,
            root_t_task=root_t_task,
            wrist_t_tool=SE3(),
            trajectory=stationary_traj,
            stiffness=SpatialVector([0, 0, 0, 0, 0, 0]),
            damping=SpatialVector([0.25, 0.25, 0.25, 0.10, 0.10, 0.10]),
            wrench_nm=SpatialForce([0, 0, 0, 0, 0, 0]),
            timeout_sec=2.0,
        )

    def override_grasp_or_carry(
        self,
        grasp_override: ApiGraspOverride.Override = ApiGraspOverride.Override.OVERRIDE_UNKNOWN,
        carry_override: ManipulatorState.CarryState = ManipulatorState.CarryState.CARRY_STATE_UNKNOWN,
    ) -> OverrideGraspOrCarry.Response:
        """
        Allows overriding of whether spot is grasping something and its behavior on stow.

        Grasp Override:

        The robot's grasp state is whether or not it's holding an object.  Usually the robot is aware of whether or not
        it is grasping something but sometimes (e.g. when the object is small and the gripper is almost closed) it makes
        mistakes.  Grasp override values:
            OVERRIDE_UNKNOWN: If this is set, this function will not request that the grasp state be changed.
            OVERRIDE_HOLDING: Grasp state will be changed to HOLDING
            OVERRIDE_NOT_HOLDING: Grasp state will be changed to NOT HOLDING

        Carry Override:

        The robot's carry state governs what the robot will do when control of the arm is requested to stow (usually by
        a hijack). This only matters if the grasp state is HOLDING (otherwise the arm will always stow) but this
        function will send the request without checking the grasp state and will not alter the grasp state unless
        requested to do so.  Carry override values:
            CARRY_STATE_UNKNOWN: If this is set, this function will not request that the carry state be changed.
            CARRY_STATE_NOT_CARRYABLE: When the arm is requested to stow, it will open the gripper (let go) first.
            CARRY_STATE_CARRIABLE: When the arm is requested to stow, it will not do anything.
            CARRY_STATE_CARRIABLE_AND_STOWABLE: When the arm is requested to stow, it will stow without attempting to
                first release the object.


        """
        request = OverrideGraspOrCarry.Request()
        request.grasp_override.value = grasp_override
        request.carry_override.value = carry_override
        try:
            return self._override_grasp_carry_client.call(request)
        except Exception as e:
            return OverrideGraspOrCarry.Response(success=False, message=f"Service call failed with error {e}")


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
