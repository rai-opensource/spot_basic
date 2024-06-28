#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
import os
import tempfile
from enum import IntEnum
from typing import Callable, List, Optional, Sequence, Tuple, Union

import bdai.interactive_choreo.choreo_builders.serialization_utils as serialization_utils
import bdai_ros2_wrappers.process as ros_process
import geometry_msgs
import numpy as np
import rclpy
from bdai_ros.utilities.manual_conversions import (
    se3_to_se3pose_proto,
    to_ros_pose,
)
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.type_hints import Action, SrvTypeRequest
from bosdyn.api import (
    arm_command_pb2,
    basic_command_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    trajectory_pb2,
)
from bosdyn.api.manipulation_api_pb2 import ApiGraspOverride
from bosdyn.api.robot_state_pb2 import ManipulatorState
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.spot.choreography_sequence_pb2 import Animation, ChoreographySequence
from bosdyn.api.spot.inverse_kinematics_pb2 import InverseKinematicsRequest, InverseKinematicsResponse
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
)
from bosdyn.client.robot_command import (
    _CLAW_GRIPPER_CLOSED_ANGLE,
    _CLAW_GRIPPER_OPEN_ANGLE,
    RobotCommandBuilder,
)
from bosdyn.util import seconds_to_duration
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
from spatialmath import SE2, SE3, SpatialForce, Twist2, Twist3, UnitQuaternion
from spatialmath.base.types import R3
from spatialmath.spatialvector import SpatialVector
from spot_msgs.action import ExecuteDance, Manipulation
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_msgs.srv import (
    ChoreographyRecordedStateToAnimation,
    ChoreographyStartRecordingState,
    ChoreographyStopRecordingState,
    DeleteSound,
    GetChoreographyStatus,
    GetGripperCameraParameters,
    GetInverseKinematicSolutions,
    GetVolume,
    GraphNavClearGraph,
    GraphNavGetLocalizationPose,
    GraphNavSetLocalization,
    GraphNavUploadGraph,
    ListAllDances,
    ListAllMoves,
    ListGraph,
    ListSounds,
    ListWorldObjects,
    LoadSound,
    OverrideGraspOrCarry,
    PlaySound,
    SetGripperCameraParameters,
    SetVolume,
    UploadAnimation,
)
from spot_utilities.robot_command_helpers import (
    timestamped_se3_poses_to_se3_trajectory,
    timestamped_spatialforces_to_wrench_protos,
)
from spot_utilities.spot_basic.base import SpotBase
from spot_utilities.spot_basic.command import SpotCommandMixin
from spot_utilities.spot_basic.command.mobility import SpotMobilityCommandMixin
from spot_utilities.spot_basic.docking import SpotDockingMixin
from spot_utilities.spot_basic.estop import SpotEStopMixin
from spot_utilities.spot_basic.lease import SpotLeaseMixin
from spot_utilities.spot_basic.power import SpotPowerMixin
from spot_utilities.spot_basic.state import FrameHint, SpotStateMixin
from spot_utilities.spot_basic.utilities import RobotComposeCommandHandle, SpotActionHandle, SpotServiceHandle
from spot_utilities.spot_dataclasses import ArmJointMap, AxisModeMap
from spot_utilities.spot_util import spot_namespaced
from std_msgs.msg import ColorRGBA as Color
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker


class SpotBasic(
    SpotBase,
    SpotLeaseMixin,
    SpotEStopMixin,
    SpotPowerMixin,
    SpotDockingMixin,
    SpotStateMixin,
    SpotMobilityCommandMixin,
    SpotCommandMixin,
):
    """Base ROS2 Node that provides interfaces to calling various commands"""

    TRIGGER_SERVICES = [
        "self_right",
        "stop_dance",
    ]

    class BatteryChangeDirHint(IntEnum):
        RIGHT = 1
        LEFT = 2

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

        self._trigger_clients = {}
        for command_name in self.TRIGGER_SERVICES:
            self._trigger_clients[command_name] = self.node.create_client(
                Trigger, spot_namespaced(self.robot_name, command_name)
            )
        self._get_choreo_status_client = self.node.create_client(
            GetChoreographyStatus, spot_namespaced(self.robot_name, "get_choreography_status")
        )
        self._start_recording_state_client = self.node.create_client(
            ChoreographyStartRecordingState, spot_namespaced(self.robot_name, "start_recording_state")
        )
        self._stop_recording_state_client = self.node.create_client(
            ChoreographyStopRecordingState, spot_namespaced(self.robot_name, "stop_recording_state")
        )
        self._ik_client = self.node.create_client(
            GetInverseKinematicSolutions, spot_namespaced(self.robot_name, "get_inverse_kinematic_solutions")
        )
        self._upload_animation_client = self.node.create_client(
            UploadAnimation, spot_namespaced(self.robot_name, "upload_animation")
        )
        self._list_all_dances_client = self.node.create_client(
            ListAllDances, spot_namespaced(self.robot_name, "list_all_dances")
        )
        self._list_all_moves_client = self.node.create_client(
            ListAllMoves, spot_namespaced(self.robot_name, "list_all_moves")
        )
        self._recorded_state_to_animation_client = self.node.create_client(
            ChoreographyRecordedStateToAnimation,
            spot_namespaced(self.robot_name, "recorded_state_to_animation"),
        )
        self._list_graph_client = self.node.create_client(ListGraph, spot_namespaced(self.robot_name, "list_graph"))
        self._list_world_objects_client = self.node.create_client(
            ListWorldObjects, spot_namespaced(self.robot_name, "list_world_objects")
        )
        self._get_parameters_client = self.node.create_client(
            GetParameters, spot_namespaced(self.robot_name, "spot_ros2/get_parameters")
        )
        self._graph_nav_upload_graph_client = self.node.create_client(
            GraphNavUploadGraph, spot_namespaced(self.robot_name, "graph_nav_upload_graph")
        )

        self._graph_nav_clear_graph_client = self.node.create_client(
            GraphNavClearGraph, spot_namespaced(self.robot_name, "graph_nav_clear_graph")
        )
        self._graph_nav_set_localization_client = self.node.create_client(
            GraphNavSetLocalization, spot_namespaced(self.robot_name, "graph_nav_set_localization")
        )
        self._graph_nav_get_localization_pose_client = self.node.create_client(
            GraphNavGetLocalizationPose, spot_namespaced(self.robot_name, "graph_nav_get_localization_pose")
        )
        self.get_gripper_camera_parameters_client = self.node.create_client(
            GetGripperCameraParameters, spot_namespaced(self.robot_name, "get_gripper_camera_parameters")
        )
        self.set_gripper_camera_parameters_client = self.node.create_client(
            SetGripperCameraParameters, spot_namespaced(self.robot_name, "set_gripper_camera_parameters")
        )
        self._list_sounds_client = self.node.create_client(ListSounds, spot_namespaced(self.robot_name, "list_sounds"))
        self._load_sound_client = self.node.create_client(LoadSound, spot_namespaced(self.robot_name, "load_sound"))
        self._override_grasp_carry_client = self.node.create_client(
            OverrideGraspOrCarry, spot_namespaced(self.robot_name, "override_grasp_or_carry")
        )
        self._play_sound_client = self.node.create_client(PlaySound, spot_namespaced(self.robot_name, "play_sound"))
        self._delete_sound_client = self.node.create_client(
            DeleteSound, spot_namespaced(self.robot_name, "delete_sound")
        )
        self._get_volume_client = self.node.create_client(GetVolume, spot_namespaced(self.robot_name, "get_volume"))
        self._set_volume_client = self.node.create_client(SetVolume, spot_namespaced(self.robot_name, "set_volume"))

        self._execute_dance_ac = ActionClientWrapper(
            ExecuteDance, spot_namespaced(robot_name, "execute_dance"), self.node
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

    def trigger_service(self, command_name: str) -> bool:
        """Triggers a command service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.node.get_logger().info(f"Trigger command '{command_name}' for {self.robot_name}")
        request = Trigger.Request()
        try:
            future = self._trigger_clients[command_name].call_async(request)
        except KeyError:
            self.node.get_logger().error(f"No trigger command {command_name}")
            return False

        if not wait_for_future(future, context=self.node.context):
            return False

        response = future.result()
        if not response.success:
            self.node.get_logger().error(
                f"Trigger command '{command_name}' for {self.robot_name} failed. Msg: {response.message}"
            )
            return False
        return True

    def trigger_service_async(
        self,
        service_name: str,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> SpotServiceHandle:
        """Triggers a service asynchronously on the SpotDriver

        Args:
            service_name: name of the service we are calling
            result_callback: callback executing on the future returned from a service call
            on_failure_callback: callback executing if a service fails

        Returns:
            A SpotServiceHandle which contains information about the state of the service
        """
        return self._call_service_async(
            service_name=f"{service_name}_async",
            request=Trigger.Request(),
            service_client=self._trigger_clients[service_name],
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
        )

    def self_right(self) -> bool:
        """Triggers the 'self_right' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("self_right")

    def rollover(self, timeout_sec: float = 5.0) -> Optional[RobotCommandFeedback]:
        command = RobotCommandBuilder.selfright_command()
        action_goal = self._convert_proto_to_robot_cmd_action_goal(command)
        return self.command("rollover", action_goal, timeout_sec=timeout_sec)

    def rollover_async(
        self,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Treats self righting as an action instead of a service and asynchronously executes it

        Args:
          result_callback: A callback to process/handle the resulting feedback from executing the command
          feedback_callback: A callback to process/handle the feedback received during the execution of the command
          on_failure_callback: A callback to process/handle when the action fails for various reasons
        """
        command = RobotCommandBuilder.selfright_command()
        action_goal = self._convert_proto_to_robot_cmd_action_goal(command)
        return self.command_async(
            "rollover_async",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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

    def execute_dance(self, path: str, timeout_sec: float = 10) -> tuple[bool, str]:
        """Upload and execute dance file (found at path) through Spot Driver
        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        self.node.get_logger().info(f"Uploading dance at '{path}' to {self.robot_name}")
        execute_dance_goal = ExecuteDance.Goal()
        try:
            dance_file = open(path, "r")
            execute_dance_goal.choreo_file_content = dance_file.read()
            dance_file.close()
        except FileNotFoundError:
            return False, "Specified filepath does not exist!"

        handle = SpotActionHandle(robot_name=self.robot_name, action_name="execute_dance")
        send_goal_future = self._execute_dance_ac.send_goal_async(execute_dance_goal)
        handle.set_send_goal_future(send_goal_future)
        handle.wait_for_result(timeout_sec=timeout_sec)

        result = send_goal_future.result().get_result().result
        if not result.success:
            self.node.get_logger().debug(
                f"Uploading dance '{path}' to '{self.robot_name}' failed. Msg: {result.message}."
            )
            return False, result.message
        return True, "success"

    def execute_dance_proto(
        self, choreography_sequence: ChoreographySequence, timeout_sec: float = 10
    ) -> tuple[bool, str]:
        """Execute a dance sequence through Spot Driver
        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """

        execute_dance_goal = ExecuteDance.Goal()
        execute_dance_goal.choreo_sequence_serialized = choreography_sequence.SerializeToString()

        handle = SpotActionHandle(robot_name=self.robot_name, action_name="execute_dance")
        send_goal_future = self._execute_dance_ac.send_goal_async(execute_dance_goal)
        handle.set_send_goal_future(send_goal_future)

        handle.wait_for_result(timeout_sec=timeout_sec)

        result = send_goal_future.result().get_result().result

        if not result.success:
            self.node.get_logger().debug(f"Executing dance on '{self.robot_name}' failed. Msg: {result.message}.")
            return False, result.message
        return True, "success"

    def stop_dance(self) -> Union[bool, SpotServiceHandle]:
        """Stops a dance sequence through Spot Driver
        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        return self.trigger_service_async("stop_dance")

    def upload_animation(self, move_name: str, path: str) -> tuple[bool, str]:
        """Upload an animation file through spot driver
        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        self.node.get_logger().info(f"Uploading animation at '{path}' to {self.robot_name}")
        request = UploadAnimation.Request()
        try:
            dance_file = open(path, "r")
            request.animation_file_content = dance_file.read()
            request.animation_name = move_name
            dance_file.close()
        except FileNotFoundError:
            return False, "Specified filepath does not exist!"

        future = self._upload_animation_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        if not response.success:
            self.node.get_logger().debug(
                f"Uploading animation '{path}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    def upload_animation_proto(self, animation: Animation) -> tuple[bool, str]:
        """Upload an animation file through spot driver
        Returns:tmp
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        self.node.get_logger().info(f"Uploading animation named '{animation.name}' to {self.robot_name}")
        request = UploadAnimation.Request()
        request.animation_proto_serialized = animation.SerializeToString()
        request.animation_name = animation.name

        future = self._upload_animation_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        if not response.success:
            self.node.get_logger().info(
                f"Uploading proto animation '{animation.name}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    def list_all_dances(self) -> tuple[bool, str, list[str]]:
        """Get list of all the dances uploaded on Spot"""
        self.node.get_logger().info(f"Sending list dances request to {self.robot_name}")
        request = ListAllDances.Request()
        future = self._list_all_dances_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out", []
        response = future.result()
        if not response.success:
            self.node.get_logger().debug(
                f"Request to list dances at '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message, []
        return True, "success", response.dances

    def list_all_moves(self) -> tuple[bool, str, list[str]]:
        """Get list of all the moves uploaded on Spot"""
        self.node.get_logger().info(f"Sending list moves request to {self.robot_name}")
        request = ListAllMoves.Request()
        future = self._list_all_moves_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out", []
        response = future.result()
        if not response.success:
            self.node.get_logger().debug(
                f"Request to list moves at '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message, []
        return True, "success", response.moves

    def get_choreo_status(self) -> tuple[bool, str, int]:
        request = GetChoreographyStatus.Request()
        future = self._get_choreo_status_client.call_async(request)
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out", GetChoreographyStatus.Request.STATUS_UNKNOWN
        response = future.result()
        if not response.success:
            self.node.get_logger().debug(f"Failed to get choreo status. Msg: {response.message}.")
            return False, response.message, GetChoreographyStatus.Request.STATUS_UNKNOWN
        return True, "success", response.status

    def start_recording_state(self, duration: float) -> tuple[bool, str]:
        request = ChoreographyStartRecordingState.Request()
        request.duration_seconds = duration
        future = self._start_recording_state_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        if not response.success:
            self.node.get_logger().error(f"Failed to start recording state. Msg: {response.message}.")
            return False, response.message
        return True, "success"

    def stop_recording_state(self) -> tuple[bool, str]:
        request = ChoreographyStopRecordingState.Request()

        future = self._stop_recording_state_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        if not response.success:
            self.node.get_logger().error(f"Failed to stop recording state. Msg: {response.message}.")
            return False, response.message
        return True, "success"

    def recorded_state_to_animation(self, has_arm: bool) -> tuple[bool, str, Animation]:
        animation_proto = Animation()

        request = ChoreographyRecordedStateToAnimation.Request()
        request.has_arm = has_arm
        future = self._recorded_state_to_animation_client.call_async(request)
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out", animation_proto
        response = future.result()
        if not response.success:
            self.node.get_logger().error(f"Failed to start recording state. Msg: {response.message}.")
            return False, response.message, animation_proto

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = "temporary_animation.cha"
            full_path = os.path.join(temp_dir, filename)
            with open(full_path, "w") as animation_file:
                animation_file.write(response.animation_file_contents)

            animation_proto = serialization_utils.load_animation(full_path)
        return True, "success", animation_proto

    def list_sounds(self) -> tuple[bool, str, list[str]]:
        """
        List sounds loaded on Spot CAM.

        Returns:
            (True, 'Success', [<sound names>]) on success
            (False, <error message>, <ignore>) on failure
        """
        request = ListSounds.Request()
        future = self._list_sounds_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out", []
        response = future.result()
        return response.success, response.message, response.names

    def load_sound(self, name: str, wav_path: str) -> tuple[bool, str]:
        """
        Upload wav file sound from this machine to Spot CAM.

        Returns:
            (True, 'Success') on success
            (False, <error message>) on failure
        """
        request = LoadSound.Request()
        request.name = name
        request.wav_path = wav_path
        future = self._load_sound_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        return response.success, response.message

    def play_sound(self, name: str, volume_multiplier: float) -> tuple[bool, str]:
        """
        Play sound loaded on Spot CAM. Volume should be between 0 and 1. # TODO(tchemel): Is this true?

        Returns:
            (True, 'Success') on success
            (False, <error message>) on failure
        """
        request = PlaySound.Request()
        request.name = name
        request.volume_multiplier = volume_multiplier
        future = self._play_sound_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        return response.success, response.message

    def delete_sound(self, name: str) -> tuple[bool, str]:
        """
        Delete sound loaded on Spot CAM.

        Returns:
            (True, 'Success') on success
            (False, <error message>) on failure
        """
        request = DeleteSound.Request()
        request.name = name
        future = self._delete_sound_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        return response.success, response.message

    def get_volume(self) -> tuple[bool, str, float]:
        """
        Get volume of sound playback on Spot CAM.

        Returns:
            (True, 'Success', <volume between 0 and 1>) on success
            (False, <error message>, <ignore>) on failure
        """
        request = GetVolume.Request()
        future = self._get_volume_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out", 0
        response = future.result()
        return response.success, response.message, response.volume

    def set_volume(self, volume: float) -> tuple[bool, str]:
        """
        Set volume of sound playback on Spot CAM. Volume should be between 0 and 1.

        Returns:
            (True, 'Success') on success
            (False, <error message>) on failure
        """
        request = SetVolume.Request()
        request.volume = volume
        future = self._set_volume_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        return response.success, response.message

    def set_localization(self, method: str, waypoint_id: Optional[str] = None) -> bool:
        """Calls the SetLocalization RPC through spot_ros2.
        The 'waypoint_id' is only necessary if 'method'
        is 'waypoint'"""
        self.node.get_logger().info("Set localization")
        method_choices = {"fiducial", "waypoint"}
        if method not in method_choices:
            raise ValueError(f"Invalid localization method {method}.Must be in {method_choices}")
        request = GraphNavSetLocalization.Request()
        request.method = method
        if request.method == "waypoint":
            request.waypoint_id = waypoint_id
        future = self._graph_nav_set_localization_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False
        response: GraphNavSetLocalization.Response = future.result()
        return response.success

    def get_localization_pose(self) -> Optional[geometry_msgs.msg.Pose]:
        self.node.get_logger().info("Get localization pose")
        request = GraphNavGetLocalizationPose.Request()
        future = self._graph_nav_get_localization_pose_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return None
        response = future.result()
        if not response.success:
            raise RuntimeError(f"Failed to get localization pose. Reason: {response.message}")
        return response.pose

    get_localization = get_localization_pose

    def upload_graph(self, upload_filepath: str) -> bool:
        self.node.get_logger().info("Upload graph")
        request = GraphNavUploadGraph.Request()
        request.upload_filepath = upload_filepath
        future = self._graph_nav_upload_graph_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False
        response = future.result()
        return response.success

    def clear_graph(self) -> bool:
        self.node.get_logger().info("Clear graph")
        request = GraphNavClearGraph.Request()
        future = self._graph_nav_clear_graph_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return False
        response = future.result()
        return response.success

    def list_graph(self, upload_filepath: str) -> List[str]:
        """Request a list of waypoint ids"""
        self.node.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        request = ListGraph.Request()
        request.upload_filepath = upload_filepath
        future = self._list_graph_client.call_async(request)
        if not wait_for_future(future, context=self.node.context):
            return []
        response = future.result()
        return response.waypoint_ids

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

    def _get_battery_change_pose(self, dir_hint: BatteryChangeDirHint) -> RobotCommandAction.Goal:
        proto = RobotCommandBuilder.battery_change_pose_command(int(dir_hint))
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def go_to_battery_change_pose(
        self, dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send command to go to the battery change pose asynchronously

        Args:
            dir_hint: A hint for the direction to roll
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_battery_change_pose(dir_hint)
        return self.command("go_to_battery_change_pose", action_goal, timeout_sec)

    def go_to_battery_change_pose_async(
        self,
        dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send command to go to the battery change pose asynchronously

        Args:
            dir_hint: A hint for the direction to roll
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_battery_change_pose(dir_hint)
        return self.command_async(
            "go_to_battery_change_pose",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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

    def _get_arm_gaze_at_action_goal(self, point: R3, frame: FrameHint) -> RobotCommandAction.Goal:
        """Create the ROS msg to gaze at a point."""
        proto = RobotCommandBuilder.arm_gaze_command(x=point[0], y=point[1], z=point[2], frame_name=frame.value)
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def arm_gaze_at(
        self, point: R3, frame: FrameHint, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Gaze the arm/hand camera at a point relative to a frame.

        Args:
            point: A 3d vector representing a point, relative to a frame.
            frame: A frame hint representing one of spot's default frames, like vision or body.
                The point is relative to this frame.
            timeout_sec: A timeout for waiting on a response/result from the action server.

        Returns:
            The feedback from the SpotDriver about the result of the commands if it was accepted, None otherwise
        """
        return self.command("arm_gaze", self._get_arm_gaze_at_action_goal(point, frame), timeout_sec)

    def arm_gaze_at_async(
        self,
        point: R3,
        frame: FrameHint,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Gaze the arm/hand camera at a point relative to a frame.

        Args:
            point: A 3d vector representing a point, relative to a frame.
            frame: A frame hint representing one of spot's default frames, like vision or body.
                The point is relative to this frame.
            result_callback: A callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: A callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: A callback to process/handle when any of actions fail for various reasons
        """
        action_goal = self._get_arm_gaze_at_action_goal(point, frame)
        return self.command_async(
            "arm_gaze",
            action_goal,
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

    def _get_arm_stow_action_goal(self) -> RobotCommandAction.Goal:
        """Create the ROS msg to stow the arm"""
        proto = RobotCommandBuilder.arm_stow_command()
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def stow_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to stow the arm synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_stow_action_goal()
        return self.command("stow_arm", action_goal, timeout_sec)

    def stow_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to stow the arm asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_stow_action_goal()
        return self.command_async(
            "stow_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_ready_action_goal(self) -> RobotCommandAction.Goal:
        """Create the ROS msg to move the arm to the ready position"""
        proto = RobotCommandBuilder.arm_ready_command()
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def ready_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to move the arm to the ready position synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_ready_action_goal()
        return self.command("ready_arm", action_goal, timeout_sec)

    # Alias
    unstow_arm = ready_arm

    def ready_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to move the arm to the ready position asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_ready_action_goal()
        return self.command_async(
            "ready_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    # Alias
    unstow_arm_async = ready_arm_async

    def _get_arm_carry_action_goal(self) -> RobotCommandAction.Goal:
        """Create the ROS msg to move the arm to the carry position"""
        proto = RobotCommandBuilder.arm_carry_command()
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def carry_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Move the arm to the carry position synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_carry_action_goal()
        return self.command("carry_arm", action_goal, timeout_sec)

    def carry_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Move the arm to the carry position asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_carry_action_goal()
        return self.command_async(
            "carry_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_pose_action_goal(
        self,
        frame_t_goal: SE3,
        move_duration_sec: float,
        base_frame: FrameHint,
        body_follow: bool = False,
        body_hip_assist: bool = False,
        body_yaw_assist: bool = False,
        disable_collisions: bool = False,
    ) -> RobotCommandAction.Goal:
        """Create the ROS Msg for moving the end effector to specific pose.

        Args:
            frame_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            base_frame: frame that the trajectory poses are specified in
            body_follow: If True, the robot's body would automatically follow the arm to achieve the goal
            body_hip_assist: Whether or not to have the robot hips move to assist the end effector movements
            body_yaw_assist: Whether or not to have the robot yaw to assist the end effector movements

        Returns:
            ROS2 Action goal containing information about the end effector pose

        """
        quat = UnitQuaternion(frame_t_goal)
        arm_pose_command = RobotCommandBuilder.arm_pose_command(
            x=frame_t_goal.x,
            y=frame_t_goal.y,
            z=frame_t_goal.z,
            qw=quat.A[0],
            qx=quat.A[1],
            qy=quat.A[2],
            qz=quat.A[3],
            frame_name=base_frame.value,
            seconds=move_duration_sec,
        )
        if body_hip_assist or body_yaw_assist:
            stand_assist_command = SpotBasic._activate_body_assist(
                hip_assist=body_hip_assist, yaw_assist=body_yaw_assist
            )
            synchro_command = RobotCommandBuilder.build_synchro_command(stand_assist_command, arm_pose_command)
        elif body_follow:
            follow_arm_command = RobotCommandBuilder.follow_arm_command()
            synchro_command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_pose_command)
            """
            if disable_collisions:
                print('DISABLING COLLISIONS')
                obstacles = spot_command_pb2.ObstacleParams(
                    disable_vision_body_obstacle_avoidance=True,
                    disable_vision_foot_obstacle_avoidance=True,
                    disable_vision_foot_constraint_avoidance=True,
                    obstacle_avoidance_padding=0.001,
                )
                params = spot_command_pb2.MobilityParams(obstacle_params=obstacles)
                synchro_command.mobility_command.params.Pack(params)
            """
        else:
            stand_command = RobotCommandBuilder.synchro_stand_command()
            synchro_command = RobotCommandBuilder.build_synchro_command(stand_command, arm_pose_command)
        return self._convert_proto_to_robot_cmd_action_goal(synchro_command)

    def move_end_effector_cartesian(
        self,
        frame_t_goal: SE3,
        move_duration_sec: float = 5.0,
        body_follow: bool = False,
        body_hip_assist: bool = False,
        body_yaw_assist: bool = False,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
        disable_collisions: bool = False,
    ) -> Optional[RobotCommandFeedback]:
        """Send command to move the end effector to a specific pose synchronously

        Args:
            frame_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            body_follow: If True, the robot's body would automatically follow the arm to achieve the goal
            body_hip_assist: Whether or not to have the robot hips move to assist the end effector movements
            body_yaw_assist: Whether or not to have the robot yaw to assist the end effector movements
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout
            base_frame: Frame the goal is specified in. Defaults to VISION frame in get_set_base_frame

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        See:
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L1224
        """
        base_frame = self.get_set_base_frame(base_frame)
        action_goal = self._get_arm_pose_action_goal(
            frame_t_goal,
            move_duration_sec,
            base_frame,
            body_follow=body_follow,
            body_hip_assist=body_hip_assist,
            body_yaw_assist=body_yaw_assist,
            disable_collisions=disable_collisions,
        )
        return self.command("move_end_effector_cartesian", action_goal, timeout_sec)

    def move_end_effector_cartesian_async(
        self,
        frame_t_goal: SE3,
        move_duration_sec: float = 5.0,
        body_follow: bool = False,
        body_hip_assist: bool = False,
        body_yaw_assist: bool = False,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
        disable_collisions: bool = False,
    ) -> Optional[SpotActionHandle]:
        """Send command to move the end effector to a specific pose asynchronously

        Args:
            frame_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            body_follow: If True, the robot's body would automatically follow the arm to achieve the goal
            body_hip_assist: Whether or not to have the robot hips move to assist the end effector movements
            body_yaw_assist: Whether or not to have the robot yaw to assist the end effector movements
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons
            base_frame: Frame the goal is specified in. Defaults to VISION frame in get_set_base_frame

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        """
        base_frame = self.get_set_base_frame(base_frame)
        action_goal = self._get_arm_pose_action_goal(
            frame_t_goal,
            move_duration_sec,
            base_frame,
            body_follow=body_follow,
            body_hip_assist=body_hip_assist,
            body_yaw_assist=body_yaw_assist,
            disable_collisions=disable_collisions,
        )
        return self.command_async(
            "move_end_effector_cartesian",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_joint_trajectory_goal(
        self,
        joint_angles: Sequence[ArmJointMap],
        times: Sequence[float],
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        ref_time: Optional[float] = None,
        max_acc: Optional[float] = None,
        max_vel: Optional[float] = None,
    ) -> RobotCommandAction.Goal:
        """Returns the action goal to move the arm along a specified trajectory

        Args:
            joint_angles: List of joint angles to command the arm to move to. Each set of joint
                          angles must be size 6, as there are 6 joints. Max 10 points per BD
            times: List of times we want to reach each joint position relative to the reference time
            joint_velocities: List of joint velocities for the arm to achieve along the trajectory.
            ref_time: reference time to begin trajectory, relative to the robot clock.
            max_acc: maximum acceleration allowed for any joint along the trajectory (rad/s^2)
            max_vel: maximum velocity allowed for any joint along the trajectory (rad/s)

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        Raises:
            Exception if too many points in trajectory (10 is the hard limit by BD).
            Joint velocities, angles, and time do not have the same lengths

        https://github.com/boston-dynamics/spot-sdk/blob/b3e41278a7d6b22ad82a2ec4d7df6d36df029470/protos/bosdyn/api/arm_command.proto#L314
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#armjointtrajectory
        """
        if len(joint_angles) > 10:
            raise ValueError("An ArmJointTrajectory accepts only up to 10 knot points")

        if len(times) != len(joint_angles):
            raise ValueError("Must have the same number of joint angles and timestamps")

        if joint_velocities is not None:
            if len(joint_velocities) != len(joint_angles):
                raise ValueError("Must have the same number of joint angles and velocities")

        # Convert the joint angles to a list
        joint_angles_list = []
        for joint_angle in joint_angles:
            joint_angles_list.append(joint_angle.to_list())
        # Convert the joint velocities to a list
        if joint_velocities is not None:
            joint_velocities_list = []
            for joint_vel in joint_velocities:
                joint_velocities_list.append(joint_vel.to_list())
            robot_cmd = RobotCommandBuilder.arm_joint_move_helper(
                joint_positions=joint_angles_list,
                times=times,
                joint_velocities=joint_velocities_list,
                ref_time=ref_time,
                max_acc=max_acc,
                max_vel=max_vel,
            )
        else:
            robot_cmd = RobotCommandBuilder.arm_joint_move_helper(
                joint_positions=joint_angles_list, times=times, ref_time=ref_time, max_acc=max_acc, max_vel=max_vel
            )
        return self._convert_proto_to_robot_cmd_action_goal(robot_cmd)

    def move_arm_joint_trajectory(
        self,
        joint_angles: Sequence[ArmJointMap],
        times: Sequence[float],
        *,
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        ref_time: Optional[float] = None,
        max_acc: Optional[float] = None,
        max_vel: Optional[float] = None,
        timeout_sec: float = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Move the joint angles of the arm along a specified trajectory.

        Args:
            joint_angles: List of joint angles to command the arm to move to. Each set of joint
                          angles must be size 6, as there are 6 joints. Max 10 points per BD
            times: List of times we want to reach each joint position relative to the reference time
            joint_velocities: List of joint velocities for the arm to achieve along the trajectory.
            ref_time: reference time to begin trajectory, relative to the robot clock.
            max_acc: maximum acceleration allowed for any joint along the trajectory (rad/s^2)
            max_vel: maximum velocity allowed for any joint along the trajectory (rad/s)
            timeout_sec: seconds between receive and execution until the command times out

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        Raises:
            Exception if too many points in trajectory

        https://github.com/boston-dynamics/spot-sdk/blob/b3e41278a7d6b22ad82a2ec4d7df6d36df029470/protos/bosdyn/api/arm_command.proto#L314
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#armjointtrajectory
        """
        action_goal = self._get_arm_joint_trajectory_goal(
            joint_angles=joint_angles,
            times=times,
            joint_velocities=joint_velocities,
            ref_time=ref_time,
            max_acc=max_acc,
            max_vel=max_vel,
        )
        return self.command("move_arm_joint_trajectory", action_goal, timeout_sec)

    def move_arm_joint_trajectory_async(
        self,
        joint_angles: Sequence[ArmJointMap],
        times: Sequence[float],
        *,
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        ref_time: Optional[float] = None,
        max_acc: Optional[float] = None,
        max_vel: Optional[float] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Move the joint angles of the arm along a specified trajectory.

        Args:
            joint_angles: List of joint angles to command the arm to move to. Each set of joint
                          angles must be size 6, as there are 6 joints. Max 10 points per BD
            times: List of times we want to reach each joint position relative to the reference time
            joint_velocities: List of joint velocities for the arm to achieve along the trajectory.
            ref_time: reference time to begin trajectory, relative to the robot clock.
            max_acc: maximum acceleration allowed for any joint along the trajectory (rad/s^2)
            max_vel: maximum velocity allowed for any joint along the trajectory (rad/s)
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        Raises:
            Exception if too many points

        https://github.com/boston-dynamics/spot-sdk/blob/b3e41278a7d6b22ad82a2ec4d7df6d36df029470/protos/bosdyn/api/arm_command.proto#L314
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#armjointtrajectory
        """
        action_goal = self._get_arm_joint_trajectory_goal(
            joint_angles=joint_angles,
            times=times,
            joint_velocities=joint_velocities,
            ref_time=ref_time,
            max_acc=max_acc,
            max_vel=max_vel,
        )
        return self.command_async(
            "move_arm_joint_trajectory_async",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_joint_move_action_goal(
        self, joint_angles: Optional[ArmJointMap], time_since_reference_secs: Optional[float] = None
    ) -> RobotCommandAction.Goal:
        """Converts an arm joint trajectory into an action goal.

        Args:
            joint_angles: the joint angles of the robot, in order from shoulder to wrist.
                          If the joint angles is None, then the arm is locked in place.
            time_since_reference_secs: duration the command needs to execute with respect to the reference

        Returns:
            Action goal representing the arm joint movements
        """
        if joint_angles is None:
            arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[arm_command_pb2.ArmJointTrajectoryPoint()])
        else:
            traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                sh0=joint_angles.sh0,
                sh1=joint_angles.sh1,
                el0=joint_angles.el0,
                el1=joint_angles.el1,
                wr0=joint_angles.wr0,
                wr1=joint_angles.wr1,
                time_since_reference_secs=time_since_reference_secs,
            )
            arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        robot_cmd = robot_command_pb2.RobotCommand()
        joint_move_cmd = robot_cmd.synchronized_command.arm_command.arm_joint_move_command
        joint_move_cmd.trajectory.CopyFrom(arm_joint_traj)
        return self._convert_proto_to_robot_cmd_action_goal(robot_cmd)

    def move_arm_to_joint_angles(
        self,
        joint_angles: Optional[ArmJointMap],
        *,
        time_since_reference_secs: Optional[float] = None,
        timeout_sec: float = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Moves the arm to a specific set of joint angles

        Args:
            joint_angles: the joint angles of the robot, in order from shoulder to wrist.
                          If the joint angles is None, then the arm is locked in place.
            time_since_reference_secs: duration the command needs to execute with respect to the reference
            timeout_sec: seconds between receive and execution until the command times out

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        # Builds the action goal
        action_goal = self._get_joint_move_action_goal(
            joint_angles=joint_angles, time_since_reference_secs=time_since_reference_secs
        )
        return self.command("move_arm_to_joint_angles", action_goal, timeout_sec)

    def move_arm_to_joint_angles_async(
        self,
        joint_angles: Optional[ArmJointMap],
        *,
        time_since_reference_secs: Optional[float] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Moves the arm to a specific set of joint angles

        Args:
            joint_angles: the joint angles of the robot, in order from shoulder to wrist.
                          If the joint angles is None, then the arm is locked in place.
            time_since_reference_secs: duration the command needs to execute with respect to the reference
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_joint_move_action_goal(
            joint_angles=joint_angles, time_since_reference_secs=time_since_reference_secs
        )
        return self.command_async(
            "move_arm_to_joint_angles_async",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def lock_arm(self) -> Optional[RobotCommandFeedback]:
        """Locks the arm joint in place.

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        return self.move_arm_to_joint_angles(joint_angles=None)

    def lock_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Locks the arm joint in place.

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise.
        """
        return self.move_arm_to_joint_angles_async(
            joint_angles=None,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_cartesian_trajectory_action_goal(
        self,
        root_frame_name: str,
        axis_mode_map: AxisModeMap,
        timestamps_between_poses: Optional[np.typing.ArrayLike] = None,
        timestamps_between_wrenches: Optional[np.typing.ArrayLike] = None,
        pose_vecs: Optional[Sequence[SE3]] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        wrench_vecs: Optional[SpatialForce] = None,
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        max_acceleration: Optional[float] = None,
        max_linear_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        max_pos_tracking_error: Optional[float] = None,
        max_rot_tracking_error: Optional[float] = None,
        stay_near_current_joint_angles: Optional[bool] = None,
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        reference_time: Optional[float] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        body_follow_arm: bool = False,
    ) -> RobotCommandAction.Goal:
        """Return a goal corresponding to the arm following a Cartesian trajectory.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid undefined behaviour.

        Args:
            root_frame_name: optional task frame that all trajectories are defined with respect to.
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
            preferred_joint_configuration: the preferred joint configuration for the arm to end up in.
            reference_time: time with respect to the robot clock of when to reference when trajectories start and end
            pose_interpolation: positional interpolation method. Defaults to cubic.
            ang_interpolation: angular interpolation method. Defaults to cubic.
            body_follow_arm: have the body follow the arm when it moves

        Returns:
            Goal commanding the robot's arm to follow the given trajectory

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """
        robot_cmd = robot_command_pb2.RobotCommand()
        arm_cartesian_command = robot_cmd.synchronized_command.arm_command.arm_cartesian_command
        arm_cartesian_command.root_frame_name = root_frame_name

        if wrist_t_tool is not None:
            arm_cartesian_command.wrist_tform_tool.CopyFrom(se3_to_se3pose_proto(wrist_t_tool))

        if root_t_task is not None:
            arm_cartesian_command.root_tform_task.CopyFrom(se3_to_se3pose_proto(root_t_task))

        if pose_vecs is not None:
            arm_trajectory = timestamped_se3_poses_to_se3_trajectory(
                timestamps_between_points=timestamps_between_poses,
                pose_vecs=pose_vecs,
                vel_vecs=vel_vecs,
                pose_interpolation=pose_interpolation,
                ang_interpolation=ang_interpolation,
                reference_time=reference_time,
            )
            arm_cartesian_command.pose_trajectory_in_task.CopyFrom(arm_trajectory)

        if wrench_vecs is not None:
            wrench_trajectory = timestamped_spatialforces_to_wrench_protos(
                timestamps_between_wrenches=timestamps_between_wrenches,
                wrench_vecs=wrench_vecs,
                reference_time=reference_time,
            )
            arm_cartesian_command.wrench_trajectory_in_task.CopyFrom(wrench_trajectory)

        if body_follow_arm:
            mobility_command = robot_cmd.synchronized_command.mobility_command
            mobility_command.follow_arm_request.CopyFrom(basic_command_pb2.FollowArmCommand.Request())

        arm_cartesian_command.x_axis = axis_mode_map.x
        arm_cartesian_command.y_axis = axis_mode_map.y
        arm_cartesian_command.z_axis = axis_mode_map.z
        arm_cartesian_command.rz_axis = axis_mode_map.rx
        arm_cartesian_command.ry_axis = axis_mode_map.ry
        arm_cartesian_command.rz_axis = axis_mode_map.rz

        if max_acceleration is not None:
            arm_cartesian_command.maximum_acceleration.value = max_acceleration

        if max_linear_velocity is not None:
            arm_cartesian_command.max_linear_velocity.value = max_linear_velocity

        if max_angular_velocity is not None:
            arm_cartesian_command.max_angular_velocity.value = max_angular_velocity

        if stay_near_current_joint_angles is not None:
            arm_cartesian_command.force_remain_near_current_joint_configuration = stay_near_current_joint_angles

        if preferred_joint_configuration is not None:
            arm_cartesian_command.preferred_joint_configuration = preferred_joint_configuration

        return self._convert_proto_to_robot_cmd_action_goal(robot_cmd)

    def move_end_effector_cartesian_trajectory(
        self,
        root_frame_name: str,
        axis_mode_map: AxisModeMap,
        *,
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
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        reference_time: Optional[float] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        body_follow_arm: bool = False,
        timeout_sec: Optional[float] = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Return a goal corresponding to the arm following a Cartesian trajectory.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid unexpected behaviour.

        Args:
            root_frame_name: optional task frame that all trajectories are defined with respect to.
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
            preferred_joint_configuration: the preferred joint configuration for the arm to end up in.
            reference_time: time with respect to the robot clock of when to reference when trajectories start and end
            pose_interpolation: positional interpolation method. Defaults to cubic.
            ang_interpolation: angular interpolation method. Defaults to cubic.
            body_follow_arm: have the body follow the arm when it moves
            timeout_sec: seconds it will attempt to run the command before timing out and returning

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """
        action_goal = self._get_arm_cartesian_trajectory_action_goal(
            root_frame_name=root_frame_name,
            axis_mode_map=axis_mode_map,
            timestamps_between_poses=timestamps_between_poses,
            timestamps_between_wrenches=timestamps_between_wrenches,
            pose_vecs=pose_vecs,
            vel_vecs=vel_vecs,
            wrench_vecs=wrench_vecs,
            wrist_t_tool=wrist_t_tool,
            root_t_task=root_t_task,
            max_acceleration=max_acceleration,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_pos_tracking_error=max_pos_tracking_error,
            max_rot_tracking_error=max_rot_tracking_error,
            stay_near_current_joint_angles=stay_near_current_joint_angles,
            preferred_joint_configuration=preferred_joint_configuration,
            reference_time=reference_time,
            pose_interpolation=pose_interpolation,
            ang_interpolation=ang_interpolation,
            body_follow_arm=body_follow_arm,
        )
        return self.command("move_arm_along_trajectory", action_goal, timeout_sec)

    def move_end_effector_cartesian_trajectory_async(
        self,
        root_frame_name: str,
        axis_mode_map: AxisModeMap,
        *,
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
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        reference_time: Optional[float] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        body_follow_arm: bool = False,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Return a goal corresponding to the arm following a Cartesian trajectory.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid unexpected behaviour.

        Args:
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            axis_mode_map: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
            timestamps_between_poses: array of timestamps of when each pose point is traversed (seconds)
            timestamps_between_wrenches: array of timestamps of when each wrench point is traversed (seconds)
            pose_vecs: vector of (spatialmath) SE3 poses (wrt the task frame) for the end effector to pass through
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
            preferred_joint_configuration: the preferred joint configuration for the arm to end up in.
            reference_time: time with respect to the robot clock of when to reference when trajectories start and end
            pose_interpolation: positional interpolation method. Defaults to cubic.
            ang_interpolation: angular interpolation method. Defaults to cubic.
            body_follow_arm: have the body follow the arm when it moves
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons
        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """

        action_goal = self._get_arm_cartesian_trajectory_action_goal(
            root_frame_name=root_frame_name,
            axis_mode_map=axis_mode_map,
            timestamps_between_poses=timestamps_between_poses,
            timestamps_between_wrenches=timestamps_between_wrenches,
            pose_vecs=pose_vecs,
            vel_vecs=vel_vecs,
            wrench_vecs=wrench_vecs,
            wrist_t_tool=wrist_t_tool,
            root_t_task=root_t_task,
            max_acceleration=max_acceleration,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_pos_tracking_error=max_pos_tracking_error,
            max_rot_tracking_error=max_rot_tracking_error,
            stay_near_current_joint_angles=stay_near_current_joint_angles,
            preferred_joint_configuration=preferred_joint_configuration,
            reference_time=reference_time,
            pose_interpolation=pose_interpolation,
            ang_interpolation=ang_interpolation,
            body_follow_arm=body_follow_arm,
        )
        return self.command_async(
            "move_arm_along_cartesian_trajectory",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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

    def get_arm_impedance_robot_cmd(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[Tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Create the robot command proto for moving the end effector to specific
        pose using hybrid position/impedance control.

        The stiffness and damping gains are defaulted to their maximum values.
        Note that this is in the task frame, so these were probably empirically obtained.
        (See code comment in here
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_impedance_control/arm_impedance_control.py.)
        """

        if stiffness is None:
            stiffness = SpatialVector([500, 500, 500, 60, 60, 60])
        if damping is None:
            damping = SpatialVector([2.5, 2.5, 2.5, 1.0, 1.0, 1.0])
        if wrench_nm is None:
            wrench_nm = SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        robot_cmd = robot_command_pb2.RobotCommand()

        impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command
        impedance_cmd.root_frame_name = root_frame_name.value

        impedance_cmd.root_tform_task.CopyFrom(se3_to_se3pose_proto(root_t_task))
        impedance_cmd.wrist_tform_tool.CopyFrom(se3_to_se3pose_proto(wrist_t_tool))
        impedance_cmd.diagonal_stiffness_matrix.CopyFrom(geometry_pb2.Vector(values=stiffness.A.tolist()))
        impedance_cmd.diagonal_damping_matrix.CopyFrom(geometry_pb2.Vector(values=damping.A.tolist()))
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.x = wrench_nm.A[0]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.y = wrench_nm.A[1]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.z = wrench_nm.A[2]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.x = wrench_nm.A[3]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.y = wrench_nm.A[4]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.z = wrench_nm.A[5]

        # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
        # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
        cmd_trajectory = impedance_cmd.task_tform_desired_tool
        for pose, duration in trajectory:
            point = cmd_trajectory.points.add()
            point.time_since_reference.CopyFrom(seconds_to_duration(duration))
            point.pose.CopyFrom(se3_to_se3pose_proto(pose))

        return robot_cmd

    def _get_arm_impedance_action_goal(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[Tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
    ) -> RobotCommandAction.Goal:
        """Create the ROS Msg for moving the end effector to specific
        pose using hybrid position/impedance control.

        The stiffness and damping gains are defaulted to their maximum values.
        Note that this is in the task frame, so these were probably empirically obtained.
        (See code comment in here
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_impedance_control/arm_impedance_control.py.)
        """
        return self._convert_proto_to_robot_cmd_action_goal(
            self.get_arm_impedance_robot_cmd(
                root_frame_name, root_t_task, wrist_t_tool, trajectory, stiffness, damping, wrench_nm
            )
        )

    def move_end_effector_impedance(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[Tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
        timeout_sec: Optional[float] = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Impedance control as variable stiffness/damping position control combined with a
            feedforward force control as a wrench.

        Args:
            root_frame_name: name of root frame as a string (without 'RobotName/' as the prefix)
            root_t_task: the transform from the root frame to the "task frame" (this doesn't need a name)
            wrist_t_tool: The tool pose relative to the parent link (arm_link_wr1).
                This can also be thought of as the "remote center" frame.
            trajectory: Discrete waypoints of where we want the tool to be relative to the task frame
                and the duration of time to slerp over.
            stiffness: the position/stiffness gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            damping: the derivative/damping gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            wrench_nm: the feedforward wrench for x, y, z, torque_x, torque_y, torque_z in the task frame
            timeout_sec: the timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_impedance_action_goal(
            root_frame_name,
            root_t_task,
            wrist_t_tool,
            trajectory,
            stiffness,
            damping,
            wrench_nm,
        )
        return self.command("move_end_effector_impedance", action_goal, timeout_sec)

    def move_end_effector_impedance_async(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[Tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Impedance control as variable stiffness/damping position control combined with a
            feed-forward force control as a wrench.

        Args:
            root_frame_name: name of root frame as a string
            root_t_task: the transform from the root frame to the "task frame" (this doesn't need a name)
            wrist_t_tool: The tool pose relative to the parent link (arm_link_wr1).
                This can also be thought of as the "remote center" frame.
            trajectory: Discrete waypoints of where we want the tool to be relative to the task frame
                and the duration of time to slerp over.
            stiffness: the position/stiffness gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            damping: the derivative/damping gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            wrench_nm: the feed-forward wrench for x, y, z, torque_x, torque_y, torque_z in the task frame
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_arm_impedance_action_goal(
            root_frame_name, root_t_task, wrist_t_tool, trajectory, stiffness, damping, wrench_nm
        )
        return self.command_async(
            "move_end_effector_impedance",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_goal_claw_gripper_open(self, theta_rad: Optional[float] = None) -> RobotCommandAction.Goal:
        if theta_rad is not None:
            if not (theta_rad >= _CLAW_GRIPPER_CLOSED_ANGLE and theta_rad <= abs(_CLAW_GRIPPER_OPEN_ANGLE)):
                raise ValueError("Invalid gripper command: theta must be within range [0, pi/2]")
            # negating theta here to match with gripper bounds from spot_sdk
            action_goal = self._convert_proto_to_robot_cmd_action_goal(
                RobotCommandBuilder.claw_gripper_open_angle_command(-theta_rad)
            )
        else:
            action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_open_command())
        return action_goal

    def claw_gripper_open(
        self, *, theta_rad: Optional[float] = None, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send a command to open the gripper

        Args:
            theta: Angle in radians (between 0 and pi/2) to open the gripper. Not passing a value defaults
                the gripper to open completely.
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_goal_claw_gripper_open(theta_rad)
        return self.command("claw_gripper_open", action_goal, timeout_sec)

    def claw_gripper_open_async(
        self,
        *,
        theta_rad: Optional[float] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to open the gripper asynchronously

        Args:
            theta: Angle in radians (between 0 and pi/2) to open the gripper. Not passing a value defaults
                the gripper to open completely.
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._get_goal_claw_gripper_open(theta_rad)
        return self.command_async(
            "claw_gripper_open",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def claw_gripper_close(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to close the gripper

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command("claw_gripper_close", action_goal, timeout_sec)

    def claw_gripper_close_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to close the gripper asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command_async(
            "claw_gripper_close",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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
