#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
from enum import Enum, IntEnum
from functools import partial
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import geometry_msgs
import numpy as np
import rclpy
import spot_driver.conversions as conv
from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai_ros.utilities.action_handle import ActionHandle
from bdai_ros.utilities.custom_types import Action
from bdai_ros.utilities.futures import wait_for_future
from bdai_ros.utilities.manual_conversions import se3_to_se3pose_proto, to_ros_pose
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.api import (
    arm_command_pb2,
    basic_command_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    trajectory_pb2,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    HAND_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
)
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import Quaternion, to_euler_zxy
from bosdyn.util import seconds_to_duration
from bosdyn_msgs.msg import (
    ListWorldObjectRequest,
    ListWorldObjectResponse,
    ManipulationApiFeedbackResponse,
    RobotCommandFeedback,
)
from geometry_msgs.msg import Twist
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sklearn.preprocessing import normalize
from spatialmath import SE2, SE3, SO3, SpatialForce, Twist2, Twist3, UnitQuaternion
from spatialmath.base.types import R3
from spatialmath.spatialvector import SpatialVector
from spot_msgs.action import Manipulation, RobotCommand
from spot_msgs.srv import (
    ClearBehaviorFault,
    DeleteSound,
    Dock,
    ExecuteDance,
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
    PlaySound,
    SetLocomotion,
    SetVelocity,
    SetVolume,
    UploadAnimation,
)
from std_msgs.msg import ColorRGBA as Color
from std_srvs.srv import SetBool, Trigger
from typing_extensions import override
from visualization_msgs.msg import Marker

from spot_utilities.robot_command_helpers import (
    AxisModeDict,
    timestamped_se2_vecs_to_robot_command_proto,
    timestamped_se3_poses_to_se3_trajectory,
    timestamped_spatialforces_to_wrench_protos,
)

Proto = TypeVar("Proto")


class SpotActionHandle(ActionHandle):
    def __init__(self, robot_name: str, command_name: str, logger: Optional[RcutilsLogger] = None):
        super().__init__(action_name=f"{robot_name}.{command_name}", logger=logger)
        self._result_callback = self._result_callback_intermediate
        self._result_callback_internal: Optional[Callable[[RobotCommandFeedback], None]] = None
        self._result_internal: Optional[RobotCommandFeedback] = None
        self._feedback_callback = self._feedback_callback_intermediate
        self._feedback_callback_internal: Optional[Callable[[RobotCommandFeedback], None]] = None

    @property
    def result(self) -> Optional[RobotCommandFeedback]:
        return self._result_internal

    @override
    def set_result_callback(self, result_callback: Callable[[RobotCommandFeedback], None]) -> None:
        """Sets the callback for processing the result from executing an Action"""
        self._result_callback_internal = result_callback

    def _result_callback_intermediate(self, result: RobotCommand.Result) -> None:
        """Callback for pulling the result out of the result and passing it to a user provided callback"""
        if not result.success:
            self._logger.error(f"Action resulted in failure. Msg: {result.message}")
            self._failure()
            return

        self._result_internal = result.result
        if self._result_callback_internal is not None:
            self._result_callback_internal(self._result_internal)

    @override
    def set_feedback_callback(self, feedback_callback: Callable[[RobotCommandFeedback], None]) -> None:
        """Sets the callback for processing the feedback during execution of an Action"""
        self._feedback_callback_internal = feedback_callback

    def _feedback_callback_intermediate(self, feedback: RobotCommand.Feedback) -> None:
        """Intermediate step to pull the feedback out of the feedback"""
        if self._feedback_callback_internal is not None:
            self._feedback_callback_internal(feedback.feedback)


class RobotComposeCommandHandle(object):
    """Handle for sequentially composing multiple asynchronous RobotCommand sent via an action client"""

    def __init__(
        self, actions: List[partial], result_callback: Optional[Callable[[RobotCommandFeedback], None]]
    ) -> None:
        self._index = 0
        self._number_actions = len(actions)
        self._action_list = actions
        self._result_callback = result_callback
        self._action_list[self._index].keywords["result_callback"] = self.result_callback
        self._action_list[self._index]()

    def result_callback(self, feedback: RobotCommandFeedback) -> None:
        if self._index < self._number_actions - 1:
            self._index += 1
            self._action_list[self._index].keywords["result_callback"] = self.result_callback
            self._action_list[self._index]()
        elif self._result_callback is not None:
            self._result_callback(feedback)


class SpotBasic(object):
    """Base ROS2 Node that provides interfaces to calling various commands"""

    TRIGGER_SERVICES = [
        "claim",
        "release",
        "stop",
        "self_right",
        "sit",
        "stand",
        "power_on",
        "power_off",
        "estop/hard",
        "estop/gentle",
        "estop/release",
        "undock",
    ]

    class LocomotionHint(IntEnum):
        """Enumeration of the locomotion hints

        See
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=mobilityparams#locomotionhint
        Note: Comments below for each enumerator come directly from the above link
        """

        UNKNOWN = 0  # Listed for completeness, but the docs say do not use
        AUTO = 1  # No hint, robot chooses an appropriate gait (typically trot.)
        TROT = 2  # Most robust gait which moves diagonal legs together.
        SPEED_SELECT_TROT = 3  # Trot which comes to a stand when not commanded to move.
        CRAWL = 4  # Slow and steady gait which moves only one foot at a time.
        SPEED_SELECT_CRAWL = 10  # Crawl which comes to a stand when not commanded to move.
        AMBLE = 5  # Four beat gait where one foot touches down at a time.
        SPEED_SELECT_AMBLE = 6  # Amble which comes to a stand when not commanded to move.
        JOG = 7  # Demo gait which moves diagonal leg pairs together with an aerial phase.
        HOP = 8  # Demo gait which hops while holding some feet in the air.

    class BatteryChangeDirHint(IntEnum):
        RIGHT = 1
        LEFT = 2

    class FrameHint(Enum):
        BODY = BODY_FRAME_NAME
        VISION = VISION_FRAME_NAME
        ODOM = ODOM_FRAME_NAME
        HAND = HAND_FRAME_NAME
        GRAV_ALIGNED_BODY = GRAV_ALIGNED_BODY_FRAME_NAME
        GROUND_PLANE = GROUND_PLANE_FRAME_NAME
        WRIST = "link_wr1"
        HEAD = "head"
        BACK = "back"
        LEFT = "left"
        RIGHT = "right"
        FRONT_LEFT = "frontleft"
        FRONT_RIGHT = "frontright"
        BACK_CAMERA = "back_fisheye"
        FRONT_LEFT_CAMERA = "frontleft_fisheye"
        FRONT_RIGHT_CAMERA = "frontright_fisheye"
        LEFT_CAMERA = "left_fisheye"
        RIGHT_CAMERA = "right_fisheye"
        HAND_CAMERA = "hand_color_image_sensor"

    def __init__(
        self,
        robot_name: str,
        parent_node_name: str,
        *,
        disable_collisions: bool = False,
        dry: bool = False,
        rviz: bool = False,
    ):
        """
        Args:
            robot_name: name of the robot
            parent_node_name: The name of the parent node containing this SpotBasic
            disable_collisions: True if collisions should be ignored, False otherwise
            dry: True if this is a dry run and commands should not actually be sent, False otherwise
            rviz: True if visuals should be published, False otherwise
        """
        self._node = rclpy.create_node(node_name=f"{parent_node_name}_spot_basic", namespace=robot_name)

        self._robot_name = robot_name
        self._disable_collisions = disable_collisions
        self._dry = dry
        self._rviz = rviz

        # A tf listener instance can be connected to the robot.
        self._tf_listener: TFListenerWrapper = None

        # TODO: are there other frame names to grab?
        self._body_frame_name = f"{self._robot_name}/{BODY_FRAME_NAME}"
        self._hand_frame_name = f"{self._robot_name}/{HAND_FRAME_NAME}"
        self._vision_frame_name = f"{self._robot_name}/{VISION_FRAME_NAME}"
        self._odom_frame_name = f"{self._robot_name}/{ODOM_FRAME_NAME}"

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = Thread(target=self._spin_executor, args=(self._executor,), daemon=True)
        self._callback_group = ReentrantCallbackGroup()

        if self._rviz:
            self._vis_pub = self._node.create_publisher(Marker, "/visualization_marker", 10)

        self._velocity_cmd_publisher = self._node.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", 10)

        self._trigger_clients = {}
        for command_name in self.TRIGGER_SERVICES:
            self._trigger_clients[command_name] = self._node.create_client(
                Trigger, f"/{self.robot_name}/{command_name}", callback_group=self._callback_group
            )
        self._dock_client = self._node.create_client(
            Dock, f"/{self.robot_name}/dock", callback_group=self._callback_group
        )
        self._stair_mode_client = self._node.create_client(
            SetBool, f"/{self.robot_name}/stair_mode", callback_group=self._callback_group
        )
        self._locomotion_mode_client = self._node.create_client(
            SetLocomotion, f"/{self.robot_name}/locomotion_mode", callback_group=self._callback_group
        )
        self._execute_dance_client = self._node.create_client(
            ExecuteDance,
            f"/{self.robot_name}/execute_dance",
            callback_group=self._callback_group,
        )
        self._upload_animation_client = self._node.create_client(
            UploadAnimation,
            f"/{self.robot_name}/upload_animation",
            callback_group=self._callback_group,
        )
        self._list_all_dances_client = self._node.create_client(
            ListAllDances,
            f"/{self.robot_name}/list_all_dances",
            callback_group=self._callback_group,
        )
        self._list_all_moves_client = self._node.create_client(
            ListAllMoves,
            f"/{self.robot_name}/list_all_moves",
            callback_group=self._callback_group,
        )
        self._max_velocity_client = self._node.create_client(
            SetVelocity, f"/{self.robot_name}/max_velocity", callback_group=self._callback_group
        )
        self._clear_behavior_fault_client = self._node.create_client(
            ClearBehaviorFault, f"/{self.robot_name}/clear_behavior_fault", callback_group=self._callback_group
        )
        self._list_graph_client = self._node.create_client(
            ListGraph, f"/{self.robot_name}/list_graph", callback_group=self._callback_group
        )
        self._list_world_objects_client = self._node.create_client(
            ListWorldObjects, f"/{self.robot_name}/list_world_objects", callback_group=self._callback_group
        )
        self._get_parameters_client = self._node.create_client(
            GetParameters, f"/{self.robot_name}/spot_ros2/get_parameters", callback_group=self._callback_group
        )
        self._graph_nav_upload_graph_client = self._node.create_client(
            GraphNavUploadGraph, f"/{self.robot_name}/graph_nav_upload_graph", callback_group=self._callback_group
        )

        self._graph_nav_clear_graph_client = self._node.create_client(
            GraphNavClearGraph, f"/{self.robot_name}/graph_nav_clear_graph", callback_group=self._callback_group
        )
        self._graph_nav_set_localization_client = self._node.create_client(
            GraphNavSetLocalization,
            f"/{self.robot_name}/graph_nav_set_localization",
            callback_group=self._callback_group,
        )
        self._graph_nav_get_localization_pose_client = self._node.create_client(
            GraphNavGetLocalizationPose,
            f"/{self._robot_name}/graph_nav_get_localization_pose",
            callback_group=self._callback_group,
        )
        self._list_sounds_client = self._node.create_client(
            ListSounds, f"/{self.robot_name}/list_sounds", callback_group=self._callback_group
        )
        self._load_sound_client = self._node.create_client(
            LoadSound, f"/{self.robot_name}/load_sound", callback_group=self._callback_group
        )
        self._play_sound_client = self._node.create_client(
            PlaySound, f"/{self.robot_name}/play_sound", callback_group=self._callback_group
        )
        self._delete_sound_client = self._node.create_client(
            DeleteSound, f"/{self.robot_name}/delete_sound", callback_group=self._callback_group
        )
        self._get_volume_client = self._node.create_client(
            GetVolume, f"/{self.robot_name}/get_volume", callback_group=self._callback_group
        )
        self._set_volume_client = self._node.create_client(
            SetVolume, f"/{self.robot_name}/set_volume", callback_group=self._callback_group
        )

        self._robot_command_action_client = ActionClient(self._node, RobotCommand, "robot_command")
        self._node.get_logger().info(f"Waiting for the 'RobotCommand' action server for {self.robot_name} ...")
        self._robot_command_action_client.wait_for_server()
        self._node.get_logger().info(f"Found the 'RobotCommand' action server for {self.robot_name}.")

        self._manipulation_action_client = ActionClient(self._node, Manipulation, "manipulation")
        self._node.get_logger().info(f"Waiting for the 'Manipulation' action server for {self.robot_name} ...")
        if self._manipulation_action_client.wait_for_server(1.0):
            self._node.get_logger().info(f"Found the 'Manipulation' action server for {self.robot_name}.")
        else:
            self._node.get_logger().info(
                f"No 'Manipulation' action server for {self.robot_name} found. Assuming it has no arm."
            )

        self._spin_thread.start()

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def body_frame_name(self) -> str:
        return self._body_frame_name

    @property
    def hand_frame_name(self) -> str:
        return self._hand_frame_name

    @property
    def vision_frame_name(self) -> str:
        return self._vision_frame_name

    @property
    def odom_frame_name(self) -> str:
        return self._odom_frame_name

    @staticmethod
    def _spin_executor(executor: SingleThreadedExecutor) -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    def initialize_robot(self) -> bool:
        """Returns true/false based on success of claiming and powering up Spot."""
        # Claim robot.
        if not self.claim():
            return False

        # Power the robot.
        if not self.power_motors_on():
            return False
        return True

    def set_tf_listener(self, tf_listener: TFListenerWrapper) -> None:
        """Attaches a TFListenerWrapper instance to the robot.

        Args:
            tf_listener: An instance of the TFListenerWrapper.
        """
        self._tf_listener = tf_listener

    def unset_tf_listener(self) -> None:
        self._tf_listener = None

    def trigger_service(self, command_name: str) -> bool:
        """Triggers a command service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self._node.get_logger().info(f"Trigger command '{command_name}' for {self.robot_name}")
        request = Trigger.Request()
        try:
            future = self._trigger_clients[command_name].call_async(request)
        except KeyError:
            self._node.get_logger().error(f"No trigger command {command_name}")
            return False

        if not wait_for_future(future):
            return False

        response = future.result()
        if not response.success:
            self._node.get_logger().error(
                f"Trigger command '{command_name}' for {self.robot_name} failed. Msg: {response.message}"
            )
            return False
        return True

    def claim(self) -> bool:
        """Triggers the 'claim' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("claim")

    def release(self) -> bool:
        """Triggers the 'release' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("release")

    def stop(self) -> bool:
        """Triggers the 'stop' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("stop")

    def dock(self, dock_id: int) -> bool:
        """Triggers the 'dock' service on the SpotDriver

        Returns:
            True if command succeeded, False otherwise
        """
        self._node.get_logger().info(f"Dock command for {self.robot_name}")
        request = Dock.Request()
        request.dock_id = dock_id
        future = self._dock_client.call_async(request)

        if not wait_for_future(future):
            return False

        response = future.result()
        if not response.success:
            self._node.get_logger().error(f"Dock command for {self.robot_name} failed. Msg: {response.message}")
            return False
        return True

    def undock(self) -> bool:
        """Triggers the 'undock' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("undock")

    def self_right(self) -> bool:
        """Triggers the 'self_right' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("self_right")

    def sit(self) -> bool:
        """Triggers the 'sit' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("sit")

    def stand(self) -> bool:
        """Triggers the 'stand' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("stand")

    def rollover(self) -> bool:
        """Triggers the 'rollover' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("rollover")

    def power_on(self) -> bool:
        """Triggers the 'power_on' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("power_on")

    # Alias for the power_on function
    power_motors_on = power_on

    def power_off(self) -> bool:
        """Triggers the 'power_off' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("power_off")

    # Alias for the power_off function
    power_motors_off = power_off

    def estop_hard(self) -> bool:
        """Triggers the 'estop/hard' service on the SpotDriver. The robot will immediately cut power to the motors

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("estop/hard")

    def estop_gentle(self) -> bool:
        """Triggers the 'estop/gentle' service on the SpotDriver. The robot will try to settle on the ground before
                cutting power to the motors.

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("estop/gentle")

    def estop_release(self) -> bool:
        """Triggers the 'estop/release' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("estop/release")

    def set_stair_mode(self, value: bool) -> bool:
        """Sets whether to be in stair_mode on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self._node.get_logger().info(f"Setting stair_mode to '{value}' for {self.robot_name}")
        request = SetBool.Request()
        request.data = value
        future = self._stair_mode_client.call_async(request)
        if not wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Setting stair_mode to '{value}' for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

    def execute_dance(self, path: str) -> tuple[bool, str]:
        """Upload and execute dance file (found at path) through Spot Driver
        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        self._node.get_logger().info(f"Uploading dance at '{path}' to {self.robot_name}")
        request = ExecuteDance.Request()
        try:
            dance_file = open(path, "r")
            request.choreo_file_content = dance_file.read()
            dance_file.close()
        except FileNotFoundError:
            return False, "Specified filepath does not exist!"

        future = self._execute_dance_client.call_async(request)
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Uploading dance '{path}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    def upload_animation(self, move_name: str, path: str) -> tuple[bool, str]:
        """Upload an animation file through spot driver
        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        self._node.get_logger().info(f"Uploading animation at '{path}' to {self.robot_name}")
        request = UploadAnimation.Request()
        try:
            dance_file = open(path, "r")
            request.animation_file_content = dance_file.read()
            request.animation_name = move_name
            dance_file.close()
        except FileNotFoundError:
            return False, "Specified filepath does not exist!"

        future = self._upload_animation_client.call_async(request)
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Uploading animation '{path}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    def list_all_dances(self) -> tuple[bool, str, list[str]]:
        """Get list of all the dances uploaded on Spot"""
        self._node.get_logger().info(f"Sending list dances request to {self.robot_name}")
        request = ListAllDances.Request()
        future = self._list_all_dances_client.call_async(request)
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out", []
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Request to list dances at '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message, []
        return True, "success", response.dances

    def list_all_moves(self) -> tuple[bool, str, list[str]]:
        """Get list of all the moves uploaded on Spot"""
        self._node.get_logger().info(f"Sending list moves request to {self.robot_name}")
        request = ListAllMoves.Request()
        future = self._list_all_moves_client.call_async(request)
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out", []
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Request to list moves at '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message, []
        return True, "success", response.moves

    def list_sounds(self) -> tuple[bool, str, list[str]]:
        """
        List sounds loaded on Spot CAM.

        Returns:
            (True, 'Success', [<sound names>]) on success
            (False, <error message>, <ignore>) on failure
        """
        request = ListSounds.Request()
        future = self._list_sounds_client.call_async(request)
        if not wait_for_future(future):
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
        if not wait_for_future(future):
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
        if not wait_for_future(future):
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
        if not wait_for_future(future):
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
        if not wait_for_future(future):
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
        if not wait_for_future(future):
            return False, "Call to wait_for_future timed out"
        response = future.result()
        return response.success, response.message

    def set_locomotion_mode(self, l_hint: LocomotionHint) -> bool:
        """Set the locomotion mode on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self._node.get_logger().info(f"Setting locomotion_mode to '{l_hint}' for {self.robot_name}")
        request = SetLocomotion.Request()
        request.locomotion_mode = int(l_hint)
        future = self._locomotion_mode_client.call_async(request)
        if not wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Setting locomotion_mode to '{l_hint}' for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

    def set_max_velocity(self, velocity_limit: Twist) -> SetVelocity.Response:
        """Set the maximum velocity on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self._node.get_logger().info(f"Setting max_velocity to '{velocity_limit}' for {self.robot_name}")
        request = SetVelocity.Request()
        request.velocity = velocity_limit
        future = self._max_velocity_client.call_async(request)
        if not wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Setting stair_mode to '{velocity_limit}' for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

    def clear_behavior_fault(self, bid: int) -> ClearBehaviorFault.Response:
        """Clear a behavior fault on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self._node.get_logger().info(f"Clearing behavior fault {bid} for {self.robot_name}")
        request = ClearBehaviorFault.Request()
        request.id = bid
        future = self._clear_behavior_fault_client.call_async(request)
        if not wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self._node.get_logger().debug(
                f"Clearing behavior fault {bid} for {self.robot_name} failed. Msg: {response.message}"
            )
            return False
        return True

    def set_localization(self, method: str, waypoint_id: Optional[str] = None) -> bool:
        """Calls the SetLocalization RPC through spot_ros2.
        The 'waypoint_id' is only necessary if 'method'
        is 'waypoint'"""
        self._node.get_logger().info("Set localization")
        method_choices = {"fiducial", "waypoint"}
        if method not in method_choices:
            raise ValueError(f"Invalid localization method {method}.Must be in {method_choices}")
        request = GraphNavSetLocalization.Request()
        request.method = method
        if request.method == "waypoint":
            request.waypoint_id = waypoint_id
        future = self._graph_nav_set_localization_client.call_async(request)
        if not wait_for_future(future):
            return False
        response: GraphNavSetLocalization.Response = future.result()
        return response.success

    def get_localization_pose(self) -> Optional[geometry_msgs.msg.Pose]:
        self._node.get_logger().info("Get localization pose")
        request = GraphNavGetLocalizationPose.Request()
        future = self._graph_nav_get_localization_pose_client.call_async(request)
        if not wait_for_future(future):
            return None
        response = future.result()
        if not response.success:
            raise RuntimeError(f"Failed to get localization pose. Reason: {response.message}")
        return response.pose

    get_localization = get_localization_pose

    def upload_graph(self, upload_filepath: str) -> bool:
        self._node.get_logger().info("Upload graph")
        request = GraphNavUploadGraph.Request()
        request.upload_filepath = upload_filepath
        future = self._graph_nav_upload_graph_client.call_async(request)
        if not wait_for_future(future):
            return False
        response = future.result()
        return response.success

    def clear_graph(self) -> bool:
        self._node.get_logger().info("Clear graph")
        request = GraphNavClearGraph.Request()
        future = self._graph_nav_clear_graph_client.call_async(request)
        if not wait_for_future(future):
            return False
        response = future.result()
        return response.success

    def list_graph(self, upload_filepath: str) -> List[str]:
        """Request a list of waypoint ids"""
        self._node.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        request = ListGraph.Request()
        request.upload_filepath = upload_filepath
        future = self._list_graph_client.call_async(request)
        if not wait_for_future(future):
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
        self._node.get_logger().info(f"List world objects for {self.robot_name}")
        request = ListWorldObjects.Request()
        request.request = p_request
        future = self._list_world_objects_client.call_async(request)
        if not wait_for_future(future):
            return None
        response = future.result()
        return response.response

    def has_arm(self) -> Optional[bool]:
        """Queries if the spot has an arm

        Returns:
            True if the spot has an arm, False if it does not, and None if the query failed
        """
        self._node.get_logger().info(f"Checking if {self.robot_name} has an arm")
        request = GetParameters.Request()
        request.names = ["has_arm"]
        future = self._get_parameters_client.call_async(request)
        if not wait_for_future(future):
            return None
        response = future.result()
        param = response.values[0]
        return param.bool_value

    def _send_action_goal_and_wait(
        self, command_name: str, goal: Action.Goal, action_client: ActionClient, timeout_sec: Optional[float] = None
    ) -> Optional[Action.Result]:
        """Sends an action goal to one of the action servers and waits for it to be done

        Args:
            command_name: A representative name of the command for logging
            goal: The Action Goal sent to the action server
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        Note: This is a blocking call
        """
        if goal is None:
            self._node.get_logger.warn("Cannot send NULL ActionGoal")
            # TODO: Is there something better that we can return with information about the failure?
            # maybe a ActionType.Result that represents failure?
            return None

        if self._dry:
            # TODO: What to return on a dry run? Just success?
            return None

        self._node.get_logger().info(f"Sending Action [{self._robot_name}.{command_name}]")

        future = action_client.send_goal_async(goal)
        if not wait_for_future(future, timeout_sec=timeout_sec):
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().error(f"Action [{self._robot_name}.{command_name}] was not accepted")
            return None

        self._node.get_logger().info(f"Action [{self._robot_name}.{command_name}] accepted")
        future = goal_handle.get_result_async()
        if not wait_for_future(future, timeout_sec=timeout_sec):
            return None

        self._node.get_logger().info(f"Received response for Action [{self._robot_name}.{command_name}]")
        return future.result().result

    def _send_action_goal_async(
        self,
        command_name: str,
        goal: Action.Goal,
        action_client: ActionClient,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        if goal is None:
            # Is there something better that we can return with information about the failure?
            return None
        if self._dry:
            # TODO: What to return on a dry run? Just success?
            return None

        handle = SpotActionHandle(robot_name=self._robot_name, command_name=command_name)
        if result_callback is not None:
            handle.set_result_callback(result_callback)

        if feedback_callback is None:
            send_goal_future = action_client.send_goal_async(goal)
        else:
            handle.set_feedback_callback(feedback_callback)
            send_goal_future = action_client.send_goal_async(goal, feedback_callback=handle.get_feedback_callback)
        handle.set_send_goal_future(send_goal_future)

        if on_failure_callback is not None:
            handle.set_on_failure_callback(on_failure_callback)

        return handle

    def command(
        self, command_name: str, goal: RobotCommand.Goal, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Runs the `RobotCommand` action

        Args:
            command_name: A representative name of the command for logging
            goal: The Action Goal sent to the action server
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if (
            action_result := self._send_action_goal_and_wait(
                command_name, goal, self._robot_command_action_client, timeout_sec
            )
        ) is not None:
            return action_result.result
        return None

    def command_async(
        self,
        command_name: str,
        goal: RobotCommand.Goal,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Runs the `RobotCommand` action asynchronously

        Args:
            command_name: A representative name of the command for logging
            goal: The Action Goal sent to the action server
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            A handle for the asynchronous lifecycle of the action

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        return self._send_action_goal_async(
            command_name,
            goal,
            self._robot_command_action_client,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def manipulation(
        self, command_name: str, goal: Manipulation.Goal, timeout_sec: Optional[float] = None
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Synchronously run a manipulation request

        Args:
            command_name
            goal
            timeout_sec

        Returns:

        """
        if not self.has_arm():
            self._node.get_logger().error("Cannot send Manipulation api request, robot has no arm.")
            return None
        if (
            action_result := self._send_action_goal_and_wait(
                command_name, goal, self._manipulation_action_client, timeout_sec
            )
        ) is not None:
            return action_result.result
        return None

    def manipulation_async(
        self,
        command_name: str,
        goal: Manipulation.Goal,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        if not self.has_arm():
            self._node.get_logger().error("Cannot send Manipulation api request, robot has no arm.")
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
        marker.header.stamp = self._node.get_clock().now().to_msg()
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
    def _convert_proto_to_robot_cmd_action_goal(proto: Proto) -> RobotCommand.Goal:
        action_goal = RobotCommand.Goal()
        conv.convert_proto_to_bosdyn_msgs_robot_command(proto, action_goal.command)
        return action_goal

    @staticmethod
    def _convert_proto_to_manip_cmd_action_goal(proto: Proto) -> RobotCommand.Goal:
        action_goal = Manipulation.Goal()
        conv.convert_proto_to_bosdyn_msgs_manipulation_api_request(proto, action_goal.command)
        return action_goal

    def _get_battery_change_pose(self, dir_hint: BatteryChangeDirHint) -> RobotCommand.Goal:
        proto = RobotCommandBuilder.battery_change_pose_command(int(dir_hint))
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def lookup_a_tform_b(
        self,
        frame_a: Union[FrameHint, str],
        frame_b: Union[FrameHint, str],
        transform_time: Optional[Union[Time, float]] = None,
        timeout: Optional[float] = None,
        wait_for_frames: bool = False,
    ) -> SE3:
        """A robot specific transform from frame_a to frame_b.

        The transform returned will be frame_a_t_frame_b, which is robot robot specific.
        So instead if frame_a is VISION, we internally append it to the robot name
        s.t. "<Robot Name>/VISION", will be called.

        Args:
            frame_a: Base frame for transform.
            frame_b: Tip frame for transform.
            transform_time: Time the transform is to be published.
            timeout: The time to wait for the transform to become available if the transform_time is beyond the most
                recent transform in the buffer.
            wait_for_frames: If true, waits timeout amount of time for a path to exist from frame_a to frame_b in the
                buffer.  If false, this will return immediately if a path does not exist even if timeout is not
                None.  Note that wait_for_transform can be used to wait indefinitely for a transform to become
                available.
        Returns:
            The timestamp from the latest recorded transform frame_a_t_frame_b
        Raises:
            All the possible TransformExceptions. ValueError if no TFListenerWrapper instance is connected.
        """
        if self._tf_listener is None:
            raise ValueError("TFListenerWrapper is not connected. Use set_tf_listener().")

        if isinstance(frame_a, self.FrameHint):
            frame_a = frame_a.value

        if isinstance(frame_b, self.FrameHint):
            frame_b = frame_b.value

        return self._tf_listener.lookup_a_tform_b(
            f"{self.robot_name}/{frame_a}", f"{self.robot_name}/{frame_b}", transform_time, timeout, wait_for_frames
        )

    def _get_basic_tform_tree_snapshot(
        self, transform_time: Optional[Union[Time, float]] = None
    ) -> geometry_pb2.FrameTreeSnapshot:
        """A tree-based collection of transformation containing common frames.

        Common frames include transformations to: BODY, VISION, ODOM, HAND, GRAV_ALIGNED_BODY, GROUND_PLANE and WRIST.
        Camera sensor related frames are also included: HEAD, BACK, LEFT, RIGHT, FRONT_LEFT, FRONT_RIGHT, BACK_CAMERA,
        FRONT_LEFT_CAMERA, FRONT_RIGHT_CAMERA, LEFT_CAMERA, RIGHT_CAMERA and HAND_CAMERA.

        Returns:
            A tree-based collection of transformation to common frames.
        """

        child_to_parent_edge_map = {}

        # Add root frame.
        child_to_parent_edge_map[self.FrameHint.BODY.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_tform_child=se3_to_se3pose_proto(SE3())
        )

        # Add head frame.
        child_to_parent_edge_map[self.FrameHint.HEAD.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.BODY.value, parent_tform_child=se3_to_se3pose_proto(SE3())
        )

        # The ground plane is relative to the odom frame.
        child_to_parent_edge_map[self.FrameHint.GROUND_PLANE.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.ODOM.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.ODOM, self.FrameHint.GROUND_PLANE, transform_time)
            ),
        )

        # Add frames relative to the body frame.
        frames_relative_to_body = [
            self.FrameHint.VISION,
            self.FrameHint.ODOM,
            self.FrameHint.GRAV_ALIGNED_BODY,
            self.FrameHint.BACK,
            self.FrameHint.LEFT,
            self.FrameHint.RIGHT,
        ]

        if self.has_arm():
            frames_relative_to_body += [self.FrameHint.WRIST]

        for target_frame in frames_relative_to_body:
            child_to_parent_edge_map[target_frame.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
                parent_frame_name=self.FrameHint.BODY.value,
                parent_tform_child=se3_to_se3pose_proto(
                    self.lookup_a_tform_b(self.FrameHint.BODY, target_frame, transform_time)
                ),
            )

        if self.has_arm():
            # Add hand frame relative to wrist frame.
            child_to_parent_edge_map[self.FrameHint.HAND.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
                parent_frame_name=self.FrameHint.WRIST.value,
                parent_tform_child=se3_to_se3pose_proto(
                    self.lookup_a_tform_b(self.FrameHint.WRIST, self.FrameHint.HAND, transform_time)
                ),
            )

            # Add hand camera frame.
            child_to_parent_edge_map[self.FrameHint.HAND_CAMERA.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
                parent_frame_name=self.FrameHint.WRIST.value,
                parent_tform_child=se3_to_se3pose_proto(
                    self.lookup_a_tform_b(self.FrameHint.WRIST, self.FrameHint.HAND_CAMERA, transform_time)
                ),
            )

        # Add frames relative to head frame.
        child_to_parent_edge_map[self.FrameHint.FRONT_LEFT.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.HEAD.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.HEAD, self.FrameHint.FRONT_LEFT, transform_time)
            ),
        )

        child_to_parent_edge_map[self.FrameHint.FRONT_RIGHT.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.HEAD.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.HEAD, self.FrameHint.FRONT_RIGHT, transform_time)
            ),
        )

        # Add camera sensor frames.
        child_to_parent_edge_map[self.FrameHint.FRONT_LEFT_CAMERA.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.FRONT_LEFT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.FRONT_LEFT, self.FrameHint.FRONT_LEFT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[self.FrameHint.FRONT_RIGHT_CAMERA.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.FRONT_RIGHT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.FRONT_RIGHT, self.FrameHint.FRONT_RIGHT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[self.FrameHint.LEFT_CAMERA.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.LEFT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.LEFT, self.FrameHint.LEFT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[self.FrameHint.RIGHT_CAMERA.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.RIGHT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.RIGHT, self.FrameHint.RIGHT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[self.FrameHint.BACK_CAMERA.value] = geometry_pb2.FrameTreeSnapshot.ParentEdge(
            parent_frame_name=self.FrameHint.BACK.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(self.FrameHint.BACK, self.FrameHint.BACK_CAMERA, transform_time)
            ),
        )

        return geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=child_to_parent_edge_map)

    def go_to_battery_change_pose(
        self, dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send command to go to the battery change pose asynchronously

        Args:
            dir_hint: A hint for the direction to roll
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        """
        action_goal = self._get_battery_change_pose(dir_hint)
        return self.command("go_to_battery_change_pose", action_goal, timeout_sec)

    def go_to_battery_change_pose_async(
        self,
        dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send command to go to the battery change pose asynchronously

        Args:
            dir_hint: A hint for the direction to roll
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self._get_battery_change_pose(dir_hint)
        return self.command_async(
            "go_to_battery_change_pose",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_mobility_params(self) -> spot_command_pb2.MobilityParams:
        """Create the MobilityParams object for use when building robot commands"""

        # NOTE so far there is no way to change this, and it might as well be set as an attribute during init.
        # but possibly we'll want to change some of these parameters on the fly someday
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=self._disable_collisions,
            disable_vision_foot_obstacle_avoidance=self._disable_collisions,
            disable_vision_foot_constraint_avoidance=self._disable_collisions,
            obstacle_avoidance_padding=0.001,
        )
        return spot_command_pb2.MobilityParams(obstacle_params=obstacles)

    def _get_walk_to_action_goal(self, vision_t_goal: SE2, height: float) -> RobotCommand.Goal:
        """TODO

        Args:
            vision_t_goal: A SE2 transform from the vision frame to where the user wants the spot to walk to
            height: The height of the body

        Returns:
            An Action.Goal representing the command for a spot to walk to a specific goal location
        """
        self._node.get_logger().info("Walking to goal...")

        if self._rviz:
            self._draw_arrow(self.vision_frame_name, self.body_frame_name, vision_t_goal, color=Color(g=1.0, a=1.0))

        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=vision_t_goal.x,
            goal_y=vision_t_goal.y,
            goal_heading=vision_t_goal.theta(),
            frame_name=VISION_FRAME_NAME,
            body_height=height,
            params=self._get_mobility_params(),
        )
        return self._convert_proto_to_robot_cmd_action_goal(proto_goal)

    def walk_to(
        self, vision_t_goal: SE2, height: float = 0, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send the command to walk to a specific SE2 goal

        Args:
            vision_t_goal: A SE2 transform from the vision frame to where the user wants the spot to walk to
            height: The height of the body
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self._get_walk_to_action_goal(vision_t_goal, height)
        return self.command("walk_to", action_goal, timeout_sec)

    def walk_to_async(
        self,
        vision_t_goal: SE2,
        height: float = 0,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send the command to walk to a specific SE2 pose

        Args:
            vision_t_goal: A SE2 transform from the vision frame to where the user wants the spot to walk to
            height: The height of the body
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self._get_walk_to_action_goal(vision_t_goal, height)
        return self.command_async(
            "walk_to",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_stand_to_action_goal(self, vision_r_goal: SO3, height: float) -> RobotCommand.Goal:
        """
        Args:
            vision_r_goal: A SO3 rotation of the body frame with respect to the footprint frame
                (gravity aligned framed with yaw computed from the stance feet)
            height: The height of the body

        Returns:
            An Action.Goal representing the command for a spot to walk to a specific goal location
        """
        self._node.get_logger().info("Moving to goal orientation...")

        vision_r_goal = UnitQuaternion(vision_r_goal)
        footprint_R_body = to_euler_zxy(
            Quaternion(
                w=vision_r_goal.A[0],
                x=vision_r_goal.A[1],
                y=vision_r_goal.A[2],
                z=vision_r_goal.A[3],
            )
        )
        proto_goal = RobotCommandBuilder.synchro_stand_command(body_height=height, footprint_R_body=footprint_R_body)
        return self._convert_proto_to_robot_cmd_action_goal(proto_goal)

    def stand_with_orientation(
        self, vision_r_goal: SO3, height: float, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send the command to stand in a specific SO3 rotation

        Args:
            vision_r_goal: A SO3 rotation from the vision frame to goal orientation.
            height: The height of the body
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal_stand = self._get_stand_to_action_goal(vision_r_goal, height=height)
        return self.command("orient", action_goal_stand, timeout_sec)

    def stand_with_orientation_async(
        self,
        vision_r_goal: SO3,
        height: float,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send the command to stand in a specific SO3 rotation

        Args:
            vision_r_goal: A SO3 rotation, expressed as EulerZXY object (yaw, pitch, roll, in radians) from the vision
                frame to where the user wants the spot to rotate to
            height: The height of the body
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal_stand = self._get_stand_to_action_goal(vision_r_goal, height=height)
        return self.command_async(
            "orient",
            action_goal_stand,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _body_gaze_at_vision_t_body_desired(
        self, body_t_target: SE3, vision_t_body: Optional[SE3] = None
    ) -> Optional[RobotCommandFeedback]:
        """
        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            vision_t_body: SE3 transform from the vision from to the current body position
            timeout_sec: A timeout for waiting on a response/result from the action server. The timeout arg is used
                on each command, so max duration is 2x timeout. Setting to None creates no timeout.

        Return:
            vision_t_body_desired: body_t_target: SE3 transform to the target you want to gaze at

        """
        x_axis = np.array([1, 0, 0])
        new_vec_norm = normalize([body_t_target.t])[0]

        # actual interesting math to compute angle and axis
        normal_axis = np.cross(x_axis, new_vec_norm)
        angle = np.arccos(np.dot(x_axis, new_vec_norm))
        body_current_t_body_pointed_at_target = SE3.AngleAxis(angle, normal_axis)

        if vision_t_body is None:
            vision_t_body = self.lookup_a_tform_b(
                self.FrameHint.VISION.value, self.FrameHint.BODY.value, wait_for_frames=True
            )

        vision_t_body_desired = vision_t_body * body_current_t_body_pointed_at_target
        return vision_t_body_desired

    def body_gaze_at(
        self, body_t_target: SE3, vision_t_body: Optional[SE3] = None, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Aligns the x-axis of the body frame along the vector from the current body position
        to the target. If you don't provide vision_t_body, you must set the tf listener for spot.

        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            vision_t_body: SE3 transform from the vision from to the current body position
            timeout_sec: A timeout for waiting on a response/result from the action server. The timeout arg is used
                on each command, so max duration is 2x timeout. Setting to None creates no timeout.

        Returns:
            The feedback from the SpotDriver about the result of the commands if it was accepted, None otherwise
        """
        vision_t_body_desired = self._body_gaze_at_vision_t_body_desired(body_t_target, vision_t_body)
        return self.walk_and_orient(vision_t_body_desired, timeout_sec=timeout_sec)

    def body_gaze_at_async(
        self,
        body_t_target: SE3,
        vision_t_body: Optional[SE3] = None,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotComposeCommandHandle]:
        """Aligns the x-axis of the body frame along the vector from the current body position
        to the target. If you don't provide vision_t_body, you must set the tf listener for spot.

        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            vision_t_body: SE3 transform from the vision from to the current body position
            result_callback: A callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: A callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: A callback to process/handle when any of actions fail for various reasons

        """
        vision_t_body_desired = self._body_gaze_at_vision_t_body_desired(body_t_target, vision_t_body)
        return self.walk_and_orient_async(
            vision_t_body_desired,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def walk_and_orient(
        self, vision_t_goal: SE3, body_height: Optional[float] = None, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send the command to walk and rotate to a specific SE3 pose

        Args:
            vision_t_goal: A SE3 transform from the vision frame to where the user wants the spot to walk to
            body_height: desired height of the body relative to spot's normal height. Passed directly into
                synchro_stand_command. If None, then height will be inferred based on vision_t_goal
            timeout_sec: A timeout for waiting on a response/result from the action server. The timeout arg is used
                on each command, so max duration is 2x timeout. Setting to None creates no timeout.

        Returns:
            The feedback from the SpotDriver about the result of the commands if it was accepted, None otherwise
        """
        if not self.walk_to(
            vision_t_goal.yaw_SE2(),
            timeout_sec=timeout_sec,
        ):
            return None

        rpy = vision_t_goal.rpy()
        new_pose_t_goal = SO3.RPY(rpy[0], rpy[1], 0)
        if body_height is None:
            if self._tf_listener is None:
                raise ValueError(
                    "unable to infer body height from vision_t_goal. "
                    "Either pass in 'body_height' explicitly, or call set_tf_listener(tflistener) "
                    "where tflistener should be a TFListenerWrapper"
                )
            body_t_vision = self.lookup_a_tform_b(
                self.FrameHint.BODY.value, self.FrameHint.VISION.value, wait_for_frames=True
            )
            body_t_goal = body_t_vision * vision_t_goal
            body_height = body_t_goal.z

        return self.stand_with_orientation(new_pose_t_goal, height=body_height, timeout_sec=timeout_sec)

    def _get_arm_gaze_at_action_goal(self, point: R3, frame: FrameHint) -> RobotCommand.Goal:
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
        on_failure_callback: Optional[Callable[[None], None]] = None,
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

    def walk_and_orient_async(
        self,
        vision_t_goal: SE3,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotComposeCommandHandle]:
        """Send the command to walk and rotate to a specific SE3 pose

        Args:
            vision_t_goal: A SE3 transform from the vision frame to where the user wants the spot to walk to
            result_callback: A callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: A callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: A callback to process/handle when any of actions fail for various reasons

        Returns:
            RobotComposeCommandHandle: A class to handle a list with multiple async RobotCommand (actions). The actions
                are sent sequentially to SpotDriver via an action client
        """
        walk_partial = partial(
            self.walk_to_async,
            vision_t_goal.yaw_SE2(),
            height=vision_t_goal.z,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )
        stand_partial = partial(
            self.stand_with_orientation_async,
            SO3(vision_t_goal),
            height=vision_t_goal.z,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )
        return RobotComposeCommandHandle([walk_partial, stand_partial], result_callback)

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
            rate_timer = self._node.create_rate(rate)
            start_time = self._node.get_clock().now()
            dt = self._node.get_clock().now() - start_time
            while dt.nanoseconds * 1e-9 < duration:
                self._publish_velocity_cmd(body_velocity)
                rate_timer.sleep()
                dt = self._node.get_clock().now() - start_time

    def _publish_velocity_cmd(self, body_velocity: Twist2) -> None:
        velocity_cmd_msg = Twist()
        velocity_cmd_msg.linear.x = body_velocity.v[0]
        velocity_cmd_msg.linear.y = body_velocity.v[1]
        velocity_cmd_msg.angular.z = body_velocity.w
        self._velocity_cmd_publisher.publish(velocity_cmd_msg)

    def _get_walk_trajectory_action_goal(
        self,
        vision_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
    ) -> RobotCommand.Goal:
        """Create an action goal for visiting a sequence of poses with a given time schedule and interpolation settings

        Args:
            vision_t_trajectory: SE2Trajectory, i.e., time-parameterization of sequence of poses in vision frame
            timestep_sec: time step used to sample poses from the trajectory
            interpolation: method used to interpolate between sampled poses (0 = none, 1 = linear, 2 = cubic)
        """
        self._node.get_logger().info("Walking along commanded trajectory...")

        proto_goal = timestamped_se2_vecs_to_robot_command_proto(
            *vision_t_trajectory(timestep_sec=timestep_sec, with_timestamps=True),
            interpolation=interpolation,
            mobility_params=self._get_mobility_params(),
            base_frame=VISION_FRAME_NAME,
        )
        return self._convert_proto_to_robot_cmd_action_goal(proto_goal)

    def walk_trajectory(
        self,
        vision_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        timeout_sec: Optional[float] = None,
    ) -> Optional[RobotCommand.Result]:
        """Send a command to walk along a trajectory synchronously

        Args:
            vision_t_trajectory: SE2Trajectory, i.e., time-parameterization of sequence of poses in vision frame
            timestep_sec: time step used to sample poses from the trajectory
            interpolation: method used to interpolate between sampled poses (0 = none, 1 = linear, 2 = cubic)
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout
        """
        action_goal = self._get_walk_trajectory_action_goal(vision_t_trajectory, timestep_sec, interpolation)
        return self.command("walk_trajectory", action_goal, timeout_sec)

    def walk_trajectory_async(
        self,
        vision_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: int = 2,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send the command to walk along a trajectory asynchronously

        Args:
            vision_t_trajectory: SE2Trajectory, i.e., time-parameterization of sequence of poses in vision frame
            timestep_sec: time step used to sample poses from the trajectory
            interpolation: method used to interpolate between sampled poses (0 = none, 1 = linear, 2 = cubic)
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self._get_walk_trajectory_action_goal(vision_t_trajectory, timestep_sec, interpolation)
        return self.command_async(
            "walk_trajectory",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_stow_action_goal(self) -> RobotCommand.Goal:
        """Create the ROS msg to stow the arm"""
        proto = RobotCommandBuilder.arm_stow_command()
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def stow_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to stow the arm synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self._get_arm_stow_action_goal()
        return self.command("stow_arm", action_goal, timeout_sec)

    def stow_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to stow the arm asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self._get_arm_stow_action_goal()
        return self.command_async(
            "stow_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_ready_action_goal(self) -> RobotCommand.Goal:
        """Create the ROS msg to move the arm to the ready position"""
        proto = RobotCommandBuilder.arm_ready_command()
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def ready_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to move the arm to the ready position synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
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
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to move the arm to the ready position asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
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

    def _get_arm_carry_action_goal(self) -> RobotCommand.Goal:
        """Create the ROS msg to move the arm to the carry position"""
        proto = RobotCommandBuilder.arm_carry_command()
        return self._convert_proto_to_robot_cmd_action_goal(proto)

    def carry_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Move the arm to the carry position synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self._get_arm_carry_action_goal()
        return self.command("carry_arm", action_goal, timeout_sec)

    def carry_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Move the arm to the carry position asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
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
        self, vision_t_goal: SE3, move_duration_sec: float, body_follow: bool = False
    ) -> RobotCommand.Goal:
        """Create the ROS Msg for moving the end effector to specific pose"""
        quat = UnitQuaternion(vision_t_goal)
        arm_pose_command = RobotCommandBuilder.arm_pose_command(
            x=vision_t_goal.x,
            y=vision_t_goal.y,
            z=vision_t_goal.z,
            qw=quat.A[0],
            qx=quat.A[1],
            qy=quat.A[2],
            qz=quat.A[3],
            frame_name=VISION_FRAME_NAME,
            seconds=move_duration_sec,
        )
        if body_follow:
            follow_arm_command = RobotCommandBuilder.follow_arm_command()
            synchro_command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_pose_command)
            return self._convert_proto_to_robot_cmd_action_goal(synchro_command)
        else:
            return self._convert_proto_to_robot_cmd_action_goal(arm_pose_command)

    def move_end_effector_cartesian(
        self,
        vision_t_goal: SE3,
        move_duration_sec: float = 5.0,
        body_follow: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Send command to move the end effector to a specific pose synchronously

        Args:
            vision_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout
            body_follow: If True, the robot's body would automatically follow the arm to achieve the goal

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        See:
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L1224
        """
        action_goal = self._get_arm_pose_action_goal(vision_t_goal, move_duration_sec, body_follow=body_follow)
        return self.command("move_end_effector_cartesian", action_goal, timeout_sec)

    def move_end_effector_cartesian_async(
        self,
        vision_t_goal: SE3,
        move_duration_sec: float = 5.0,
        body_follow: bool = False,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send command to move the end effector to a specific pose asynchronously

        Args:
            vision_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO

        Returns:
            TODO
        """
        action_goal = self._get_arm_pose_action_goal(vision_t_goal, move_duration_sec, body_follow=body_follow)
        return self.command_async(
            "move_end_effector_cartesian",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def _get_arm_trajectory_action_goal(
        self,
        root_frame_name: str,
        axis_mode_dict: AxisModeDict,
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
    ) -> RobotCommand.Goal:
        """Return a goal corresponding to the arm following a Cartesian trajectory.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid undefined behaviour.

        Args:
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            axis_mode_dict: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
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

        arm_cartesian_command.x_axis = axis_mode_dict.x
        arm_cartesian_command.y_axis = axis_mode_dict.y
        arm_cartesian_command.z_axis = axis_mode_dict.z
        arm_cartesian_command.rz_axis = axis_mode_dict.rx
        arm_cartesian_command.ry_axis = axis_mode_dict.ry
        arm_cartesian_command.rz_axis = axis_mode_dict.rz

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
        axis_mode_dict: AxisModeDict,
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
            axis_mode_dict: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
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
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """
        action_goal = self._get_arm_trajectory_action_goal(
            root_frame_name=root_frame_name,
            axis_mode_dict=axis_mode_dict,
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
        axis_mode_dict: AxisModeDict,
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
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Return a goal corresponding to the arm following a Cartesian trajectory.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid unexpected behaviour.

        Args:
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            axis_mode_dict: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
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
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons
        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """

        action_goal = self._get_arm_trajectory_action_goal(
            root_frame_name=root_frame_name,
            axis_mode_dict=axis_mode_dict,
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
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        raise NotImplementedError()

    def move_end_effector_force_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """TODO

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        raise NotImplementedError()

    def _get_arm_impedance_action_goal(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[Tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
    ) -> RobotCommand.Goal:
        """Create the ROS Msg for moving the end effector to specific
        pose using hybrid position/impedance control.

        The stiffness and damping gains are defaulted to their maximum values.
        Note that this is in the task frame, so these were probably empirically obtained.
        (See code comment in here
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_impedance_control/arm_impedance_control.py.)
        """  # noqa: E501

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

        return self._convert_proto_to_robot_cmd_action_goal(robot_cmd)

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
            wrist_t_tool: The tool pose relative to the parent link (link_wr1).
                This can also be thought of as the "remote center" frame.
            trajectory: Discrete waypoints of where we want the tool to be relative to the task frame
                and the duration of time to slerp over.
            stiffness: the position/stiffness gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            damping: the derivative/damping gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            wrench_nm: the feedforward wrench for x, y, z, torque_x, torque_y, torque_z in the task frame
            timeout_sec: the timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self._get_arm_impedance_action_goal(
            root_frame_name, root_t_task, wrist_t_tool, trajectory, stiffness, damping, wrench_nm
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
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Impedance control as variable stiffness/damping position control combined with a
            feed-forward force control as a wrench.

        Args:
            root_frame_name: name of root frame as a string
            root_t_task: the transform from the root frame to the "task frame" (this doesn't need a name)
            wrist_t_tool: The tool pose relative to the parent link (link_wr1).
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
            TODO
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

    def claw_gripper_open(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to open the gripper

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_open_command())
        return self.command("claw_gripper_open", action_goal, timeout_sec)

    def claw_gripper_open_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to open the gripper asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_open_command())
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
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command("claw_gripper_close", action_goal, timeout_sec)

    def claw_gripper_close_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to close the gripper asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self._convert_proto_to_robot_cmd_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command_async(
            "claw_gripper_close",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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
        """  # noqa: E501

        # Build manipulation command.
        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_data.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_data.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_data.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_data.k[5]

        pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=self._get_basic_tform_tree_snapshot(image_timestamp),
            frame_name_image_sensor=camera_data.header.frame_id,
            camera_model=pinhole_model,
        )

        # We can specify where in the gripper we want to grasp.
        grasp.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio

        if apply_grasp_params_func is not None:
            apply_grasp_params_func(grasp)

        # Build the proto.
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        action_goal = self._convert_proto_to_manip_cmd_action_goal(grasp_request)
        return self.manipulation("grasp_in_image", action_goal, 10.0)

    def grasp_in_image_async(
        self,
        pixel_xy: List[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
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
        """  # noqa: E501

        # Build manipulation command.
        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_data.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_data.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_data.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_data.k[5]

        pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=self._get_basic_tform_tree_snapshot(image_timestamp),
            frame_name_image_sensor=camera_data.header.frame_id,
            camera_model=pinhole_model,
        )

        # We can specify where in the gripper we want to grasp.
        grasp.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio

        if apply_grasp_params_func is not None:
            apply_grasp_params_func(grasp)

        # Build the proto.
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        action_goal = self._convert_proto_to_manip_cmd_action_goal(grasp_request)
        return self.manipulation_async(
            "grasp_in_image",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def palm_grasp_in_image_top_down(
        self,
        pixel_xy: List[float],
        image_data: Any,
    ) -> RobotCommand.Goal:
        """Grasp an item in an image taken with a camera on spot with the palm
        pressed into the object.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            image_data: Image data taken using spot (need to figure out data type).
        """
        raise NotImplementedError()
        # return self.grasp_in_image(pixel_xy, image_data, 0.0, self.set_tight_top_down_grasp)

    def grasp_along_ray(
        self,
        task_frame_t_ray_start: R3,
        task_frame_t_ray_end: R3,
        task_frame_name: str,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Cast a ray in the world and pick the first object found along the ray.
        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        Example: You see the object you are interested in with the gripper's camera.
        To grasp it, you cast a ray from the camera out to 4 meters (well past the object).
        To do this you'd set:
            ray_start_rt_frame: camera's position
            ray_end_rt_frame: camera's position + unit vector along ray of interest * 4 meters
        """

        grasp = manipulation_api_pb2.PickObject(frame_name=task_frame_name)
        grasp.object_rt_frame.CopyFrom(
            geometry_pb2.Vec3(x=task_frame_t_ray_end[0], y=task_frame_t_ray_end[1], z=task_frame_t_ray_end[2])
        )
        # grasp.frame_name = task_frame_name
        # Don't move the robot base, but automatically look at the grasp target with the hand camera.
        # grasp.walk_gaze_mode = manipulation_api_pb2.PICK_AUTO_GAZE

        # We can specify where in the gripper we want to grasp.
        grasp.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio

        if apply_grasp_params_func is not None:
            apply_grasp_params_func(grasp)

        # Build the proto
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object=grasp)

        action_goal = self._convert_proto_to_manip_cmd_action_goal(grasp_request)
        return self.manipulation("grasp_long_ray", action_goal, 10.0)

    def shutdown(self) -> None:
        """Shutdown all the internal components"""
        self._executor.shutdown()
        self._node.destroy_node()
        try:
            # seconds
            self._spin_thread.join(1)
        except KeyboardInterrupt:
            pass

        # TODO: Is there anything else that we should shutdown?


def main() -> None:
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    args = parser.parse_args()

    node = rclpy.create_node("spot_basic_cli")
    spot_basic = SpotBasic(robot_name=args.robot_name, parent_node_name=node.get_name())

    while rclpy.ok():
        cmd = input("Please enter a command:\n" + " ".join(SpotBasic.TRIGGER_SERVICES) + " exit\n> ")
        if cmd == "exit":
            break
        success = spot_basic.trigger_service(cmd)
        if not success:
            print(f"Error with command: {cmd}")
        else:
            print(f"Successfully executed command: {cmd}")

    spot_basic.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
