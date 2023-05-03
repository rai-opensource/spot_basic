#!/usr/bin/env python3
#  Copyright [2023] Boston Dynamics AI Institute, Inc.
import sys
from enum import Enum, IntEnum
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import py_trees
import rclpy
import spot_driver.conversions as conv
from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai.utilities.math_helpers import SE2Pose, SE2Velocity, SE3Pose
from bdai_ros.utilities.manual_conversions import se3pose_to_proto, to_ros_pose
from bosdyn.api import geometry_pb2, manipulation_api_pb2, robot_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.trajectory_pb2 import POS_INTERP_CUBIC, PositionalInterpolation
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.util import seconds_to_duration
from bosdyn_msgs.msg import ListWorldObjectRequest, ListWorldObjectResponse, RobotCommandFeedback
from geometry_msgs.msg import Twist
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future
from spot_msgs.action import RobotCommand
from spot_msgs.srv import ClearBehaviorFault, ListGraph, ListWorldObjects, SetLocomotion, SetVelocity
from std_msgs.msg import ColorRGBA as Color
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker

from spot_utilities.robot_command_helpers import timestamped_se2_vecs_to_robot_command_proto


class RobotCommandHandle(object):
    """Handle for an asynchronous RobotCommand sent via an action client"""

    def __init__(self, robot_name: str, command_name: str):
        """Constructor

        Arg:
            robot_name: Name of the robot
            command_name: representative name of the command used for logging
        """
        self.__robot_name = robot_name
        self.__command_name = command_name

        self.__send_goal_future: Optional[Future] = None
        self.__goal_handle = None
        self.__get_result_future: Optional[Future] = None
        self.__feedback_callback_internal: Optional[Callable[[RobotCommandFeedback], None]] = None
        self.__result_callback_internal: Optional[Callable[[RobotCommandFeedback], None]] = None
        self.__on_failure_callback: Optional[Callable] = None
        self.result: Optional[RobotCommandFeedback] = None

        self.__logger = RcutilsLogger(name=f"{self.__robot_name}.{self.__command_name}")

    def set_send_goal_future(self, send_goal_future: Future) -> None:
        """Sets the future received from sending the Action.Goal and sets up the callback for when a response is
        received
        """
        self.__send_goal_future = send_goal_future
        self.__send_goal_future.add_done_callback(self.__goal_response_callback)

    def set_feedback_callback(self, feedback_callback: Callable[[RobotCommandFeedback], None]) -> None:
        """Sets the callback used to process feedback received while a RobotCommand is being executed"""
        self.__feedback_callback_internal = feedback_callback

    def set_result_callback(self, result_callback: Callable[[RobotCommandFeedback], None]) -> None:
        """Sets the callback for processing the resulting feedback from executing a RobotCommand"""
        self.__result_callback_internal = result_callback

    def set_on_failure_callback(self, on_failure_callback: Callable) -> None:
        """ """
        self.__on_failure_callback = on_failure_callback

    def __goal_response_callback(self, future: Future) -> None:
        """Callback that handles receiving a response from the ActionServer

        Note: This does not handle receiving a result directly
        """
        if future.result() is None:
            self.__logger.error(f"RobotCommand request failed: {future.exception():!r}")
            self.__failure()
            return
        self.__goal_handle = future.result()
        if not self.__goal_handle.accepted:
            self.__logger.error("RobotCommand rejected")
            self.__failure()
            return

        self.__logger.info("RobotCommand accepted")

        self.__get_result_future = self.__goal_handle.get_result_async()
        self.__get_result_future.add_done_callback(self.__get_result_callback)

    def get_feedback_callback(self, feedback: RobotCommand.Feedback) -> None:
        """Public feedback callback given to the ActionClient

        Currently just pulls the feedback from the future and passes it to the user provided callback
        """
        if self.__feedback_callback_internal is not None:
            self.__feedback_callback_internal(feedback.feedback)

    def __get_result_callback(self, future: Future) -> None:
        """Callback for pulling the resulting feedback out of the future and passing it to a user provided callback"""
        result = future.result().result
        if not result.success:
            self.__logger.error(f"RobotCommand resulted in failure. Msg: {result.message}")
            self.__failure()
            return

        self.result = result.result
        if self.__result_callback_internal is not None:
            self.__result_callback_internal(self.result)
        # TODO: cleanup

    def cancel(self) -> None:
        """Send the desire to cancel a command"""
        self.__logger.debug("Canceling RobotCommand")

        if self.__goal_handle is not None:
            future = self.__goal_handle.cancel_goal_async()
            future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future: Future) -> None:
        """Callback for handling the response to attempting to cancel a command"""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.__logger.debug("RobotCommand successfully canceled")
        else:
            self.__logger.debug("RobotCommand failed to canceled")
        # TODO: cleanup

    def __failure(self) -> None:
        if self.__on_failure_callback is not None:
            self.__on_failure_callback()


class RobotCommandBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name: str):
        super().__init__(name)
        self._spot: Optional[SpotBasic] = None
        self._action_handle: Optional[RobotCommandHandle] = None
        self._status: py_trees.common.Status = py_trees.common.Status.INVALID

    def setup(self, **kwargs: Dict[str, Any]) -> None:
        self._spot = kwargs.get("spot")

    def update(self) -> py_trees.common.Status:
        return self._status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if self._status == py_trees.common.Status.RUNNING and new_status == py_trees.common.Status.INVALID:
            self._action_handle.cancel()

    # pylint: disable=unused-argument
    def _get_result_callback(self, feedback: RobotCommandFeedback) -> None:
        self._status = py_trees.common.Status.SUCCESS

    # pylint: enable=unused-argument

    def _on_failure_callback(self) -> None:
        self._status = py_trees.common.Status.FAILURE


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
        self.__node = rclpy.create_node(node_name=f"{parent_node_name}_spot_basic", namespace=robot_name)

        self.__robot_name = robot_name
        self.__disable_collisions = disable_collisions
        self.__dry = dry
        self.__rviz = rviz

        # TODO: are there other frame names to grab?
        self.__body_frame_name = f"{self.robot_name}/{BODY_FRAME_NAME}"
        self.__hand_frame_name = f"{self.robot_name}/{HAND_FRAME_NAME}"
        self.__vision_frame_name = f"{self.robot_name}/{VISION_FRAME_NAME}"

        self.__executor = SingleThreadedExecutor()
        self.__executor.add_node(self.__node)
        self.__spin_thread = Thread(target=self.__spin_executor, args=(self.__executor,), daemon=True)
        self.__callback_group = ReentrantCallbackGroup()

        if self.__rviz:
            self.__vis_pub = self.__node.create_publisher(Marker, "/visualization_marker", 10)

        self.__velocity_cmd_publisher = self.__node.create_publisher(Twist, f"/{self.__robot_name}/cmd_vel", 10)

        self.__trigger_clients = {}
        for command_name in self.TRIGGER_SERVICES:
            self.__trigger_clients[command_name] = self.__node.create_client(
                Trigger, f"/{self.__robot_name}/{command_name}", callback_group=self.__callback_group
            )
        self.__stair_mode_client = self.__node.create_client(
            SetBool, f"/{self.__robot_name}/stair_mode", callback_group=self.__callback_group
        )
        self.__locomotion_mode_client = self.__node.create_client(
            SetLocomotion, f"/{self.__robot_name}/locomotion_mode", callback_group=self.__callback_group
        )
        self.__max_velocity_client = self.__node.create_client(
            SetVelocity, f"/{self.__robot_name}/max_velocity", callback_group=self.__callback_group
        )
        self.__clear_behavior_fault_client = self.__node.create_client(
            ClearBehaviorFault, f"/{self.__robot_name}/clear_behavior_fault", callback_group=self.__callback_group
        )
        self.__list_graph_client = self.__node.create_client(
            ListGraph, f"/{self.__robot_name}/list_graph", callback_group=self.__callback_group
        )
        self.__list_world_objects_client = self.__node.create_client(
            ListWorldObjects, f"/{self.__robot_name}/list_world_objects", callback_group=self.__callback_group
        )
        self.__get_parameters_client = self.__node.create_client(
            GetParameters, f"/{self.__robot_name}/spot_ros2/get_parameters", callback_group=self.__callback_group
        )

        self.__robot_command_action_client = ActionClient(self.__node, RobotCommand, "robot_command")
        self.__node.get_logger().info(f"Waiting for the 'RobotCommand' action server for {self.__robot_name}")
        self.__robot_command_action_client.wait_for_server()
        self.__spin_thread.start()

    @property
    def robot_name(self):
        return self.__robot_name

    @property
    def body_frame_name(self):
        return self.__body_frame_name

    @property
    def hand_frame_name(self):
        return self.__hand_frame_name

    @property
    def vision_frame_name(self):
        return self.__vision_frame_name

    @staticmethod
    def __spin_executor(executor: SingleThreadedExecutor) -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    @staticmethod
    def __wait_for_future(future: Future, timeout_sec: Optional[float] = None) -> bool:
        """Blocks while waiting for a future to become done

        Args:
            future: The future to be waited on
            timeout_sec: An optional timeout for who long to wait

        Returns:
            True if successful, False if the timeout was triggered
        """
        event = Event()

        def done_callback(_):
            nonlocal event
            event.set()

        future.add_done_callback(done_callback)
        if timeout_sec is None:
            return event.wait()
        else:
            return event.wait(timeout_sec)

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
        self.__node.get_logger().info(f"Trigger command '{command_name}' for {self.robot_name}")
        request = Trigger.Request()
        try:
            future = self.__trigger_clients[command_name].call_async(request)
        except KeyError:
            self.__node.get_logger().error(f"No trigger command {command_name}")
            return False

        if not self.__wait_for_future(future):
            return False

        response = future.result()
        if not response.success:
            self.__node.get_logger().error(
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
        self.__node.get_logger().info(f"Setting stair_mode to '{value}' for {self.robot_name}")
        request = SetBool.Request()
        request.data = value
        future = self.__stair_mode_client.call_async(request)
        if not self.__wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Setting stair_mode to '{value}' for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

    def set_locomotion_mode(self, l_hint: LocomotionHint) -> bool:
        """Set the locomotion mode on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.__node.get_logger().info(f"Setting locomotion_mode to '{l_hint}' for {self.robot_name}")
        request = SetLocomotion.Request()
        request.locomotion_mode = int(l_hint)
        future = self.__locomotion_mode_client.call_async(request)
        if not self.__wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Setting locomotion_mode to '{l_hint}' for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

    def set_max_velocity(self, velocity_limit: Twist) -> SetVelocity.Response:
        """Set the maximum velocity on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.__node.get_logger().info(f"Setting max_velocity to '{velocity_limit}' for {self.robot_name}")
        request = SetVelocity.Request()
        request.velocity = velocity_limit
        future = self.__max_velocity_client.call_async(request)
        if not self.__wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Setting stair_mode to '{velocity_limit}' for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True

    def clear_behavior_fault(self, bid: int) -> ClearBehaviorFault.Response:
        """Clear a behavior fault on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        # TODO: how to set the bid?
        self.__node.get_logger().info(f"Clearing behavior fault {bid} for {self.robot_name}")
        request = ClearBehaviorFault.Request()
        request.id = bid
        future = self.__clear_behavior_fault_client.call_async(request)
        if not self.__wait_for_future(future):
            return False
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Clearing behavior fault {bid} for {self.robot_name} failed. Msg: {response.message}"
            )
            return False
        return True

    def list_graph(self, upload_filepath: str) -> List[str]:
        """Request a list of waypoint ids"""
        self.__node.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        request = ListGraph.Request()
        request.upload_filepath = upload_filepath
        future = self.__list_graph_client.call_async(request)
        if not self.__wait_for_future(future):
            return []
        response = future.result()
        return response.waypoint_ids

    def list_world_objects(self, p_request: ListWorldObjectRequest) -> ListWorldObjectResponse:
        """Requests a list of world objects

        TODO: not use ROS msgs
        """
        self.__node.get_logger().info(f"List world objects for {self.robot_name}")
        request = ListWorldObjects.Request()
        request.request = p_request
        future = self.__list_world_objects_client.call_async(request)
        if not self.__wait_for_future(future):
            return None
        response = future.result()
        return response.response

    def has_arm(self) -> Optional[bool]:
        """Queries if the spot has an arm

        Returns:
            True if the spot has an arm, False if it does not, and None if the query failed
        """
        self.__node.get_logger().info(f"Checking if {self.robot_name} has an arm")
        request = GetParameters.Request()
        request.names = ["has_arm"]
        future = self.__get_parameters_client.call_async(request)
        if not self.__wait_for_future(future):
            return None
        response = future.result()
        param = response.values[0]
        return param.bool_value

    def __send_command_goal_and_wait(
        self, command_name: str, goal: RobotCommand.Goal, timeout_sec=None
    ) -> Optional[RobotCommandFeedback]:
        """Sends a goal to the robot command action server and waits for it to be done

        Args:
            command_name: A representative name of the command for logging
            goal: The Action Goal sent to the action server
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        Note: This is a blocking call
        """
        self.__node.get_logger().info(f"Sending RobotCommand [{self.__robot_name}.{command_name}]")

        future = self.__robot_command_action_client.send_goal_async(goal)
        if not self.__wait_for_future(future, timeout_sec=timeout_sec):
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.__node.get_logger().error(f"RobotCommand [{self.__robot_name}.{command_name}] was not accepted")
            return None

        self.__node.get_logger().info(f"RobotCommand [{self.__robot_name}.{command_name}] accepted")
        future = goal_handle.get_result_async()
        if not self.__wait_for_future(future, timeout_sec=timeout_sec):
            return None

        self.__node.get_logger().info(f"Received response for RobotCommand [{self.__robot_name}.{command_name}]")
        return future.result()

    def command(self, command_name: str, goal: RobotCommand.Goal, timeout_sec=None) -> Optional[RobotCommandFeedback]:
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
        if goal is None:
            self.__node.get_logger.warn("Cannot send NULL ActionGoal")
            # TODO: Is there something better that we can return with information about the failure?
            # maybe a RobotCommand.Result that represents failure?
            return None
        if not self.__dry:
            return self.__send_command_goal_and_wait(command_name, goal, timeout_sec)
        # TODO: What to return on a dry run? Just success?
        return None

    def command_async(
        self,
        command_name: str,
        goal: RobotCommand.Goal,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
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
        if goal is None:
            # Is there something better that we can return with information about the failure?
            return None
        if not self.__dry:
            handle = RobotCommandHandle(robot_name=self.__robot_name, command_name=command_name)
            if result_callback is not None:
                handle.set_result_callback(result_callback)

            if feedback_callback is None:
                send_goal_future = self.__robot_command_action_client.send_goal_async(goal)
            else:
                handle.set_feedback_callback(feedback_callback)
                send_goal_future = self.__robot_command_action_client.send_goal_async(
                    goal, feedback_callback=handle.get_feedback_callback
                )
            handle.set_send_goal_future(send_goal_future)

            if on_failure_callback is not None:
                handle.set_on_failure_callback(on_failure_callback)

            return handle
        # TODO: What to return on a dry run? Just success?
        return None

    def __draw_arrow(self, frame_id: str, ns: str, frame_t_arrow: SE2Pose | SE3Pose, color: Color) -> None:
        """Draw an arrow in RVIZ"""
        marker = Marker()
        marker.header.stamp = self.__node.get_clock().now().to_msg()
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
        self.__vis_pub.publish(marker)

    @staticmethod
    def __convert_proto_to_action_goal(proto) -> RobotCommand.Goal:
        action_goal = RobotCommand.Goal()
        conv.convert_proto_to_bosdyn_msgs_robot_command(proto, action_goal.command)
        return action_goal

    def __get_battery_change_pose(self, dir_hint: BatteryChangeDirHint):
        proto = RobotCommandBuilder.battery_change_pose_command(int(dir_hint))
        return self.__convert_proto_to_action_goal(proto)

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
        action_goal = self.__get_battery_change_pose(dir_hint)
        return self.command("go_to_battery_change_pose", action_goal, timeout_sec)

    def go_to_battery_change_pose_async(
        self,
        dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """Send command to go to the battery change pose asynchronously

        Args:
            dir_hint: A hint for the direction to roll
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self.__get_battery_change_pose(dir_hint)
        return self.command_async(
            "go_to_battery_change_pose",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def __get_mobility_params(self) -> spot_command_pb2.MobilityParams:
        """Create the MobilityParams object for use when building robot commands"""

        # NOTE so far there is no way to change this and it might as well be set as an attribute during init.
        # but possibly we'll want to change some of these parameters on the fly someday
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=self.__disable_collisions,
            disable_vision_foot_obstacle_avoidance=self.__disable_collisions,
            disable_vision_foot_constraint_avoidance=self.__disable_collisions,
            obstacle_avoidance_padding=0.001,
        )
        return spot_command_pb2.MobilityParams(obstacle_params=obstacles)

    def __get_walk_to_action_goal(self, vision_t_goal: SE2Pose, height: float) -> RobotCommand.Goal:
        """TODO

        Args:
            vision_t_goal: A SE2 transform from the vision frame to where the user wants the spot to walk to
            height: The height of the body

        Returns:
            An Action.Goal representing the command for a spot to walk to a specific goal location
        """
        self.__node.get_logger().info("Walking to goal...")

        if self.__rviz:
            self.__draw_arrow(self.vision_frame_name, self.body_frame_name, vision_t_goal, color=Color(g=1.0, a=1.0))

        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=vision_t_goal.x,
            goal_y=vision_t_goal.y,
            goal_heading=vision_t_goal.yaw,
            frame_name=VISION_FRAME_NAME,
            body_height=height,
            params=self.__get_mobility_params(),
        )
        return self.__convert_proto_to_action_goal(proto_goal)

    def walk_to(
        self, vision_t_goal: SE2Pose, height: float = 0, timeout_sec: Optional[float] = None
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
        action_goal = self.__get_walk_to_action_goal(vision_t_goal, height)
        return self.command("walk_to", action_goal, timeout_sec)

    def walk_to_async(
        self,
        vision_t_goal: SE2Pose,
        height: float = 0,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
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
        action_goal = self.__get_walk_to_action_goal(vision_t_goal, height)
        return self.command_async(
            "walk_to",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def walk_velocity(self, body_velocity: SE2Velocity) -> None:
        """Send the command to walk/turn in a specific body-frame direction at a specific speed

        Args:
            body_velocity: the desired SE2 in the body frame
        """
        velocity_cmd_msg = Twist()
        velocity_cmd_msg.linear.x = body_velocity.linear_velocity_x
        velocity_cmd_msg.linear.y = body_velocity.linear_velocity_y
        velocity_cmd_msg.angular.z = body_velocity.angular_velocity
        self.__velocity_cmd_publisher.publish(velocity_cmd_msg)

    def __get_walk_trajectory_action_goal(
        self,
        vision_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: PositionalInterpolation = POS_INTERP_CUBIC,
    ) -> RobotCommand.Goal:
        """Create an action goal for visiting a sequence of poses with a given time schedule and interpolation settings

        Args:
            vision_t_trajectory: SE2Trajectory, i.e., time-parameterization of sequence of poses in vision frame
            timestep_sec: time step used to sample poses from the trajectory
            interpolation: method used to interpolate between sampled poses (0 = none, 1 = linear, 2 = cubic)
        """
        self.__node.get_logger().info("Walking along commanded trajectory...")

        proto_goal = timestamped_se2_vecs_to_robot_command_proto(
            *vision_t_trajectory(timestep_sec=timestep_sec, with_timestamps=True),
            interpolation=interpolation,
            mobility_params=self.__get_mobility_params(),
            base_frame=VISION_FRAME_NAME,
        )
        return self.__convert_proto_to_action_goal(proto_goal)

    def walk_trajectory(
        self,
        vision_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: PositionalInterpolation = POS_INTERP_CUBIC,
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
        action_goal = self.__get_walk_trajectory_action_goal(vision_t_trajectory, timestep_sec, interpolation)
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
    ) -> Optional[RobotCommandHandle]:
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
        action_goal = self.__get_walk_trajectory_action_goal(vision_t_trajectory, timestep_sec, interpolation)
        return self.command_async(
            "walk_trajectory",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def __get_arm_stow_action_goal(self):
        """Create the ROS msg to stow the arm"""
        proto = RobotCommandBuilder.arm_stow_command()
        return self.__convert_proto_to_action_goal(proto)

    def stow_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to stow the arm synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self.__get_arm_stow_action_goal()
        return self.command("stow_arm", action_goal, timeout_sec)

    def stow_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """Send a command to stow the arm asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self.__get_arm_stow_action_goal()
        return self.command_async(
            "stow_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def __get_arm_ready_action_goal(self):
        """Create the ROS msg to move the arm to the ready position"""
        proto = RobotCommandBuilder.arm_ready_command()
        return self.__convert_proto_to_action_goal(proto)

    def ready_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Send a command to move the arm to the ready position synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self.__get_arm_ready_action_goal()
        return self.command("ready_arm", action_goal, timeout_sec)

    # Alias
    unstow_arm = ready_arm

    def ready_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """Send a command to move the arm to the ready position asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self.__get_arm_ready_action_goal()
        return self.command_async(
            "ready_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    # Alias
    unstow_arm_async = ready_arm_async

    def __get_arm_carry_action_goal(self):
        """Create the ROS msg to move the arm to the carry position"""
        proto = RobotCommandBuilder.arm_carry_command()
        return self.__convert_proto_to_action_goal(proto)

    def carry_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Move the arm to the carry position synchronously

        Args:
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise
        """
        action_goal = self.__get_arm_carry_action_goal()
        return self.command("carry_arm", action_goal, timeout_sec)

    def carry_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """Move the arm to the carry position asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self.__get_arm_carry_action_goal()
        return self.command_async(
            "carry_arm",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def __get_arm_pose_action_goal(self, vision_t_goal: SE3Pose, move_duration_sec: float) -> RobotCommand.Goal:
        """Create the ROS Msg for moving the end effector to specific pose"""
        proto = RobotCommandBuilder.arm_pose_command(
            x=vision_t_goal.x,
            y=vision_t_goal.y,
            z=vision_t_goal.z,
            qw=vision_t_goal.rotation.w,
            qx=vision_t_goal.rotation.x,
            qy=vision_t_goal.rotation.y,
            qz=vision_t_goal.rotation.z,
            frame_name=VISION_FRAME_NAME,
            seconds=move_duration_sec,
        )

        return self.__convert_proto_to_action_goal(proto)

    def move_end_effector_cartesian(
        self, vision_t_goal: SE3Pose, move_duration_sec: float = 5.0, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Send command to move the end effector to a specific pose synchronously

        Args:
            vision_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted, None otherwise

        See:
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L1224
        """
        action_goal = self.__get_arm_pose_action_goal(vision_t_goal, move_duration_sec)
        return self.command("move_end_effector_cartesian", action_goal, timeout_sec)

    def move_end_effector_cartesian_async(
        self,
        vision_t_goal: SE3Pose,
        move_duration_sec: float = 5.0,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
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
        action_goal = self.__get_arm_pose_action_goal(vision_t_goal, move_duration_sec)
        return self.command_async(
            "move_end_effector_cartesian",
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
        pass

    def move_end_effector_force_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """TODO

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        pass

    def __get_arm_impedance_action_goal(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3Pose,
        wrist_t_tool: SE3Pose,
        trajectory: List[Tuple[SE3Pose, float]],
        stiffness: List[float] = None,
        damping: List[float] = None,
        wrench_nm: List[float] = None,
    ) -> RobotCommand.Goal:
        """Create the ROS Msg for moving the end effector to specific
        pose using hybrid position/impedance control.

        The stiffness and damping gains are defaulted to their maximum values.
        Note that this is in the task frame, so these were probably empirically obtained.
        (See code comment in here
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_impedance_control/arm_impedance_control.py.)
        """  # noqa: E501

        if stiffness is None:
            stiffness = [500, 500, 500, 60, 60, 60]
        if damping is None:
            damping = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
        if wrench_nm is None:
            wrench_nm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Make sure we keep adjusting the body for the arm
        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                enable_hip_height_assist=True, enable_body_yaw_assist=False
            )
        )
        stand_command = RobotCommandBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control)
        )
        robot_cmd = robot_command_pb2.RobotCommand()
        robot_cmd.CopyFrom(stand_command)

        impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command
        impedance_cmd.root_frame_name = root_frame_name.value

        impedance_cmd.root_tform_task.CopyFrom(se3pose_to_proto(root_t_task))
        impedance_cmd.wrist_tform_tool.CopyFrom(se3pose_to_proto(wrist_t_tool))
        impedance_cmd.diagonal_stiffness_matrix.CopyFrom(geometry_pb2.Vector(values=stiffness))
        impedance_cmd.diagonal_damping_matrix.CopyFrom(geometry_pb2.Vector(values=damping))
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.x = wrench_nm[0]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.y = wrench_nm[1]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.force.z = wrench_nm[2]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.x = wrench_nm[3]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.y = wrench_nm[4]
        impedance_cmd.feed_forward_wrench_at_tool_in_desired_tool.torque.z = wrench_nm[5]

        # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
        # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
        cmd_trajectory = impedance_cmd.task_tform_desired_tool
        for pose, duration in trajectory:
            point = cmd_trajectory.points.add()
            point.time_since_reference.CopyFrom(seconds_to_duration(duration))
            point.pose.CopyFrom(se3pose_to_proto(pose))

        return self.__convert_proto_to_action_goal(robot_cmd)

    def move_end_effector_impedance(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3Pose,
        wrist_t_tool: SE3Pose,
        trajectory: List[Tuple[SE3Pose, float]],
        stiffness: List[float] = None,
        damping: List[float] = None,
        wrench_nm: List[float] = None,
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
        action_goal = self.__get_arm_impedance_action_goal(
            root_frame_name, root_t_task, wrist_t_tool, trajectory, stiffness, damping, wrench_nm
        )
        return self.command("move_end_effector_impedance", action_goal, timeout_sec)

    def move_end_effector_impedance_async(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3Pose,
        wrist_t_tool: SE3Pose,
        trajectory: List[Tuple[SE3Pose, float]],
        stiffness: List[float] = None,
        damping: List[float] = None,
        wrench_nm: List[float] = None,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
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
        action_goal = self.__get_arm_impedance_action_goal(
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
        action_goal = self.__convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_open_command())
        return self.command("claw_gripper_open", action_goal, timeout_sec)

    def claw_gripper_open_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """Send a command to open the gripper asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self.__convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_open_command())
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
        action_goal = self.__convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command("claw_gripper_close", action_goal, timeout_sec)

    def claw_gripper_close_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Optional[RobotCommandHandle]:
        """Send a command to close the gripper asynchronously

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            TODO
        """
        action_goal = self.__convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command_async(
            "claw_gripper_close",
            action_goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @staticmethod
    def set_tight_top_down_grasp(grasp):
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
        image_data: Any,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
    ):
        """Grasp an item in an image taken with a camera on spot.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            image_data: Image data taken using spot (need to figure out data type).
            palm_placement_ratio: a float [0.0, 1.0] that indicates how far
                in the edge of the object is into the palm cavity.
                (0.0 will place the base of palm with camera against the object,
                1.0 will be a 'finger pinch')
            apply_grasp_params_func: A helper function that will apply grasp parameters
                to modify default grasping behavior.
                (see https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=pickobjectinimage#graspparams)
        """  # noqa: E501
        pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image_data.shot.transforms_snapshot,
            frame_name_image_sensor=image_data.shot.frame_name_image_sensor,
            camera_model=image_data.source.pinhole,
        )

        # We can specify where in the gripper we want to grasp.
        grasp.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio

        if apply_grasp_params_func is not None:
            apply_grasp_params_func(grasp)

        # Build the proto
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        return self.__convert_proto_to_action_goal(grasp_request)

    def palm_grasp_in_image_top_down(
        self,
        pixel_xy: List[float],
        image_data: Any,
    ):
        """Grasp an item in an image taken with a camera on spot with the palm
        pressed into the object.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            image_data: Image data taken using spot (need to figure out data type).
        """
        return self.grasp_in_image(pixel_xy, image_data, 0.0, self.set_tight_top_down_grasp)

    def shutdown(self) -> None:
        """Shutdown all the internal components"""
        try:
            self.__spin_thread.join()
        except KeyboardInterrupt:
            pass
        self.__robot_command_action_client.destroy()
        self.__node.destroy_node()
        # TODO: Is there anything else that we should shutdown?


def main():
    import argparse

    # import math
    # import time
    #
    # from bdai.utilities.math_helpers import Quaternion
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "robot_name",
        help="Name of the robot if the ROS driver is inside that namespace",
    )
    parser.add_argument(
        "--disable-collisions",
        action="store_true",
        help="Flag for disabling collision avoidance",
    )
    parser.add_argument("--dry", action="store_true", help="Flag for if this is a dry run")
    options = parser.parse_args()

    rclpy.init()
    spot_basic = SpotBasic(
        robot_name=options.robot_name,
        parent_node_name="dummy_node",
        disable_collisions=options.disable_collisions,
        dry=options.dry,
    )
    logger = RcutilsLogger(name="main")

    # TESTING CODE GOES HERE
    # DS recommendation: make a file in spot_utilities/examples
    logger.info(f"has arm: {spot_basic.has_arm()}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
