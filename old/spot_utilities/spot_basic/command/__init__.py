# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.

from enum import IntEnum
from typing import Any, Callable, Optional, Protocol, TypeAlias, Union, overload

import bosdyn_api_msgs.msg
import builtin_interfaces.msg
import rclpy.duration
from bosdyn.api import robot_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn_api_msgs.msg import RobotCommandFeedback
from geometry_msgs.msg import Twist
from rclpy.task import Future
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_msgs.srv import ClearBehaviorFault, SetLocomotion, SetVelocity
from spot_msgs.srv import RobotCommand as RobotCommandService
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.state import SpotStateProtocol
from spot_utilities.spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
    SpotActionHandle,
    SpotServiced,
    SpotServicedProtocol,
    SpotServiceHandle,
)
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import SetBool, Trigger
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike

from bdai.deprecation import deprecated


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


RobotCommandActionableProtocol: TypeAlias = SpotActionableProtocol[
    RobotCommandAction.Goal, RobotCommandAction.Result, RobotCommandAction.Feedback
]


class SpotCommandProtocol(SpotStateProtocol, Protocol):
    """Spot base command protocol."""

    class ExecuteProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for blocking command execution."""

        @overload
        def synchronously(
            self,
            command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand],
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Execute a robot command synchronously.

            This method will wait for robot command completion.
            See `SpotCommandProtocol.StartExecuteProtocol.synchronously` for a
            method that does not.

            Args:
                command: a robot command to execute, in its Protobuf or ROS 2 form.
                feedback_callback: optional action feedback callback.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the execution request will be cancelled and the call will raise. Note this
                timeout is local to the caller. It may take some time for the action to be cancelled,
                if cancellation is not rejected by the action server.

            Returns:
                the execution result.

            Raises:
                ActionTimeout: if the execution request timed out.
                ActionRejected: if the execution request was not accepted.
                ActionCancelled: if the execution request was cancelled.
                ActionAborted: if the execution request was aborted.
            """

        @overload
        def synchronously(
            self,
            command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand],
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Execute a robot command synchronously.

            This method will wait for robot command completion.
            See `SpotCommandProtocol.StartExecuteProtocol.synchronously` for a
            method that does not.

            Args:
                command: a robot command to execute, in its Protobuf or ROS 2 form.
                feedback_callback: optional action feedback callback.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the execution request will be cancelled and the call will raise. Note this
                timeout is local to the caller. It may take some time for the action to be cancelled,
                if cancellation is not rejected by the action server.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the execution result or None on timeout or failure.
            """

        def asynchronously(
            self,
            command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand],
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Execute a robot command asynchronously.

            The returned future will be resolved upon robot command completion
            (or failure). See `SpotCommandProtocol.StartExecuteProtocol.synchronously`
            for a future that is resolved as soon as the robot command is acknowledged.

            Args:
                command: a robot command to execute, in its Protobuf or ROS 2 form.
                track_feedback: whether and how to track action feedback. Other
                than a boolean to enable or disable tracking, a positive integer
                may be provided to cap feedback buffer size.

            Returns:
                the future execution outcome.
            """

    class StartExecuteProtocol(
        SpotServicedProtocol[RobotCommandService.Request, RobotCommandService.Response], Protocol
    ):
        """Protocol to start command execution (without blocking)."""

        def synchronously(
            self,
            command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand],
            duration: Union[builtin_interfaces.msg.Duration, rclpy.duration.Duration],
            *,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandService.Response:
            """Start executing a robot command synchronously.

            This method will not wait for robot command completion but return as soon as it
            is acknowledged. See `SpotCommandProtocol.ExecuteProtocol.synchronously` for a
            method that does wait for completion.

            Args:
                command: a robot command to execute, in its Protobuf or ROS 2 form.
                duration: the intended robot command duration.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the execution request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the execution request, including the robot command id.

            Raises:
                ServiceError: if it fails to start execution.
                ServiceTimeout: if it times out while starting execution.
            """

        def asynchronously(
            self,
            command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand],
            duration: Union[builtin_interfaces.msg.Duration, rclpy.duration.Duration],
        ) -> FutureLike[RobotCommandService.Response]:
            """Start executing a robot command asynchronously.

            The returned future will be resolved as soon as the robot command is acknowledged.
            See `SpotCommandProtocol.ExecuteProtocol.asynchronously` for a future that is
            resolved upon robot command completion (or failure).

            Args:
                command: a robot command to execute, in its Protobuf or ROS 2 form.
                duration: the intended robot command duration.

            Returns:
                a future response to the execution request.
            """

    class StopExecuteProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Protocol to stop command execution."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Stop an ongoing robot command synchronously.

            Args:
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the stop request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the stop request.

            Raises:
                ServiceError: if it fails to stop execution.
                ServiceTimeout: if it times out while stopping execution.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Stop an ongoing robot command asynchronously.

            Returns:
                a future response to the stop request.
            """

    class SetLocomotionModeProtocol(SpotServicedProtocol[SetLocomotion.Request, SetLocomotion.Response], Protocol):
        """Protocol to set the robot locomotion mode."""

        @overload
        def synchronously(
            self, locomotion_hint: LocomotionHint, *, timeout_sec: Optional[float] = None
        ) -> SetLocomotion.Response:
            """Set the robot locomotion mode synchronously.

            Args:
                locomotion_hint: intended robot locomotion mode.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the setting change request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the setting change request.

            Raises:
                ServiceError: if it fails to set the mode.
                ServiceTimeout: if it times out while setting the mode.
            """

        @overload
        def synchronously(
            self, locomotion_hint: LocomotionHint, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[SetLocomotion.Response]:
            """Set the robot locomotion mode synchronously.

            Args:
                locomotion_hint: intended robot locomotion mode.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the setting change request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the setting change request or None on timeout or failure.
            """

        def asynchronously(self, locomotion_hint: LocomotionHint) -> FutureLike[SetLocomotion.Response]:
            """Set the robot locomotion mode asynchronously.

            Args:
                locomotion_hint: intended robot locomotion mode.

            Returns:
                a future response to the stop request.
            """

    class SetStairModeProtocol(SpotServicedProtocol[SetBool.Request, SetBool.Response], Protocol):
        """Protocol to toggle the robot stair mode."""

        @overload
        def synchronously(self, enabled: bool, /, *, timeout_sec: Optional[float] = None) -> SetBool.Response:
            """Toggle the robot in and out of stair mode synchronously.

            Args:
                enabled: whether the stair mode should be enabled or disabled.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the toggling request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the toggling request.

            Raises:
                ServiceError: if it fails to toggle.
                ServiceTimeout: if it times out while toggling.
            """

        @overload
        def synchronously(
            self, enabled: bool, /, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[SetBool.Response]:
            """Toggle the robot in and out of stair mode synchronously.

            Args:
                enabled: whether the stair mode should be enabled or disabled.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the toggling request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the toggling request or None on timeout or failure.
            """

        def asynchronously(self, enabled: bool, /) -> FutureLike[SetBool.Response]:
            """Toggle the robot in and out of stair mode asynchronously.

            Args:
                enabled: whether the stair mode should be enabled or disabled.

            Returns:
                a future response to the toggling request.
            """

    class SetMaxVelocityProtocol(SpotServicedProtocol[SetVelocity.Request, SetVelocity.Response], Protocol):
        """Protocol to set the robot velocity limits."""

        @overload
        def synchronously(self, velocity_limit: Twist, *, timeout_sec: Optional[float] = None) -> SetVelocity.Response:
            """Set the robot velocity limits synchronously.

            Args:
                velocity_limit: linear and angular velocity limits.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the setting change request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the setting change request.

            Raises:
                ServiceError: if it fails to set velocity limits.
                ServiceTimeout: if it times out while setting velocity limits.
            """

        @overload
        def synchronously(
            self, velocity_limit: Twist, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[SetVelocity.Response]:
            """Set the robot velocity limits synchronously.

            Args:
                velocity_limit: linear and angular velocity limits.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the setting change request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the setting change request or None on timeout or failure.
            """

        def asynchronously(self, velocity_limit: Twist) -> FutureLike[SetVelocity.Response]:
            """Set the robot velocity limits asynchronously.

            Args:
                velocity_limit: linear and angular velocity limits.

            Returns:
                a future response to the setting change request.
            """

    class ClearBehaviorFaultProtocol(
        SpotServicedProtocol[ClearBehaviorFault.Request, ClearBehaviorFault.Response], Protocol
    ):
        """Protocol to clear robot behavior faults."""

        @overload
        def synchronously(self, id: int, /, *, timeout_sec: Optional[float] = None) -> ClearBehaviorFault.Response:
            """Clear a robot behavior fault synchronously.

            Args:
                id: behavior fault id to clear.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the clearing request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the clearing request.

            Raises:
                ServiceError: if it fails to clear the fault.
                ServiceTimeout: if it times out while clearing the fault.
            """

        @overload
        def synchronously(
            self, id: int, /, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ClearBehaviorFault.Response]:
            """Clear a robot behavior fault synchronously.

            Args:
                id: behavior fault id to clear.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the clearing request will be cancelled and the call will raise. Note this
                timeout is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the clearing request or None on timeout or failure.
            """

        def asynchronously(self, id: int, /) -> FutureLike[ClearBehaviorFault.Response]:
            """Clear a robot behavior fault asynchronously.

            Args:
                id: behavior fault id to clear.

            Returns:
                a future response to the clearing request.
            """

    @property
    def mobility_params(self) -> spot_command_pb2.MobilityParams:
        """Get the robot mobility params."""

    def command(
        self, command_name: str, goal: RobotCommandAction.Goal, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]: ...

    def command_async(
        self,
        command_name: str,
        goal: RobotCommandAction.Goal,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]: ...

    execute: GeneralizedMethodLike[ExecuteProtocol]
    start_executing: GeneralizedMethodLike[StartExecuteProtocol]
    stop: GeneralizedMethodLike[StopExecuteProtocol]

    set_locomotion_mode: GeneralizedMethodLike[SetLocomotionModeProtocol]
    set_max_velocity: GeneralizedMethodLike[SetMaxVelocityProtocol]
    set_stair_mode: GeneralizedMethodLike[SetStairModeProtocol]

    clear_behavior_fault: GeneralizedMethodLike[ClearBehaviorFaultProtocol]


class SpotCommandBuilder:
    """Base command builder."""

    def __init__(self, target: SpotCommandProtocol) -> None:
        """Initialize command builder.

        Args:
            target: robot command interface for which to build commands.
        """
        self.target = target


class SpotCommandMixin(SpotCommandProtocol):
    """Mixin that enables execution of Spot commands."""

    LocomotionHint: TypeAlias = LocomotionHint

    def __init__(self, *args: Any, dry: bool = False, disable_collisions: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=disable_collisions,
            disable_vision_foot_obstacle_avoidance=disable_collisions,
            disable_vision_foot_constraint_avoidance=disable_collisions,
            obstacle_avoidance_padding=0.001,
        )
        self._mobility_params = spot_command_pb2.MobilityParams(obstacle_params=obstacles)
        self._dry = dry

        SpotCommandMixin.set_locomotion_mode.rebind(
            self, SpotServiced(SetLocomotion, spot_namespaced(self.robot_name, "locomotion_mode"), node=self.node)
        )
        SpotCommandMixin.set_stair_mode.rebind(
            self, SpotServiced(SetBool, spot_namespaced(self.robot_name, "stair_mode"), node=self.node)
        )
        SpotCommandMixin.set_max_velocity.rebind(
            self, SpotServiced(SetVelocity, spot_namespaced(self.robot_name, "max_velocity"), node=self.node)
        )
        SpotCommandMixin.start_executing.rebind(
            self, SpotServiced(RobotCommandService, spot_namespaced(self.robot_name, "robot_command"), node=self.node)
        )
        SpotCommandMixin.execute.rebind(
            self, SpotActionable(RobotCommandAction, spot_namespaced(self.robot_name, "robot_command"), node=self.node)
        )
        SpotCommandMixin.stop.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "stop"), node=self.node)
        )
        SpotCommandMixin.clear_behavior_fault.rebind(
            self,
            SpotServiced(ClearBehaviorFault, spot_namespaced(self.robot_name, "clear_behavior_fault"), node=self.node),
        )

    @property
    def mobility_params(self) -> spot_command_pb2.MobilityParams:
        """Get the robot mobility params."""
        return self._mobility_params

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.SetLocomotionModeProtocol])
    def set_locomotion_mode(self, locomotion_hint: LocomotionHint, **kwargs: Any) -> SetLocomotion.Response:
        """Set the robot locomotion mode.

        Args:
            locomotion_hint: intended robot locomotion mode.

        See `SpotCommandProtocol.SetLocomotionModeProtocol` for further reference.
        """

    @set_locomotion_mode.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.set_locomotion_mode.synchronously",
    )
    def _legacy_set_locomotion_mode(self, locomotion_hint: LocomotionHint) -> bool:
        """Set the locomotion mode for the robot.

        Returns:
           A boolean to flag its outcome and an error message on failure.
        """
        self.node.get_logger().info(f"Setting locomotion_mode to '{locomotion_hint}' for {self.robot_name}")
        response = self.set_locomotion_mode.synchronously(locomotion_hint, nothrow=True)
        if response is None:
            return False
        if not response.success:
            self.node.get_logger().debug(
                f"Setting locomotion_mode to '{locomotion_hint}' for "
                f"{self.robot_name} failed. Msg: {response.message}."
            )
        return response.success

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.SetStairModeProtocol])
    def set_stair_mode(self, enabled: bool, /, **kwargs: Any) -> SetBool.Response:
        """Toggle the robot in and out of stair mode synchronously.

        Args:
            enabled: whether the stair mode should be enabled or disabled.

        See `SpotCommandProtocol.SetStairModeProtocol` for further reference.
        """

    @set_stair_mode.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.set_stair_mode.synchronously",
    )
    def _legacy_set_stair_mode(self, value: bool) -> bool:
        """Enable (or disable) the stair mode for the robot.

        Returns:
           A boolean to flag its outcome and an error message on failure.
        """
        self.node.get_logger().info(f"Setting stair_mode to '{value}' for {self.robot_name}")
        response = self.set_stair_mode.synchronously(value, nothrow=True)
        if response is None:
            return False
        if not response.success:
            self.node.get_logger().debug(
                f"Setting stair_mode to '{value}' for {self.robot_name} failed. Msg: {response.message}."
            )
        return response.success

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.SetMaxVelocityProtocol])
    def set_max_velocity(self, velocity_limit: Twist, **kwargs: Any) -> SetVelocity.Response:
        """Set the robot velocity limits.

        Args:
            velocity_limit: linear and angular velocity limits.

        See `SpotCommandProtocol.SetMaxVelocityProtocol` for further reference.
        """

    @set_max_velocity.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.set_max_velocity.synchronously",
    )
    def _legacy_set_max_velocity(self, velocity_limit: Twist) -> bool:
        """Set the maximum velocity for the robot.

        Returns:
           A boolean to flag its outcome and an error message on failure.
        """
        self.node.get_logger().info(f"Setting max_velocity to '{velocity_limit}' for {self.robot_name}")
        response = self.set_max_velocity.synchronously(velocity_limit, nothrow=True)
        if response is None:
            return False
        if not response.success:
            self.node.get_logger().debug(
                f"Setting stair_mode to '{velocity_limit}' for {self.robot_name} failed. Msg: {response.message}."
            )
        return response.success

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.ClearBehaviorFaultProtocol])
    def clear_behavior_fault(self, id: int, /, **kwargs: Any) -> ClearBehaviorFault.Response:
        """Clear a robot behavior fault.

        Args:
            id: behavior fault id to clear.

        See `SpotCommandProtocol.ClearBehaviorFaultProtocol` for further reference.
        """

    @clear_behavior_fault.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.clear_behavior.synchronously",
    )
    def _legacy_clear_behavior_fault(self, bid: int) -> bool:
        """Clear a behavior fault on the robot.

        Returns:
           A boolean to flag its outcome and an error message on failure.
        """
        self.node.get_logger().info(f"Clearing behavior fault {bid} for {self.robot_name}")
        response = self.clear_behavior_fault.synchronously(bid, nothrow=True)
        if response is None:
            return False
        if not response.success:
            self.node.get_logger().debug(
                f"Clearing behavior fault {bid} for {self.robot_name} failed. Msg: {response.message}"
            )
        return response.success

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.StartExecuteProtocol])
    def start_executing(
        self,
        command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand],
        duration: Union[builtin_interfaces.msg.Duration, rclpy.duration.Duration],
        **kwargs: Any,
    ) -> RobotCommandService.Response:
        """Start executing a robot command.

        Args:
            command: a robot command to execute, in its Protobuf or ROS 2 form.
            duration: the intended robot command duration.

        See `SpotCommandProtocol.StartExecuteProtocol` for further reference.
        """

    @start_executing.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.start_executing.synchronously",
    )
    def _legacy_start_executing(
        self,
        command: robot_command_pb2.RobotCommand,
        duration_sec: Optional[float] = None,
        *,
        timeout_sec: Optional[float] = None,
    ) -> int:
        """Start executing a `command` on the robot.

        Args:
            command: Robot command to start executing.
            duration_sec: Optional duration for the command, in seconds.

        Returns:
            Robot command ID for the command started.
        """
        duration = rclpy.duration.Duration(seconds=duration_sec if duration_sec else 0)
        response = self.start_executing.synchronously(command, duration, timeout_sec=timeout_sec)
        return response.robot_command_id

    def start_executing_async(
        self,
        command: robot_command_pb2.RobotCommand,
        duration_sec: Optional[float] = None,
    ) -> Future:
        """Start executing a `command` on the robot asynchronously.

        Args:
            command: Robot command to start executing.
            duration_sec: Optional duration for the command, in seconds.

        Returns:
            A future robot command ID for the command started.
        """
        duration = rclpy.duration.Duration(seconds=duration_sec if duration_sec else 0)
        return self.start_executing.asynchronously(command, duration)

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.ExecuteProtocol])
    def execute(
        self, command: Union[robot_command_pb2.RobotCommand, bosdyn_api_msgs.msg.RobotCommand], **kwargs: Any
    ) -> RobotCommandAction.Result:
        """Execute a robot command.

        Args:
            command: a robot command to execute, in its Protobuf or ROS 2 form.

        See `SpotCommandProtocol.ExecuteProtocol` for further reference.
        """

    @execute.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.execute.synchronously",
    )
    def _legacy_execute(
        self, command: robot_command_pb2.RobotCommand, *, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Execute the given robot `command` and wait for its completion.

        Args:
            command: Robot command to be executed.

        Returns:
            Outcome as command feedback.
        """
        response = self.execute.synchronously(command, timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return None
        return response.result

    @generalized_method(spec=GeneralizedMethodLike[SpotCommandProtocol.StopExecuteProtocol])
    def stop(self, **kwargs: Any) -> Trigger.Response:
        """Stop an ongoing robot command.

        See `SpotCommandProtocol.StopExecuteProtocol` for further reference.
        """

    @stop.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.stop.synchronously",
    )
    def _legacy_stop(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Stop command execution, if any.

        Returns:
            A boolean to flag its outcome and an error message on failure.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "stop",
            self.stop.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    def command(
        self, command_name: str, goal: RobotCommandAction.Goal, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Run the `RobotCommand` action.

        Args:
            command_name: A representative name of the command for logging
            goal: The Action Goal sent to the action server
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        failed = False

        def on_failure() -> None:
            nonlocal failed
            failed = True

        handle = self.command_async(command_name, goal, on_failure_callback=on_failure)
        if handle is None or not handle.wait_for_result(timeout_sec) or failed:
            return None
        return handle.result

    def command_async(
        self,
        command_name: str,
        goal: RobotCommandAction.Goal,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Run the `RobotCommand` action asynchronously.

        Args:
            command_name: A representative name of the command for logging
            goal: The Action Goal sent to the action server
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            Optional[SpotActionHandle]: A handle for the asynchronous lifecycle of the action

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if self._dry:
            return None
        return SpotActionHandle.call(
            self.robot_name,
            command_name,
            self.execute.action_client,
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
