# Copyright (c) 2024-2025 Boston Dynamics AI Institute LLC. All rights reserved.

from enum import IntEnum
from typing import Any, Callable, Optional, Protocol, TypeAlias, Union, overload

from bosdyn.api import full_body_command_pb2, robot_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import RobotCommandFeedback
from spatialmath import SE3
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_utilities.spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_utilities.spot_basic.command.arm import SpotArmCommandBuilder
from spot_utilities.spot_basic.command.gripper import SpotGripperCommandBuilder
from spot_utilities.spot_basic.command.mobility import SpotMobilityCommandBuilder
from spot_utilities.spot_basic.state import ArmJointMap
from spot_utilities.spot_basic.utilities import SpotActionableProtocol, SpotActionHandle
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedMethodLike, generalized_method

from bdai.deprecation import deprecated


class BatteryChangeDirHint(IntEnum):
    RIGHT = 1
    LEFT = 2


class SpotFullBodyCommandBuilder(SpotCommandBuilder):
    """A builder for Spot full body commands."""

    def roll_to_battery_change_pose_command(
        self,
        direction_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
        *,
        mobility_params: Optional[spot_command_pb2.MobilityParams] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Build a go to battery change pose command.

        Args:
            direction_hint: optional hint for a rolling direction, will default to a right roll.
            mobility_params: optional mobility params, defaulting to target
            mobility params if none are provided.
        """
        if mobility_params is None:
            mobility_params = self.target.mobility_params
        full_body_command = full_body_command_pb2.FullBodyCommand.Request()
        full_body_command.battery_change_pose_request.direction_hint = direction_hint
        full_body_command.params.Pack(mobility_params)
        return robot_command_pb2.RobotCommand(full_body_command=full_body_command)

    def self_right_command(
        self, *, mobility_params: Optional[spot_command_pb2.MobilityParams] = None
    ) -> robot_command_pb2.RobotCommand:
        """Build a self right command.

        Args:
            mobility_params: optional mobility params, defaulting to target
            mobility params if none are provided.
        """
        if mobility_params is None:
            mobility_params = self.target.mobility_params
        full_body_command = full_body_command_pb2.FullBodyCommand.Request()
        full_body_command.selfright_request.SetInParent()
        full_body_command.params.Pack(mobility_params)
        return robot_command_pb2.RobotCommand(full_body_command=full_body_command)

    def whole_body_command(
        self,
        torso_pose: Union[SE3Pose, SE3],
        joint_angles: Union[ArmJointMap, list[float]],
        duration: Optional[float] = None,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        gripper_angle: Optional[bool] = None,
        gripper_max_torque: Optional[float] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Creates a robot command that takes the torso, arm, and gripper to the desired state without moving the
        feet. (If the SE3 pose is out of range, this will get as close as possible without taking a step).

        See arm_joint_command(), torso_pose_command(), and claw_gripper_set_angle_command() documentation for
        further reference.
        """
        arm_builder = SpotArmCommandBuilder(self.target)
        mobility_builder = SpotMobilityCommandBuilder(self.target)
        torso_command = mobility_builder.torso_pose_command(torso_pose, duration)
        arm_command = arm_builder.arm_joint_command(joint_angles, duration, max_velocity, max_acceleration)
        if gripper_angle is None:
            return RobotCommandBuilder.build_synchro_command(torso_command, arm_command)
        gripper_builder = SpotGripperCommandBuilder(self.target)
        gripper_command = gripper_builder.claw_gripper_set_angle_command(
            theta_rad=gripper_angle, max_torque=gripper_max_torque
        )
        return RobotCommandBuilder.build_synchro_command(torso_command, arm_command, gripper_command)


RobotCommandActionableProtocol: TypeAlias = SpotActionableProtocol[
    RobotCommandAction.Goal, RobotCommandAction.Result, RobotCommandAction.Feedback
]


class SpotFullBodyCommandProtocol(SpotCommandProtocol, Protocol):
    """Spot full body mobility capabilities protocol."""

    class SelfRightProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot self-righting.

        Takes the robot from being dropped on the floor to an upright position.
        """

        @overload
        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot self right synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotFullBodyCommandProtocol.self_right_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot self right synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotFullBodyCommandProtocol.self_right_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot self right asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotFullBodyCommandProtocol.self_right_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class RollToBatteryChangePoseProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot rollover.

        Takes the robot down to the floor, in an upside position that allows battery replacement.
        """

        @overload
        def synchronously(
            self,
            direction_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot roll to its battery change pose synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotFullBodyCommandProtocol.roll_to_battery_change_pose_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            direction_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot roll to its battery change pose synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotFullBodyCommandProtocol.roll_to_battery_change_pose_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            direction_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot roll to its battery change pose asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotFullBodyCommandProtocol.roll_to_battery_change_pose_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    self_right: GeneralizedMethodLike[SelfRightProtocol]
    roll_to_battery_change_pose: GeneralizedMethodLike[RollToBatteryChangePoseProtocol]


class SpotFullBodyCommandMixin(SpotCommandProtocol):
    """Mixin that eases Spot full body command execution."""

    BatteryChangeDirHint: TypeAlias = BatteryChangeDirHint

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotFullBodyCommandBuilder(self)
        SpotFullBodyCommandMixin.self_right.rebind(self, self.execute.compose(command_builder.self_right_command))
        SpotFullBodyCommandMixin.go_to_battery_change_pose.rebind(
            self, self.execute.compose(command_builder.roll_to_battery_change_pose_command)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotFullBodyCommandProtocol.SelfRightProtocol])
    def self_right(self, **kwargs: Any) -> RobotCommandAction.Result:
        """Make the robot self right.

        See `SpotFullBodyCommandProtocol.SelfRightProtocol` for further reference.
        """

    @self_right.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.self_right.synchronously",
    )
    def _legacy_self_right(self, timeout_sec: float = 5.0) -> bool:
        """Make the robot self right."""
        response = self.self_right.synchronously(timeout_sec=timeout_sec, nothrow=True)
        return response is not None and response.success

    def rollover(self, timeout_sec: float = 5.0) -> Optional[RobotCommandFeedback]:
        """Make the robot rollover (ie. self right)."""
        response = self.self_right.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None or not response.success:
            return None
        return response.result

    def rollover_async(
        self,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Treats rollover (self righting) as an action instead of a service and asynchronously executes it

        Args:
          result_callback: A callback to process/handle the resulting feedback from executing the command
          feedback_callback: A callback to process/handle the feedback received during the execution of the command
          on_failure_callback: A callback to process/handle when the action fails for various reasons
        """
        command_builder = SpotFullBodyCommandBuilder(self)
        command = command_builder.self_right_command()
        goal = RobotCommandAction.Goal()
        convert(command, goal.command)
        return self.command_async(
            "rollover_async",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotFullBodyCommandProtocol.RollToBatteryChangePoseProtocol])
    def roll_to_battery_change_pose(
        self, direction_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT, **kwargs: Any
    ) -> RobotCommandAction.Result:
        """Make the robot roll to its battery change pose.

        See `SpotFullBodyCommandProtocol.RollToBatteryChangePoseProtocol` for further reference.
        """

    @roll_to_battery_change_pose.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.roll_to_battery_change_pose.synchronously",
    )
    def _legacy_roll_to_battery_change_pose(
        self, dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Make the robot roll to its battery change pose.

        Args:
            dir_hint: A hint for the direction to roll.
        """
        response = self.roll_to_battery_change_pose.synchronously(dir_hint, timeout_sec=timeout_sec, nothrow=True)
        if response is None or not response.success:
            return None
        return response.result

    go_to_battery_change_pose = roll_to_battery_change_pose

    def go_to_battery_change_pose_async(
        self,
        dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send command to go to battery change pose asynchronously.

        Args:
            dir_hint: A hint for the direction to roll
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        command_builder = SpotFullBodyCommandBuilder(self)
        command = command_builder.roll_to_battery_change_pose_command(dir_hint)
        goal = RobotCommandAction.Goal()
        convert(command, goal.command)
        return self.command_async(
            "go_to_battery_change_pose",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )
