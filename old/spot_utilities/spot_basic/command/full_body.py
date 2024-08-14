# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from enum import IntEnum
from typing import Any, Callable, Optional, TypeAlias, Union

from bdai_ros2_wrappers.callables import generalized_method
from bosdyn.api import (
    full_body_command_pb2,
    robot_command_pb2,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import RobotCommandFeedback
from spatialmath import SE3
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_utilities.spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_utilities.spot_basic.command.arm import SpotArmCommandBuilder
from spot_utilities.spot_basic.command.mobility import SpotMobilityCommandBuilder
from spot_utilities.spot_basic.state import ArmJointMap
from spot_utilities.spot_basic.utilities import SpotActionHandle


class SpotFullBodyCommandBuilder(SpotCommandBuilder):
    """A builder for Spot full body commands."""

    class BatteryChangeDirHint(IntEnum):
        RIGHT = 1
        LEFT = 2

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

    def torso_and_arm_command(
        self,
        torso_pose: Union[SE3Pose, SE3],
        joint_angles: Union[ArmJointMap, list[float]],
        duration: Optional[float] = None,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Creates a robot command that takes both robot torso and arm to the desired pose without moving the robot's
        feet. (If the SE3 pose is out of range, this will get as close as possible without taking a step).

        See arm_joint_command() and torso_pose_command() documentation for further reference.
        """
        arm_builder = SpotArmCommandBuilder(self)
        mobility_builder = SpotMobilityCommandBuilder(self)
        return RobotCommandBuilder.build_synchro_command(
            mobility_builder.torso_pose_command(torso_pose, duration),
            arm_builder.arm_joint_command(joint_angles, duration, max_velocity, max_acceleration),
        )


class SpotFullBodyCommandMixin(SpotCommandProtocol):
    """Mixin that eases Spot full body command execution."""

    BatteryChangeDirHint: TypeAlias = SpotFullBodyCommandBuilder.BatteryChangeDirHint

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotFullBodyCommandBuilder(self)
        SpotFullBodyCommandMixin.self_right.rebind(self, self.execute.compose(command_builder.self_right_command))
        SpotFullBodyCommandMixin.go_to_battery_change_pose.rebind(
            self, self.execute.compose(command_builder.roll_to_battery_change_pose_command)
        )

    @generalized_method(transitional=True)
    def self_right(self, timeout_sec: float = 5.0) -> bool:
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

    @generalized_method(transitional=True)
    def roll_to_battery_change_pose(
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


# Alias
BatteryChangeDirHint: TypeAlias = SpotFullBodyCommandMixin.BatteryChangeDirHint
