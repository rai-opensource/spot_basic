# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Optional

from bdai_ros2_wrappers.callables import GeneralizedGuard, generalized_method
from bosdyn.api import robot_command_pb2
from bosdyn.client.robot_command import _CLAW_GRIPPER_CLOSED_ANGLE, _CLAW_GRIPPER_OPEN_ANGLE, RobotCommandBuilder
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import RobotCommandFeedback
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_utilities.spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_utilities.spot_basic.utilities import SpotActionHandle


class SpotGripperCommandBuilder(SpotCommandBuilder):
    """A builder for Spot gripper commands."""

    def claw_gripper_open_command(self, theta_rad: Optional[float] = None) -> robot_command_pb2.RobotCommand:
        """Build a claw gripper opening command.

        Args:
            theta_rad: opening angle for the claw gripper, in radians. Must lie in the [0, pi / 2] interval,
            where pi / 2 is the fully open angle. Defaults to a fully open claw if none is given.
        """
        if theta_rad is None:
            return RobotCommandBuilder.claw_gripper_open_command()

        theta_rad = -theta_rad  # negating to match gripper bounds in Spot SDK
        if theta_rad > _CLAW_GRIPPER_CLOSED_ANGLE or theta_rad < _CLAW_GRIPPER_OPEN_ANGLE:
            raise ValueError("Invalid gripper command: theta must be within range [0, pi/2]")
        return RobotCommandBuilder.claw_gripper_open_angle_command(theta_rad)

    def claw_gripper_close_command(self) -> robot_command_pb2.RobotCommand:
        """Build a claw gripper closing command."""
        return RobotCommandBuilder.claw_gripper_close_command()


class SpotGripperCommandMixin(SpotCommandProtocol):
    """Mixin that eases basic Spot gripper command execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotGripperCommandBuilder(self)

        do_claw_gripper_open = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.claw_gripper_open_command),
            f"{self.robot_name} has no arm, no gripper to open",
        )
        SpotGripperCommandMixin.claw_gripper_open.rebind(self, do_claw_gripper_open)

        do_claw_gripper_close = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.claw_gripper_close_command),
            f"{self.robot_name} has no arm, no gripper to close",
        )
        SpotGripperCommandMixin.claw_gripper_close.rebind(self, do_claw_gripper_close)

    @generalized_method(transitional=True)
    def claw_gripper_open(
        self, *, theta_rad: Optional[float] = None, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Open the robot claw gripper.

        Args:
            theta_rad: opening angle for the claw gripper, in radians. Must lie in the [0, pi / 2] interval,
            where pi / 2 is the fully open angle. Defaults to a fully open claw if none is given.

        Returns:
            outcome as command feedback.
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotGripperCommandBuilder(self)
        convert(command_builder.claw_gripper_open_command(theta_rad), goal.command)
        return self.command("claw_gripper_open", goal, timeout_sec)

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
        goal = RobotCommandAction.Goal()
        command_builder = SpotGripperCommandBuilder(self)
        convert(command_builder.claw_gripper_open_command(theta_rad), goal.command)
        return self.command_async(
            "claw_gripper_open",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def claw_gripper_close(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Close the robot claw gripper.

        Returns:
            outcome as command feedback.
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotGripperCommandBuilder(self)
        convert(command_builder.claw_gripper_close_command(), goal.command)
        return self.command("claw_gripper_close", goal, timeout_sec)

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
        goal = RobotCommandAction.Goal()
        command_builder = SpotGripperCommandBuilder(self)
        convert(command_builder.claw_gripper_close_command(), goal.command)
        return self.command_async(
            "claw_gripper_close",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )
