# Copyright (c) 2024-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any, Callable, Optional, Protocol, TypeAlias, Union

from bosdyn.api import robot_command_pb2
from bosdyn.client.robot_command import _CLAW_GRIPPER_CLOSED_ANGLE, _CLAW_GRIPPER_OPEN_ANGLE, RobotCommandBuilder
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import RobotCommandFeedback
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_utilities.spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_utilities.spot_basic.utilities import SpotActionableProtocol, SpotActionHandle
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedGuard, GeneralizedMethodLike, generalized_method

from bdai.deprecation import deprecated


class SpotGripperCommandBuilder(SpotCommandBuilder):
    """A builder for Spot gripper commands."""

    def claw_gripper_open_command(self) -> robot_command_pb2.RobotCommand:
        """Build a claw gripper opening command."""
        return RobotCommandBuilder.claw_gripper_open_command()

    def claw_gripper_close_command(
        self, max_torque: Optional[float] = None, disable_force_on_contact: bool = False
    ) -> robot_command_pb2.RobotCommand:
        """Build a claw gripper closing command.

        Args:
            max_torque: The maximum allowed torque applied in Nm. If unspecified, a default value of 5.5 will be used
        """
        return RobotCommandBuilder.claw_gripper_close_command(
            max_torque=max_torque, disable_force_on_contact=disable_force_on_contact
        )

    def claw_gripper_set_angle_command(
        self, theta_rad: float, max_torque: Optional[float] = None
    ) -> robot_command_pb2.RobotCommand:
        """Build a claw gripper command to set to an arbitrary angle.

        Args:
            theta_rad: opening angle for the claw gripper, in radians. Must lie in the [-pi/2, 0] interval,
            where -pi/2 is the fully open angle and 0 is closed. (This matches the joint angle in joint_states).
            max_torque: maximum allowed torque applied in Nm. If unspecified, a default value of 5.5 will be used.
        """
        if theta_rad > _CLAW_GRIPPER_CLOSED_ANGLE or theta_rad < _CLAW_GRIPPER_OPEN_ANGLE:
            raise ValueError("Invalid gripper command: theta must be within range [-pi/2, 0]")
        return RobotCommandBuilder.claw_gripper_open_angle_command(gripper_q=theta_rad, max_torque=max_torque)


RobotCommandActionableProtocol: TypeAlias = SpotActionableProtocol[
    RobotCommandAction.Goal, RobotCommandAction.Result, RobotCommandAction.Feedback
]


class SpotGripperCommandProtocol(SpotCommandProtocol, Protocol):
    """Spot arm gripper capabilities protocol.

    Note that the robot must have an arm for any of these capabilities to be available.
    """

    class ClawGripperOpenProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol to open the robot claw gripper."""

        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Open the robot claw gripper synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.claw_gripper_open_command` synchronously.
            See documentation for both for further reference on arguments taken.

            Raises:
                RuntimeError: if the robot has no arm.
            """

        def asynchronously(
            self,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Open the robot claw gripper asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.claw_gripper_open_command` asynchronously.
            See documentation for both for further reference on arguments taken.

            Raises:
                RuntimeError: if the robot has no arm.
            """

    class ClawGripperCloseProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol to close the robot claw gripper."""

        def synchronously(
            self,
            max_torque: Optional[float] = None,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Close the robot claw gripper synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.claw_gripper_close_command` synchronously.
            See documentation for both for further reference on arguments taken.

            Raises:
                RuntimeError: if the robot has no arm.
            """

        def asynchronously(
            self,
            max_torque: Optional[float] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Close the robot claw gripper asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.claw_gripper_close_command` asynchronously.
            See documentation for both for further reference on arguments taken.

            Raises:
                RuntimeError: if the robot has no arm.
            """

    class ClawGripperSetAngleProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol to set the robot claw gripper to a given joint angle."""

        def synchronously(
            self,
            theta_rad: float,
            max_torque: Optional[float] = None,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Set the claw gripper joint angle synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.claw_gripper_set_angle_command` synchronously.
            See documentation for both for further reference on arguments taken.

            Raises:
                RuntimeError: if the robot has no arm.
            """

        def asynchronously(
            self,
            theta_rad: float,
            max_torque: Optional[float] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Set the claw gripper joint angle asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.claw_gripper_set_angle_command` asynchronously.
            See documentation for both for further reference on arguments taken.

            Raises:
                RuntimeError: if the robot has no arm.
            """


class SpotGripperCommandMixin(SpotCommandProtocol):
    """Mixin that enables basic Spot gripper command execution.

    Note that the robot must have an arm and a gripper claw for any of these APIs to be available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotGripperCommandBuilder(self)

        SpotGripperCommandMixin.claw_gripper_open.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.execute.compose(command_builder.claw_gripper_open_command),
                f"{self.robot_name} has no arm, no gripper to open",
            ),
        )

        SpotGripperCommandMixin.claw_gripper_close.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.execute.compose(command_builder.claw_gripper_close_command),
                f"{self.robot_name} has no arm, no gripper to close",
            ),
        )

        SpotGripperCommandMixin.claw_gripper_set_angle.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.execute.compose(command_builder.claw_gripper_set_angle_command),
                f"{self.robot_name} has no arm, no gripper to set the angle of",
            ),
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotGripperCommandProtocol.ClawGripperOpenProtocol])
    def claw_gripper_open(self, **kwargs: Any) -> RobotCommandAction.Result:
        """Open robot claw gripper.

        See `SpotGripperCommandProtocol.ClawGripperOpenProtocol` for further reference.
        """

    @claw_gripper_open.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.claw_gripper_open.synchronously",
    )
    def _legacy_claw_gripper_open(self, *, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Fully open the robot claw gripper.

        Returns:
            outcome as command feedback.
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotGripperCommandBuilder(self)
        convert(command_builder.claw_gripper_open_command(), goal.command)
        return self.command("claw_gripper_open", goal, timeout_sec)

    def claw_gripper_open_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to fully open the gripper asynchronously

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
        convert(command_builder.claw_gripper_open_command(), goal.command)
        return self.command_async(
            "claw_gripper_open",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotGripperCommandProtocol.ClawGripperCloseProtocol])
    def claw_gripper_close(
        self, max_torque: Optional[float] = None, disable_force_on_contact: bool = False, **kwargs: Any
    ) -> RobotCommandAction.Result:
        """Close robot claw gripper.

        See `SpotGripperCommandProtocol.ClawGripperCloseProtocol` for further reference.
        """

    @claw_gripper_close.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.claw_gripper_close.synchronously",
    )
    def _legacy_claw_gripper_close(
        self, timeout_sec: Optional[float] = None, max_torque: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Close the robot claw gripper.

        Returns:
            outcome as command feedback.
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotGripperCommandBuilder(self)
        convert(command_builder.claw_gripper_close_command(max_torque), goal.command)
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

    @generalized_method(spec=GeneralizedMethodLike[SpotGripperCommandProtocol.ClawGripperSetAngleProtocol])
    def claw_gripper_set_angle(
        self, theta_rad: float, max_torque: Optional[float] = None, **kwargs: Any
    ) -> RobotCommandAction.Result:
        """Set the joint angle of the robot claw gripper.

        See `SpotGripperCommandProtocol.ClawGripperSetAngleProtocol` for further reference.
        """
