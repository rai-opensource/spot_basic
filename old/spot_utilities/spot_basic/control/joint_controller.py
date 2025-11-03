# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from functools import cached_property

from sensor_msgs.msg import JointState
from spot_msgs.msg import JointCommand
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.control.base import SpotControllerInterface
from spot_utilities.spot_util import spot_namespaced
from synchros2.subscription import Subscription


class SpotJointControllerInterface(SpotControllerInterface):
    """Joint controller interface for Spot."""

    def __init__(self, spot: SpotBaseProtocol) -> None:
        """Initialize controller interface.

        Args:
            spot: base Spot driver interface.
        """
        super().__init__(spot, ["joint_state_broadcaster", "spot_joint_controller"])
        self._joints_command_publisher = self.spot.node.create_publisher(
            JointCommand, spot_namespaced(self.spot.robot_name, "spot_joint_controller/joint_commands"), 1
        )

    @cached_property
    def joint_states_feedback(self) -> Subscription[JointState]:
        """Get joint states feedback."""
        return Subscription(
            JointState, spot_namespaced(self.spot.robot_name, "low_level/joint_states"), node=self.spot.node
        )

    def update_joints_command(self, command: JointCommand = None) -> None:
        """Update joint states command, one setpoint per joint per interface configured for the controller.

        Typical interfaces are position, velocity, and effort. Missing setpoints will default to 0.

        Args:
            command: A spot_msgs/msg/JointCommand message that has the names of the desired joints to
            command, desired position, velocity, effort, and gain values. They should be in the same order.
            Note: Needs <robot_name> prefix for joint names (i.e. Opal/arm_sh0)
            See https://github.com/bdaiinstitute/spot_ros2/tree/main/spot_controllers#spotjointcontroller
            for more info on using the SpotJointController

        Raises:
            RuntimeError: if the controller interface is not attached or command is None
        """
        if self.configuration is None:
            raise RuntimeError("controller interface must be attached to update setpoints")

        if command is None:
            raise ValueError("A JointCommand command must be given.")

        self._joints_command_publisher.publish(command)
