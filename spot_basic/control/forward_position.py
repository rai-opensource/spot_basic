# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections.abc import Sequence
from functools import cached_property
from typing import Optional

from sensor_msgs.msg import JointState
from spot_basic.base import SpotBaseProtocol
from spot_basic.control.base import SpotControllerInterface
from spot_basic.utilities import spot_namespaced
from std_msgs.msg import Float64MultiArray
from synchros2.subscription import Subscription


class SpotForwardPositionControllerInterface(SpotControllerInterface):
    """Forward position controller interface for Spot."""

    def __init__(self, spot: SpotBaseProtocol) -> None:
        """Initialize controller interface.

        Args:
            spot: base Spot driver interface.
        """
        super().__init__(spot, ["joint_state_broadcaster", "forward_position_controller"])
        self._joint_position_command_publisher = self.spot.node.create_publisher(
            Float64MultiArray, spot_namespaced(self.spot.robot_name, "forward_position_controller/commands"), 1
        )

    @cached_property
    def joint_states_feedback(self) -> Subscription[JointState]:
        """Get joint states feedback."""
        return Subscription(
            JointState, spot_namespaced(self.spot.robot_name, "low_level/joint_states"), node=self.spot.node
        )

    def update_joint_position_command(self, command: Optional[Sequence[float]] = None, **kwargs: float) -> None:
        """Update joint position command, one setpoint per joint configured for the controller.

        Missing setpoints will default to 0.

        Args:
            command: full setpoint array. This array follows standard `ros2_control` ordering conventions i.e.
            `[<position for joint 0>, <position for joint 1>, ..., <position for joint M>]`.
            kwargs: these are deduced from controller configuration, allowing commands to be set on a per joint basis.

        Raises:
            RuntimeError: if the controller interface is not attached
        """
        if self.configuration is None:
            raise RuntimeError("controller interface must be attached to update setpoints")

        joints = self.configuration["forward_position_controller"]["joints"]
        joints = list(map(lambda name: name.removeprefix(self.spot.robot_name).strip("/"), joints))

        if command is not None:
            if len(command) != len(joints):
                raise ValueError(f"command must have {len(joints)} elements")
            data = list(command)
        elif all(name in kwargs for name in joints):
            data = [0.0] * len(joints)
        else:
            raise ValueError("either a whole-body command or a complete set of per joint setpoints must be given")

        for i, joint in enumerate(joints):
            if joint in kwargs:
                data[i] = kwargs[joint]

        self._joint_position_command_publisher.publish(Float64MultiArray(data=data))
