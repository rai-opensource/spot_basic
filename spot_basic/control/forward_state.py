# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Forward state controller interface for Spot joint state commands."""

from collections.abc import Sequence
from functools import cached_property
from typing import Optional

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from synchros2.subscription import Subscription

from spot_basic.base import SpotBaseProtocol
from spot_basic.control.base import SpotControllerInterface
from spot_basic.utilities import spot_namespaced


class SpotForwardStateControllerInterface(SpotControllerInterface):
    """Forward state controller interface for Spot."""

    def __init__(self, spot: SpotBaseProtocol) -> None:
        """Initialize controller interface.

        Args:
            spot: base Spot driver interface.
        """
        super().__init__(spot, ["joint_state_broadcaster", "forward_state_controller"])
        self._joint_states_command_publisher = self.spot.node.create_publisher(
            Float64MultiArray, spot_namespaced(self.spot.robot_name, "forward_state_controller/commands"), 1
        )

    @cached_property
    def joint_states_feedback(self) -> Subscription[JointState]:
        """Get joint states feedback."""
        return Subscription(
            JointState, spot_namespaced(self.spot.robot_name, "low_level/joint_states"), node=self.spot.node
        )

    def update_joint_states_command(self, command: Optional[Sequence[float]] = None, **kwargs: Sequence[float]) -> None:
        """Update joint states command, one setpoint per joint per interface configured for the controller.

        Typical interfaces are position, velocity, and effort. Missing setpoints will default to 0.

        Args:
            command: full setpoint array. This array follows standard `ros2_control`
                ordering conventions i.e. `[<interface 0 for joints 0 to M>, <interface
                1 for joints 0 to M>, ..., <interface N for joints 0 to M>]`.
            kwargs: these are deduced from controller configuration, allowing commands
                to be set on a per interface and a per joint basis. The former must
                contain M setpoints, whereas the latter must contain N setpoints.

        Raises:
            RuntimeError: if the controller interface is not attached
        """
        if self.configuration is None:
            raise RuntimeError("controller interface must be attached to update setpoints")

        joints = self.configuration["forward_state_controller"]["joints"]
        joints = list(map(lambda name: name.removeprefix(self.spot.robot_name).strip("/"), joints))
        interfaces = self.configuration["forward_state_controller"]["interface_names"]

        if command is not None:
            if len(command) != len(joints) * len(interfaces):
                raise ValueError(f"command must have {len(joints) * len(interfaces)} elements")
            data = list(command)
        elif all(name in kwargs for name in joints) or all(name in kwargs for name in interfaces):
            data = [0.0] * len(joints) * len(interfaces)
        else:
            raise ValueError(
                "either a whole-body command or a complete set of per joint or per interface setpoints must be given"
            )

        for i, interface in enumerate(interfaces):
            if interface in kwargs:
                command = kwargs[interface]
                if len(command) != len(joints):
                    raise ValueError(f"{interface} command must have {len(joints)} elements, one per joint")
                data[i * len(joints) : (i + 1) * len(joints)] = command

        for i, joint in enumerate(joints):
            if joint in kwargs:
                command = kwargs[joint]
                if len(command) != len(interfaces):
                    raise ValueError(f"{joint} command must have {len(interfaces)} elements, one per interface")
                for j, value in enumerate(command):
                    data[j * len(joints) + i] = value

        self._joint_states_command_publisher.publish(Float64MultiArray(data=data))
