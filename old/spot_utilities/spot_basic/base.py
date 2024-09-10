# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional, Protocol

import bdai_ros2_wrappers.scope as ros_scope
from bdai_ros2_wrappers.service import Serviced
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from spot_utilities.spot_util import spot_namespaced


class SpotBaseProtocol(Protocol):
    """Spot base protocol."""

    @property
    def robot_name(self) -> str:
        """Target robot name."""

    @property
    def node(self) -> Node:
        """Parent ROS 2 node."""

    def has_arm(self) -> Optional[bool]:
        """Checks if the robot has an arm."""


class SpotBase(SpotBaseProtocol):
    """Base Spot interface for interacting with its ROS 2 driver."""

    def __init__(self, robot_name: str, *args: Any, node: Optional[Node] = None, **kwargs: Any) -> None:
        """Initialize the robot interface.

        Args:
            robot_name: name of the robot.
            node: optional parent node, defaults to the current process node if none is given.
        """
        self._robot_name = robot_name
        node = node or ros_scope.node()
        if node is None:
            raise ValueError("no process-wide ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")
        self._node = node
        self._get_spot_parameters: Serviced[GetParameters.Request, GetParameters.Response] = Serviced(
            GetParameters, spot_namespaced(self.robot_name, "spot_ros2/get_parameters"), node=self.node
        )
        # make sure robot name and node properties are initialized before constructing mixins
        super().__init__(*args, **kwargs)

    @property
    def robot_name(self) -> str:
        """Target robot name."""
        return self._robot_name

    @property
    def node(self) -> Node:
        """Parent ROS 2 node."""
        return self._node

    def has_arm(self) -> bool:
        """Check if the robot has an arm.

        Returns:
            True if the robot has an arm, False if it does not.
        """
        self.node.get_logger().debug(f"Checking if {self.robot_name} has an arm")

        if not self._get_spot_parameters.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f"No {self.robot_name} driver found, assuming there is no arm")
            return False

        response = self._get_spot_parameters(GetParameters.Request(names=["has_arm"]), timeout_sec=5.0)
        param = response.values[0]
        return param.bool_value
