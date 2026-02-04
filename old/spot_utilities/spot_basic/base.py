# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any, Optional, Protocol

import synchros2.scope as ros_scope
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from spot_utilities.spot_util import spot_namespaced
from std_msgs.msg import Bool
from synchros2.futures import wait_for_future
from synchros2.subscription import Subscription
from synchros2.utilities import ensure

HAS_ARM_WAIT_TIME_S = 1.0
N_HAS_ARM_RETRIES = 5


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
            raise ValueError("no process-wide ROS 2 node available (did you use synchros2.process.main?)")
        self._node = node
        self._has_arm_status: Subscription[Bool] = Subscription(
            Bool,
            spot_namespaced(self.robot_name, "status/has_arm"),
            qos_profile=QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1),
            node=self.node,
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
            True if the robot has an arm, False if it does not (or it cannot be verified if so).
        """
        self.node.get_logger().debug(f"Checking if {self.robot_name} has an arm")
        if not wait_for_future(self._has_arm_status.latest_update, timeout_sec=5.0):
            self.node.get_logger().error(f"No {self.robot_name} driver found, assuming there is no arm")
            return False
        message = ensure(self._has_arm_status.latest)
        return message.data
