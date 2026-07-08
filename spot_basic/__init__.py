# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import contextlib
from typing import Generator, Optional, Protocol

from rclpy.node import Node

from spot_basic.arm_surface_contact import SpotArmSurfaceContactMixin
from spot_basic.base import SpotBase
from spot_basic.cam.audio import SpotCAMAudioMixin
from spot_basic.command import SpotCommandMixin
from spot_basic.command.arm import SpotArmCommandMixin
from spot_basic.command.full_body import SpotFullBodyCommandMixin
from spot_basic.command.gripper import SpotGripperCommandMixin
from spot_basic.command.mobility import SpotMobilityCommandMixin
from spot_basic.docking import SpotDockingMixin
from spot_basic.estop import SpotEStopMixin
from spot_basic.graph_nav import SpotGraphNavMixin
from spot_basic.gripper_camera import SpotGripperCameraMixin
from spot_basic.ik import SpotIKMixin
from spot_basic.image import SpotImageMixin
from spot_basic.lease import SpotLeaseMixin
from spot_basic.manipulation import SpotManipulationMixin
from spot_basic.power import SpotPowerMixin
from spot_basic.state import FrameHint, SpotStateMixin
from spot_basic.utilities import *  # noqa
from spot_basic.utilities import spot_namespaced
from spot_basic.world import SpotWorldMixin


class SpotBasic(
    SpotBase,
    SpotLeaseMixin,
    SpotPowerMixin,
    SpotEStopMixin,
    SpotDockingMixin,
    SpotStateMixin,
    SpotCAMAudioMixin,
    SpotGraphNavMixin,
    SpotIKMixin,
    SpotWorldMixin,
    SpotArmCommandMixin,
    SpotGripperCommandMixin,
    SpotGripperCameraMixin,
    SpotMobilityCommandMixin,
    SpotFullBodyCommandMixin,
    SpotManipulationMixin,
    SpotCommandMixin,
    SpotArmSurfaceContactMixin,
    SpotImageMixin,
):
    """Basic interface to Spot robots over ROS 2."""

    def __init__(
        self,
        robot_name: str,
        *,
        node: Optional[Node] = None,
        disable_collisions: bool = False,
    ):
        """Initialize the Spot interface.

        Args:
            robot_name: name of the robot
            node: optional parent node, defaults to the current process node
            disable_collisions: True if collisions should be ignored, False otherwise
            rviz: True if visuals should be published, False otherwise
        """
        super().__init__(robot_name, disable_collisions=disable_collisions, node=node)

    @contextlib.contextmanager
    def leash(self, weak: bool = False, drop_on_exit: bool = False, timeout_sec: Optional[float] = None) -> Generator:
        """Leash the robot for immediate use.

        This amounts to taking the robot body lease, powering the robot, and getting it ready for action.
        It will also ensure relevant ROS 2 interfaces are up and ready to be used.

        Args:
            weak: if weak, the leash will not have the robot stand up.
            drop_on_exit: if dropped on exit, the robot will be left powered on and up on exit.
            timeout_sec: optional timeout, in seconds. May block indefinitely if none is provided.
        """
        with self.lease(timeout_sec=timeout_sec):
            self.wait_for_servers(timeout_sec)
            self.power_on.synchronously(timeout_sec=timeout_sec)
            try:
                if not weak:
                    self.stand.synchronously(timeout_sec=timeout_sec)
                try:
                    yield
                finally:
                    if not drop_on_exit and not weak:
                        self.stop.synchronously(timeout_sec=timeout_sec)
                        if self.has_arm():
                            self.stow_arm.synchronously(timeout_sec=timeout_sec)
                        self.sit.synchronously(timeout_sec=timeout_sec)
            finally:
                if not drop_on_exit:
                    self.power_off.synchronously(timeout_sec=timeout_sec)

    def wait_for_servers(self, timeout_sec: Optional[float] = None) -> bool:
        """Wait for the robot's ROS 2 action servers to become available.

        Waits for the RobotCommand action server, and if the robot has an arm,
        also waits for the Manipulation action server.

        Args:
            timeout_sec: optional timeout per server, in seconds. May block indefinitely if none is provided.

        Returns:
            True if all expected servers were found, False if any were not available within the timeout.
        """
        all_found = True
        self.node.get_logger().info(f"Waiting for the 'RobotCommand' action server for {self.robot_name} ...")
        if self.execute.wait_for_server(timeout_sec):
            self.node.get_logger().info(f"Found the 'RobotCommand' action server for {self.robot_name}.")
        else:
            self.node.get_logger().warn(f"No robot command interface found for {self.robot_name}.")
            all_found = False
        if self.has_arm():
            self.node.get_logger().info(f"Waiting for the 'Manipulation' action server for {self.robot_name} ...")
            if self.perform.wait_for_server(timeout_sec):
                self.node.get_logger().info(f"Found the 'Manipulation' action server for {self.robot_name}.")
            else:
                self.node.get_logger().warn(f"No manipulation command interface found for {self.robot_name}.")
                all_found = False
        return all_found


class _Successful(Protocol):
    success: bool


def successful(result: Optional[_Successful]) -> bool:
    return result is not None and result.success
