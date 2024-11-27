# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import contextlib
from typing import Generator, Optional

from rclpy.node import Node
from spot_utilities.spot_basic.base import SpotBase
from spot_utilities.spot_basic.cam.audio import SpotCAMAudioMixin
from spot_utilities.spot_basic.choreography import SpotChoreographyMixin
from spot_utilities.spot_basic.command import SpotCommandMixin
from spot_utilities.spot_basic.command.arm import SpotArmCommandMixin
from spot_utilities.spot_basic.command.full_body import SpotFullBodyCommandMixin
from spot_utilities.spot_basic.command.gripper import SpotGripperCommandMixin
from spot_utilities.spot_basic.command.mobility import SpotMobilityCommandMixin
from spot_utilities.spot_basic.docking import SpotDockingMixin
from spot_utilities.spot_basic.estop import SpotEStopMixin
from spot_utilities.spot_basic.graph_nav import SpotGraphNavMixin
from spot_utilities.spot_basic.gripper_camera import SpotGripperCameraMixin
from spot_utilities.spot_basic.ik import SpotIKMixin
from spot_utilities.spot_basic.image import SpotImageMixin
from spot_utilities.spot_basic.lease import SpotLeaseMixin
from spot_utilities.spot_basic.legacy import SpotLegacyMixin
from spot_utilities.spot_basic.manipulation import SpotManipulationMixin
from spot_utilities.spot_basic.power import SpotPowerMixin
from spot_utilities.spot_basic.state import FrameHint, SpotStateMixin
from spot_utilities.spot_basic.utilities import *  # noqa
from spot_utilities.spot_basic.utilities import SpotActionHandle
from spot_utilities.spot_basic.world import SpotWorldMixin
from spot_utilities.spot_util import spot_namespaced


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
    SpotLegacyMixin,
    SpotArmCommandMixin,
    SpotGripperCommandMixin,
    SpotGripperCameraMixin,
    SpotMobilityCommandMixin,
    SpotFullBodyCommandMixin,
    SpotChoreographyMixin,
    SpotManipulationMixin,
    SpotCommandMixin,
    SpotImageMixin,
):
    """Basic interface to Spot robots over ROS 2."""

    def __init__(
        self,
        robot_name: str,
        *,
        node: Optional[Node] = None,
        disable_collisions: bool = False,
        dry: bool = False,
        rviz: bool = False,
    ):
        """Initialize the Spot interface.

        Args:
            robot_name: name of the robot
            node: optional parent node, defaults to the current process node
            disable_collisions: True if collisions should be ignored, False otherwise
            dry: True if this is a dry run and commands should not actually be sent, False otherwise
            rviz: True if visuals should be published, False otherwise
        """
        super().__init__(robot_name, dry=dry, disable_collisions=disable_collisions, node=node)
        assert not rviz, "Aid visuals are deprecated"

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
            if not self.execute.wait_for_server(timeout_sec):
                self.node.get_logger().warn(f"No robot command interface found for {self.robot_name}.")
            if self.has_arm() and not self.perform.wait_for_server(timeout_sec):
                self.node.get_logger().warn(f"No manipulation command interface found for {self.robot_name}.")
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
