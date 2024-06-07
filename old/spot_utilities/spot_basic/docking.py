# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Optional, Union

from bdai_ros2_wrappers.callables import generalized_method
from spot_msgs.srv import Dock
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServiceHandle
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotDockingMixin(SpotBaseProtocol):
    """Mixin that enables Spot docking maneuvers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotDockingMixin.dock.rebind(self, SpotServiced(Dock, spot_namespaced(self.robot_name, "dock"), node=self.node))
        SpotDockingMixin.undock.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "undock"), node=self.node)
        )

    @generalized_method(transitional=True)
    def dock(
        self,
        dock_id: int,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Dock.Response], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Dock the robot.

        Args:
            dock_id: ID for dock fiducial.

        Returns:
            A boolean to flag its outcome and an error message on failure.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "dock",
            self.dock.client,
            Dock.Request(dock_id=dock_id),
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    @generalized_method(transitional=True)
    def undock(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Trigger.Response], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Undock the robot.

        Returns:
            A boolean to flag its outcome and an error message on failure.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "undock",
            self.undock.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success
