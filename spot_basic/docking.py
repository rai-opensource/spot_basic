# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any, Optional, Protocol

from spot_msgs.srv import Dock
from std_srvs.srv import Trigger
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike

from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import SpotServiced, SpotServicedProtocol, spot_namespaced


class SpotDockingProtocol(SpotBaseProtocol, Protocol):
    """Spot docking behavior protocol."""

    class DockProtocol(SpotServicedProtocol[Dock.Request, Dock.Response], Protocol):
        """Docking protocol."""

        def synchronously(self, dock_id: int, *, timeout_sec: Optional[float] = None) -> Dock.Response:
            """Dock the robot synchronously.

            Args:
                dock_id: target dock fiducial id. It must be visible from the robot POV.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the docking request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response of to the docking request.

            Raises:
                ServiceError: if docking fails.
                ServiceTimeout: if docking times out.
            """

        def asynchronously(self, dock_id: int) -> FutureLike[Dock.Response]:
            """Dock the robot asynchronously.

            Args:
                dock_id: target dock fiducial id. It must be visible from the robot POV.

            Returns:
                a future response to the undocking request.
            """

    class UndockProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Undocking protocol."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Undock the robot synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the undocking request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response of to the undocking request.

            Raises:
                ServiceError: if undocking fails.
                ServiceTimeout: if undocking times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Undock the robot asynchronously.

            Returns:
                a future response to the undocking request.
            """

    dock: GeneralizedMethodLike[DockProtocol]
    undock: GeneralizedMethodLike[UndockProtocol]


class SpotDockingMixin(SpotBaseProtocol):
    """Mixin that enables Spot docking maneuvers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotDockingMixin.dock.rebind(self, SpotServiced(Dock, spot_namespaced(self.robot_name, "dock"), node=self.node))
        SpotDockingMixin.undock.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "undock"), node=self.node)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotDockingProtocol.DockProtocol])
    def dock(self, dock_id: int, **kwargs: Any) -> Dock.Response:
        """Dock the robot.

        Args:
            dock_id: target dock fiducial id. It must be visible from the robot POV.

        See `SpotDockingProtocol.DockProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotDockingProtocol.UndockProtocol])
    def undock(self, **kwargs: Any) -> Trigger.Response:
        """Undock the robot.

        See `SpotDockingProtocol.UndockProtocol` for further reference.
        """
