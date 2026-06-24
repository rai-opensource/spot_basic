# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections.abc import Sequence
from typing import Any, Dict, Optional, Protocol, TypeVar

from controller_manager_msgs.srv import (
    ConfigureController,
    LoadController,
    SetHardwareComponentState,
    SwitchController,
    UnloadController,
)
from lifecycle_msgs.msg import State
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.parameter import parameter_value_to_python
from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import SpotServiced, spot_namespaced
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike, wait_for_future


class SpotControllerProtocol(Protocol):
    """Base Spot (low-level) controller protocol"""

    class AttachProtocol(Protocol):
        """Protocol for hardware interface and controller attachment."""

        def synchronously(
            self,
            *,
            timeout_sec: Optional[float] = None,
        ) -> None:
            """Attach hardware interface and controller to Spot synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the attach request will be cancelled and the call will raise. Note this
                timeout is local to the caller and interruption may occur in between service
                requests.

            Raises:
                RuntimeError: if controller attachment fails or times out. Failure does not imply
                rollback, it is on the user to retry or undo attachment.
            """

        def asynchronously(self) -> FutureLike[None]:
            """Attach hardware interface and controller to Spot asynchronously.

            Returns:
                a future for this operation.

            Raises:
                RuntimeError: if controller attachment fails. Failure does not imply
                rollback, it is on the user to retry or undo attachment.
            """

    class DetachProtocol(Protocol):
        """Protocol for hardware interface and controller detachment."""

        def synchronously(
            self,
            *,
            timeout_sec: Optional[float] = None,
        ) -> None:
            """Detach hardware interface and controller to Spot synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the attach request will be cancelled and the call will raise. Note this
                timeout is local to the caller and interruption may occur in between service
                requests.

            Raises:
                RuntimeError: if controller detachment fails or times out. Failure does not imply
                rollback, it is on the user to retry or undo detachment.
            """

        def asynchronously(self) -> FutureLike[None]:
            """Detach hardware interface and controller to Spot asynchronously.

            Returns:
               a future for this operation.

            Raises:
                RuntimeError: if controller detachment fails. Failure does not imply
                rollback, it is on the user to retry or undo detachment.
            """


ControllerInterfaceT = TypeVar("ControllerInterfaceT", bound="SpotControllerInterface")


class SpotControllerInterface(SpotControllerProtocol):
    """Base interface to low level controllers for Spot, typically ROS 2 control driven."""

    SPOT_HARDWARE_INTERFACE = "SpotSystem"

    def __init__(self, spot: SpotBaseProtocol, controller_names: Sequence[str]) -> None:
        """Initialize controller interface.

        Args:
            spot: base Spot driver interface.
            controller_names: controllers to interface with, as configured for the Spot ROS 2's controller manager.
        """
        self.spot = spot
        self.executor = spot.node.executor
        self.controller_names = controller_names
        self.configuration: Optional[Dict[str, Any]] = None

        self._load_controller: SpotServiced[LoadController.Request, LoadController.Response] = SpotServiced(
            LoadController,
            spot_namespaced(self.spot.robot_name, "controller_manager/load_controller"),
            node=self.spot.node,
        )
        self._unload_controller: SpotServiced[UnloadController.Request, UnloadController.Response] = SpotServiced(
            UnloadController,
            spot_namespaced(self.spot.robot_name, "controller_manager/unload_controller"),
            node=self.spot.node,
        )
        self._configure_controller: SpotServiced[ConfigureController.Request, ConfigureController.Response] = (
            SpotServiced(
                ConfigureController,
                spot_namespaced(self.spot.robot_name, "controller_manager/configure_controller"),
                node=self.spot.node,
            )
        )
        self._switch_controllers: SpotServiced[SwitchController.Request, SwitchController.Response] = SpotServiced(
            SwitchController,
            spot_namespaced(self.spot.robot_name, "controller_manager/switch_controller"),
            node=self.spot.node,
        )
        self._set_hardware_interface_state: SpotServiced[
            SetHardwareComponentState.Request, SetHardwareComponentState.Response
        ] = SpotServiced(
            SetHardwareComponentState,
            spot_namespaced(self.spot.robot_name, "controller_manager/set_hardware_component_state"),
            node=self.spot.node,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotControllerProtocol.AttachProtocol])
    async def attach(self) -> None:
        """Attach (ie. activate) low level hardware interface and associated controller for Spot.

        See `SpotControllerInterface.AttachProtocol` for further reference.
        """
        response = await self._set_hardware_interface_state.asynchronously(
            name=self.SPOT_HARDWARE_INTERFACE, target_state=State(id=State.PRIMARY_STATE_ACTIVE)
        )
        if not response.ok:
            raise RuntimeError("failed to activate hardware interface")

        self.configuration = {}
        for name in self.controller_names:
            response = await self._load_controller.asynchronously(name=name)
            if not response.ok:
                raise RuntimeError(f"failed to load {name}")

            response = await self._configure_controller.asynchronously(name=name)
            if not response.ok:
                raise RuntimeError(f"failed to configure {name}")

            list_controller_parameters: SpotServiced[ListParameters.Request, ListParameters.Response] = SpotServiced(
                ListParameters, spot_namespaced(self.spot.robot_name, f"{name}/list_parameters"), node=self.spot.node
            )
            listed = await list_controller_parameters.asynchronously()

            get_controller_parameters: SpotServiced[GetParameters.Request, GetParameters.Response] = SpotServiced(
                GetParameters, spot_namespaced(self.spot.robot_name, f"{name}/get_parameters"), node=self.spot.node
            )
            result = await get_controller_parameters.asynchronously(names=listed.result.names)

            self.configuration[name] = dict(
                zip(listed.result.names, map(parameter_value_to_python, result.values), strict=False)
            )

        response = await self._switch_controllers.asynchronously(
            activate_controllers=self.controller_names, strictness=SwitchController.Request.STRICT
        )
        if not response.ok:
            raise RuntimeError("failed to activate controllers")

    @attach.sync_overload
    def _attach_synchronously(self, *, timeout_sec: Optional[float] = None) -> None:
        future = self.attach.asynchronously()
        if not wait_for_future(future, timeout_sec):
            future.cancel()
            raise RuntimeError("timed out attaching interface")
        return future.result()

    @generalized_method(spec=GeneralizedMethodLike[SpotControllerProtocol.DetachProtocol])
    async def detach(self, **kwargs: Any) -> None:
        """Detach (ie. deactivate) low level hardware interface and associated controller for Spot.

        See `SpotControllerInterface.DetachProtocol` for further reference.
        """
        self.configuration = None

        response = await self._switch_controllers.asynchronously(
            deactivate_controllers=self.controller_names, strictness=SwitchController.Request.STRICT
        )
        if not response.ok:
            raise RuntimeError("failed to deactivate controllers")

        for name in self.controller_names:
            response = await self._unload_controller.asynchronously(name=name)
            if not response.ok:
                raise RuntimeError(f"failed to unload {name}")

        response = await self._set_hardware_interface_state.asynchronously(
            name=self.SPOT_HARDWARE_INTERFACE, target_state=State(id=State.PRIMARY_STATE_UNCONFIGURED)
        )
        if not response.ok:
            raise RuntimeError("failed to deactivate hardware interface")

    @detach.sync_overload
    def _detach_synchronously(self, timeout_sec: Optional[float] = None) -> None:
        future = self.detach.asynchronously()
        if not wait_for_future(future, timeout_sec):
            future.cancel()
            raise RuntimeError("timed out detaching interface")
        return future.result()

    def __enter__(self: ControllerInterfaceT) -> ControllerInterfaceT:
        """Attach when entering context."""
        self.attach.synchronously()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Detach when leaving context."""
        self.detach.synchronously()
