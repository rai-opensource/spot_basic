# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import contextlib
from typing import Any, Callable, Generator, Optional, Protocol, Union

from rclpy.task import Future
from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import SpotServiced, SpotServicedProtocol, SpotServiceHandle, spot_namespaced
from spot_msgs.msg import PowerState
from std_srvs.srv import Trigger
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike
from synchros2.subscription import Subscription

from bdai.deprecation import deprecated


class SpotPowerProtocol(SpotBaseProtocol, Protocol):
    """Spot power management protocol."""

    class PowerOnProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Motor power on protocol."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Power on the robot (motors) synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the power on request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the power on request.

            Raises:
                ServiceError: if the power on request fails.
                ServiceTimeout: if the power on request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Power on the robot (motors) asynchronously.

            Returns:
                a future response to the power on request.
            """

    class PowerOffProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Motor power off protocol."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Power off the robot (motors) synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the power off request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the power off request.

            Raises:
                ServiceError: if the power off request fails.
                ServiceTimeout: if the power off request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Power off the robot (motors) asynchronously.

            Returns:
                a future response to the power off request.
            """

    power_on: GeneralizedMethodLike[PowerOnProtocol]
    power_motors_on: GeneralizedMethodLike[PowerOnProtocol]
    power_off: GeneralizedMethodLike[PowerOffProtocol]
    power_motors_off: GeneralizedMethodLike[PowerOffProtocol]


class SpotPowerMixin(SpotBaseProtocol):
    """Mixin offering Spot power management support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.power_states = Subscription(
            PowerState, spot_namespaced(self.robot_name, "status/power_states"), node=self.node
        )
        SpotPowerMixin.power_on.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "power_on"), node=self.node)
        )
        SpotPowerMixin.power_off.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "power_off"), node=self.node)
        )

    @contextlib.contextmanager
    def power(self, timeout_sec: Optional[float] = None) -> Generator:
        """Maintain power for the robot while in scope.

        Args:
            timeout_sec: optional timeout, in seconds. May block indefinitely if none is provided.
        """
        self.power_on.synchronously(timeout_sec=timeout_sec)
        try:
            yield
        finally:
            self.power_off.synchronously(timeout_sec=timeout_sec)

    @generalized_method(spec=GeneralizedMethodLike[SpotPowerProtocol.PowerOnProtocol])
    def power_on(self, **kwargs: Any) -> Trigger.Response:
        """Power on the robot (motors).

        See `SpotPowerProtocol.PowerOnProtocol` for further reference.
        """

    @power_on.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.power_on.synchronously",
    )
    def _legacy_power_on(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Power on the robot (motors).

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the power on request is asynchronous or not.
            result_callback: optional callback for power on request response.
            on_failure_callback: optional callback for power on request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "power_on",
            self.power_on.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    power_motors_on = power_on

    @generalized_method(spec=GeneralizedMethodLike[SpotPowerProtocol.PowerOffProtocol])
    def power_off(self, **kwargs: Any) -> Trigger.Response:
        """Power off the robot (motors).

        See `SpotPowerProtocol.PowerOffProtocol` for further reference.
        """

    @power_off.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.power_off.synchronously",
    )
    def _legacy_power_off(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Power off the robot (motors).

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the power off request is asynchronous or not.
            result_callback: optional callback for power off request response.
            on_failure_callback: optional callback for power off request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "power_off",
            self.power_off.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    power_motors_off = power_off
