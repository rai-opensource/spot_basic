# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Optional, Protocol, Union

from bdai_ros2_wrappers.callables import GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike
from rclpy.task import Future
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServicedProtocol, SpotServiceHandle
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotEStopProtocol(SpotBaseProtocol, Protocol):
    """Spot emergency stop management protocol."""

    class HardEStopProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Hard emergency stop protocol.

        A hard emergency stop cuts all power immediately.
        """

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Cut all power synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the emergency stop request will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the emergency stop request.

            Raises:
                ServiceError: if the emergency stop request fails.
                ServiceTimeout: if the emergency stop request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Cut all power asynchronously.

            Returns:
                a future response to the emergency stop request.
            """

    class GentleEStopProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Gentle emergency stop protocol.

        A gentle emergency stop attempts to stop all motion
        and sit down before cutting power.
        """

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Sit down and cut all power synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the emergency stop request will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the emergency stop request.

            Raises:
                ServiceError: if the emergency stop request fails.
                ServiceTimeout: if the emergency stop request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Sit down and cut all power asynchronously.

            Returns:
                a future response to the emergency stop request.
            """

    class ReleaseEStopProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Release emergency stop protocol.

        Releasing the emergency stop allows the robot to be powered on but it does not turn it on.
        To power on the robot, check the `SpotPowerProtocol`.
        """

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Release emergency stop synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the emergency stop release request will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the emergency stop release request.

            Raises:
                ServiceError: if the emergency stop release request fails.
                ServiceTimeout: if the emergency stop release request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Release emergency stop asynchronously.

            Returns:
                a future response to the emergency stop release request.
            """

    estop_hard: GeneralizedMethodLike[HardEStopProtocol]
    estop_gentle: GeneralizedMethodLike[GentleEStopProtocol]
    estop_release: GeneralizedMethodLike[ReleaseEStopProtocol]


class SpotEStopMixin(SpotBaseProtocol):
    """Mixin offering Spot emergency stops management support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotEStopMixin.estop_hard.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "estop/hard"), node=self.node)
        )
        SpotEStopMixin.estop_gentle.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "estop/gentle"), node=self.node)
        )
        SpotEStopMixin.estop_release.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "estop/release"), node=self.node)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotEStopProtocol.HardEStopProtocol])
    def estop_hard(self, **kwargs: Any) -> Trigger.Response:
        """Pull a hard emergency stop on the robot.

        See `SpotEStopProtocol.HardEStopProtocol` for further reference.
        """

    @estop_hard.legacy
    def _legacy_estop_hard(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Pull a hard emergency stop on the robot.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the emergency stop request is asynchronous or not.
            result_callback: optional callback for emergency stop request response.
            on_failure_callback: optional callback for emergency stop request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "estop_hard",
            self.estop_hard.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    @generalized_method(spec=GeneralizedMethodLike[SpotEStopProtocol.GentleEStopProtocol])
    def estop_gentle(self, **kwargs: Any) -> Trigger.Response:
        """Pull a gentle emergency stop on the robot.

        See `SpotEStopProtocol.GentleEStopProtocol` for further reference.
        """

    @estop_gentle.legacy
    def _legacy_estop_gentle(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Pull a gentle emergency stop on the robot.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the emergency stop request is asynchronous or not.
            result_callback: optional callback for emergency stop request response.
            on_failure_callback: optional callback for emergency stop request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "estop_gentle",
            self.estop_gentle.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    @generalized_method(spec=GeneralizedMethodLike[SpotEStopProtocol.ReleaseEStopProtocol])
    def estop_release(self, **kwargs: Any) -> Trigger.Response:
        """Release an active emergency stop on the robot.

        See `SpotEStopProtocol.ReleaseEStopProtocol` for further reference.
        """

    @estop_release.legacy
    def _legacy_estop_release(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Release an active emergency stop on the robot.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the emergency stop release request is asynchronous or not.
            result_callback: optional callback for emergency stop release request response.
            on_failure_callback: optional callback for emergency stop release request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "estop_release",
            self.estop_release.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success
