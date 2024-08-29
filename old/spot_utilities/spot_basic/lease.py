# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import contextlib
from typing import Any, Callable, Generator, Optional, Protocol, Union

from bdai_ros2_wrappers.callables import GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike
from rclpy.task import Future
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServicedProtocol, SpotServiceHandle
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotLeaseProtocol(SpotBaseProtocol, Protocol):
    """Spot lease management protocol."""

    class ClaimProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Lease claim protocol."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Claim the robot (body) lease synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the claim request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the claim request.

            Raises:
                ServiceError: if the claim request fails.
                ServiceTimeout: if the claim request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Claim the robot (body) lease asynchronously.

            Returns:
                a future response to the claim request.
            """

    class ReleaseProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Lease release protocol."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Release the robot (body) lease synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the release request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the release request.

            Raises:
                ServiceError: if the release request fails.
                ServiceTimeout: if the release request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Release the robot (body) lease asynchronously.

            Returns:
                a future response to the release request.
            """

    claim: GeneralizedMethodLike[ClaimProtocol]
    release: GeneralizedMethodLike[ReleaseProtocol]


class SpotLeaseMixin(SpotBaseProtocol):
    """Mixin offering Spot lease management support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotLeaseMixin.claim.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "claim"), node=self.node)
        )
        SpotLeaseMixin.release.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "release"), node=self.node)
        )

    @contextlib.contextmanager
    def lease(self, timeout_sec: Optional[float] = None) -> Generator:
        """Maintain a lease on the robot while in scope.

        Args:
            timeout_sec: optional timeout, in seconds. May block indefinitely
            if none is provided.
        """
        self.claim.synchronously(timeout_sec=timeout_sec)
        try:
            yield
        finally:
            self.release.synchronously(timeout_sec=timeout_sec)

    @generalized_method(spec=GeneralizedMethodLike[SpotLeaseProtocol.ClaimProtocol])
    def claim(self, **kwargs: Any) -> Trigger.Response:
        """Claim the robot by acquiring its body lease.

        See `SpotLeaseProtocol.ClaimProtocol` for further reference.
        """

    @claim.legacy
    def _legacy_claim(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Claim the robot by acquiring its body lease.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the claim request is asynchronous or not.
            result_callback: optional callback for claim request response.
            on_failure_callback: optional callback for claim request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "claim",
            self.claim.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success

    @generalized_method(spec=GeneralizedMethodLike[SpotLeaseProtocol.ReleaseProtocol])
    def release(self, **kwargs: Any) -> Trigger.Response:
        """Release the robot by returning its body lease.

        See `SpotLeaseProtocol.ReleaseProtocol` for further reference.
        """

    @release.legacy
    def _legacy_release(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Release the robot by returning its body lease.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            asynchronous: whether the release request is asynchronous or not.
            result_callback: optional callback for release request response.
            on_failure_callback: optional callback for release request failure.

        Returns:
            a boolean to flag its outcome for synchronous calls, and
            a service handle for asynchronous calls.
        """
        handle = SpotServiceHandle.call(
            self.robot_name,
            "release",
            self.release.client,
            result_callback=result_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )
        if asynchronous:
            return handle
        if not handle.wait():
            return False
        return handle.result is not None and handle.result.success
