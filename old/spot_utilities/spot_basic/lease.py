# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import contextlib
from typing import Any, Callable, Generator, Optional, Union

from bdai_ros2_wrappers.callables import generalized_method
from rclpy.task import Future
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServiceHandle
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotLeaseMixin(SpotBaseProtocol):
    """Mixin that manages Spot leases."""

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
        """Maintain a lease on the robot while in scope."""
        self.claim.synchronously(timeout_sec=timeout_sec)
        try:
            yield
        finally:
            self.release.synchronously(timeout_sec=timeout_sec)

    @generalized_method(transitional=True)
    def claim(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Claim the robot by acquiring its lease.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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

    @generalized_method(transitional=True)
    def release(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Release the robot by returning its lease.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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
