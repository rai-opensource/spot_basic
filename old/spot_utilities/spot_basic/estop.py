# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Optional, Union

from bdai_ros2_wrappers.callables import generalized_method
from rclpy.task import Future
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServiceHandle
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotEStopMixin(SpotBaseProtocol):
    """Mixin that manages Spot emergency stops (or eStops)."""

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

    @generalized_method(transitional=True)
    def estop_hard(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Pull a hard emergency stop on the robot.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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

    @generalized_method(transitional=True)
    def estop_gentle(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Pull a gentle emergency stop on the robot.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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

    @generalized_method(transitional=True)
    def estop_release(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Release an active emergency stop on the robot.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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
