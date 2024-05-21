# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Optional, Union

from bdai_ros2_wrappers.callables import generalized_method
from bdai_ros2_wrappers.subscription import Subscription
from rclpy.task import Future
from spot_msgs.msg import PowerState
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServiceHandle
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotPowerMixin(SpotBaseProtocol):
    """Mixin that manages Spot power."""

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

    @generalized_method(transitional=True)
    def power_on(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Turn on the robot.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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

    @generalized_method(transitional=True)
    def power_off(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Union[bool, SpotServiceHandle]:
        """Turn off the robot.

        Returns:
            A boolean to flag its outcome and an error message on failure.
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
