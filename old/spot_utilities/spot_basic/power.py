# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

from bdai_ros2_wrappers.subscription import Subscription
from spot_msgs.msg import PowerState
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotPowerMixin(SpotBaseProtocol):
    """Mixin that manages Spot power."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.power_states = Subscription(
            PowerState, spot_namespaced(self.robot_name, "status/power_states"), node=self.node
        )
        self.power_on = self.power_motors_on = SpotServiced(
            Trigger, spot_namespaced(self.robot_name, "power_on"), node=self.node
        )
        self.power_off = self.power_motors_off = SpotServiced(
            Trigger, spot_namespaced(self.robot_name, "power_off"), node=self.node
        )
