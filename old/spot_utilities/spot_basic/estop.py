# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotEStopMixin(SpotBaseProtocol):
    """Mixin that manages Spot emergency stops (or eStops)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.estop_hard = SpotServiced(Trigger, spot_namespaced(self.robot_name, "estop/hard"), node=self.node)
        self.estop_gentle = SpotServiced(Trigger, spot_namespaced(self.robot_name, "estop/gentle"), node=self.node)
        self.estop_release = SpotServiced(Trigger, spot_namespaced(self.robot_name, "estop/release"), node=self.node)
