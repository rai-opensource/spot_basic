# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotLeaseMixin:
    """Mixin that manages Spot leases."""

    def __init__(self: SpotBaseProtocol, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.claim = SpotServiced(Trigger, spot_namespaced(self.robot_name, "claim"), node=self.node)
        self.release = SpotServiced(Trigger, spot_namespaced(self.robot_name, "release"), node=self.node)
