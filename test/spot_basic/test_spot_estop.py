# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pytest
from bosdyn.api.estop_pb2 import EstopStopLevel
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.scope import ROSAwareScope

from spot_basic import SpotBasic


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_hard_estop(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()

    assert spot_interface.claim.synchronously().success
    assert simple_spot.api.estop_status.stop_level == EstopStopLevel.ESTOP_LEVEL_NONE

    assert spot_interface.estop_hard.synchronously().success
    assert simple_spot.api.estop_status.stop_level == EstopStopLevel.ESTOP_LEVEL_CUT

    assert spot_interface.estop_release.synchronously().success
    assert simple_spot.api.estop_status.stop_level == EstopStopLevel.ESTOP_LEVEL_NONE

    assert spot_interface.release.synchronously().success


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_gentle_estop(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()

    assert spot_interface.claim.synchronously().success
    assert simple_spot.api.estop_status.stop_level == EstopStopLevel.ESTOP_LEVEL_NONE

    assert spot_interface.estop_gentle.synchronously().success
    assert simple_spot.api.estop_status.stop_level == EstopStopLevel.ESTOP_LEVEL_SETTLE_THEN_CUT

    assert spot_interface.estop_release.synchronously().success
    assert simple_spot.api.estop_status.stop_level == EstopStopLevel.ESTOP_LEVEL_NONE

    assert spot_interface.release.synchronously().success
