# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pytest
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.scope import ROSAwareScope

from spot_basic import SpotBasic


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_has_arm(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.robot_name == simple_spot.api.name
    assert spot_interface.node is ros.node
    assert spot_interface.has_arm()


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_spot_base_has_no_arm(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.robot_name == simple_spot_without_arm.api.name
    assert spot_interface.node is ros.node
    assert not spot_interface.has_arm()
