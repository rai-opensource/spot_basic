# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture

from bdai_ros2_wrappers.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_has_arm(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.robot_name == simple_spot.api.name
    assert spot_interface.node is ros.node
    assert spot_interface.has_arm()


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_spot_base_has_no_arm(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.robot_name == simple_spot_without_arm.api.name
    assert spot_interface.node is ros.node
    assert not spot_interface.has_arm()
