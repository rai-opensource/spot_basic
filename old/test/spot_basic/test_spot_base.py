# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai_ros2_wrappers.scope import ROSAwareScope
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.reconfigurable_spot_graph_description)
def test_spot_base_interface(reconfigurable_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(reconfigurable_spot.api.name, node=ros.node)
    assert spot_interface.robot_name == reconfigurable_spot.api.name
    assert spot_interface.node is ros.node
    manipulator_state = reconfigurable_spot.api.robot_state.manipulator_state
    assert spot_interface.has_arm() == manipulator_state.is_gripper_holding_item
