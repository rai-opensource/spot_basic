# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


import pytest
from bosdyn.api.manipulation_api_pb2 import ApiGraspOverrideResponse
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture

from bdai_ros2_wrappers.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_override_grasp_or_carry(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    api_response = ApiGraspOverrideResponse()
    simple_spot.api.OverrideGrasp.future.returns(api_response)
    response = spot_interface.override_grasp_or_carry(grasp_override=2, carry_override=2)
    assert response.success
