# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai_ros2_wrappers.futures import unwrap_future
from bdai_ros2_wrappers.scope import ROSAwareScope
from bosdyn.api.spot.inverse_kinematics_pb2 import InverseKinematicsRequest, InverseKinematicsResponse
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.state import FrameHint
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_inverse_kinematics_request(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    request = InverseKinematicsRequest()
    request.root_frame_name = FrameHint.VISION.value
    request.fixed_stance.fl_rt_scene.x = 1.0
    request.fixed_stance.fr_rt_scene.x = 1.0
    future = spot_interface.get_inverse_kinematic_solutions.asynchronously(request)

    call = simple_spot.api.InverseKinematics.serve(timeout=2.0)
    assert call is not None
    assert call.request.root_frame_name == request.root_frame_name
    assert call.request.fixed_stance.fl_rt_scene.x == request.fixed_stance.fl_rt_scene.x
    assert call.request.fixed_stance.fr_rt_scene.x == request.fixed_stance.fr_rt_scene.x
    call.returns(InverseKinematicsResponse(status=InverseKinematicsResponse.Status.STATUS_OK))

    response = unwrap_future(future, timeout_sec=5.0).response
    assert response.status.value == InverseKinematicsResponse.Status.STATUS_OK
