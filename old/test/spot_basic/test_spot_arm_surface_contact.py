# Copyright (c) 2024-2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

import pytest
from bosdyn.api.arm_surface_contact_pb2 import ArmSurfaceContact as ArmSurfaceContactProto
from bosdyn.api.arm_surface_contact_service_pb2 import ArmSurfaceContactResponse
from bosdyn.api.header_pb2 import CommonError
from spatialmath import SE3
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import wait_for_future
from synchros2.scope import ROSAwareScope

from bdai.trajectory.se3_trajectory import SE3Trajectory, SE3TrajectoryPoint


def _create_dummy_args() -> list[Any]:
    pos_mode = ArmSurfaceContactProto.Request.AxisMode.AXIS_MODE_POSITION
    force_mode = ArmSurfaceContactProto.Request.AxisMode.AXIS_MODE_FORCE
    return [
        SpotBasic.FrameHint.VISION,
        SE3Trajectory(
            [
                SE3TrajectoryPoint(task_t_waypoint=SE3(), time_from_start=0.1),
                SE3TrajectoryPoint(task_t_waypoint=SE3(x=0.0, y=0.0, z=1.0), time_from_start=0.3),
            ]
        ),
        pos_mode,
        pos_mode,
        force_mode,
        [0.0, 0.0, -0.5],
    ]


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_with_no_arm(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)

    with pytest.raises(RuntimeError):
        spot_interface.arm_surface_contact.synchronously(*_create_dummy_args())


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_surface_contact(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = ArmSurfaceContactResponse()
    # Check this runs asynchronously by first calling and then serving the response
    action = spot_interface.arm_surface_contact.asynchronously(*_create_dummy_args())
    # Serve the response
    simple_spot.api.ArmSurfaceContact.future.returns(command_response)
    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    # Test synchronously
    simple_spot.api.ArmSurfaceContact.future.returns(command_response)
    result = spot_interface.arm_surface_contact.synchronously(*_create_dummy_args())
    assert result.success

    # Test an error
    command_response.header.error.code = CommonError.CODE_INVALID_REQUEST
    simple_spot.api.ArmSurfaceContact.future.returns(command_response)
    result = spot_interface.arm_surface_contact.synchronously(*_create_dummy_args(), nothrow=True)
    assert not result.success
