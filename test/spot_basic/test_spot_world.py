# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pytest
from bosdyn.api.world_object_pb2 import (
    AprilTagProperties,
    ListWorldObjectRequest,
    ListWorldObjectResponse,
    WorldObject,
    WorldObjectType,
)
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.scope import ROSAwareScope

from spot_basic import SpotBasic


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_world_inspection(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    expected_world_object = WorldObject(id=1234)
    expected_apriltag_properties = expected_world_object.apriltag_properties
    expected_apriltag_properties.tag_id = 521
    expected_apriltag_properties.fiducial_filtered_pose_status = AprilTagProperties.AprilTagPoseStatus.STATUS_OK
    response = ListWorldObjectResponse(world_objects=[expected_world_object])
    simple_spot.api.ListWorldObjects.future.returns(response).forever()

    request = ListWorldObjectRequest()
    request.object_type.append(WorldObjectType.WORLD_OBJECT_APRILTAG)
    response = spot_interface.list_world_objects.synchronously(request, timeout_sec=5.0)

    assert len(response.response.world_objects) == 1
    world_object = response.response.world_objects[0]
    assert expected_world_object.id == world_object.id
    apriltag_properties = expected_world_object.apriltag_properties
    assert expected_apriltag_properties.tag_id == apriltag_properties.tag_id
    assert apriltag_properties.fiducial_filtered_pose_status == (
        expected_apriltag_properties.fiducial_filtered_pose_status
    )
