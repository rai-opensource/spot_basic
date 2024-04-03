# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai_ros2_wrappers.scope import ROSAwareScope
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.usefixtures("simple_spot_node")
def test_spot_leasing(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    assert spot_interface.claim()
    for resource in simple_spot.api.leasable_resources:
        if resource.resource == "body":
            client_name = resource.lease_owner.client_name
            assert client_name.startswith("ros_spot")
            assert not resource.is_stale

    assert spot_interface.release()
    for resource in simple_spot.api.leasable_resources:
        if resource.resource == "body":
            assert resource.lease_owner.client_name == ""
            assert resource.is_stale
