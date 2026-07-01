# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pytest
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.scope import ROSAwareScope

from spot_basic import SpotBasic


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_manual_spot_leasing(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()

    assert spot_interface.claim.synchronously().success
    for resource in simple_spot.api.leasable_resources:
        if resource.resource == "body":
            client_name = resource.lease_owner.client_name
            assert client_name.startswith("ros_spot")
            assert not resource.is_stale

    assert spot_interface.release.synchronously().success
    for resource in simple_spot.api.leasable_resources:
        if resource.resource == "body":
            assert resource.lease_owner.client_name == ""
            assert resource.is_stale


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_take_lease(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()

    assert spot_interface.take_lease.synchronously()
    for resource in simple_spot.api.leasable_resources:
        if resource.resource == "body":
            client_name = resource.lease_owner.client_name
            assert client_name.startswith("ros_spot")
            assert not resource.is_stale
