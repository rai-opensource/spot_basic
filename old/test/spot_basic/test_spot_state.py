# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import asdict

import pytest
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope
from numpy.testing import assert_equal
from spatialmath import SO3
from spot_utilities.spot_basic import FrameHint, SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_arm_joint_positions(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    kinematic_state = simple_spot.api.robot_state.kinematic_state
    expected_arm_joint_positions = {"sh0": 0.1, "sh1": -0.1, "el0": 0.2, "el1": -0.2, "wr0": 0.3, "wr1": -0.3}
    for basename, position in expected_arm_joint_positions.items():
        joint_state = kinematic_state.joint_states.add()
        joint_state.name = f"arm0.{basename}"
        joint_state.position.value = position

    assert spot_interface.initialize_robot()

    assert wait_for_future(spot_interface.joint_states.update, timeout_sec=10.0)

    arm_joint_positions = asdict(spot_interface.arm_joint_positions)
    assert arm_joint_positions == expected_arm_joint_positions


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_leg_joint_positions(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    kinematic_state = simple_spot.api.robot_state.kinematic_state

    expected_leg_joint_positions = {
        "fl.hx": 0.1,
        "fl.hy": -0.1,
        "fl.kn": 0.2,
        "fr.hx": -0.2,
        "fr.hy": 0.3,
        "fr.kn": -0.3,
        "hl.hx": 0.4,
        "hl.hy": -0.4,
        "hl.kn": 0.5,
        "hr.hx": -0.5,
        "hr.hy": 0.6,
        "hr.kn": -0.6,
    }
    for name, position in expected_leg_joint_positions.items():
        joint_state = kinematic_state.joint_states.add()
        joint_state.name = name
        joint_state.position.value = position

    assert spot_interface.initialize_robot()

    assert wait_for_future(spot_interface.joint_states.update, timeout_sec=10.0)

    assert spot_interface.leg_joint_positions.to_list() == list(expected_leg_joint_positions.values())


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_transform_lookup(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    kinematic_state = simple_spot.api.robot_state.kinematic_state
    kinematic_state.acquisition_timestamp.GetCurrentTime()

    transforms_snapshot = kinematic_state.transforms_snapshot
    parent_edge = transforms_snapshot.child_to_parent_edge_map[FrameHint.BODY.value]
    parent_edge.parent_frame_name = FrameHint.ODOM.value
    parent_edge.parent_tform_child.position.x = 1.0
    parent_edge.parent_tform_child.rotation.w = 1.0

    assert spot_interface.initialize_robot()

    transform = spot_interface.lookup_a_tform_b(FrameHint.ODOM, FrameHint.BODY, timeout=10.0, wait_for_frames=True)
    assert_equal(transform.t, (1.0, 0.0, 0.0))
    assert_equal(transform.R, SO3().A)  # identity
