# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import contextlib
import itertools

import pytest
from bosdyn.api.robot_command_pb2 import JointControlStreamResponse, RobotCommandResponse
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.control import (
    SpotForwardPositionControllerInterface,
    SpotForwardStateControllerInterface,
)
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.scope import ROSAwareScope

ORDERED_JOINTS = [
    "front_left_hip_x",
    "front_left_hip_y",
    "front_left_knee",
    "front_right_hip_x",
    "front_right_hip_y",
    "front_right_knee",
    "rear_left_hip_x",
    "rear_left_hip_y",
    "rear_left_knee",
    "rear_right_hip_x",
    "rear_right_hip_y",
    "rear_right_knee",
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

NUM_JOINTS = len(ORDERED_JOINTS)


def zero_robot_state_stream_response() -> RobotStateStreamResponse:
    zero_response = RobotStateStreamResponse()
    zero_response.joint_states.position.extend([0.0] * NUM_JOINTS)
    zero_response.joint_states.velocity.extend([0.0] * NUM_JOINTS)
    zero_response.joint_states.load.extend([0.0] * NUM_JOINTS)
    zero_response.inertial_state.packets.add()  # hardware interface expects at least a packet
    zero_response.contact_states.extend([0] * 4)  # hardware interface expects all 4 feet to have a contact state
    return zero_response


@pytest.mark.launch(fixture=pytest.simple_controllable_spot_graph_description)
def test_spot_full_state_control(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    simple_spot.api.GetRobotStateStream.future.returns(itertools.repeat(zero_robot_state_stream_response()))

    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(response)

    with SpotForwardStateControllerInterface(spot_interface) as ifc:
        call = simple_spot.api.JointControlStream.serve(timeout=2.0)
        assert call is not None

        with pytest.raises(ValueError):
            ifc.update_joint_states_command()

        with pytest.raises(ValueError):
            short_command = [0.0]
            ifc.update_joint_states_command(short_command)

        expected_command = list(range(1, NUM_JOINTS * 3 + 1))

        with pytest.raises(ValueError):
            short_position_command = [0.0]
            ifc.update_joint_states_command(expected_command, position=short_position_command)

        with pytest.raises(ValueError):
            short_arm_sh0_command = [0.0]
            ifc.update_joint_states_command(expected_command, arm_sh0=short_arm_sh0_command)

        ifc.update_joint_states_command(expected_command)

        ifc.update_joint_states_command(
            position=expected_command[0::3], velocity=expected_command[1::3], effort=expected_command[2::3]
        )

        ifc.update_joint_states_command(
            **{
                name: (
                    expected_command[i],
                    expected_command[NUM_JOINTS + i],
                    expected_command[2 * NUM_JOINTS + i],
                )
                for i, name in enumerate(ORDERED_JOINTS)
            }
        )

        requests = list(itertools.islice(call.request, 300))

    response = JointControlStreamResponse()
    response.status = JointControlStreamResponse.Status.STATUS_OK
    call.returns(response)

    assert len(requests) > 0
    assert requests[-1].joint_command.position == expected_command[:NUM_JOINTS]
    assert requests[-1].joint_command.velocity == expected_command[NUM_JOINTS : NUM_JOINTS * 2]
    assert requests[-1].joint_command.load == expected_command[NUM_JOINTS * 2 :]


@pytest.mark.launch(fixture=pytest.simple_controllable_spot_graph_description)
def test_spot_joint_position_control(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    simple_spot.api.GetRobotStateStream.future.returns(itertools.repeat(zero_robot_state_stream_response()))

    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(response)

    with SpotForwardPositionControllerInterface(spot_interface) as ifc:
        call = simple_spot.api.JointControlStream.serve(timeout=2.0)
        assert call is not None

        with pytest.raises(ValueError):
            ifc.update_joint_position_command()

        with pytest.raises(ValueError):
            short_command = [0.0]
            ifc.update_joint_position_command(short_command)

        expected_command = list(range(1, NUM_JOINTS + 1))
        ifc.update_joint_position_command(expected_command)

        ifc.update_joint_position_command(**{name: expected_command[i] for i, name in enumerate(ORDERED_JOINTS)})

        requests = list(itertools.islice(call.request, 300))

    response = JointControlStreamResponse()
    response.status = JointControlStreamResponse.Status.STATUS_OK
    call.returns(response)

    assert len(requests) > 0
    assert requests[-1].joint_command.position == expected_command


@pytest.mark.launch(fixture=pytest.simple_controllable_spot_graph_description)
def test_spot_full_state_control_feedback(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    expected_response = RobotStateStreamResponse()
    expected_response.joint_states.position.extend(range(1, NUM_JOINTS + 1))
    expected_response.joint_states.velocity.extend(range(1, NUM_JOINTS + 1))
    expected_response.joint_states.load.extend(range(1, NUM_JOINTS + 1))
    expected_response.inertial_state.packets.add()  # hardware interface expects at least a packet
    expected_response.contact_states.extend([0] * 4)  # hardware interface expects all 4 feet to have a contact state
    simple_spot.api.GetRobotStateStream.future.returns(itertools.repeat(expected_response))

    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(response)

    with SpotForwardStateControllerInterface(spot_interface) as ifc:
        call = simple_spot.api.JointControlStream.serve(timeout=2.0)
        assert call is not None

        with contextlib.closing(ifc.joint_states_feedback.stream()) as joint_states_stream:
            joint_states_batch = list(itertools.islice(joint_states_stream, 300))

        assert len(joint_states_batch) > 0
        for i, joint_name in enumerate(joint_states_batch[-1].name):
            j = ORDERED_JOINTS.index(joint_name.removeprefix(simple_spot.api.name).strip("/"))
            assert joint_states_batch[-1].position[i] == expected_response.joint_states.position[j]
            assert joint_states_batch[-1].velocity[i] == expected_response.joint_states.velocity[j]
            assert joint_states_batch[-1].effort[i] == expected_response.joint_states.load[j]

    response = JointControlStreamResponse()
    response.status = JointControlStreamResponse.Status.STATUS_OK
    call.returns(response)
