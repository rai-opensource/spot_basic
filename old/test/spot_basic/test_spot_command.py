# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus, SitCommand
from bosdyn.api.robot_command_pb2 import ClearBehaviorFaultResponse, RobotCommandFeedbackResponse, RobotCommandResponse
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder
from geometry_msgs.msg import Twist, Vector3
from spatialmath import Twist2
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.command import LocomotionHint
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_mobility_params(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    assert spot_interface.set_stair_mode.synchronously(True, timeout_sec=5.0)
    assert spot_interface.set_locomotion_mode.synchronously(LocomotionHint.TROT, timeout_sec=5.0)
    assert spot_interface.set_max_velocity.synchronously(
        Twist(linear=Vector3(x=1.0), angular=Vector3(z=1.0)), timeout_sec=5.0
    )

    spot_interface.walk_velocity(Twist2.Tx(1.0))
    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "se2_velocity_request"
    mobility_params = spot_command_pb2.MobilityParams()
    synchro_command.mobility_command.params.Unpack(mobility_params)
    assert mobility_params.stair_hint is True
    assert mobility_params.locomotion_hint == spot_command_pb2.HINT_TROT
    assert mobility_params.vel_limit.max_vel.linear.x == 1.0
    assert mobility_params.vel_limit.max_vel.linear.y == 0.0
    assert mobility_params.vel_limit.max_vel.angular == 1.0
    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(response)


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_clear_behavior_fault(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    future = spot_interface.clear_behavior_fault.asynchronously(127)

    call = simple_spot.api.ClearBehaviorFault.serve(timeout=2.0)
    assert call is not None
    assert call.request.behavior_fault_id == 127
    response = ClearBehaviorFaultResponse()
    response.status = ClearBehaviorFaultResponse.Status.STATUS_CLEARED
    call.returns(response)

    assert wait_for_future(future, timeout_sec=5.0)
    result = future.result()
    assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_stop(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    future = spot_interface.stop.asynchronously()

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "full_body_command"
    full_body_command = call.request.command.full_body_command
    assert full_body_command.WhichOneof("command") == "stop_request"
    assert full_body_command.HasField("stop_request")
    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(response)

    assert wait_for_future(future, timeout_sec=5.0)
    result = future.result()
    assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_execute(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    action = spot_interface.execute.asynchronously(RobotCommandBuilder.synchro_sit_command())

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "sit_request"
    assert synchro_command.mobility_command.HasField("sit_request")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    sit_feedback = mobility_command_feedback.sit_feedback
    sit_feedback.status = SitCommand.Feedback.Status.STATUS_IS_SITTING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=5.0)
    assert action.succeeded


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_start_execution(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    future = spot_interface.start_executing.asynchronously(RobotCommandBuilder.synchro_sit_command())

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "sit_request"
    assert synchro_command.mobility_command.HasField("sit_request")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    assert wait_for_future(future, timeout_sec=5.0)
    result = future.result()
    assert result.success, result.message
    assert result.robot_command_id == command_response.robot_command_id
