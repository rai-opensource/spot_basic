# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math

import pytest
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.gripper_camera_param_pb2 import (
    GripperCameraGetParamResponse,
    GripperCameraParamRequest,
    GripperCameraParamResponse,
)
from bosdyn.api.gripper_command_pb2 import ClawGripperCommand
from bosdyn.api.robot_command_pb2 import RobotCommandFeedbackResponse, RobotCommandResponse
from bosdyn_api_msgs.msg import ClawGripperCommandFeedbackStatus
from spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import unwrap_future, wait_for_future
from synchros2.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_no_gripper_opening(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    with pytest.raises(RuntimeError):
        spot_interface.claw_gripper_open.synchronously()


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_opening(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.claw_gripper_open.asynchronously()

    command_call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert command_call is not None
    assert command_call.request.command.WhichOneof("command") == "synchronized_command"
    gripper_command = command_call.request.command.synchronized_command.gripper_command
    assert gripper_command.WhichOneof("command") == "claw_gripper_command"
    assert len(gripper_command.claw_gripper_command.trajectory.points) == 1
    gripper_goal = gripper_command.claw_gripper_command.trajectory.points[0]
    assert math.isclose(gripper_goal.point, -math.pi / 2, abs_tol=1e-5)
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    command_call.returns(command_response)

    feedback_call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    gripper_command_feedback = synchro_feedback.gripper_command_feedback
    gripper_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    gripper_command_feedback.claw_gripper_feedback.status = ClawGripperCommand.Feedback.Status.STATUS_AT_GOAL
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    claw_gripper_feedback = synchro_feedback.gripper_command_feedback.command.claw_gripper_feedback
    assert claw_gripper_feedback.status.value == ClawGripperCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_no_gripper_closing(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    with pytest.raises(RuntimeError):
        spot_interface.claw_gripper_close.synchronously()


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_closing(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.claw_gripper_close.asynchronously()

    command_call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert command_call is not None
    assert command_call.request.command.WhichOneof("command") == "synchronized_command"
    gripper_command = command_call.request.command.synchronized_command.gripper_command
    assert gripper_command.WhichOneof("command") == "claw_gripper_command"
    assert len(gripper_command.claw_gripper_command.trajectory.points) == 1
    gripper_goal = gripper_command.claw_gripper_command.trajectory.points[0]
    assert gripper_goal.point == 0.0
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1324
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    command_call.returns(command_response)

    feedback_call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    gripper_command_feedback = synchro_feedback.gripper_command_feedback
    gripper_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    gripper_command_feedback.claw_gripper_feedback.status = ClawGripperCommand.Feedback.Status.STATUS_AT_GOAL
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    claw_gripper_feedback = synchro_feedback.gripper_command_feedback.command.claw_gripper_feedback
    assert claw_gripper_feedback.status.value == ClawGripperCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_camera_changes(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    request = GripperCameraParamRequest()
    request.params.brightness.value = 0.8
    request.params.contrast.value = 0.2
    request.params.exposure_auto.value = True
    future = spot_interface.set_gripper_camera_parameters.asynchronously(request)

    call = simple_spot.api.SetParams.serve(timeout=2.0)
    assert call is not None
    assert call.request.params.brightness.value == request.params.brightness.value
    assert call.request.params.contrast.value == request.params.contrast.value
    assert call.request.params.exposure_auto.value == request.params.exposure_auto.value
    call.returns(GripperCameraParamResponse())

    future = spot_interface.get_gripper_camera_parameters.asynchronously()

    call = simple_spot.api.GetParams.serve(timeout=2.0)
    assert call is not None
    params_response = GripperCameraGetParamResponse()
    params_response.params.brightness.value = 0.2
    params_response.params.contrast.value = 0.8
    params_response.params.exposure_auto.value = False
    call.returns(params_response)

    response = unwrap_future(future, timeout_sec=5.0)
    assert response.success, response.message
    assert response.response.params.brightness.data == params_response.params.brightness.value
    assert response.response.params.contrast.data == params_response.params.contrast.value
    assert response.response.params.exposure_auto.data == params_response.params.exposure_auto.value


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_set_angle(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    # request a gripper angle of -30 degrees
    requested_angle = -math.radians(30.0)
    action = spot_interface.claw_gripper_set_angle.asynchronously(requested_angle)

    command_call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert command_call is not None
    assert command_call.request.command.WhichOneof("command") == "synchronized_command"
    gripper_command = command_call.request.command.synchronized_command.gripper_command
    assert gripper_command.WhichOneof("command") == "claw_gripper_command"
    assert len(gripper_command.claw_gripper_command.trajectory.points) == 1
    gripper_goal = gripper_command.claw_gripper_command.trajectory.points[0]
    assert math.isclose(gripper_goal.point, requested_angle)
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1423
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    command_call.returns(command_response)

    feedback_call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    gripper_command_feedback = synchro_feedback.gripper_command_feedback
    gripper_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    gripper_command_feedback.claw_gripper_feedback.status = ClawGripperCommand.Feedback.Status.STATUS_AT_GOAL
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    claw_gripper_feedback = synchro_feedback.gripper_command_feedback.command.claw_gripper_feedback
    assert claw_gripper_feedback.status.value == ClawGripperCommandFeedbackStatus.STATUS_AT_GOAL
