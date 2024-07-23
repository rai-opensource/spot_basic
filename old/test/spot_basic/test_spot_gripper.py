# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math

import pytest
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.gripper_command_pb2 import ClawGripperCommand
from bosdyn.api.robot_command_pb2 import RobotCommandFeedbackResponse, RobotCommandResponse
from bosdyn_api_msgs.msg import ClawGripperCommandFeedbackStatus
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_no_gripper_opening(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    with pytest.raises(RuntimeError):
        spot_interface.claw_gripper_open.synchronously()


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_opening(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    claw_gripper_feedback = synchro_feedback.gripper_command_feedback.command.claw_gripper_feedback
    assert claw_gripper_feedback.status.value == ClawGripperCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_partial_opening(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    action = spot_interface.claw_gripper_open.asynchronously(math.radians(30.0))

    command_call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert command_call is not None
    assert command_call.request.command.WhichOneof("command") == "synchronized_command"
    gripper_command = command_call.request.command.synchronized_command.gripper_command
    assert gripper_command.WhichOneof("command") == "claw_gripper_command"
    assert len(gripper_command.claw_gripper_command.trajectory.points) == 1
    gripper_goal = gripper_command.claw_gripper_command.trajectory.points[0]
    assert math.isclose(gripper_goal.point, -math.radians(30.0))
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 4321
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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    claw_gripper_feedback = synchro_feedback.gripper_command_feedback.command.claw_gripper_feedback
    assert claw_gripper_feedback.status.value == ClawGripperCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_no_gripper_closing(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    with pytest.raises(RuntimeError):
        spot_interface.claw_gripper_close.synchronously()


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_gripper_closing(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    claw_gripper_feedback = synchro_feedback.gripper_command_feedback.command.claw_gripper_feedback
    assert claw_gripper_feedback.status.value == ClawGripperCommandFeedbackStatus.STATUS_AT_GOAL
