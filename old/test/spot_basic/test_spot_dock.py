# Copyright (c) 2024-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pytest
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus, StandCommand
from bosdyn.api.docking.docking_pb2 import PREP_POSE_UNDOCK, DockingCommandFeedbackResponse, DockingCommandResponse
from bosdyn.api.robot_command_pb2 import RobotCommandFeedbackResponse, RobotCommandResponse
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import wait_for_future
from synchros2.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_docking(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    with spot_interface.lease():
        assert spot_interface.dock.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        future = spot_interface.dock.asynchronously(dock_id=23)

        call = simple_spot.api.RobotCommand.serve(timeout=2.0)
        assert call is not None
        synchro_command = call.request.command.synchronized_command
        assert synchro_command.mobility_command.HasField("stand_request")
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
        stand_feedback = mobility_command_feedback.stand_feedback
        stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
        call.returns(feedback_response)

        call = simple_spot.api.DockingCommand.serve(timeout=2.0)
        assert call is not None
        assert call.request.docking_station_id == 23
        response = DockingCommandResponse()
        response.status = DockingCommandResponse.Status.STATUS_OK
        response.docking_command_id = 1235
        call.returns(response)

        call = simple_spot.api.DockingCommandFeedback.serve(timeout=2.0)
        assert call is not None
        assert call.request.docking_command_id == response.docking_command_id
        response = DockingCommandFeedbackResponse()
        response.status = DockingCommandFeedbackResponse.Status.STATUS_DOCKED
        call.returns(response)

        assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
        response = future.result()
        assert response.success, response.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_undocking(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    with spot_interface.lease():
        assert spot_interface.undock.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        future = spot_interface.undock.asynchronously()

        call = simple_spot.api.DockingCommand.serve(timeout=2.0)
        assert call is not None
        assert call.request.docking_station_id == 0
        assert call.request.prep_pose_behavior == PREP_POSE_UNDOCK
        response = DockingCommandResponse()
        response.status = DockingCommandResponse.Status.STATUS_OK
        response.docking_command_id = 1234
        call.returns(response)

        call = simple_spot.api.DockingCommandFeedback.serve(timeout=2.0)
        assert call is not None
        assert call.request.docking_command_id == response.docking_command_id
        response = DockingCommandFeedbackResponse()
        response.status = DockingCommandFeedbackResponse.Status.STATUS_AT_PREP_POSE
        call.returns(response)

        assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
        response = future.result()
        assert response.success, response.message
