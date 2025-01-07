# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pathlib

import pytest
from bosdyn.api.spot.choreography_sequence_pb2 import (
    Animation,
    ChoreographySequence,
    ChoreographyStateLog,
    ChoreographyStatusResponse,
    DownloadRobotStateLogRequest,
    DownloadRobotStateLogResponse,
    ExecuteChoreographyResponse,
    ListAllMovesResponse,
    ListAllSequencesResponse,
    StartRecordingStateResponse,
    StopRecordingStateResponse,
    UploadAnimatedMoveResponse,
    UploadChoreographyResponse,
)
from bosdyn.choreography.client.animation_file_to_proto import convert_animation_file_to_proto
from spot_msgs.srv import ChoreographyStartRecordingState, GetChoreographyStatus
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import wait_for_future
from synchros2.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_choreography_execution(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    assert spot_interface.execute_dance.wait_for_server(timeout_sec=2.0)
    action = spot_interface.execute_dance.asynchronously(choreo_sequence=ChoreographySequence(name="hula"))

    call = simple_spot.api.UploadChoreography.serve(timeout=2.0)
    assert call is not None
    assert call.request.choreography_sequence.name == "hula"
    call.returns(UploadChoreographyResponse())

    call = simple_spot.api.ExecuteChoreography.serve(timeout=2.0)
    assert call is not None
    assert call.request.choreography_sequence_name == "hula"
    response = ExecuteChoreographyResponse(execution_id=123)
    response.status = ExecuteChoreographyResponse.Status.STATUS_OK
    call.returns(response)

    call = simple_spot.api.ChoreographyStatus.serve(timeout=2.0)
    assert call is not None
    response = ChoreographyStatusResponse(execution_id=123)
    response.status = ChoreographyStatusResponse.Status.STATUS_COMPLETED_SEQUENCE
    call.returns(response)

    assert wait_for_future(action.finalization, timeout_sec=5.0)
    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_choreography_introspection(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    assert spot_interface.list_all_moves.wait_for_service(timeout_sec=2.0)
    future = spot_interface.list_all_moves.asynchronously()
    call = simple_spot.api.ListAllMoves.serve(timeout=2.0)
    assert call is not None
    response = ListAllMovesResponse()
    move_info = response.moves.add()
    move_info.name = "jump"
    call.returns(response)

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message
    assert result.moves == ["jump"]

    assert spot_interface.list_all_dances.wait_for_service(timeout_sec=2.0)
    future = spot_interface.list_all_dances.asynchronously()
    call = simple_spot.api.ListAllSequences.serve(timeout=2.0)
    assert call is not None
    response = ListAllSequencesResponse()
    sequence_info = response.sequence_info.add()
    sequence_info.name = "hula"
    call.returns(response)

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message
    assert result.dances == ["hula"]


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_choreography_uploading(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    animation = Animation(name="backflip")
    assert spot_interface.upload_animation.wait_for_service(timeout_sec=2.0)
    future = spot_interface.upload_animation.asynchronously(animation)

    call = simple_spot.api.UploadAnimatedMove.serve(timeout=2.0)
    assert call is not None
    assert call.request.animated_move.name == "backflip"
    response = UploadAnimatedMoveResponse()
    response.status = UploadAnimatedMoveResponse.Status.STATUS_OK
    call.returns(response)

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_choreography_recording(tmp_path: pathlib.Path, simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    assert spot_interface.start_recording_state.wait_for_service(timeout_sec=2.0)
    future = spot_interface.start_recording_state.asynchronously(duration_seconds=10.0)

    call = simple_spot.api.StartRecordingState.serve(timeout=2.0)
    assert call is not None
    assert call.request.continue_recording_duration.seconds == 10
    response = StartRecordingStateResponse()
    response.status = StartRecordingStateResponse.Status.STATUS_OK
    response.recording_session_id = 321
    call.returns(response)

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message
    assert result.status == ChoreographyStartRecordingState.Request.STATUS_OK
    assert result.recording_session_id == 321

    assert spot_interface.recorded_state_to_animation.wait_for_service(timeout_sec=2.0)
    future = spot_interface.recorded_state_to_animation.asynchronously(has_arm=False)

    call = simple_spot.api.DownloadRobotStateLog.serve(timeout=2.0)
    assert call is not None
    assert call.request.log_type == DownloadRobotStateLogRequest.LOG_TYPE_MANUAL
    response = DownloadRobotStateLogResponse()
    response.status = DownloadRobotStateLogResponse.Status.STATUS_OK
    state_log = ChoreographyStateLog()
    response.chunk.data = state_log.SerializeToString()
    response.chunk.total_size = len(response.chunk.data)
    call.returns([response])

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message
    animation_file_path = tmp_path / "animation.cha"
    animation_file_path.write_text(result.animation_file_contents)
    assert convert_animation_file_to_proto(animation_file_path) is not None

    assert spot_interface.stop_recording_state.wait_for_service(timeout_sec=2.0)
    future = spot_interface.stop_recording_state.asynchronously()

    call = simple_spot.api.StopRecordingState.serve(timeout=2.0)
    assert call is not None
    call.returns(StopRecordingStateResponse())

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_choreography_status(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    assert spot_interface.get_choreo_status.wait_for_service(timeout_sec=2.0)
    future = spot_interface.get_choreo_status.asynchronously()

    call = simple_spot.api.ChoreographyStatus.serve(timeout=2.0)
    assert call is not None
    response = ChoreographyStatusResponse(execution_id=4231)
    response.status = ChoreographyStatusResponse.Status.STATUS_POWERED_OFF
    call.returns(response)

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message
    assert result.execution_id == 4231
    assert result.status == GetChoreographyStatus.Request.STATUS_POWERED_OFF
