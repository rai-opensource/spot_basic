# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pathlib

import pytest
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope
from bdai_ros2_wrappers.utilities import ensure
from bosdyn.api.graph_nav.graph_nav_pb2 import (
    ClearGraphResponse,
    DownloadGraphResponse,
    GetLocalizationStateResponse,
    SetLocalizationRequest,
    SetLocalizationResponse,
    UploadGraphResponse,
)
from bosdyn.api.graph_nav.map_pb2 import Graph
from bosdyn.api.graph_nav.nav_pb2 import Localization
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_clear_graph(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    with spot_interface.lease():
        future = spot_interface.clear_graph.asynchronously()
        call = simple_spot.api.ClearGraph.serve(timeout=2.0)
        assert call is not None
        response = ClearGraphResponse()
        response.status = ClearGraphResponse.Status.STATUS_OK
        call.returns(response)
        assert wait_for_future(future, timeout_sec=2.0)
        result = future.result()
        assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_upload_graph(tmp_path: pathlib.Path, simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    almost_empty_graph = Graph()
    waypoint = almost_empty_graph.waypoints.add()
    waypoint.id = "wp-0"
    waypoint.annotations.name = "origin"

    graph_filepath = tmp_path / "graph"
    graph_filepath.write_bytes(almost_empty_graph.SerializeToString())

    with spot_interface.lease():
        future = spot_interface.upload_graph.asynchronously(str(tmp_path))
        call = simple_spot.api.UploadGraph.serve(timeout=2.0)
        assert call is not None
        assert len(call.request.graph.waypoints) == 1
        assert call.request.graph.waypoints[0].id == waypoint.id
        call.returns(UploadGraphResponse(status=UploadGraphResponse.Status.STATUS_OK))

        response = GetLocalizationStateResponse()
        response.localization.waypoint_id = waypoint.id
        simple_spot.api.GetLocalizationState.future.returns(response).forever()
        while simple_spot.api.GetLocalizationState.pending:
            call = simple_spot.api.GetLocalizationState.serve(timeout=2.0)
            assert call is not None
            call.returns(response)

        assert wait_for_future(future, timeout_sec=2.0)
        result = future.result()
        assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_get_localization(tmp_path: pathlib.Path, simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    with spot_interface.lease():
        localization = Localization()
        localization.waypoint_id = "wp-0"
        localization.waypoint_tform_body.rotation.w = 1.0
        localization.seed_tform_body.rotation.w = 1.0
        localization.seed_tform_body.position.x = 1.0
        localization.seed_tform_body.rotation.w = 1.0

        response = GetLocalizationStateResponse(localization=localization)
        simple_spot.api.GetLocalizationState.future.returns(response).forever()
        while simple_spot.api.GetLocalizationState.pending:
            call = simple_spot.api.GetLocalizationState.serve(timeout=2.0)
            assert call is not None
            call.returns(response)

        response = spot_interface.get_localization.synchronously(timeout_sec=2.0)
        assert response.success, response.message
        assert response.pose.pose.position.x == 1.0
        assert response.pose.pose.orientation.w == 1.0

        assert wait_for_future(spot_interface.localization.update, timeout_sec=2.0)
        pose = ensure(spot_interface.localization.latest)
        assert pose.pose.position.x == 1.0


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_set_localization(tmp_path: pathlib.Path, simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    with spot_interface.lease():
        simple_spot.api.DownloadGraph.future.returns(DownloadGraphResponse(graph=Graph())).forever()
        future = spot_interface.set_localization.asynchronously(method="fiducial")
        call = simple_spot.api.SetLocalization.serve(timeout=2.0)
        assert call is not None
        assert call.request.initial_guess.waypoint_id == ""
        assert call.request.fiducial_init == SetLocalizationRequest.FiducialInit.FIDUCIAL_INIT_NEAREST
        call.returns(SetLocalizationResponse(status=SetLocalizationResponse.Status.STATUS_OK))
        assert wait_for_future(future, timeout_sec=2.0)
        result = future.result()
        assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_list_graph(tmp_path: pathlib.Path, simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    graph = Graph()
    waypoint = graph.waypoints.add()
    waypoint.id = "wp-0"
    waypoint.annotations.name = "origin"

    localization = Localization()
    localization.waypoint_id = "wp-0"
    localization.waypoint_tform_body.rotation.w = 1.0
    localization.seed_tform_body.rotation.w = 1.0
    localization.seed_tform_body.position.x = 1.0
    localization.seed_tform_body.rotation.w = 1.0

    graph_filepath = tmp_path / "graph"
    graph_filepath.write_bytes(graph.SerializeToString())

    with spot_interface.lease():
        simple_spot.api.ClearGraph.future.returns(
            ClearGraphResponse(status=ClearGraphResponse.Status.STATUS_OK)
        ).forever()
        simple_spot.api.UploadGraph.future.returns(
            UploadGraphResponse(status=UploadGraphResponse.Status.STATUS_OK)
        ).forever()
        simple_spot.api.DownloadGraph.future.returns(DownloadGraphResponse(graph=graph)).forever()
        simple_spot.api.GetLocalizationState.future.returns(
            GetLocalizationStateResponse(localization=localization)
        ).forever()

        response = spot_interface.list_graph.synchronously(str(graph_filepath))
        assert len(response.waypoint_ids) == 1
        assert response.waypoint_ids[0] == "wp-0"
