# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pathlib

import numpy as np
import open3d
import pytest
from cv_bridge import CvBridge
from numpy.testing import assert_almost_equal
from spatialmath import SE2, SO3
from synchros2.futures import unwrap_future

from spot_basic import FrameHint, SpotBasic
from spot_basic.manual_conversions import to_se3
from spot_basic.testing_prompt_session import TestingPromptSession


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_spot_scan_in_place(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    routine = [SO3.Rz(15.0, unit="deg"), SO3.Rz(-15.0, unit="deg"), SO3.Ry(-15.0, unit="deg"), SO3()]

    pointcloud = hardware_spot_interface.scan_in_place(routine=routine, timeout_sec=30.0).voxel_down_sample(
        voxel_size=0.02
    )

    open3d.visualization.draw_geometries([pointcloud], window_name="Scanned pointcloud - close when done inspecting!")
    assert prompt_session.boolean_ask("was the scan complete and correct?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_spot_cameras(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    cv_bridge = CvBridge()
    for camera in (
        hardware_spot_interface.cameras.back.rgb,
        hardware_spot_interface.cameras.left.rgb,
        hardware_spot_interface.cameras.right.rgb,
        hardware_spot_interface.cameras.front_left.rgb,
        hardware_spot_interface.cameras.front_right.rgb,
    ):
        image_message = unwrap_future(camera.image.update, timeout_sec=5.0)
        open3d.visualization.draw_geometries(
            [open3d.geometry.Image(cv_bridge.imgmsg_to_cv2(image_message, "rgb8"))],
            window_name=f"{camera.name} image - close when done inspecting!",
        )
        assert prompt_session.boolean_ask(f"was the image shown consistent with the {camera.name} camera?")


# NOTE: requires a graphnav map at test/data/graphnav_session (not bundled with the package).
@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_graphnav_localization(
    test_data_path: pathlib.Path, hardware_spot_interface: SpotBasic, hardware_spot_parameters: dict
) -> None:
    if not hardware_spot_parameters["publish_graph_nav_pose"]:
        pytest.skip("Cannot test graphnav localization without graphnav enabled")

    graphnav_session = str(test_data_path / "graphnav_session")

    response = hardware_spot_interface.upload_graph.synchronously(graphnav_session, timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.set_localization.synchronously("fiducial", timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.get_localization.synchronously(timeout_sec=5.0)
    assert response.success, response.message
    start_pose = to_se3(response.pose)

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY).yaw_SE2()
    response = hardware_spot_interface.walk.synchronously(vision_t_body * SE2(1.0, 0.0), timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.get_localization.synchronously(timeout_sec=5.0)
    assert response.success, response.message
    end_pose = to_se3(response.pose)

    displacement = start_pose.inv() * end_pose
    assert_almost_equal(displacement.t, np.array([1.0, 0.0, 0.0]), decimal=1)
    assert_almost_equal(displacement.R, np.eye(3), decimal=2)

    response = hardware_spot_interface.clear_graph.synchronously(timeout_sec=5.0)
    assert response.success, response.message


# NOTE: requires a graphnav map at test/data/graphnav_session (not bundled with the package).
@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_graphnav_introspection(
    test_data_path: pathlib.Path, hardware_spot_interface: SpotBasic, hardware_spot_parameters: dict
) -> None:
    if not hardware_spot_parameters["publish_graph_nav_pose"]:
        pytest.skip("Cannot test graphnav localization without graphnav enabled")

    graphnav_session = str(test_data_path / "graphnav_session")

    response = hardware_spot_interface.list_graph.synchronously(graphnav_session, timeout_sec=5.0)
    assert len(response.waypoint_ids) == 16
