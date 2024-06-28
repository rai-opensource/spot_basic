# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import open3d
import pytest
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from bdai_ros2_wrappers.futures import unwrap_future
from cv_bridge import CvBridge
from spatialmath import SO3
from spot_utilities.spot_basic import SpotBasic


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
