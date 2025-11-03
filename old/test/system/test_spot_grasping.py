# Copyright (c) 2024-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.


import numpy as np
import pytest
from rclpy.time import Time
from spatialmath import SE3, SO3
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.state import FrameHint
from spot_utilities.spot_basic.viz import prompt_for_click

from bdai.test_utilities.testing_prompt_session import TestingPromptSession


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_gripper_exercise(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.claw_gripper_open.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.claw_gripper_close.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot open and close its arm gripper?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_grasp_object_within_fov(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    camera = prompt_session.choose(
        "which camera shall we use?", {camera.location.value: camera for camera in hardware_spot_interface.cameras}
    )

    image_pixel_xy, image, camera_info = prompt_for_click("Click object to approach!", camera.rgb)
    image_timestamp = Time.from_msg(image.header.stamp)

    response = hardware_spot_interface.walk_to_object_in_image.synchronously(
        image_pixel_xy=image_pixel_xy,
        image_timestamp=image_timestamp,
        camera_info=camera_info,
        offset_distance=1.0,
        timeout_sec=20.0,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot walk towards the object?")

    image_pixel_xy, image, camera_info = prompt_for_click("Click object to grasp!", camera.rgb)
    image_timestamp = Time.from_msg(image.header.stamp)

    response = hardware_spot_interface.pick_object_in_image.synchronously(
        image_pixel_xy=image_pixel_xy,
        image_timestamp=image_timestamp,
        camera_info=camera_info,
        timeout_sec=30.0,
    )
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot grasp the object?")

    response = hardware_spot_interface.claw_gripper_open.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.claw_gripper_close.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot drop the object?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_fetch_play(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    vision_t_body_base = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, timeout=5.0)

    camera = prompt_session.choose(
        "which camera shall we use?", {camera.location.value: camera for camera in hardware_spot_interface.cameras}
    )
    image_pixel_xy, image, camera_info = prompt_for_click("Click object to grasp!", camera.rgb)
    image_timestamp = Time.from_msg(image.header.stamp)

    response = hardware_spot_interface.pick_object_in_image.synchronously(
        image_pixel_xy=image_pixel_xy,
        image_timestamp=image_timestamp,
        camera_info=camera_info,
        timeout_sec=20.0,
    )
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.walk.synchronously(vision_t_body_base.yaw_SE2(), timeout_sec=20.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot fetch the object?")

    response = hardware_spot_interface.move_end_effector.synchronously(
        SE3.Rt(SO3.Ry(np.pi / 2.0), [1.0, 0.0, -0.2]),
        duration_sec=3.0,
        base_frame=FrameHint.BODY,
        timeout_sec=20.0,
        body_hip_assist=True,
    )
    assert response.success, response.message

    vision_t_object = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.HAND, timeout=5.0)

    response = hardware_spot_interface.claw_gripper_open.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.walk.synchronously(vision_t_body_base.yaw_SE2(), timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot drop the object?")

    ray_end_wrt_vision = vision_t_object.t
    ray_end_wrt_vision[2] -= 0.5
    ray_start_wrt_vision = np.copy(ray_end_wrt_vision)
    ray_start_wrt_vision[2] += 1.0

    response = hardware_spot_interface.pick_object_ray_in_world.synchronously(
        ray_start_wrt_vision, ray_end_wrt_vision, frame=FrameHint.VISION, timeout_sec=20.0, nothrow=True
    )
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.walk.synchronously(vision_t_body_base.yaw_SE2(), timeout_sec=20.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot grab the object again?")

    response = hardware_spot_interface.claw_gripper_open.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=10.0)
    assert response.success, response.message
