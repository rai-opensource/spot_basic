# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import pytest
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from bdai_ros2_wrappers.utilities import ensure
from hil_utilities import prompt_for_click, visual_aid
from rclpy.time import Time
from spatialmath import SE3, SO3, SpatialForce
from spot_utilities.spot_basic import FrameHint, SpotBasic
from spot_utilities.spot_basic.command.arm import ArmJointMap, AxisModeMap


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_basic_hand_motion(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("is the robot standing with its arm out?")

    body_t_goal = hardware_spot_interface.lookup_a_tform_b(
        hardware_spot_interface.body_frame_name, hardware_spot_interface.hand_frame_name, timeout=2.0
    ) * SE3.Trans([0.1, 0.1, -0.1])
    response = hardware_spot_interface.move_end_effector.synchronously(
        body_t_goal,
        base_frame=FrameHint.BODY,
        duration_sec=1.5,
        timeout_sec=5.0,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot hand move?")
    assert not prompt_session.boolean_ask("did the robot body follow the motion?")

    body_t_goal = hardware_spot_interface.lookup_a_tform_b(
        hardware_spot_interface.body_frame_name, hardware_spot_interface.hand_frame_name, timeout=2.0
    ) * SE3.Rx(-np.pi / 3)
    response = hardware_spot_interface.move_end_effector.synchronously(
        body_t_goal,
        base_frame=FrameHint.BODY,
        duration_sec=1.5,
        timeout_sec=5.0,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot hand rotate?")
    assert not prompt_session.boolean_ask("did the robot body follow the motion?")

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_hand_below_body(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.move_end_effector.synchronously(
        SE3.Rt(SO3.Ry(np.pi / 2), [1.0, 0.0, -0.2]),
        duration_sec=3.0,
        base_frame=FrameHint.BODY,
        timeout_sec=20.0,
        body_hip_assist=True,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot hand move below body?")

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.sit.synchronously(timeout_sec=10.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_body_follows_hand(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("is the robot standing with its arm out?")

    odom_t_goal = hardware_spot_interface.lookup_a_tform_b(
        hardware_spot_interface.odom_frame_name, hardware_spot_interface.hand_frame_name, timeout=2.0
    ) * SE3.Trans([-0.4, 0.4, -0.2])
    response = hardware_spot_interface.move_end_effector.synchronously(
        odom_t_goal, duration_sec=5.0, base_frame=FrameHint.ODOM, body_follow=True, timeout_sec=20.0
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did both robot hand and body move?")

    odom_t_goal = hardware_spot_interface.lookup_a_tform_b(
        hardware_spot_interface.odom_frame_name, hardware_spot_interface.hand_frame_name, timeout=2.0
    ) * SE3.Rt(SO3.Rz(np.pi / 2), [-0.2, 0.2, 0.0])
    response = hardware_spot_interface.move_end_effector.synchronously(
        odom_t_goal,
        duration_sec=5.0,
        base_frame=FrameHint.ODOM,
        body_hip_assist=True,
        body_yaw_assist=True,
        timeout_sec=20.0,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did both robot hand and body rotate?")

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_hand_snake(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("is the robot standing with its arm out?")

    num_periods = 6
    period_sec = 2.0
    duration_sec = num_periods * period_sec

    motion_extent_x = 0.4
    motion_extent_z = 0.1
    motion_extent_p = np.deg2rad(20.0)
    linear_velocity_x = 2 * motion_extent_x / duration_sec

    timestamps = np.linspace(0.0, duration_sec, 100)

    midpoint = len(timestamps) // 2
    trajectory_x = linear_velocity_x * timestamps
    trajectory_x[midpoint:] = 2.0 * motion_extent_x - trajectory_x[midpoint:]

    trajectory_z = motion_extent_z * np.sin(2.0 * np.pi * timestamps / period_sec)

    trajectory_p = -motion_extent_p * np.cos(2.0 * np.pi * timestamps / period_sec)

    body_t_hand = hardware_spot_interface.lookup_a_tform_b(FrameHint.BODY, FrameHint.HAND, timeout=2.0)
    pose_trajectory = [
        body_t_hand * SE3.Rt(SO3.Ry(p), [x, 0.0, z])
        for x, z, p in zip(trajectory_x, trajectory_z, trajectory_p, strict=True)
    ]

    response = hardware_spot_interface.move_end_effector_along_trajectory.synchronously(
        pose_vecs=pose_trajectory,
        timestamps_between_poses=timestamps,
        axis_mode_map=AxisModeMap.position_mode(),
        root_frame=FrameHint.BODY,
        timeout_sec=30.0,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot hand describe a wave forward and backward?")

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_arm_joint_perturbations(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    base_configuration = np.array([0.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 0.0, -np.pi / 2.0])

    response = hardware_spot_interface.move_arm.synchronously(
        ArmJointMap.from_list(list(base_configuration)), timeout_sec=10.0
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot arm move away from the body?")

    joint_positions = []
    times = np.linspace(0.0, 2.0, len(base_configuration) + 1) + 0.5  # cope with latency
    for i in range(len(base_configuration)):
        perturbed_configuration = np.array(base_configuration)
        perturbed_configuration[i] -= 0.15
        perturbed_configuration[-i] += 0.15
        joint_positions.append(ArmJointMap.from_list(list(perturbed_configuration)))
    joint_positions.append(ArmJointMap.from_list(list(base_configuration)))

    response = hardware_spot_interface.move_arm_along_trajectory.synchronously(
        joint_positions=joint_positions, times=times, max_acceleration=1.0, timeout_sec=20.0
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot slowly move its arm joints?")

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_hand_gazing(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.claw_gripper_open.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    camera = ensure(hardware_spot_interface.cameras.hand)

    image_pixel_xy, image, camera_info = prompt_for_click("Click to gaze!", camera.rgb)

    vision_t_camera = hardware_spot_interface.lookup_a_tform_b(
        FrameHint.VISION, camera_info.header.frame_id, timeout=5.0
    )

    x, y = image_pixel_xy
    fx, _, cx, _, fy, cy, *_ = camera_info.k
    point_wrt_camera = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
    point_wrt_vision = vision_t_camera * point_wrt_camera

    response = hardware_spot_interface.arm_gaze.synchronously(
        np.squeeze(point_wrt_vision), frame=FrameHint.VISION, timeout_sec=10.0
    )
    assert response.success, response.message

    with visual_aid("Centered image", camera.rgb):
        assert prompt_session.boolean_ask("is robot hand gazing at the clicked point?")

    response = hardware_spot_interface.claw_gripper_close.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_arm_strength(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    camera = prompt_session.choose(
        "which camera shall we use?", {camera.location.value: camera for camera in hardware_spot_interface.cameras}
    )
    image_pixel_xy, image, camera_info = prompt_for_click("Click object to grasp!", camera.rgb)
    image_timestamp = Time.from_msg(image.header.stamp)

    response = hardware_spot_interface.pick_object_in_image.synchronously(
        image_pixel_xy=image_pixel_xy,
        image_timestamp=image_timestamp,
        camera_info=camera_info,
        timeout_sec=30.0,
    )
    assert response.success, response.message

    response = hardware_spot_interface.carry_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot grab the object?")

    wrench_trajectory = [
        SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        SpatialForce([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    timestamps = [0.0, 2.5, 5.0]

    response = hardware_spot_interface.move_end_effector_along_trajectory.synchronously(
        wrench_vecs=wrench_trajectory,
        timestamps_between_poses=timestamps,
        axis_mode_map=AxisModeMap.force_mode(),
        root_frame=FrameHint.GRAV_ALIGNED_BODY,
        timeout_sec=20.0,
    )
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot hand move up and down?")

    response = hardware_spot_interface.claw_gripper_open.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.claw_gripper_close.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.stow_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_arm_zero_g(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not hardware_spot_interface.has_arm():
        pytest.skip(f"{hardware_spot_interface.robot_name} has no arm")

    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.ready_arm.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.arm_zero_g.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("does the robot arm feel weightless to touch?")
