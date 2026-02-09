# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math

import pytest
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_api_msgs.msg import SE2TrajectoryCommandFeedbackStatus
from spatialmath import SE2, SE3, SO3
from spot_basic import FrameHint, SpotBasic
from spot_basic.command import LocomotionHint
from spot_basic.trajectory.se2_trajectory import SE2Trajectory

from bdai.interactive_choreo.choreo_builders.animation_builder import AnimationBuilder
from bdai.interactive_choreo.choreo_builders.sequence_builder import SequenceBuilder
from bdai.test_utilities.testing_prompt_session import TestingPromptSession


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_basic_motion_inplace(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message
    assert prompt_session.boolean_ask("did the robot stand up?")
    assert hardware_spot_interface.ready_arm(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("was the robot arm deployed?")
    assert hardware_spot_interface.stow_arm(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("was the robot arm stowed?")
    response = hardware_spot_interface.sit.synchronously(timeout_sec=5.0)
    assert response.success, response.message
    assert prompt_session.boolean_ask("did the robot sit down?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_walk_and_orient(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    responses = hardware_spot_interface.walk_and_orient.synchronously(vision_t_body, timeout_sec=5.0)
    assert all(r.success for r in responses), [r.message for r in responses]
    assert prompt_session.boolean_ask("did the robot stand up in its place?")

    vision_t_goal = vision_t_body * SE3.Rt(R=SO3.RPY([0.0, 0.0, 90.0], unit="deg"), t=[0.0, 0.0, 0.5])
    responses = hardware_spot_interface.walk_and_orient.synchronously(vision_t_goal, timeout_sec=5.0)
    assert all(r.success for r in responses), [r.message for r in responses]
    assert prompt_session.boolean_ask("did the robot rotate 90 degrees CCW?")
    assert prompt_session.boolean_ask("did the robot raise its body?")

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    vision_t_goal = vision_t_body * SE3.Rt(R=SO3.RPY([0.0, 0.0, -90.0], unit="deg"), t=[0.0, 0.0, -0.25])
    responses = hardware_spot_interface.walk_and_orient.synchronously(vision_t_goal, timeout_sec=5.0)
    assert all(r.success for r in responses), [r.message for r in responses]
    assert prompt_session.boolean_ask("did the robot lower its body?")
    assert prompt_session.boolean_ask("did the robot rotate 90 degrees CW?")

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    vision_t_goal = vision_t_body * SE3.Rt(R=SO3.RPY([0.0, 30.0, 0.0], unit="deg"), t=[0.0, 0.0, 0.0])
    responses = hardware_spot_interface.walk_and_orient.synchronously(vision_t_goal, timeout_sec=5.0)
    assert all(r.success for r in responses), [r.message for r in responses]
    assert prompt_session.boolean_ask("did the robot tilt 30 degrees down?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_crawl_in_a_circle(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY).yaw_SE2()

    radius = 0.4
    vision_t_goal_sequence = []
    for i in range(9):
        h_offset = i * 45.0
        x_offset = radius * math.cos(math.radians(h_offset))
        y_offset = radius * math.sin(math.radians(h_offset))
        vision_t_goal_sequence.append(vision_t_body * SE2(x_offset, y_offset, 90.0 + h_offset, unit="deg"))
    vision_t_goal_sequence.append(vision_t_body)  # go back to center

    mobility_params = RobotCommandBuilder.mobility_params(locomotion_hint=LocomotionHint.CRAWL)
    for vision_t_goal in vision_t_goal_sequence:
        response = hardware_spot_interface.walk.synchronously(
            vision_t_goal, mobility_params=mobility_params, timeout_sec=10.0
        )
        assert response.success, response.message
        mobilty_command_feedback = response.result.command.synchronized_feedback.mobility_command_feedback
        se2_trajectory_feedback = mobilty_command_feedback.feedback.se2_trajectory_feedback
        assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_STOPPED

    assert prompt_session.boolean_ask("did the robot crawl in a circle?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_back_and_forth_parade(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.stand.synchronously(timeout_sec=5.0)
    assert response.success, response.message

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY).yaw_SE2()

    vision_t_goal_sequence = [
        vision_t_body * body_t_goal for body_t_goal in (SE2(), SE2(0.6, 0.0, 0.0), SE2(), SE2(-0.6, 0.0, 0.0), SE2())
    ]

    trajectory = SE2Trajectory(poses=vision_t_goal_sequence, duration_sec=8.0)
    mobility_params = RobotCommandBuilder.mobility_params(locomotion_hint=LocomotionHint.TROT)
    response = hardware_spot_interface.walk_trajectory.synchronously(
        trajectory, mobility_params=mobility_params, timeout_sec=20.0
    )
    assert response.success, response.message

    mobilty_command_feedback = response.result.command.synchronized_feedback.mobility_command_feedback
    se2_trajectory_feedback = mobilty_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_STOPPED

    assert prompt_session.boolean_ask("did the robot move back and forth?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_docking_cycle(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not prompt_session.boolean_ask("is the robot docked?"):
        pytest.skip("Cannot test docking cycle out of a dock")

    dock_id = prompt_session.integer_ask("what is your robot's dock ID?")

    response = hardware_spot_interface.undock.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot undock?")

    body_t_dock_facing_pose = SE2(0.0, 0.0, 180.0, unit="deg")
    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY).yaw_SE2()
    response = hardware_spot_interface.walk.synchronously(vision_t_body * body_t_dock_facing_pose, timeout_sec=10.0)
    assert response.success, response.message

    response = hardware_spot_interface.dock.synchronously(dock_id, timeout_sec=30.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot re-dock?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_minimal_choreography(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    if not prompt_session.boolean_ask("does the robot have a choreography license)?"):
        pytest.skip("Cannot test choreography without a license")
    if not hardware_spot_interface.has_arm():
        pytest.skip("Cannot test choreography without an arm")

    animation_builder = AnimationBuilder()
    animation_builder.start_from_empty(name="gripper_open_close")
    animation_builder.make_name_unique()
    animation_builder.animate_gripper(start_time=0, gripper_angle=0.0)
    animation_builder.animate_gripper(start_time=1, gripper_angle=-1.0)
    animation_builder.animate_gripper(start_time=2, gripper_angle=0.0)
    animation = animation_builder.build_animation()

    response = hardware_spot_interface.upload_animation.synchronously(animation, timeout_sec=10.0)
    assert response.success, response.message

    sequence_builder = SequenceBuilder()
    sequence_builder.start_from_empty(name="minimal_sequence")
    sequence_builder.add_animation(animation, start_time=0.0)
    sequence = sequence_builder.build_sequence()

    response = hardware_spot_interface.execute_dance.synchronously(sequence, timeout_sec=10.0)
    assert response.success, response.message
    assert prompt_session.boolean_ask("did the robot open and close its gripper?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_rollover(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    response = hardware_spot_interface.roll_to_battery_change_pose.synchronously(timeout_sec=30.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot roll down to the floor?")

    response = hardware_spot_interface.self_right.synchronously(timeout_sec=30.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot roll back up?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_spot_party(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    gravity_aligned_orientation = SO3()
    response = hardware_spot_interface.stand_with_orientation.synchronously(
        gravity_aligned_orientation, body_height=0.5, timeout_sec=10.0
    )
    assert response.success, response.message

    response = hardware_spot_interface.stand_with_orientation.synchronously(
        gravity_aligned_orientation, body_height=-0.5, timeout_sec=10.0
    )
    assert response.success, response.message

    response = hardware_spot_interface.stand_with_orientation.synchronously(
        gravity_aligned_orientation, body_height=0.5, timeout_sec=10.0
    )
    assert response.success, response.message

    response = hardware_spot_interface.stand.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot squat up and down?")

    response = hardware_spot_interface.body_gaze.synchronously(
        target=[1.0, 0.0, 0.75], frame=FrameHint.BODY, timeout_sec=10.0
    )
    assert response.success, response.message

    response = hardware_spot_interface.body_gaze.synchronously(
        target=[1.0, 0.0, -0.75], frame=FrameHint.BODY, timeout_sec=10.0
    )
    assert response.success, response.message

    response = hardware_spot_interface.stand.synchronously(timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did the robot bend up and down?")
