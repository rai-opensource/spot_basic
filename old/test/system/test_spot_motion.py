# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from typing import List

import pytest
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_api_msgs.msg import SE2TrajectoryCommandFeedbackStatus
from spatialmath import SE3, SO3
from spot_utilities.spot_basic import FrameHint, SpotBasic
from spot_utilities.spot_basic.command import LocomotionHint


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_basic_motion_inplace(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    assert hardware_spot_interface.stand(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("did the robot stand up?")
    assert hardware_spot_interface.ready_arm(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("was the robot arm deployed?")
    assert hardware_spot_interface.stow_arm(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("was the robot arm stowed?")
    assert hardware_spot_interface.sit(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("did the robot sit down?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_walk_and_orient(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    assert hardware_spot_interface.stand(timeout_sec=5.0) is not None

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    assert hardware_spot_interface.walk_and_orient(vision_t_body) is not None
    assert prompt_session.boolean_ask("did the robot stand up in its place?")

    vision_t_goal = vision_t_body * SE3.Rt(R=SO3.RPY([0.0, 0.0, 90.0], unit="deg"), t=[0.0, 0.0, 0.5])
    assert hardware_spot_interface.walk_and_orient(vision_t_goal) is not None
    assert prompt_session.boolean_ask("did the robot rotate 90 degrees CCW?")
    assert prompt_session.boolean_ask("did the robot raise its body?")

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    vision_t_goal = vision_t_body * SE3.Rt(R=SO3.RPY([0.0, 0.0, -90.0], unit="deg"), t=[0.0, 0.0, -0.25])
    assert hardware_spot_interface.walk_and_orient(vision_t_goal) is not None
    assert prompt_session.boolean_ask("did the robot rotate 90 degrees CW?")
    assert prompt_session.boolean_ask("did the robot lower its body?")

    vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    vision_t_goal = vision_t_body * SE3.Rt(R=SO3.RPY([0.0, 30.0, 0.0], unit="deg"), t=[0.0, 0.0, 0.0])
    assert hardware_spot_interface.walk_and_orient(vision_t_goal) is not None
    assert prompt_session.boolean_ask("did the robot tilt 30 degrees down?")


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_crawl_in_a_circle(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    radius = 0.4

    body_t_goal_sequence: List[SE3] = []
    for i in range(9):
        offset_angle = i * 45.0
        h_offset = SO3.RPY([0.0, 0.0, 90.0 + offset_angle], unit="deg")
        x_offset = radius * math.cos(math.radians(offset_angle))
        y_offset = radius * math.sin(math.radians(offset_angle))
        body_t_goal_sequence.append(SE3.Rt(R=h_offset, t=[x_offset, y_offset, 0.0]))
    body_t_goal_sequence.append(SE3())  # go back to center

    assert hardware_spot_interface.stand(timeout_sec=5.0) is not None
    initial_vision_t_body = hardware_spot_interface.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY)
    for body_t_goal in body_t_goal_sequence:
        vision_t_goal = initial_vision_t_body * body_t_goal
        command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=vision_t_goal.x,
            goal_y=vision_t_goal.y,
            goal_heading=vision_t_goal.yaw_SE2().theta(),
            frame_name=FrameHint.VISION.value,
            locomotion_hint=LocomotionHint.CRAWL,
        )
        response = hardware_spot_interface.execute.synchronously(command, timeout_sec=10.0)
        mobilty_command_feedback = response.result.command.synchronized_feedback.mobility_command_feedback
        se2_trajectory_feedback = mobilty_command_feedback.feedback.se2_trajectory_feedback
        assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL

    assert prompt_session.boolean_ask("did the robot crawl in a circle?")
