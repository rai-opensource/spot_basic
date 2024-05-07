# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from spatialmath import SE3, SO3
from spot_utilities.spot_basic import FrameHint, SpotBasic


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
    assert prompt_session.boolean_ask("did the robot stay in place?")

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
