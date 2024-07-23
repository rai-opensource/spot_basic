# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


import pytest
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from spot_utilities.spot_basic import SpotBasic


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
