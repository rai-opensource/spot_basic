# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from spot_utilities.spot_basic import SpotBasic


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_basic_motion_inplace(hardware_spot_interface: SpotBasic, prompt_session: TestingPromptSession) -> None:
    assert hardware_spot_interface.initialize_robot()
    assert hardware_spot_interface.stand(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("has the robot stood up?")
    assert hardware_spot_interface.ready_arm(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("has the robot arm been deployed?")
    assert hardware_spot_interface.stow_arm(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("has the robot arm been stowed?")
    assert hardware_spot_interface.sit(timeout_sec=5.0) is not None
    assert prompt_session.boolean_ask("has the robot sat down?")
