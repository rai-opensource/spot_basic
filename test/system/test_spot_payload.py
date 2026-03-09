# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pathlib

import pytest
from spot_basic import SpotBasic
from spot_basic.testing_prompt_session import TestingPromptSession


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_spot_audio_playback(
    test_data_path: pathlib.Path,
    hardware_spot_interface: SpotBasic,
    hardware_spot_parameters: dict,
    prompt_session: TestingPromptSession,
) -> None:
    if not hardware_spot_parameters["initialize_spot_cam"]:
        pytest.skip("Cannot test audio playback without a Spot CAM")

    response = hardware_spot_interface.load_sound.synchronously(
        "test-tone", str(test_data_path / "tone.wav"), timeout_sec=5.0
    )
    assert response.success, response.message

    response = hardware_spot_interface.list_sounds.synchronously(timeout_sec=5.0)
    assert response.success, response.message
    assert "test-tone" in response.names

    response = hardware_spot_interface.set_volume.synchronously(50.0, timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.play_sound.synchronously("test-tone", timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("did you hear a tone?")

    response = hardware_spot_interface.set_volume.synchronously(100.0, timeout_sec=5.0)
    assert response.success, response.message

    response = hardware_spot_interface.play_sound.synchronously("test-tone", timeout_sec=10.0)
    assert response.success, response.message

    assert prompt_session.boolean_ask("was it louder now?")

    response = hardware_spot_interface.delete_sound.synchronously("test-tone", timeout_sec=5.0)
    assert response.success, response.message
