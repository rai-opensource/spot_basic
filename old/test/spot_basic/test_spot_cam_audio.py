# Copyright (c) 2024-2025 Boston Dynamics AI Institute LLC. All rights reserved.

import pathlib

import pytest
from bosdyn.api.spot_cam.audio_pb2 import (
    DeleteSoundResponse,
    GetVolumeResponse,
    ListSoundsResponse,
    LoadSoundResponse,
    PlaySoundResponse,
    SetVolumeResponse,
    Sound,
)
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import wait_for_future
from synchros2.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_sound_management(test_data_path: pathlib.Path, simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    with spot_interface.lease():
        # Load a sound
        test_wav_path = test_data_path / "tone.wav"
        assert spot_interface.load_sound.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        future = spot_interface.load_sound.asynchronously("tone", str(test_wav_path))
        call = simple_spot.api.LoadSound.serve(timeout=2.0)
        assert call is not None
        sound_data = b""
        for chunk in call.request:
            assert chunk.sound.name == "tone"
            sound_data += chunk.data.data
        with test_wav_path.open("rb") as f:
            test_wav_data = f.read()
        assert sound_data == test_wav_data
        call.returns(LoadSoundResponse())
        assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
        result = future.result()
        assert result.success, result.message

        # List sound
        simple_spot.api.ListSounds.future.returns(ListSoundsResponse(sounds=[Sound(name="tone")]))
        assert spot_interface.list_sounds.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        response = spot_interface.list_sounds.synchronously(timeout_sec=pytest.DEFAULT_TIMEOUT)
        assert response.success, response.message
        assert response.names == ["tone"]

        # Play it
        assert spot_interface.play_sound.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        future = spot_interface.play_sound.asynchronously("tone", 0.5)
        call = simple_spot.api.PlaySound.serve(timeout=2.0)
        assert call is not None
        assert call.request.sound.name == "tone"
        assert call.request.gain.value == 0.5
        call.returns(PlaySoundResponse())
        assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
        result = future.result()
        assert result.success, result.message

        # Delete it
        assert spot_interface.delete_sound.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        future = spot_interface.delete_sound.asynchronously("tone")
        call = simple_spot.api.DeleteSound.serve(timeout=2.0)
        assert call is not None
        assert call.request.sound.name == "tone"
        call.returns(DeleteSoundResponse())
        assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
        result = future.result()
        assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_volume_control(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)

    with spot_interface.lease():
        assert spot_interface.set_volume.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        future = spot_interface.set_volume.asynchronously(75.0)

        call = simple_spot.api.SetVolume.serve(timeout=2.0)
        assert call is not None
        assert call.request.volume == 75.0
        call.returns(SetVolumeResponse())

        assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
        result = future.result()
        assert result.success, result.message

        simple_spot.api.GetVolume.future.returns(GetVolumeResponse(volume=75.0))
        assert spot_interface.get_volume.wait_for_service(timeout_sec=pytest.DEFAULT_TIMEOUT)
        response = spot_interface.get_volume.synchronously(timeout_sec=pytest.DEFAULT_TIMEOUT)
        assert response.success, response.message
        assert response.volume == 75.0
