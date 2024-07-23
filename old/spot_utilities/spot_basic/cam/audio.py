# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

from bdai_ros2_wrappers.callables import generalized_method
from spot_msgs.srv import DeleteSound, GetVolume, ListSounds, LoadSound, PlaySound, SetVolume
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced


class SpotCAMAudioMixin(SpotBaseProtocol):
    """Mixin that manages access to the Spot CAM audio system."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotCAMAudioMixin.list_sounds.rebind(
            self, SpotServiced(ListSounds, spot_namespaced(self.robot_name, "list_sounds"), node=self.node)
        )
        SpotCAMAudioMixin.load_sound.rebind(
            self, SpotServiced(LoadSound, spot_namespaced(self.robot_name, "load_sound"), node=self.node)
        )
        SpotCAMAudioMixin.play_sound.rebind(
            self, SpotServiced(PlaySound, spot_namespaced(self.robot_name, "play_sound"), node=self.node)
        )
        SpotCAMAudioMixin.delete_sound.rebind(
            self, SpotServiced(DeleteSound, spot_namespaced(self.robot_name, "delete_sound"), node=self.node)
        )
        SpotCAMAudioMixin.get_volume.rebind(
            self, SpotServiced(GetVolume, spot_namespaced(self.robot_name, "get_volume"), node=self.node)
        )
        SpotCAMAudioMixin.set_volume.rebind(
            self, SpotServiced(SetVolume, spot_namespaced(self.robot_name, "set_volume"), node=self.node)
        )

    @generalized_method(transitional=True)
    def list_sounds(self) -> tuple[bool, str, list[str]]:
        """List sounds loaded on Spot CAM."""
        response = self.list_sounds.synchronously(nothrow=True)
        if response is None:
            return False, "Internal error listing sounds", []
        return response.success, response.message, response.names

    @generalized_method(transitional=True)
    def load_sound(self, name: str, wav_path: str) -> tuple[bool, str]:
        """Upload wav file sound from this machine to Spot CAM."""
        response = self.load_sound.synchronously(name, wav_path, nothrow=True)
        if response is None:
            return False, "Internal error loading sound"
        return response.success, response.message

    @generalized_method(transitional=True)
    def play_sound(self, name: str, volume_multiplier: float) -> tuple[bool, str]:
        """Play sound loaded on Spot CAM. Volume is a percentage."""
        response = self.play_sound.synchronously(name, volume_multiplier, nothrow=True)
        if response is None:
            return False, "Internal error playing sound"
        return response.success, response.message

    @generalized_method(transitional=True)
    def delete_sound(self, name: str) -> tuple[bool, str]:
        """Delete sound loaded on Spot CAM."""
        response = self.delete_sound.synchronously(name, nothrow=True)
        if response is None:
            return False, "Internal error deleting sound"
        return response.success, response.message

    @generalized_method(transitional=True)
    def get_volume(self) -> tuple[bool, str, float]:
        """Get volume of sound playback on Spot CAM as a percentage."""
        response = self.get_volume.synchronously(nothrow=True)
        if response is None:
            return False, "Internal error getting volume", 0.0
        return response.success, response.message, response.volume

    @generalized_method(transitional=True)
    def set_volume(self, volume: float) -> tuple[bool, str]:
        """Set volume of sound playback on Spot CAM. Volume is a percentage."""
        response = self.set_volume.synchronously(volume)
        if response is None:
            return False, "Internal error setting volume"
        return response.success, response.message
