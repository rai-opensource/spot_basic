# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
from typing import Any, Optional, Protocol, overload

from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import SpotServiced, SpotServicedProtocol, spot_namespaced
from spot_msgs.srv import DeleteSound, GetVolume, ListSounds, LoadSound, PlaySound, SetVolume
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike


class SpotCAMAudioProtocol(SpotBaseProtocol, Protocol):
    """Spot CAM audio system protocol.

    Note that the robot must have a Spot CAM payload for any of these capabilities to be available.
    """

    class ListProtocol(SpotServicedProtocol[ListSounds.Request, ListSounds.Response], Protocol):
        """Protocol to list sounds."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> ListSounds.Response:
            """List available sounds on the Spot CAM synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled and the call will raise.
                Note this timeout is local to the caller -- the robot is unaware and will
                complete the computation before discarding the response.

            Returns:
                the response to the listing request.

            Raises:
                ServiceError: if it fails to list sounds.
                ServiceTimeout: if it times out while listing sounds.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ListSounds.Response]:
            """List available sounds on the Spot CAM synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled. Note this timeout is local
                to the caller  -- the robot is unaware and will complete the computation
                before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the listing request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[ListSounds.Response]:
            """List available sounds on the Spot CAM asynchronously.

            Returns:
                a future response to the listing request.
            """

    class LoadProtocol(SpotServicedProtocol[LoadSound.Request, LoadSound.Response], Protocol):
        """Protocol to load a new sound."""

        @overload
        def synchronously(
            self, name: str, wav_path: os.PathLike, *, timeout_sec: Optional[float] = None
        ) -> LoadSound.Response:
            """Load a new sound to the Spot CAM synchronously.

            Args:
                name: a name (or label) to id the sound. It will be required to play or delete it.
                wav_path: a path to a .wav file with the sound to be loaded.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled. Note this timeout is local to
                the caller -- the robot is unaware and will complete the computation before
                discarding the response.

            Returns:
                a response the load request.

            Raises:
                ServiceError: if it fails to load the sound.
                ServiceTimeout: if it times out while loading the sound.
            """

        @overload
        def synchronously(
            self, name: str, wav_path: os.PathLike, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[LoadSound.Response]:
            """Load a new sound to the Spot CAM synchronously.

            Args:
                name: a name (or label) to id the sound. It will be required to play or delete it.
                wav_path: a path to a .wav file with the sound to be loaded.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled. Note this timeout is local to
                the caller -- the robot is unaware and will complete the computation before
                discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response the load request or None on timeout or failure.
            """

        def asynchronously(self, name: str, wav_path: os.PathLike) -> FutureLike[LoadSound.Response]:
            """Load a new sound to the Spot CAM asynchronously.

            Args:
                name: a name (or label) to id the sound. It will be required to play or delete it.
                wav_path: a path to a .wav file with the sound to be loaded.

            Returns:
                a future response to the load request.
            """

    class PlaybackProtocol(SpotServicedProtocol[PlaySound.Request, PlaySound.Response], Protocol):
        """Protocol to play a sound."""

        @overload
        def synchronously(
            self, name: str, volume_multiplier: float, *, timeout_sec: Optional[float] = None
        ) -> PlaySound.Response:
            """Play a sound with the Spot CAM synchronously.

            The sound must be loaded already.

            Args:
                name: name of the sound to play.
                volume_multiplier: speaker volume gain for sound playback.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, playback will be cancelled and the call will raise. Note this timeout
                is local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the playback request.

            Raises:
                ServiceError: if it fails to play the sound.
                ServiceTimeout: if it times out while playing the sound.
            """

        @overload
        def synchronously(
            self, name: str, volume_multiplier: float, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[PlaySound.Response]:
            """Play a sound with the Spot CAM synchronously.

            The sound must be loaded already.

            Args:
                name: name of the sound to play.
                volume_multiplier: speaker volume gain for sound playback. Must be nonnegative.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, playback will be cancelled. Note this timeout is local to the caller -- the
                robot is unaware and will complete the computation before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the playback request or None on timeout or failure.
            """

        def asynchronously(self, name: str, volume_multiplier: float) -> FutureLike[PlaySound.Response]:
            """Play a sound with the Spot CAM asynchronously.

            The sound must be loaded already.

            Args:
                name: name of the sound to play.
                volume_multiplier: speaker volume gain for sound playback.

            Returns:
                a future response to the playback request.
            """

    class DeleteProtocol(SpotServicedProtocol[DeleteSound.Request, DeleteSound.Response], Protocol):
        """Protocol to delete an existing sound."""

        @overload
        def synchronously(self, name: str, *, timeout_sec: Optional[float] = None) -> DeleteSound.Response:
            """Delete a sound synchronously.

            Args:
                name: name of the sound to delete.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note this timeout
                is local to the caller  -- the robot is unaware and will complete the computation
                before discarding the response.

            Returns:
                the response to the request.

            Raises:
                ServiceError: if it fails to delete the sound.
                ServiceTimeout: if it times out while deleting the sound.
            """

        @overload
        def synchronously(
            self, name: str, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[DeleteSound.Response]:
            """Delete a sound synchronously.

            Args:
                name: name of the sound to delete.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled. Note this timeout is local to the
                caller -- the robot is unaware and will complete the computation before
                discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the request or None on timeout or failure.
            """

        def asynchronously(self, name: str) -> FutureLike[DeleteSound.Response]:
            """Delete a sound asynchronously.

            Args:
                name: name of the sound to play.

            Returns:
                a future response to the request.
            """

    class SetVolumeProtocol(SpotServicedProtocol[SetVolume.Request, SetVolume.Response], Protocol):
        """Protocol to set the speaker volume."""

        @overload
        def synchronously(self, volume: float, *, timeout_sec: Optional[float] = None) -> SetVolume.Response:
            """Change the Spot CAM speaker volume synchronously.

            Args:
                volume: speaker volume as a percentage.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the volume setting request will be cancelled and the call will raise.
                Note this timeout is local to the caller -- the robot is unaware and will complete
                the computation before discarding the response.

            Returns:
                the response to the volume setting request.

            Raises:
                ServiceError: if it fails to set the volume.
                ServiceTimeout: if it times out while setting the volume.
            """

        @overload
        def synchronously(
            self, volume: float, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[SetVolume.Response]:
            """Change the Spot CAM speaker volume synchronously.

            Args:
                volume: speaker volume as a percentage.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the volume setting request will be cancelled. Note this timeout is
                local to the caller -- the robot is unaware and will complete the computation
                before discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the volume setting request or None on timeout or failure.
            """

        def asynchronously(self, volume: float) -> FutureLike[SetVolume.Response]:
            """Change the Spot CAM speaker volume asynchronously.

            Args:
                volume: speaker volume as a percentage.

            Returns:
                a future response to the volume setting request.
            """

    class GetVolumeProtocol(SpotServicedProtocol[GetVolume.Request, GetVolume.Response], Protocol):
        """Protocol to get the speaker volume."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> SetVolume.Response:
            """Get the Spot CAM speaker volume synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the volume request will be cancelled and the call will raise. Note
                this timeout is local to the caller -- the robot is unaware and will complete
                the computation before discarding the response.

            Returns:
                the response to the volume request, including the speaker volume as a percentage.

            Raises:
                ServiceError: if it fails to fetch the volume.
                ServiceTimeout: if it times out while fetching the volume.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[GetVolume.Response]:
            """Get the Spot CAM speaker volume synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the volume request will be cancelled. Note this timeout is local to
                the caller -- the robot is unaware and will complete the computation before
                discarding the response.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the volume request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[GetVolume.Response]:
            """Get the Spot CAM speaker volume asynchronously.

            Returns:
                a future response to the volume request.
            """

    list_sounds: GeneralizedMethodLike[ListProtocol]
    load_sound: GeneralizedMethodLike[LoadProtocol]
    play_sound: GeneralizedMethodLike[PlaybackProtocol]
    delete_sound: GeneralizedMethodLike[DeleteProtocol]
    set_volume: GeneralizedMethodLike[SetVolumeProtocol]
    get_volume: GeneralizedMethodLike[GetVolumeProtocol]


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

    @generalized_method(spec=GeneralizedMethodLike[SpotCAMAudioProtocol.ListProtocol])
    def list_sounds(self, **kwargs: Any) -> ListSounds.Response:
        """List available sounds on the Spot CAM.

        See `SpotCAMAudioProtocol.ListProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotCAMAudioProtocol.LoadProtocol])
    def load_sound(self, name: str, wav_path: os.PathLike, **kwargs: Any) -> LoadSound.Response:
        """Load a new sound to the Spot CAM.

        Args:
            name: a name (or label) to id the sound. It will be required to play or delete it.
            wav_path: a path to a .wav file with the sound to be loaded.

        See `SpotCAMAudioProtocol.LoadProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotCAMAudioProtocol.PlaybackProtocol])
    def play_sound(self, name: str, volume_multiplier: float, **kwargs: Any) -> PlaySound.Response:
        """Play a sound with the Spot CAM.

        Args:
            name: name of the sound to play.
            volume_multiplier: speaker volume gain for sound playback. Must be nonnegative.

        See `SpotCAMAudioProtocol.PlaybackProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotCAMAudioProtocol.DeleteProtocol])
    def delete_sound(self, name: str, **kwargs: Any) -> DeleteSound.Response:
        """Delete a sound from the Spot CAM.

        Args:
            name: name of the sound to delete.

        See `SpotCAMAudioProtocol.DeleteProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotCAMAudioProtocol.GetVolumeProtocol])
    def get_volume(self, **kwargs: Any) -> GetVolume.Response:
        """Get the Spot CAM speaker volume.

        See `SpotCAMAudioProtocol.GetVolumeProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotCAMAudioProtocol.SetVolumeProtocol])
    def set_volume(self, volume: float, **kwargs: Any) -> SetVolume.Response:
        """Set the Spot CAM speaker volume.

        Args:
            volume: speaker volume as a percentage.

        See `SpotCAMAudioProtocol.SetVolumeProtocol` for further reference.
        """
