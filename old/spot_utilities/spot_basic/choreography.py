# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import tempfile
from typing import Any, Callable, Optional, Protocol, Union, cast, overload

import bdai.interactive_choreo.choreo_builders.serialization_utils as serialization_utils
from bdai_ros2_wrappers.action import ActionFuture
from bdai_ros2_wrappers.callables import GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike
from bosdyn.api.spot.choreography_sequence_pb2 import Animation, ChoreographySequence
from spot_msgs.action import ExecuteDance
from spot_msgs.srv import (
    ChoreographyRecordedStateToAnimation,
    ChoreographyStartRecordingState,
    ChoreographyStopRecordingState,
    GetChoreographyStatus,
    ListAllDances,
    ListAllMoves,
    UploadAnimation,
)
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
    SpotServiced,
    SpotServicedProtocol,
    SpotServiceHandle,
    make_message,
)
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


class SpotChoreographyProtocol(SpotBaseProtocol, Protocol):
    """Spot choreographic maneuvers protocol.

    Note that in order to execute a choregraphy, the corresponding
    license must be installed on the robot. Execution will fail
    otherwise.
    """

    class ExecuteProtocol(
        SpotActionableProtocol[ExecuteDance.Goal, ExecuteDance.Result, ExecuteDance.Feedback], Protocol
    ):
        """Choreography execution protocol."""

        def synchronously(
            self,
            choreo_sequence: Optional[ChoreographySequence] = None,
            choreo_file_path: Optional[os.PathLike] = None,
            *,
            feedback_callback: Optional[Callable[[ExecuteDance.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            **kwargs: Any,
        ) -> ExecuteDance.Result:
            """Execute choreography synchronously.

            Either a choreography sequence or a path to a serialized choregraphy sequence
            must be provided, but not both.

            Args:
                choreo_sequence: choreography sequence in its Protobuf form.
                choreo_file_path: path to a serialized choreography sequence in its Protobuf form.
                feedback_callback: optional action feedback callback.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                expires, the manipulation request will be cancelled and the call will raise. Note this
                timeout is local to the caller. It may take some time for the action to be cancelled,
                if cancellation is not rejected by the action server.
                kwargs: optional keyword-only arguments to initialize the underlying
                `spot_msgs.action.ExecuteDance.Goal` instance.

            Returns:
                the execution result.

            Raises:
                ActionTimeout: if the choregraphic execution timed out.
                ActionRejected: if the choregraphic execution was not accepted.
                ActionCancelled: if the choregraphic execution was cancelled.
                ActionAborted: if the choregraphic execution was aborted.
            """

        def asynchronously(
            self,
            choreo_sequence: Optional[ChoreographySequence] = None,
            choreo_file_path: Optional[os.PathLike] = None,
            *,
            track_feedback: Union[int, bool] = False,
            **kwargs: Any,
        ) -> ActionFuture[ExecuteDance.Result, ExecuteDance.Feedback]:
            """Execute choreography asynchronously.

            Either a choreography sequence or a path to a serialized choregraphy sequence
            may be provided but not both.

            Args:
                choreo_sequence: choreography sequence in its Protobuf form.
                choreo_file_path: path to a serialized choreography sequence in its Protobuf form.
                track_feedback: whether and how to track action feedback. Other
                than a boolean to enable or disable tracking, a positive integer
                may be provided to cap feedback buffer size.
                kwargs: optional keyword-only arguments to initialize the underlying
                `spot_msgs.action.ExecuteDance.Goal` instance.

            Returns:
                a future execution outcome.
            """

    class StopProtocol(SpotServicedProtocol[Trigger.Request, Trigger.Response], Protocol):
        """Choreography stopping protocol."""

        def synchronously(self, *, timeout_sec: Optional[float] = None) -> Trigger.Response:
            """Stop an ongoing choreography synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified
                and it expires, the stop request will be cancelled and the call will
                raise. Note this timeout is local to the caller (and it may take a
                while for the robot to react).

            Returns:
                a response to the stop request.

            Raises:
                ServiceError: if the stop request fails.
                ServiceTimeout: if the stop request times out.
            """

        def asynchronously(self) -> FutureLike[Trigger.Response]:
            """Stop an ongoing choreography asynchronously.

            Returns:
                a future response to the stop request.
            """

    class AnimationUploadProtocol(SpotServicedProtocol[UploadAnimation.Request, UploadAnimation.Response], Protocol):
        """Protocol to upload new robot moves."""

        @overload
        def synchronously(
            self,
            animation: Optional[Animation] = None,
            animation_file_path: Optional[os.PathLike] = None,
            *,
            animation_name: Optional[str] = None,
            timeout_sec: Optional[float] = None,
        ) -> UploadAnimation.Response:
            """Upload animation synchronously.

            Either an animation or a path to a serialized animation may be provided but not both.

            Args:
                animation: optional animation in its Protobuf form.
                animation_file_path: optional path to a serialized animation in its Protobuf form.
                animation_name: optional name (or label) for the animation. Note this only applies
                when uploading an already serialized animation.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it expires,
                the upload request will be cancelled and the call will raise. Note this timeout is
                local to the caller (and it may take a while for the robot to acknowledge it).

            Returns:
                a response to the upload request.

            Raises:
                ServiceError: if it fails to upload.
                ServiceTimeout: if it times out while uploading.
            """

        @overload
        def synchronously(
            self,
            animation: Optional[Animation] = None,
            animation_file_path: Optional[os.PathLike] = None,
            *,
            animation_name: Optional[str] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[UploadAnimation.Response]:
            """Upload animation synchronously.

            Either an animation or a path to a serialized animation may be provided but not both.

            Args:
                animation: optional animation in its Protobuf form.
                animation_file_path: optional path to a serialized animation in its Protobuf form.
                animation_name: optional name (or label) for the animation. Note this only applies
                when uploading an already serialized animation.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it expires,
                the upload request will be cancelled and the call will raise. Note this timeout is
                local to the caller (and it may take a while for the robot to acknowledge it).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the upload request or None on timeout or failure.
            """

        def asynchronously(
            self,
            animation: Optional[Animation] = None,
            animation_file_path: Optional[os.PathLike] = None,
            *,
            animation_name: Optional[str] = None,
        ) -> FutureLike[UploadAnimation.Response]:
            """Upload animation asynchronously.

            Args:
                animation: optional animation in its Protobuf form.
                animation_file_path: optional path to a serialized animation in its Protobuf form.
                animation_name: optional name (or label) for the animation. Note this only applies
                when uploading an already serialized animation.

            Returns:
                a future response to the upload request.
            """

    class MoveListingProtocol(SpotServicedProtocol[ListAllMoves.Request, ListAllMoves.Response], Protocol):
        """Protocol to list available robot moves."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> ListAllMoves.Response:
            """List all available moves synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled and the call will raise.

            Returns:
                a response to the listing request, including the list of moves.

            Raises:
                ServiceError: if it fails to list.
                ServiceTimeout: if it times out while listing.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ListAllMoves.Response]:
            """List all available moves synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled and the call will raise.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the listing request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[ListAllMoves.Response]:
            """List all available moves asynchronously.

            Returns:
                a future response to the listing request, including the list of moves.
            """

    class DanceListingProtocol(SpotServicedProtocol[ListAllDances.Request, ListAllDances.Response], Protocol):
        """Protocol to list available robot dances."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> ListAllDances.Response:
            """List all available dances synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled and the call will raise.

            Returns:
                a response to the listing request, including the list of dances.

            Raises:
                ServiceError: if it fails to list.
                ServiceTimeout: if it times out while listing.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ListAllDances.Response]:
            """List all available dances synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the listing request will be cancelled and the call will raise.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the listing request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[ListAllDances.Response]:
            """List all available dances asynchronously.

            Returns:
                a future response to the listing request, including the list of dances.
            """

    class StatusProtocol(SpotServicedProtocol[GetChoreographyStatus.Request, GetChoreographyStatus.Response], Protocol):
        """Choreography status check protocol."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> GetChoreographyStatus.Response:
            """Check the status of an ongoing choreography synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the status check request will be cancelled and the call will raise.

            Returns:
                a response to the status check request, including the status.

            Raises:
                ServiceError: if the status check fails.
                ServiceTimeout: if the status check times out.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[GetChoreographyStatus.Response]:
            """Check the status of an ongoing choreography synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the status check request will be cancelled and the call will raise.
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the status check request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[GetChoreographyStatus.Response]:
            """Check the status of an ongoing choreography asynchronously.

            Returns:
                a future response to the status check request, including the status.
            """

    class StartRecordingStateProtocol(
        SpotServicedProtocol[ChoreographyStartRecordingState.Request, ChoreographyStartRecordingState.Response],
        Protocol,
    ):
        """State recording start protocol."""

        @overload
        def synchronously(
            self, duration_seconds: float, *, timeout_sec: Optional[float] = None
        ) -> ChoreographyStartRecordingState.Response:
            """Start recording robot state synchronously.

            Args:
                duration_seconds: the intended duration of the recording, in seconds.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the start request will be cancelled and the call will raise.  Note
                this timeout is local to the caller (and it may take a while for the robot
                to acknowledge it).

            Returns:
                a response to the start request.

            Raises:
                ServiceError: if the start request fails.
                ServiceTimeout: if the start request times out.
            """

        @overload
        def synchronously(
            self, duration_seconds: float, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ChoreographyStartRecordingState.Response]:
            """Start recording robot state synchronously.

            Args:
                duration_seconds: the intended duration of the recording, in seconds.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the start request will be cancelled and the call will raise.  Note
                this timeout is local to the caller (and it may take a while for the robot
                to acknowledge it).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the start request or None on timeout or failure.
            """

        def asynchronously(self, duration_seconds: float) -> FutureLike[ChoreographyStartRecordingState.Response]:
            """Start recording robot state asynchronously.

            Args:
                duration_seconds: the intended duration of the recording, in seconds.

            Returns:
                a future response to the start request.
            """

    class StopRecordingStateProtocol(
        SpotServicedProtocol[ChoreographyStopRecordingState.Request, ChoreographyStopRecordingState.Response], Protocol
    ):
        """State recording stop protocol."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> ChoreographyStopRecordingState.Response:
            """Stop recording robot state synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the stop request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to acknowledge it).

            Returns:
                a response to the stop request.

            Raises:
                ServiceError: if the stop request fails.
                ServiceTimeout: if the stop request times out.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ChoreographyStopRecordingState.Response]:
            """Stop recording robot state synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the stop request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to acknowledge it).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the stop request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[ChoreographyStopRecordingState.Response]:
            """Stop recording robot state asynchronously.

            Returns:
                a future response to the stop request.
            """

    class AnimationBuildProtocol(
        SpotServicedProtocol[
            ChoreographyRecordedStateToAnimation.Request, ChoreographyRecordedStateToAnimation.Response
        ],
        Protocol,
    ):
        """Protocol to build an animation from recorded state."""

        @overload
        def synchronously(
            self, has_arm: bool = False, *, timeout_sec: Optional[float] = None
        ) -> ChoreographyRecordedStateToAnimation.Response:
            """Build an animation from recorded state synchronously.

            Args:
                has_arm: whether to include arm motion, if any, while building the animation.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the build request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to acknowledge it).

            Returns:
                a response to the build request, including the animation that was built.

            Raises:
                ServiceError: if the build fails.
                ServiceTimeout: if the build times out.
            """

        @overload
        def synchronously(
            self, has_arm: bool = False, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ChoreographyRecordedStateToAnimation.Response]:
            """Build an animation from recorded state synchronously.

            Args:
                has_arm: whether to include arm motion, if any, while building the animation.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the build request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to acknowledge it).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                a response to the build request or None on timeout or failure.
            """

        def asynchronously(self, has_arm: bool = False) -> FutureLike[ChoreographyRecordedStateToAnimation.Response]:
            """Build an animation from recorded state asynchronously.

            Args:
                has_arm: whether to include arm motion, if any, while building the animation.

            Returns:
                a future response to the build request, including the animation that was built.
            """

    execute_dance: GeneralizedMethodLike[ExecuteProtocol]
    stop_dance: GeneralizedMethodLike[StopProtocol]

    upload_animation: GeneralizedMethodLike[AnimationUploadProtocol]
    list_all_dances: GeneralizedMethodLike[DanceListingProtocol]
    list_all_moves: GeneralizedMethodLike[MoveListingProtocol]

    get_choreography_status: GeneralizedMethodLike[StatusProtocol]

    start_recording_state: GeneralizedMethodLike[StartRecordingStateProtocol]
    stop_recording_state: GeneralizedMethodLike[StopRecordingStateProtocol]
    build_animation_from_recorded_state: GeneralizedMethodLike[AnimationBuildProtocol]


class SpotChoreographyMixin(SpotBaseProtocol):
    """Mixin that enables execution of Spot choreographies."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        def make_execute_dance_goal(
            choreo_sequence: Optional[ChoreographySequence] = None,
            choreo_file_path: Optional[os.PathLike] = None,
            **kwargs: Any,
        ) -> ExecuteDance.Goal:
            if choreo_sequence is not None:
                kwargs["choreo_sequence_serialized"] = choreo_sequence.SerializeToString()
            if choreo_file_path is not None:
                with open(choreo_file_path, "r") as f:
                    kwargs["choreo_file_content"] = f.read()
            return make_message(ExecuteDance.Goal, **kwargs)

        SpotChoreographyMixin.execute_dance.rebind(
            self,
            SpotActionable(ExecuteDance, spot_namespaced(self.robot_name, "execute_dance"), node=self.node).compose(
                make_execute_dance_goal
            ),
        )

        SpotChoreographyMixin.stop_dance.rebind(
            self, SpotServiced(Trigger, spot_namespaced(self.robot_name, "stop_dance"), node=self.node)
        )

        def make_upload_animation_request(
            animation: Optional[Animation] = None, animation_file_path: Optional[os.PathLike] = None, **kwargs: Any
        ) -> UploadAnimation.Request:
            if animation is not None:
                kwargs["animation_proto_serialized"] = animation.SerializeToString()
                kwargs["animation_name"] = animation.name
            if animation_file_path is not None:
                with open(animation_file_path, "r") as f:
                    kwargs["animation_file_content"] = f.read()
            return make_message(UploadAnimation.Request, **kwargs)

        SpotChoreographyMixin.upload_animation.rebind(
            self,
            SpotServiced(UploadAnimation, spot_namespaced(self.robot_name, "upload_animation"), node=self.node).compose(
                make_upload_animation_request
            ),
        )

        SpotChoreographyMixin.get_choreo_status.rebind(
            self,
            SpotServiced(
                GetChoreographyStatus, spot_namespaced(self.robot_name, "get_choreography_status"), node=self.node
            ),
        )

        SpotChoreographyMixin.start_recording_state.rebind(
            self,
            SpotServiced(
                ChoreographyStartRecordingState,
                spot_namespaced(self.robot_name, "start_recording_state"),
                node=self.node,
            ),
        )

        SpotChoreographyMixin.stop_recording_state.rebind(
            self,
            SpotServiced(
                ChoreographyStopRecordingState, spot_namespaced(self.robot_name, "stop_recording_state"), node=self.node
            ),
        )

        SpotChoreographyMixin.recorded_state_to_animation.rebind(
            self,
            SpotServiced(
                ChoreographyRecordedStateToAnimation,
                spot_namespaced(self.robot_name, "recorded_state_to_animation"),
                node=self.node,
            ),
        )

        SpotChoreographyMixin.list_all_dances.rebind(
            self, SpotServiced(ListAllDances, spot_namespaced(self.robot_name, "list_all_dances"), node=self.node)
        )

        SpotChoreographyMixin.list_all_moves.rebind(
            self, SpotServiced(ListAllMoves, spot_namespaced(self.robot_name, "list_all_moves"), node=self.node)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.ExecuteProtocol])
    def execute_dance(
        self,
        choreo_sequence: Optional[ChoreographySequence] = None,
        choreo_file_path: Optional[os.PathLike] = None,
        **kwargs: Any,
    ) -> ExecuteDance.Result:
        """Execute a choreography with the robot.

        Args:
            choreo_sequence: choreography sequence in its Protobuf form.
            choreo_file_path: path to a serialized choreography sequence in its Protobuf form.

        See `SpotChoreographyProtocol.ExecuteProtocol` for further reference.
        """

    @execute_dance.legacy
    def _legacy_execute_dance(self, path: str, timeout_sec: float = 10) -> tuple[bool, str]:
        """Upload and execute a dance on the robot."""
        self.node.get_logger().info(f"Uploading dance at '{path}' to {self.robot_name}")
        response = self.execute_dance.synchronously(
            choreo_file_path=cast(os.PathLike, path), timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return False, "Timed out executing dance"
        if not response.success:
            self.node.get_logger().debug(
                f"Uploading dance '{path}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    def execute_dance_proto(
        self, choreography_sequence: ChoreographySequence, timeout_sec: float = 10.0
    ) -> tuple[bool, str]:
        """Execute a dance sequence through Spot Driver.

        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        response = self.execute_dance.synchronously(
            choreography_sequence=choreography_sequence, timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return False, "Timed out executing dance"
        if not response.success:
            self.node.get_logger().debug(f"Executing dance on '{self.robot_name}' " "failed. Msg: {response.message}.")
            return False, response.message
        return True, "success"

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.StopProtocol])
    def stop_dance(self, **kwargs: Any) -> Trigger.Response:
        """Stop an ongoing choreography.

        See `SpotChoreographyProtocol.StopProtocol` for further reference.
        """

    @stop_dance.legacy
    def _legacy_stop_dance(self) -> SpotServiceHandle:
        """Stop an ongoing robot dance."""
        return SpotServiceHandle.call(
            self.robot_name,
            "stop_dance",
            self.stop_dance.client,
            context=self.node.context,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.AnimationUploadProtocol])
    def upload_animation(
        self,
        animation: Optional[Animation] = None,
        animation_file_path: Optional[os.PathLike] = None,
        *,
        animation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> UploadAnimation.Response:
        """Upload a new robot move.

        Args:
            animation: optional animation in its Protobuf form.
            animation_file_path: optional path to a serialized animation in its Protobuf form.
            animation_name: optional name (or label) for the animation. Note this only applies
            when uploading an already serialized animation.

        See `SpotChoreographyProtocol.AnimationUploadProtocol` for further reference.
        """

    @upload_animation.legacy
    def _legacy_upload_animation(self, move_name: str, path: str, timeout_sec: float = 10.0) -> tuple[bool, str]:
        """Upload an animation to the robot."""
        self.node.get_logger().info(f"Uploading animation at '{path}' to {self.robot_name}")
        response = self.upload_animation.synchronously(
            animation_name=move_name, animation_file_path=cast(os.PathLike, path), timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return False, "Timed out uploading animation"
        if not response.success:
            self.node.get_logger().debug(
                f"Uploading animation '{path}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    def upload_animation_proto(self, animation: Animation, timeout_sec: float = 10.0) -> tuple[bool, str]:
        """Upload an animation file through spot driver

        Returns:
            (True, "success") if the command succeeded, (False, <error_message>) otherwise
        """
        self.node.get_logger().info(f"Uploading animation named '{animation.name}' to {self.robot_name}")
        response = self.upload_animation.synchronously(animation=animation, timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out uploading animation"
        if not response.success:
            self.node.get_logger().info(
                f"Uploading proto animation '{animation.name}' to '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message
        return True, "success"

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.DanceListingProtocol])
    def list_all_dances(self, **kwargs: Any) -> ListAllDances.Response:
        """List all available robot dances.

        See `SpotChoreographyProtocol.DanceListingProtocol` for further reference.
        """

    @list_all_dances.legacy
    def _legacy_list_all_dances(self, timeout_sec: float = 10.0) -> tuple[bool, str, list[str]]:
        """List all the dances uploaded to the robot."""
        self.node.get_logger().info(f"Sending list dances request to {self.robot_name}")
        response = self.list_all_dances.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out listing dances", []
        if not response.success:
            self.node.get_logger().debug(
                f"Request to list dances at '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message, []
        return True, "success", response.dances

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.MoveListingProtocol])
    def list_all_moves(self, **kwargs: Any) -> ListAllMoves.Response:
        """List all available robot moves.

        See `SpotChoreographyProtocol.MoveListingProtocol` for further reference.
        """

    @list_all_moves.legacy
    def _legacy_list_all_moves(self, timeout_sec: float = 10.0) -> tuple[bool, str, list[str]]:
        """List all the moves uploaded to the robot."""
        self.node.get_logger().info(f"Sending list moves request to {self.robot_name}")
        response = self.list_all_moves.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out listing moves", []
        if not response.success:
            self.node.get_logger().debug(
                f"Request to list moves at '{self.robot_name}' failed. Msg: {response.message}."
            )
            return False, response.message, []
        return True, "success", response.moves

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.StatusProtocol])
    def get_choreography_status(self, **kwargs: Any) -> GetChoreographyStatus.Response:
        """Check an ongoing choreography status.

        See `SpotChoreographyProtocol.StatusProtocol` for further reference.
        """

    @get_choreography_status.legacy
    def _legacy_get_choreography_status(self, timeout_sec: float = 10.0) -> tuple[bool, str, int]:
        """Get robot choregraphy status, if any."""
        response = self.get_choreo_status.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out getting chore status", GetChoreographyStatus.Request.STATUS_UNKNOWN
        if not response.success:
            self.node.get_logger().debug(f"Failed to get choreo status. Msg: {response.message}.")
            return False, response.message, GetChoreographyStatus.Request.STATUS_UNKNOWN
        return True, "success", response.status

    get_choreo_status = get_choreography_status

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.StartRecordingStateProtocol])
    def start_recording_state(self, duration_seconds: float, **kwargs: Any) -> ChoreographyStartRecordingState.Response:
        """Start recording robot state (for animation building).

        Args:
            duration_seconds: the intended duration of the recording, in seconds.

        See `SpotChoreographyProtocol.StartRecordingStateProtocol` for further reference.
        """

    @start_recording_state.legacy
    def _legacy_start_recording_state(self, duration: float, timeout_sec: float = 10.0) -> tuple[bool, str]:
        """Start recording robot state."""
        response = self.start_recording_state.synchronously(
            duration_seconds=duration, timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return False, "Timed out starting state recording"
        if not response.success:
            self.node.get_logger().error(f"Failed to start recording state. Msg: {response.message}.")
            return False, response.message
        return True, "success"

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.StopRecordingStateProtocol])
    def stop_recording_state(self, **kwargs: Any) -> ChoreographyStopRecordingState.Response:
        """Stop recording robot state forcefully.

        See `SpotChoreographyProtocol.StopRecordingStateProtocol` for further reference.
        """

    @stop_recording_state.legacy
    def _legacy_stop_recording_state(self, timeout_sec: float = 10.0) -> tuple[bool, str]:
        """Stop recording robot state."""
        response = self.stop_recording_state.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out stoping state recording"
        if not response.success:
            self.node.get_logger().error(f"Failed to stop recording state. Msg: {response.message}.")
            return False, response.message
        return True, "success"

    @generalized_method(spec=GeneralizedMethodLike[SpotChoreographyProtocol.AnimationBuildProtocol])
    def build_animation_from_recorded_state(
        self, has_arm: Optional[bool] = False, **kwargs: Any
    ) -> ChoreographyRecordedStateToAnimation.Response:
        """Build an animated robot move from recorded state.

        Args:
            has_arm: whether to include arm motion, if any, while building the animation.

        See `SpotChoreographyProtocol.AnimationBuildProtocol` for further reference.
        """

    recorded_state_to_animation = build_animation_from_recorded_state

    @recorded_state_to_animation.legacy
    def _legacy_recorded_state_to_animation(
        self, has_arm: bool, timeout_sec: float = 10.0
    ) -> tuple[bool, str, Animation]:
        """Fetch recorded state as an animation."""
        animation_proto = Animation()
        response = self.recorded_state_to_animation.synchronously(
            has_arm=has_arm, timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return False, "Timed out converting recorded state to animation", animation_proto
        if not response.success:
            self.node.get_logger().error(f"Failed to convert recorded state to animation. Msg: {response.message}.")
            return False, response.message, animation_proto

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = "temporary_animation.cha"
            full_path = os.path.join(temp_dir, filename)
            with open(full_path, "w") as animation_file:
                animation_file.write(response.animation_file_contents)

            animation_proto = serialization_utils.load_animation(full_path)
        return True, "success", animation_proto
