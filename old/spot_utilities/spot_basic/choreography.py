# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import tempfile
from typing import Any, Optional

import bdai.interactive_choreo.choreo_builders.serialization_utils as serialization_utils
from bdai_ros2_wrappers.callables import generalized_method
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
    SpotServiced,
    SpotServiceHandle,
    make_message,
)
from spot_utilities.spot_util import spot_namespaced
from std_srvs.srv import Trigger


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

        execute_dance = SpotActionable(
            ExecuteDance, spot_namespaced(self.robot_name, "execute_dance"), node=self.node
        ).compose(make_execute_dance_goal)
        SpotChoreographyMixin.execute_dance.rebind(self, execute_dance)

        stop_dance = SpotServiced(Trigger, spot_namespaced(self.robot_name, "stop_dance"), node=self.node)
        SpotChoreographyMixin.stop_dance.rebind(self, stop_dance)

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

        upload_animation = SpotServiced(
            UploadAnimation, spot_namespaced(self.robot_name, "upload_animation"), node=self.node
        ).compose(make_upload_animation_request)
        SpotChoreographyMixin.upload_animation.rebind(self, upload_animation)

        get_choreography_status = SpotServiced(
            GetChoreographyStatus, spot_namespaced(self.robot_name, "get_choreography_status"), node=self.node
        )
        SpotChoreographyMixin.get_choreo_status.rebind(self, get_choreography_status)

        start_recording_state = SpotServiced(
            ChoreographyStartRecordingState, spot_namespaced(self.robot_name, "start_recording_state"), node=self.node
        )
        SpotChoreographyMixin.start_recording_state.rebind(self, start_recording_state)

        stop_recording_state = SpotServiced(
            ChoreographyStopRecordingState, spot_namespaced(self.robot_name, "stop_recording_state"), node=self.node
        )
        SpotChoreographyMixin.stop_recording_state.rebind(self, stop_recording_state)

        recorded_state_to_animation = SpotServiced(
            ChoreographyRecordedStateToAnimation,
            spot_namespaced(self.robot_name, "recorded_state_to_animation"),
            node=self.node,
        )
        SpotChoreographyMixin.recorded_state_to_animation.rebind(self, recorded_state_to_animation)

        list_all_dances = SpotServiced(
            ListAllDances, spot_namespaced(self.robot_name, "list_all_dances"), node=self.node
        )
        SpotChoreographyMixin.list_all_dances.rebind(self, list_all_dances)

        list_all_moves = SpotServiced(ListAllMoves, spot_namespaced(self.robot_name, "list_all_moves"), node=self.node)
        SpotChoreographyMixin.list_all_moves.rebind(self, list_all_moves)

    @generalized_method(transitional=True)
    def execute_dance(self, path: str, timeout_sec: float = 10) -> tuple[bool, str]:
        """Upload and execute a dance on the robot."""
        self.node.get_logger().info(f"Uploading dance at '{path}' to {self.robot_name}")
        response = self.execute_dance.synchronously(choreo_file_path=path, timeout_sec=timeout_sec, nothrow=True)
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

    @generalized_method(transitional=True)
    def stop_dance(self) -> SpotServiceHandle:
        """Stop an ongoing robot dance."""
        return SpotServiceHandle.call(
            self.robot_name,
            "stop_dance",
            self.stop_dance.client,
            context=self.node.context,
        )

    @generalized_method(transitional=True)
    def upload_animation(self, move_name: str, path: str, timeout_sec: float = 10.0) -> tuple[bool, str]:
        """Upload an animation to the robot."""
        self.node.get_logger().info(f"Uploading animation at '{path}' to {self.robot_name}")
        response = self.upload_animation.synchronously(
            animation_name=move_name, animation_file_path=path, timeout_sec=timeout_sec, nothrow=True
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

    @generalized_method(transitional=True)
    def list_all_dances(self, timeout_sec: float = 10.0) -> tuple[bool, str, list[str]]:
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

    @generalized_method(transitional=True)
    def list_all_moves(self, timeout_sec: float = 10.0) -> tuple[bool, str, list[str]]:
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

    @generalized_method(transitional=True)
    def get_choreo_status(self, timeout_sec: float = 10.0) -> tuple[bool, str, int]:
        """Get robot choregraphy status, if any."""
        response = self.get_choreo_status.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out getting chore status", GetChoreographyStatus.Request.STATUS_UNKNOWN
        if not response.success:
            self.node.get_logger().debug(f"Failed to get choreo status. Msg: {response.message}.")
            return False, response.message, GetChoreographyStatus.Request.STATUS_UNKNOWN
        return True, "success", response.status

    @generalized_method(transitional=True)
    def start_recording_state(self, duration: float, timeout_sec: float = 10.0) -> tuple[bool, str]:
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

    @generalized_method(transitional=True)
    def stop_recording_state(self, timeout_sec: float = 10.0) -> tuple[bool, str]:
        """Stop recording robot state."""
        response = self.stop_recording_state.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return False, "Timed out stoping state recording"
        if not response.success:
            self.node.get_logger().error(f"Failed to stop recording state. Msg: {response.message}.")
            return False, response.message
        return True, "success"

    @generalized_method(transitional=True)
    def recorded_state_to_animation(self, has_arm: bool, timeout_sec: float = 10.0) -> tuple[bool, str, Animation]:
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
