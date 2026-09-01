# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Arm manipulation, grasping, and object interaction helpers."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Protocol, Sequence, Union

import bosdyn_api_msgs.msg
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import ApiGraspOverride
from bosdyn.api.robot_state_pb2 import ManipulatorState
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from spatialmath.base.types import R3
from spot_msgs.action import Manipulation
from spot_msgs.srv import OverrideGraspOrCarry
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedGuard, GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike

from spot_basic.state import FrameHint, SpotStateProtocol
from spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
    SpotServiced,
    SpotServicedProtocol,
    spot_namespaced,
)


class SpotManipulationCommandBuilder:
    """A builder for Spot manipulation commands."""

    def __init__(self, target: SpotStateProtocol) -> None:
        """Initialize the manipulation command builder.

        Args:
            target: SpotBasic object used to look up robot-specific transforms for manipulation requests.
        """
        self.target = target

    def tight_top_down_grasp_params(self) -> manipulation_api_pb2.GraspParams:
        """Return grasping parameters that constraint the grasp to be top down.

        The object will be grasped from the top, moving down along the vertical axis,
        with high accuracy relative to the user requested position and constraints.
        """
        grasp_params = manipulation_api_pb2.GraspParams()

        # Specify the frame we're using.
        grasp_params.grasp_params_frame_name = FrameHint.BODY.value

        # The axis on the gripper is the x-axis.
        axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

        # The axis in the frame is the negative z-axis
        axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

        # Add the vector constraint to our proto.
        constraint = grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vision
        )

        # We'll take anything within about 1.0 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = math.radians(1.0)

        grasp_params.position_constraint = manipulation_api_pb2.GRASP_POSITION_CONSTRAINT_FIXED_AT_USER_POSITION

        return grasp_params

    def pick_object_in_image_command(
        self,
        image_pixel_xy: Sequence[float],
        camera_info: CameraInfo,
        image_timestamp: Optional[Union[Time, float]] = None,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Build an object grasping command given the object location in one of the robot camera feeds.

        Args:
            image_pixel_xy: pixel location in the image for the robot to perform object detection.
            image_timestamp: image timestamp, used to fetch the associated transformation tree snapshot.
            camera_info: camera intrinsics and extrinsics assocated with the image the robot took.
            grasp_params: optional object grasping parameters.
            walk_gaze_mode: optional mode for walking and gazing before grasping.

        Returns:
            the object grasping manipulation request.
        """
        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_info.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_info.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_info.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_info.k[5]

        frame_name_image_sensor = camera_info.header.frame_id
        if frame_name_image_sensor.startswith(self.target.robot_name):
            frame_name_image_sensor = frame_name_image_sensor[len(self.target.robot_name) + 1 :]

        transforms_snapshot = self.target.get_transforms_tree_snapshot(image_timestamp, timeout_sec=2.0)

        proto = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=image_pixel_xy[0], y=image_pixel_xy[1]),
            transforms_snapshot_for_camera=transforms_snapshot,
            frame_name_image_sensor=frame_name_image_sensor,
            camera_model=pinhole_model,
        )

        if grasp_params is not None:
            proto.grasp_params.CopyFrom(grasp_params)
        if walk_gaze_mode is not None:
            proto.walk_gaze_mode = walk_gaze_mode

        return manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=proto)

    def walk_to_object_in_image_command(
        self,
        image_pixel_xy: Sequence[float],
        camera_info: CameraInfo,
        image_timestamp: Optional[Union[Time, float]] = None,
        offset_distance: Optional[float] = None,
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Build a walk-to-object command given the object location in one of the robot camera feeds.

        Args:
            image_pixel_xy: pixel location in the image for the robot to perform object detection.
            image_timestamp: image timestamp, used to fetch the associated transformation tree snapshot.
            camera_info: camera intrinsics and extrinsics assocated with the image the robot took.
            offset_distance: optional distance between the robot's CoM and the object.

        Returns:
            the walk-to-object manipulation request.
        """
        pinhole_model = image_pb2.ImageSource.PinholeModel()
        pinhole_model.intrinsics.focal_length.x = camera_info.k[0]
        pinhole_model.intrinsics.focal_length.y = camera_info.k[4]
        pinhole_model.intrinsics.principal_point.x = camera_info.k[2]
        pinhole_model.intrinsics.principal_point.y = camera_info.k[5]

        frame_name_image_sensor = camera_info.header.frame_id
        if frame_name_image_sensor.startswith(self.target.robot_name):
            frame_name_image_sensor = frame_name_image_sensor[len(self.target.robot_name) + 1 :]

        transforms_snapshot = self.target.get_transforms_tree_snapshot(image_timestamp, timeout_sec=2.0)

        proto = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=image_pixel_xy[0], y=image_pixel_xy[1]),
            transforms_snapshot_for_camera=transforms_snapshot,
            frame_name_image_sensor=frame_name_image_sensor,
            camera_model=pinhole_model,
        )

        if offset_distance is not None:
            proto.offset_distance.value = offset_distance

        return manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=proto)

    def walk_to_object_ray_in_world_command(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame: Union[str, FrameHint],
        offset_distance: Optional[float] = None,
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Build a walk-to-object command, the first object found along a given ray.

        Useful to prepare for grasping or manipulating something.

        Args:
            ray_start_rt_frame: ray source position in the given `frame`.
            ray_end_rt_frame: ray target position in the given `frame`.
            frame: frame of reference w.r.t. which the ray is defined.
            offset_distance: optional distance between the robot's CoM and the object.

        Returns:
            the walk-to-object manipulation request.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto#L20
        """
        proto = manipulation_api_pb2.WalkToObjectRayInWorld(
            ray_start_rt_frame=geometry_pb2.Vec3(
                x=ray_start_rt_frame[0], y=ray_start_rt_frame[1], z=ray_start_rt_frame[2]
            ),
            ray_end_rt_frame=geometry_pb2.Vec3(x=ray_end_rt_frame[0], y=ray_end_rt_frame[1], z=ray_end_rt_frame[2]),
            frame_name=frame.value if isinstance(frame, FrameHint) else frame,
        )
        if offset_distance is not None:
            proto.offset_distance.value = offset_distance
        return manipulation_api_pb2.ManipulationApiRequest(walk_to_object_ray_in_world=proto)

    def pick_object_ray_in_world_command(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame: Union[str, FrameHint],
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
    ) -> manipulation_api_pb2.ManipulationApiRequest:
        """Build an object grasping command, the first object found along the given ray.

        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        Args:
            ray_start_rt_frame: ray source position in the given `frame`.
            ray_end_rt_frame: ray target position in the given `frame`.
            frame: frame of reference w.r.t. which the ray is defined.
            grasp_params: optional object grasping parameters.
            walk_gaze_mode: optional mode for walking and gazing before grasping.

        Returns:
            the object grasping manipulation request.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto
        """
        proto = manipulation_api_pb2.PickObjectRayInWorld(
            ray_start_rt_frame=geometry_pb2.Vec3(
                x=ray_start_rt_frame[0], y=ray_start_rt_frame[1], z=ray_start_rt_frame[2]
            ),
            ray_end_rt_frame=geometry_pb2.Vec3(x=ray_end_rt_frame[0], y=ray_end_rt_frame[1], z=ray_end_rt_frame[2]),
            frame_name=frame.value if isinstance(frame, FrameHint) else frame,
        )
        if grasp_params is not None:
            proto.grasp_params.CopyFrom(grasp_params)
        if walk_gaze_mode is not None:
            proto.walk_gaze_mode = walk_gaze_mode
        return manipulation_api_pb2.ManipulationApiRequest(pick_object_ray_in_world=proto)


class SpotManipulationProtocol(SpotStateProtocol, Protocol):
    """Spot manipulation capabilities protocol."""

    class PerformProtocol(
        SpotActionableProtocol[Manipulation.Goal, Manipulation.Result, Manipulation.Feedback], Protocol
    ):
        """Manipulation request execution protocol."""

        def synchronous(
            self,
            request: Union[bosdyn_api_msgs.msg.ManipulationApiRequest, manipulation_api_pb2.ManipulationApiRequest],
            *,
            feedback_callback: Optional[Callable[[Manipulation.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> Manipulation.Result:
            """Perform manipulation request synchronously.

            Args:
                request: manipulation request to perform, in its Protobuf or ROS 2 form.
                feedback_callback: optional action feedback callback.
                timeout_sec: optional action timeout, in seconds. If a timeout is specified and it
                    expires, the manipulation request will be cancelled and the call will raise.
                    Note this timeout is local to the caller. It may take some time for the action
                    to be cancelled, if cancellation is not rejected by the action server.

            Returns:
                the manipulation result.

            Raises:
                ActionTimeout: if the manipulation request timed out.
                ActionRejected: if the manipulation request was not accepted.
                ActionCancelled: if the manipulation request was cancelled.
                ActionAborted: if the manipulation request was aborted.
                RuntimeError: if the robot has no arm.
            """

        def asynchronous(
            self,
            request: Union[bosdyn_api_msgs.msg.ManipulationApiRequest, manipulation_api_pb2.ManipulationApiRequest],
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[Manipulation.Result, Manipulation.Feedback]:
            """Perform manipulation request asynchronously.

            Args:
                request: manipulation request to perform, in its Protobuf or ROS 2 form.
                track_feedback: whether and how to track action feedback. Other than a boolean to enable
                    or disable tracking, a positive integer may be provided to cap feedback buffer size.

            Returns:
                the future manipulation outcome.

            Raises:
                RuntimeError: if the robot has no arm.
            """

    class WalkToObjectInImageProtocol(
        SpotActionableProtocol[Manipulation.Goal, Manipulation.Result, Manipulation.Feedback], Protocol
    ):
        """Protocol for walking towards an object seen by the robot."""

        def synchronously(
            self,
            image_pixel_xy: Sequence[float],
            camera_info: CameraInfo,
            image_timestamp: Optional[Union[Time, float]] = None,
            offset_distance: Optional[float] = None,
            *,
            feedback_callback: Optional[Callable[[Manipulation.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> Manipulation.Result:
            """Walk to an object synchronously.

            The object is referenced by a pixel on an image taken by the robot.

            This method is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.walk_to_object_in_image_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            image_pixel_xy: Sequence[float],
            camera_info: CameraInfo,
            image_timestamp: Optional[Union[Time, float]] = None,
            offset_distance: Optional[float] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[Manipulation.Result, Manipulation.Feedback]:
            """Walk to an object asynchronously.

            The object is referenced by a pixel on an image taken by the robot.

            This method is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.walk_to_object_in_image_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class PickObjectInImageProtocol(
        SpotActionableProtocol[Manipulation.Goal, Manipulation.Result, Manipulation.Feedback], Protocol
    ):
        """Protocol for grasping an object seen by the robot."""

        def synchronously(
            self,
            image_pixel_xy: Sequence[float],
            camera_info: CameraInfo,
            image_timestamp: Optional[Union[Time, float]] = None,
            grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
            walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
            *,
            feedback_callback: Optional[Callable[[Manipulation.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> Manipulation.Result:
            """Grasp an object synchronously.

            The object is referenced by a pixel on an image taken by the robot.

            This is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.pick_object_in_image_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            image_pixel_xy: Sequence[float],
            camera_info: CameraInfo,
            image_timestamp: Optional[Union[Time, float]] = None,
            grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
            walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[Manipulation.Result, Manipulation.Feedback]:
            """Grasp an object asynchronously.

            The object is referenced by a pixel on an image taken by the robot.

            This is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.pick_object_in_image_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class WalkToObjectRayInWorldProtocol(
        SpotActionableProtocol[Manipulation.Goal, Manipulation.Result, Manipulation.Feedback], Protocol
    ):
        """Protocol for walking towards an object along a ray in the world."""

        def synchronously(
            self,
            ray_start_rt_frame: R3,
            ray_end_rt_frame: R3,
            frame: Union[str, FrameHint],
            offset_distance: Optional[float] = None,
            *,
            feedback_callback: Optional[Callable[[Manipulation.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> Manipulation.Result:
            """Walk to an object synchronously.

            The object is the first one found along a given ray in the world.

            This method is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.walk_to_object_ray_in_world_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            ray_start_rt_frame: R3,
            ray_end_rt_frame: R3,
            frame: Union[str, FrameHint],
            offset_distance: Optional[float] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[Manipulation.Result, Manipulation.Feedback]:
            """Walk to an object asynchronously.

            The object is the first one found along a given ray in the world.

            This method is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.walk_to_object_ray_in_world_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class PickObjectRayInWorldProtocol(
        SpotActionableProtocol[Manipulation.Goal, Manipulation.Result, Manipulation.Feedback], Protocol
    ):
        """Protocol for grasping an object along a ray in the world."""

        def synchronously(
            self,
            ray_start_rt_frame: R3,
            ray_end_rt_frame: R3,
            frame: Union[str, FrameHint],
            grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
            walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
            *,
            feedback_callback: Optional[Callable[[Manipulation.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> Manipulation.Result:
            """Grasp an object synchronously.

            The object is the first one found along a given ray in the world.

            This is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.pick_object_ray_in_world_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            ray_start_rt_frame: R3,
            ray_end_rt_frame: R3,
            frame: Union[str, FrameHint],
            grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
            walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[Manipulation.Result, Manipulation.Feedback]:
            """Grasp an object asynchronously.

            The object is the first one found along a given ray in the world.

            This is typically implemented by `SpotManipulationProtocol.perform`ing a
            `SpotManipulationCommandBuilder.pick_object_ray_in_world_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class OverrideGraspOrCarryProtocol(
        SpotServicedProtocol[OverrideGraspOrCarry.Request, OverrideGraspOrCarry.Response], Protocol
    ):
        """Protocol for overriding grasp or carry behavior.

        The robot's grasp state is whether or not it's holding an object. Usually the robot is aware of whether or not
        it is grasping something but sometimes (e.g. when the object is small and the gripper is almost closed) it makes
        mistakes. Valid override values are:

            - OVERRIDE_UNKNOWN: If this is set, this function will not request that the grasp state be changed.
            - OVERRIDE_HOLDING: Grasp state will be changed to HOLDING
            - OVERRIDE_NOT_HOLDING: Grasp state will be changed to NOT HOLDING

        The robot's carry state governs what the robot will do when control of the arm is requested to stow (usually by
        a hijack). This only matters if the grasp state is HOLDING (otherwise the arm will always stow). Overriding
        the carry state will not affect the grasp state. Valid override values are:

            - CARRY_STATE_UNKNOWN: If this is set, this function will not request that the carry state be changed.
            - CARRY_STATE_NOT_CARRYABLE: When the arm is requested to stow, it will open the gripper (let go) first.
            - CARRY_STATE_CARRIABLE: When the arm is requested to stow, it will not do anything.
            - CARRY_STATE_CARRIABLE_AND_STOWABLE: When the arm is requested to stow, it will stow without attempting to
                first release the object.
        """

        def synchronous(
            self,
            grasp_override: Optional[ApiGraspOverride.Override] = None,
            carry_override: Optional[ManipulatorState.CarryState] = None,
            *,
            timeout_sec: Optional[float] = None,
        ) -> OverrideGraspOrCarry.Response:
            """Override grasping state and arm behavior on stow synchronously.

            Args:
                grasp_override: optional grasp state override.
                carry_override: optional carry state override.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it expires,
                    the override request will be cancelled and the call will raise. Note this
                    timeout is local to the caller (and it may take a while for the robot to react).

            Returns:
                the response to the override request.

            Raises:
                ServiceError: if the override request fails.
                ServiceTimeout: if the override request times out.
                RuntimeError: if the robot has no arm.
            """

        def asynchronous(
            self,
            grasp_override: Optional[ApiGraspOverride.Override] = None,
            carry_override: Optional[ManipulatorState.CarryState] = None,
        ) -> FutureLike[OverrideGraspOrCarry.Response]:
            """Override grasping state and arm behavior on stow synchronously.

            Args:
                grasp_override: optional grasp state override.
                carry_override: optional carry state override.

            Returns:
                a future response to the override request.

            Raises:
                RuntimeError: if the robot has no arm.
            """

    perform: GeneralizedMethodLike[PerformProtocol]
    walk_to_object_in_image: GeneralizedMethodLike[WalkToObjectInImageProtocol]
    pick_object_in_image: GeneralizedMethodLike[PickObjectInImageProtocol]
    walk_to_object_ray_in_world: GeneralizedMethodLike[WalkToObjectRayInWorldProtocol]
    pick_object_ray_in_world: GeneralizedMethodLike[PickObjectRayInWorldProtocol]
    override_grasp_or_carry: GeneralizedMethodLike[OverrideGraspOrCarryProtocol]


class SpotManipulationMixin(SpotStateProtocol):
    """Mixin offering Spot manipulation capabilities.

    Note that the robot must have an arm for any of these APIs to be available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize manipulation clients and bind generalized methods.

        Creates concrete ROS 2 action and service clients for manipulation and grasp override
        requests, then rebinds generalized methods through arm-availability guards. The rebound
        APIs raise `RuntimeError` when invoked on a robot without an arm.

        Args:
            args: positional arguments forwarded to the next class in the MRO.
            kwargs: keyword arguments forwarded to the next class in the MRO.
        """
        super().__init__(*args, **kwargs)
        command_builder = SpotManipulationCommandBuilder(self)

        SpotManipulationMixin.perform.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                SpotActionable(Manipulation, spot_namespaced(self.robot_name, "manipulation"), node=self.node),
                f"{self.robot_name} has no arm, cannot manipulate",
            ),
        )

        SpotManipulationMixin.walk_to_object_in_image.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.perform.compose(command_builder.walk_to_object_in_image_command),
                f"{self.robot_name} has no arm, cannot manipulate",
            ),
        )

        SpotManipulationMixin.pick_object_in_image.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.perform.compose(command_builder.pick_object_in_image_command),
                f"{self.robot_name} has no arm, cannot manipulate",
            ),
        )

        SpotManipulationMixin.walk_to_object_ray_in_world.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.perform.compose(command_builder.walk_to_object_ray_in_world_command),
                f"{self.robot_name} has no arm, cannot manipulate",
            ),
        )

        SpotManipulationMixin.pick_object_ray_in_world.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                self.perform.compose(command_builder.pick_object_ray_in_world_command),
                f"{self.robot_name} has no arm, cannot manipulate",
            ),
        )

        SpotManipulationMixin.override_grasp_or_carry.rebind(
            self,
            GeneralizedGuard(
                self.has_arm,
                SpotServiced(
                    OverrideGraspOrCarry, spot_namespaced(self.robot_name, "override_grasp_or_carry"), node=self.node
                ),
                f"{self.robot_name} has no arm, nothing to override",
            ),
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.PerformProtocol])
    def perform(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a manipulation command with the robot.

        See `SpotManipulationProtocol.PerformProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.WalkToObjectInImageProtocol])
    def walk_to_object_in_image(
        self,
        image_pixel_xy: Sequence[float],
        camera_info: CameraInfo,
        image_timestamp: Optional[Union[Time, float]] = None,
        offset_distance: Optional[float] = None,
        **kwargs: Any,
    ) -> Manipulation.Result:
        """Walk to an object seen by the robot.

        See `SpotManipulationProtocol.WalkToObjectInImageProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.PickObjectInImageProtocol])
    def pick_object_in_image(
        self,
        image_pixel_xy: Sequence[float],
        camera_info: CameraInfo,
        image_timestamp: Optional[Union[Time, float]] = None,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
        **kwargs: Any,
    ) -> Manipulation.Result:
        """Grasp an object seen by the robot.

        See `SpotManipulationProtocol.PickObjectInImageProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.WalkToObjectRayInWorldProtocol])
    def walk_to_object_ray_in_world(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame: Union[str, FrameHint],
        offset_distance: Optional[float] = None,
        **kwargs: Any,
    ) -> Manipulation.Result:
        """Walk to an object along a ray in the world.

        See `SpotManipulationProtocol.WalkToObjectRayInWorldProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.PickObjectRayInWorldProtocol])
    def pick_object_ray_in_world(self, *args: Any, **kwargs: Any) -> Manipulation.Result:
        """Grasp an object along a ray in the world.

        See `SpotManipulationProtocol.PickObjectRayInWorldProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.OverrideGraspOrCarryProtocol])
    def override_grasp_or_carry(self, *args: Any, **kwargs: Any) -> OverrideGraspOrCarry.Response:
        """Override grasping state and arm behavior on stow.

        See `SpotManipulationProtocol.OverrideGraspOrCarryProtocol` for further reference.
        """
