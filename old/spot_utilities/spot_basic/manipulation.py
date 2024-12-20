# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Protocol, Sequence, Union

import bosdyn_api_msgs.msg
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import ApiGraspOverride
from bosdyn.api.robot_state_pb2 import ManipulatorState
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import ManipulationApiFeedbackResponse
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from spatialmath.base.types import R3
from spot_msgs.action import Manipulation
from spot_msgs.srv import OverrideGraspOrCarry
from spot_utilities.spot_basic.state import FrameHint, SpotStateProtocol
from spot_utilities.spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
    SpotActionHandle,
    SpotServiced,
    SpotServicedProtocol,
)
from spot_utilities.spot_util import spot_namespaced

from bdai.deprecation import deprecated
from bdai_ros2_wrappers.action import ActionFuture
from bdai_ros2_wrappers.callables import GeneralizedGuard, GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike


class SpotManipulationCommandBuilder:
    """A builder for Spot manipulation commands."""

    def __init__(self, target: SpotStateProtocol) -> None:
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
                expires, the manipulation request will be cancelled and the call will raise. Note this
                timeout is local to the caller. It may take some time for the action to be cancelled,
                if cancellation is not rejected by the action server.

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
                track_feedback: whether and how to track action feedback. Other
                than a boolean to enable or disable tracking, a positive integer
                may be provided to cap feedback buffer size.

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
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the override request will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to react).

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

    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.manipulation.synchronously",
    )
    def manipulation(
        self, command_name: str, goal: Manipulation.Goal, timeout_sec: Optional[float] = None
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Synchronously run a manipulation request.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            command_name (str): A representative name of the command for logging
            goal (Manipulation.Goal): The Action Goal sent to the action server
            timeout_sec (Optional[float]): A timeout for waiting on a response/result from the action server. Setting to
                None creates no timeout

        Returns:
            Optional[ManipulationApiFeedbackResponse]: The feedback from the SpotDriver about the result of the command
                if it was accepted and a result was successfully received, None otherwise

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """

        if not self.has_arm():
            self.node.get_logger().error("Cannot send Manipulation api request, robot has no arm.")
            return None

        failed = False

        def on_failure() -> None:
            nonlocal failed
            failed = True

        handle = self.manipulation_async(command_name, goal, on_failure_callback=on_failure)
        if handle is None or not handle.wait_for_result(timeout_sec) or failed:
            return None
        return handle.result

    def manipulation_async(
        self,
        command_name: str,
        goal: Manipulation.Goal,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        if not self.has_arm():
            self.node.get_logger().error("Cannot send Manipulation api request, robot has no arm.")
            return None
        return SpotActionHandle.call(
            self.robot_name,
            command_name,
            self.perform.action_client,
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
            context=self.node.context,
        )

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

    @walk_to_object_in_image.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.walk_to_object_in_image.synchronously",
    )
    def _legacy_walk_to_object_in_image(
        self,
        pixel_xy: Sequence[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        offset: Optional[float] = None,
        timeout_sec: Optional[float] = 10.0,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Walk to an object in an image taken with a camera on spot.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            pixel_xy: The point in the image where spot will detect the object.
            camera_data: Camera data associated with the image spot took.
            image_timestamp: Uses the timestamp to get the associated transformation tree snapshot.
            offset: Target distance between spot's CoM and the object.
            timeout_sec: Timeout for response from action server

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.walk_to_object_in_image_command(
            image_pixel_xy=pixel_xy, image_timestamp=image_timestamp, camera_info=camera_data, offset_distance=offset
        )
        convert(command, goal.command)
        return self.manipulation("walk_to_object_image", goal, timeout_sec=timeout_sec)

    walk_to_object_image = walk_to_object_in_image

    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.walk_to_object_in_image.asynchronously",
    )
    def walk_to_object_image_async(
        self,
        pixel_xy: Sequence[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        offset: Optional[float] = None,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Walk to an object in an image taken with a camera on spot asynchronously.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            pixel_xy: The point in the image where spot will detect the object.
            camera_data: Camera data associated with the image spot took.
            image_timestamp: Uses the timestamp to get the associated transformation tree snapshot.
            offset: Target distance between spot's CoM and the object.
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.walk_to_object_in_image_command(
            image_pixel_xy=pixel_xy, image_timestamp=image_timestamp, camera_info=camera_data, offset_distance=offset
        )
        convert(command, goal.command)
        return self.manipulation_async(
            "walk_to_object_image",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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

    @pick_object_in_image.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.pick_object_in_image.synchronously",
    )
    def _legacy_pick_object_in_image(
        self,
        pixel_xy: Sequence[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Grasp an object in an image taken with a camera on spot.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            camera_data: Camera data associate with the image spot took.
            image_timestamp: Uses the timestamp to get retrieve the associated transformation tree snapshot.
            palm_placement_ratio: a float [0.0, 1.0] that indicates how far
                in the edge of the object is into the palm cavity.
                (0.0 will place the base of palm with camera against the object,
                1.0 will be a 'finger pinch')
            apply_grasp_params_func: A helper function that will apply grasp parameters
                to modify default grasping behavior.
                (see https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=pickobjectinimage#graspparams)

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.pick_object_in_image_command(
            image_pixel_xy=pixel_xy, image_timestamp=image_timestamp, camera_info=camera_data
        )
        command.pick_object_in_image.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio
        if apply_grasp_params_func is not None:
            apply_grasp_params_func(command.pick_object_in_image)
        convert(command, goal.command)
        return self.manipulation("grasp_in_image", goal, 10.0)

    grasp_in_image = pick_object_in_image

    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.walk_to_object_in_image.asynchronously",
    )
    def grasp_in_image_async(
        self,
        pixel_xy: Sequence[float],
        image_timestamp: Union[Time, float],
        camera_data: CameraInfo,
        palm_placement_ratio: float,
        apply_grasp_params_func: Optional[Callable] = None,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Grasp an item in an image taken with a camera on spot asynchronously.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            pixel_xy: The point in the image where spot will do object detection,
                and then grasp the object.
            camera_data: Camera data associate with the image spot took.
            image_timestamp: Uses the timestamp to get retrieve the associated transformation tree snapshot.
            palm_placement_ratio: a float [0.0, 1.0] that indicates how far
                in the edge of the object is into the palm cavity.
                (0.0 will place the base of palm with camera against the object,
                1.0 will be a 'finger pinch')
            apply_grasp_params_func: A helper function that will apply grasp parameters
                to modify default grasping behavior.
                (see https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=pickobjectinimage#graspparams)
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            An ManipulationApiFeedbackResponse if the robot has an arm.
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.pick_object_in_image_command(
            image_pixel_xy=pixel_xy, image_timestamp=image_timestamp, camera_info=camera_data
        )
        command.pick_object_in_image.grasp_params.grasp_palm_to_fingertip = palm_placement_ratio
        if apply_grasp_params_func is not None:
            apply_grasp_params_func(command.pick_object_in_image)
        convert(command, goal.command)
        return self.manipulation_async(
            "grasp_in_image",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

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

    @walk_to_object_ray_in_world.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.walk_to_object_ray_in_world.synchronously",
    )
    def _legacy_walk_to_object_ray_in_world(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        offset_distance: float,
        timeout_sec: Optional[float] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Walks the robot up to an object.  Useful to prepare to grasp or manipulate something.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto#L20
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.walk_to_object_ray_in_world_command(
            ray_start_rt_frame, ray_end_rt_frame, frame_name, offset_distance
        )
        convert(command, goal.command)
        return self.manipulation("walk_to_object_ray_in_world", goal, timeout_sec)

    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.walk_to_object_ray_in_world.asynchronously",
    )
    def walk_to_object_ray_in_world_async(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        offset_distance: float,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Walks the robot up to an object.  Useful to prepare to grasp or manipulate something.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto#L20
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.walk_to_object_ray_in_world_command(
            ray_start_rt_frame, ray_end_rt_frame, frame_name, offset_distance
        )
        convert(command, goal.command)
        return self.manipulation_async(
            "walk_to_object_ray_in_world",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.PickObjectRayInWorldProtocol])
    def pick_object_ray_in_world(self, *args: Any, **kwargs: Any) -> Manipulation.Result:
        """Grasp an object along a ray in the world.

        See `SpotManipulationProtocol.PickObjectRayInWorldProtocol` for further reference.
        """

    @pick_object_ray_in_world.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.pick_object_ray_in_world.synchronously",
    )
    def _legacy_pick_object_ray_in_world(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
        timeout_sec: Optional[float] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Cast a ray in the world and pick the first object found along the ray.

        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.pick_object_ray_in_world_command(
            ray_start_rt_frame, ray_end_rt_frame, frame_name, grasp_params, walk_gaze_mode
        )
        convert(command, goal.command)
        return self.manipulation("grasp_along_ray", goal, timeout_sec)

    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.pick_object_ray_in_world.asynchronously",
    )
    def pick_object_ray_in_world_async(
        self,
        ray_start_rt_frame: R3,
        ray_end_rt_frame: R3,
        frame_name: str,
        grasp_params: Optional[manipulation_api_pb2.GraspParams] = None,
        walk_gaze_mode: Optional[manipulation_api_pb2.WalkGazeMode] = None,
        *,
        result_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        feedback_callback: Optional[Callable[[ManipulationApiFeedbackResponse], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[ManipulationApiFeedbackResponse]:
        """Cast a ray in the world and pick the first object found along the ray.

        This is the lowest-level grasping message; all other grasp options internally
        use this message to trigger a grasp.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        See: https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/manipulation_api.proto
        """
        goal = Manipulation.Goal()
        command_builder = SpotManipulationCommandBuilder(self)
        command = command_builder.pick_object_ray_in_world_command(
            ray_start_rt_frame, ray_end_rt_frame, frame_name, grasp_params, walk_gaze_mode
        )
        convert(command, goal.command)
        return self.manipulation_async(
            "grasp_along_ray",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotManipulationProtocol.OverrideGraspOrCarryProtocol])
    def override_grasp_or_carry(self, *args: Any, **kwargs: Any) -> OverrideGraspOrCarry.Response:
        """Override grasping state and arm behavior on stow.

        See `SpotManipulationProtocol.OverrideGraspOrCarryProtocol` for further reference.
        """

    @override_grasp_or_carry.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.override_grasp_or_carry.synchronously",
    )
    def _legacy_override_grasp_or_carry(
        self,
        grasp_override: ApiGraspOverride.Override = ApiGraspOverride.Override.OVERRIDE_UNKNOWN,
        carry_override: ManipulatorState.CarryState = ManipulatorState.CarryState.CARRY_STATE_UNKNOWN,
    ) -> OverrideGraspOrCarry.Response:
        """
        Allows overriding of whether spot is grasping something and its behavior on stow.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Grasp Override:

        The robot's grasp state is whether or not it's holding an object.  Usually the robot is aware of whether or not
        it is grasping something but sometimes (e.g. when the object is small and the gripper is almost closed) it makes
        mistakes.  Grasp override values:
            OVERRIDE_UNKNOWN: If this is set, this function will not request that the grasp state be changed.
            OVERRIDE_HOLDING: Grasp state will be changed to HOLDING
            OVERRIDE_NOT_HOLDING: Grasp state will be changed to NOT HOLDING

        Carry Override:

        The robot's carry state governs what the robot will do when control of the arm is requested to stow (usually by
        a hijack). This only matters if the grasp state is HOLDING (otherwise the arm will always stow) but this
        function will send the request without checking the grasp state and will not alter the grasp state unless
        requested to do so.  Carry override values:
            CARRY_STATE_UNKNOWN: If this is set, this function will not request that the carry state be changed.
            CARRY_STATE_NOT_CARRYABLE: When the arm is requested to stow, it will open the gripper (let go) first.
            CARRY_STATE_CARRIABLE: When the arm is requested to stow, it will not do anything.
            CARRY_STATE_CARRIABLE_AND_STOWABLE: When the arm is requested to stow, it will stow without attempting to
                first release the object.
        """
        try:
            return self.override_grasp_or_carry.synchronously(grasp_override, carry_override)
        except Exception as e:
            return OverrideGraspOrCarry.Response(success=False, message=f"Service call failed with error {e}")
