# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import functools
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Protocol, TypeAlias, Union, overload

import numpy as np
from bosdyn.api import robot_command_pb2, trajectory_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import Quaternion, to_euler_zxy
from bosdyn.util import seconds_to_duration
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import RobotCommandFeedback
from geometry_msgs.msg import Twist
from spatialmath import SE2, SE3, SO3, Twist2, UnitQuaternion
from spatialmath.base.types import R3
from spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_basic.command.helpers import timestamped_se2_vecs_to_robot_command_proto
from spot_basic.state import FrameHint
from spot_basic.trajectory.se2_trajectory import SE2Trajectory
from spot_basic.utilities import RobotComposeCommandHandle, SpotActionableProtocol, SpotActionHandle, spot_namespaced
from spot_msgs.action import RobotCommand as RobotCommandAction
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike

from bdai.deprecation import deprecated
from bdai_ros.utilities.manual_conversions import se3_to_se3pose


class SpotMobilityCommandBuilder(SpotCommandBuilder):
    """A builder for Spot mobility commands."""

    def sit_command(
        self, *, mobility_params: Optional[spot_command_pb2.MobilityParams] = None
    ) -> robot_command_pb2.RobotCommand:
        """Build a sit command.

        Args:
            mobility_params: optional mobility params, defaulting to target
            mobility params if none are provided.
        """
        if mobility_params is None:
            mobility_params = self.target.mobility_params
        return RobotCommandBuilder.synchro_sit_command(mobility_params)

    def stand_command(
        self, *, mobility_params: Optional[spot_command_pb2.MobilityParams] = None
    ) -> robot_command_pb2.RobotCommand:
        """Build a stand command.

        Args:
            mobility_params: optional mobility params, defaulting to target
            mobility params if none are provided.
        """
        if mobility_params is None:
            mobility_params = self.target.mobility_params
        return RobotCommandBuilder.synchro_stand_command(mobility_params)

    def stand_with_orientation_command(
        self, gravbody_r_goal: SO3, body_height: float = 0.0
    ) -> robot_command_pb2.RobotCommand:
        """Build a stand command with target body height and orientation.

        Args:
            gravbody_r_goal: body frame orientation w.r.t. to a gravity aligned framed
            with yaw computed from the stance feet.
            body_height: body height, in meters, relative to the nominal stand height.
        """
        gravbody_r_goal = UnitQuaternion(gravbody_r_goal)
        footprint_R_body = to_euler_zxy(
            Quaternion(**{key: val for key, val in zip("xyzw", gravbody_r_goal.vec_xyzs, strict=True)})
        )
        return RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body, body_height=body_height)

    def walk_command(
        self,
        frame_t_goal: SE2,
        body_height: float = 0.0,
        *,
        base_frame: Optional[FrameHint] = None,
        mobility_params: Optional[spot_command_pb2.MobilityParams] = None,
        disable_obstacle_avoidance: Optional[bool] = False,
    ) -> robot_command_pb2.RobotCommand:
        """Build a walk to SE(2) pose command.

        Args:
            frame_t_goal: pose to walk to, defined w.r.t. the given `base_frame`.
            body_height: body height, in meters, relative to the nominal stand height.
            base_frame: optional base frame, defaults to the VISION frame.
        """
        if base_frame is None:
            base_frame = FrameHint.VISION
        if mobility_params is None:
            mobility_params = deepcopy(self.target.mobility_params)
        if disable_obstacle_avoidance:
            self._logger.warning("Caution: attempting to send mobility trajectory without obstacle avoidance")
            mobility_params.obstacle_params.disable_vision_body_obstacle_avoidance = (disable_obstacle_avoidance,)
        return RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=frame_t_goal.x,
            goal_y=frame_t_goal.y,
            goal_heading=frame_t_goal.theta(),
            body_height=body_height,
            frame_name=base_frame.value,
            params=mobility_params,
        )

    def walk_trajectory_command(
        self,
        frame_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        *,
        base_frame: Optional[FrameHint] = None,
        mobility_params: Optional[spot_command_pb2.MobilityParams] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Build a walk along an SE(2) trajectory command.

        The trajectory is defined as a time-parameterized, optionally interpolated sequence of SE(2) poses.

        Args:
            frame_t_trajectory: time-parameterize sequence of poses, defined in the given `base_frame`.
            timestep_sec: time step to be used for trajectory sampling.
            interpolation: method to be used for trajectory interpolation.
            base_frame: optional base frame, defaults to the VISION frame.
        """
        if base_frame is None:
            base_frame = FrameHint.VISION
        if mobility_params is None:
            mobility_params = self.target.mobility_params
        return timestamped_se2_vecs_to_robot_command_proto(
            *frame_t_trajectory(timestep_sec=timestep_sec, with_timestamps=True),
            interpolation=interpolation,
            base_frame=base_frame.value,
            mobility_params=mobility_params,
        )

    def torso_pose_command(
        self,
        pose: Union[SE3Pose, SE3],
        duration: Optional[float] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Creates a stand command that takes the robot torso to desired pose while keeping the feet
        in contact with the ground.

        The reference frame is the footprint pseudo-frame, which is a gravity-aligned frame with
        its origin located at the geometric center of the feet in the X-Y axis, and at the nominal
        height of the hips in the Z axis. The yaw of the frame (wrt the world) is calculated by the
        average foot locations, and is aligned with the feet.

        Args:
            pose: A SE3Pose object representing the desired torso pose in the footprint pseudo-frame.
            duration: The duration over which the motion to the new pose should be executed, in seconds.

        Returns:
            A stand command towards the desired torso pose.
        """
        if isinstance(pose, SE3):
            pose = se3_to_se3pose(pose)
        # Create the trajectory point with the desired pose and duration
        if duration is not None:
            duration = seconds_to_duration(duration)
        trajectory_point = trajectory_pb2.SE3TrajectoryPoint(
            pose=pose.to_proto(),
            time_since_reference=duration,
        )

        # Build the trajectory proto by combining the point
        trajectory = trajectory_pb2.SE3Trajectory(points=[trajectory_point])

        # Build a custom mobility params to specify absolute body control
        body_control = spot_command_pb2.BodyControlParams(base_offset_rt_footprint=trajectory)
        mobility_params = spot_command_pb2.MobilityParams(body_control=body_control)

        # Issue the command via the RobotCommandClient
        return RobotCommandBuilder.synchro_stand_command(params=mobility_params)

    def walk_velocity_command(
        self, v_x: float, v_y: float, v_rot: float, params: Optional[spot_command_pb2.MobilityParams] = None
    ) -> robot_command_pb2.RobotCommand:
        """Creates a velocity command.

        Args:
            v_x: Linear velocity along the x-axis.
            v_y: Linear velocity along the y-axis.
            v_rot: Angular velocity about the z-axis.
            params: Mobility params for the robot. Defaults to the mobility params of the target.

        Returns:
            A velocity command with the desired velocity.
        """
        params = params or self.target.mobility_params
        return RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot, params=params)

    def walk_and_orient_maneuver(
        self,
        frame_t_goal: SE3,
        body_height: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
    ) -> Iterable[robot_command_pb2.RobotCommand]:
        """Build a walk to SE(2) x SO(3) pose maneuver.

        The manuever comprises a walk to SE(2) pose command followed
        by a stand command with target body height and SO(3) orientation.

        Args:
            frame_t_goal: pose to walk to, defined w.r.t. the given `base_frame`. Assumed planar,
            its SE(2) equivalent disregarding roll and pitch is used for walking, and the full
            SO(3) rotation is used for standing.
            body_height: body height, in meters, relative to the nominal stand height.
            base_frame: optional base frame, defaults to the VISION frame.
        """
        # first walk
        yield self.walk_command(frame_t_goal.yaw_SE2(), base_frame=base_frame)

        # then orient
        if base_frame is None:
            base_frame = FrameHint.VISION
        frame_t_orien_se3 = SE3.RPY(frame_t_goal.rpy())
        gravbody_t_frame = self.target.lookup_a_tform_b(
            FrameHint.GRAV_ALIGNED_BODY.value, base_frame.value, wait_for_frames=True
        )
        gravbody_t_orien_se3 = gravbody_t_frame * frame_t_orien_se3
        rpy = gravbody_t_orien_se3.rpy()
        gravbody_t_orien = SE3.RPY(rpy[0], rpy[1], 0)
        if body_height is None:
            body_t_frame = self.target.lookup_a_tform_b(FrameHint.BODY.value, base_frame.value, wait_for_frames=True)
            body_t_goal = body_t_frame * frame_t_goal
            body_height = body_t_goal.z

        yield self.stand_with_orientation_command(gravbody_t_orien, body_height)

    def body_gaze_command(self, target: R3, frame: Optional[FrameHint] = None) -> RobotCommandFeedback:
        """Build a body gazing command.

        This command is basically stand with orientation command defined in terms of a point in space
        the robot should align its body x-axis with.

        Args:
            target: Point in space to gaze, defined w.r.t. the given frame.
            frame: optional frame, defaults to the VISION frame if none is provided.
        """
        if frame is None:
            frame = FrameHint.VISION

        gravbody_t_frame = self.target.lookup_a_tform_b(
            FrameHint.GRAV_ALIGNED_BODY, frame, timeout=2.0, wait_for_frames=True
        )
        target_wrt_gravbody = np.squeeze(gravbody_t_frame * target)

        x_axis = np.array([1, 0, 0])
        target_axis = target_wrt_gravbody / np.linalg.norm(target_wrt_gravbody)
        normal_axis = np.cross(x_axis, target_axis)
        angle = np.arccos(np.dot(x_axis, target_axis))
        gravbody_r_target = SO3.AngleAxis(angle, normal_axis)

        return self.stand_with_orientation_command(gravbody_r_target)


RobotCommandActionableProtocol: TypeAlias = SpotActionableProtocol[
    RobotCommandAction.Goal, RobotCommandAction.Result, RobotCommandAction.Feedback
]


class SpotMobilityCommandProtocol(SpotCommandProtocol, Protocol):
    """Spot general mobility capabilities protocol."""

    class SitProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot sitting."""

        @overload
        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot sit synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.sit_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot sit synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.sit_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot sit asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.sit_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class StandProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot standing."""

        @overload
        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot stand synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.stand_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot stand synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.stand_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot stand asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.stand_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class StandWithOrientationProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot standing with a given orientation."""

        @overload
        def synchronously(
            self,
            gravbody_r_goal: SO3,
            body_height: float = 0.0,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot stand with target body height and orientation synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.stand_with_orientation_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            gravbody_r_goal: SO3,
            body_height: float = 0.0,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot stand with target body height and orientation synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.stand_with_orientation_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            gravbody_r_goal: SO3,
            body_height: float = 0.0,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot stand with target body height and orientation asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.stand_with_orientation_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class WalkProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot walking."""

        @overload
        def synchronously(
            self,
            frame_t_goal: SE2,
            body_height: float = 0.0,
            *,
            base_frame: Optional[FrameHint] = None,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot walk to an SE(2) pose synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.walk_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            frame_t_goal: SE2,
            body_height: float = 0.0,
            *,
            base_frame: Optional[FrameHint] = None,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot walk to an SE(2) pose synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.walk_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            frame_t_goal: SE2,
            body_height: float = 0.0,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot walk to an SE(2) pose asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.walk_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class WalkTrajectoryProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot trajectory walking."""

        @overload
        def synchronously(
            self,
            frame_t_trajectory: SE2Trajectory,
            timestep_sec: float = 0.1,
            interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
            *,
            base_frame: Optional[FrameHint] = None,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot walk along an SE(2) trajectory synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.walk_trajectory_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            frame_t_trajectory: SE2Trajectory,
            timestep_sec: float = 0.1,
            interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
            *,
            base_frame: Optional[FrameHint] = None,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot walk along an SE(2) trajectory synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.walk_trajectory_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            frame_t_trajectory: SE2Trajectory,
            timestep_sec: float = 0.1,
            interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
            *,
            base_frame: Optional[FrameHint] = None,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot walk along an SE(2) trajectory asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.walk_trajectory_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    class WalkAndOrientProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for the robot walking and reorienting maneuver.

        The manuever comprises a walk to SE(2) pose command followed
        by a stand command with target body height and SO(3) orientation.
        """

        @overload
        def synchronously(
            self,
            frame_t_goal: SE3,
            body_height: Optional[float] = None,
            *,
            base_frame: Optional[FrameHint] = None,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> List[RobotCommandAction.Result]:
            """Make the robot walk to an SE(2) x SO(3) pose synchronously.

            This is typically implemented by sequentially `SpotCommandProtocol.execute`ing
            a `SpotMobilityCommandBuilder.walk_and_orient_maneuver` synchronously.
            See documentation for both for further reference on arguments taken.

            Returns:
                the maneuver results, one for walking and one for standing.

            Raises:
                ActionTimeout: if either command during the maneuver timed out.
                ActionRejected: if either command during the maneuver was not accepted.
                ActionCancelled: if either command during the maneuver was cancelled.
                ActionAborted: if either command during the maneuver was aborted.
            """

        @overload
        def synchronously(
            self,
            frame_t_goal: SE3,
            body_height: Optional[float] = None,
            *,
            base_frame: Optional[FrameHint] = None,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[List[RobotCommandAction.Result]]:
            """Make the robot walk to an SE(2) x SO(3) pose synchronously.

            This is typically implemented by sequentially `SpotCommandProtocol.execute`ing
            a `SpotMobilityCommandBuilder.walk_and_orient_maneuver` synchronously.
            See documentation for both for further reference on arguments taken.

            Returns:
                the maneuver results or None on timeout or failure.
            """

        def asynchronously(
            self,
            frame_t_goal: SE3,
            body_height: Optional[float] = None,
            *,
            base_frame: Optional[FrameHint] = None,
        ) -> FutureLike[List[RobotCommandAction.Result]]:
            """Make the robot walk to an SE(2) x SO(3) pose synchronously.

            This is typically implemented by sequentially `SpotCommandProtocol.execute`ing
            a `SpotMobilityCommandBuilder.walk_and_orient_maneuver` asynchronously.
            See documentation for both for further reference on arguments taken.

            Returns:
                the future maneuver results, one for walking, one for standing.
            """

    class BodyGazeProtocol(RobotCommandActionableProtocol, Protocol):
        """Protocol for robot fully body gazing.

        Body gazing is basically standing with an orientation defined in terms
        of a point in space the robot should align its body frame x-axis with.
        """

        @overload
        def synchronously(
            self,
            target: R3,
            frame: Optional[FrameHint] = None,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
        ) -> RobotCommandAction.Result:
            """Make the robot point its body towards some point in space synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.body_gaze_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        @overload
        def synchronously(
            self,
            target: R3,
            frame: Optional[FrameHint] = None,
            *,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[RobotCommandAction.Result]:
            """Make the robot point its body towards some point in space synchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.body_gaze_command` synchronously.
            See documentation for both for further reference on arguments taken.
            """

        def asynchronously(
            self,
            target: R3,
            frame: Optional[FrameHint] = None,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[RobotCommandAction.Result, RobotCommandAction.Feedback]:
            """Make the robot point its body towards some point in space asynchronously.

            This is typically implemented by `SpotCommandProtocol.execute`ing a
            `SpotMobilityCommandBuilder.body_gaze_command` asynchronously.
            See documentation for both for further reference on arguments taken.
            """

    sit: GeneralizedMethodLike[SitProtocol]
    stand: GeneralizedMethodLike[StandProtocol]
    stand_with_orientation: GeneralizedMethodLike[StandWithOrientationProtocol]

    walk: GeneralizedMethodLike[WalkProtocol]
    walk_trajectory: GeneralizedMethodLike[WalkTrajectoryProtocol]
    walk_and_orient: GeneralizedMethodLike[WalkAndOrientProtocol]

    body_gaze: GeneralizedMethodLike[BodyGazeProtocol]


class SpotMobilityCommandMixin(SpotCommandProtocol):
    """Mixin that eases basic Spot mobility command execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotMobilityCommandBuilder(self)
        SpotMobilityCommandMixin.sit.rebind(self, self.execute.compose(command_builder.sit_command))
        SpotMobilityCommandMixin.stand.rebind(self, self.execute.compose(command_builder.stand_command))
        SpotMobilityCommandMixin.stand_with_orientation.rebind(
            self, self.execute.compose(command_builder.stand_with_orientation_command)
        )
        SpotMobilityCommandMixin.walk.rebind(self, self.execute.compose(command_builder.walk_command))
        SpotMobilityCommandMixin.walk_trajectory.rebind(
            self, self.execute.compose(command_builder.walk_trajectory_command)
        )
        SpotMobilityCommandMixin.walk_and_orient.rebind(
            self, self.execute.vectorized.compose(command_builder.walk_and_orient_maneuver)
        )
        SpotMobilityCommandMixin.body_gaze.rebind(self, self.execute.compose(command_builder.body_gaze_command))
        self._velocity_command_publisher = self.node.create_publisher(
            Twist, spot_namespaced(self.robot_name, "cmd_vel"), 10
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.SitProtocol])
    def sit(self, **kwargs: Any) -> RobotCommandAction.Result:
        """Make the robot sit.

        See `SpotMobilityCommandProtocol.SitProtocol` for further reference.
        """

    @sit.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.sit.synchronously",
    )
    def _legacy_sit(self, timeout_sec: float = 5.0) -> Optional[RobotCommandFeedback]:
        """Make the robot sit.

        Returns:
            outcome as command feedback.
        """
        response = self.sit.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return None
        return response.result

    def sit_async(
        self,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Treats sitting as an action instead of a service and asynchronously executes it

        Args:
          result_callback: A callback to process/handle the resulting feedback from executing the command
          feedback_callback: A callback to process/handle the feedback received during the execution of the command
          on_failure_callback: A callback to process/handle when the action fails for various reasons
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(command_builder.sit_command(), goal.command)
        return self.command_async(
            "sit_async",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.StandProtocol])
    def stand(self, **kwargs: Any) -> RobotCommandAction.Result:
        """Make the robot stand.

        See `SpotMobilityCommandProtocol.StandProtocol` for further reference.
        """

    @stand.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.stand.synchronously",
    )
    def _legacy_stand(self, timeout_sec: float = 5.0) -> Optional[RobotCommandFeedback]:
        """Make the robot stand.

        Returns:
            outcome as command feedback.
        """
        response = self.stand.synchronously(timeout_sec=timeout_sec, nothrow=True)
        if response is None:
            return None
        return response.result

    def stand_async(
        self,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Treats standing as an action instead of a service and asynchronously executes it

        Args:
          result_callback: A callback to process/handle the resulting feedback from executing the command
          feedback_callback: A callback to process/handle the feedback received during the execution of the command
          on_failure_callback: A callback to process/handle when the action fails for various reasons
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(command_builder.stand_command(), goal.command)
        return self.command_async(
            "stand_async",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.StandWithOrientationProtocol])
    def stand_with_orientation(
        self, gravbody_r_goal: SO3, body_height: float = 0.0, **kwargs: Any
    ) -> RobotCommandAction.Result:
        """Make the robot stand with target body height and orientation.

        See `SpotMobilityCommandProtocol.StandWithOrientationProtocol` for further reference.
        """

    @stand_with_orientation.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.stand_with_orientation.synchronously",
    )
    def _legacy_stand_with_orientation(
        self, gravbody_r_goal: SO3, body_height: float = 0.0, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Make the robot stand with target body height and orientation.

        Args:
            gravbody_r_goal: body frame orientation w.r.t. to a gravity aligned
            framed with yaw computed from the stance feet.
            body_height: body height, in meters, relative to the nominal stand height.

        Returns:
            outcome as command feedback.
        """
        response = self.stand_with_orientation.synchronously(
            gravbody_r_goal, body_height, timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return None
        return response.result

    def stand_with_orientation_async(
        self,
        gravbody_r_goal: SO3,
        body_height: float,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send the command to stand in a specific SO3 rotation

        Args:
            gravbody_r_goal: body frame orientation w.r.t. to a gravity aligned framed with yaw computed
            from the stance feet
            body_height: body height, in meters, relative to the nominal stand height
            result_callback: a callback to process/handle the resulting feedback from executing the command
            feedback_callback: a callback to process/handle the feedback received during the execution of the command
            on_failure_callback: a callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(command_builder.stand_with_orientation_command(gravbody_r_goal, body_height), goal.command)
        return self.command_async(
            "orient",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.WalkProtocol])
    def walk(
        self, frame_t_goal: SE2, body_height: float = 0.0, *, base_frame: Optional[FrameHint] = None, **kwargs: Any
    ) -> RobotCommandAction.Result:
        """Make the robot walk to an SE(2) pose.

        See `SpotMobilityCommandProtocol.WalkProtocol` for further reference.
        """

    @walk.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.walk.synchronously",
    )
    def _legacy_walk(
        self,
        frame_t_goal: SE2,
        body_height: float = 0.0,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Make the robot walk to an SE(2) pose.

        Args:
            frame_t_goal: pose to walk to, defined w.r.t. the given `base_frame`.
            body_height: body height, in meters, relative to the nominal stand height.
            base_frame: optional base frame, defaults to the VISION frame.

        Returns:
            outcome as command feedback.
        """
        response = self.walk.synchronously(
            frame_t_goal, body_height, base_frame=base_frame, timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return None
        return response.result

    # Alias
    walk_to = walk

    def walk_to_async(
        self,
        frame_t_goal: SE2,
        body_height: float = 0.0,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send the command to walk to a specific SE2 pose

        Args:
            frame_t_goal: pose to walk to, defined w.r.t. the given `base_frame`
            body_height: body height, in meters, relative to the nominal stand height
            base_frame: optional base frame, defaults to the VISION frame
            result_callback: a callback to process/handle the resulting feedback from executing the command
            feedback_callback: a callback to process/handle the feedback received during the execution of the command
            on_failure_callback: a callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(command_builder.walk_command(frame_t_goal, body_height, base_frame=base_frame), goal.command)
        return self.command_async(
            "walk_to",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.WalkTrajectoryProtocol])
    def walk_trajectory(
        self,
        frame_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        *,
        base_frame: Optional[FrameHint] = None,
        **kwargs: Any,
    ) -> RobotCommandAction.Result:
        """Make the robot walk along an SE(2) trajectory.

        See `SpotMobilityCommandProtocol.WalkTrajectoryProtocol` for further reference.
        """

    @walk_trajectory.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.walk_trajectory.synchronously",
    )
    def _legacy_walk_trajectory(
        self,
        frame_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Make the robot walk along an SE(2) trajectory.

        Args:
            frame_t_trajectory: time-parameterize sequence of poses, defined in the given `base_frame`.
            timestep_sec: time step to be used for trajectory sampling.
            interpolation: method to be used for trajectory interpolation.
            base_frame: optional base frame, defaults to the VISION frame.

        Returns:
            outcome as command feedback.
        """
        response = self.walk_trajectory.synchronously(
            frame_t_trajectory,
            timestep_sec,
            interpolation,
            base_frame=base_frame,
            timeout_sec=timeout_sec,
            nothrow=True,
        )
        if response is None:
            return None
        return response.result

    def walk_trajectory_async(
        self,
        frame_t_trajectory: SE2Trajectory,
        timestep_sec: float = 0.1,
        interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send the command to walk along a trajectory asynchronously

        Args:
            frame_t_trajectory: time-parameterize sequence of poses, defined in the given `base_frame`
            timestep_sec: time step to be used for trajectory sampling
            interpolation: method to be used for trajectory interpolation
            base_frame: optional base frame, defaults to the VISION frame
            result_callback: a callback to process/handle the resulting feedback from executing the command
            feedback_callback: a callback to process/handle the feedback received during the execution of the command
            on_failure_callback: a callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(
            command_builder.walk_trajectory_command(
                frame_t_trajectory, timestep_sec, interpolation, base_frame=base_frame
            ),
            goal.command,
        )
        return self.command_async(
            "walk_trajectory",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.WalkAndOrientProtocol])
    def walk_and_orient(
        self,
        frame_t_goal: SE3,
        body_height: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
        **kwargs: Any,
    ) -> RobotCommandAction.Result:
        """Make the robot walk to an SE(2) x SO(3) pose.

        See `SpotMobilityCommandProtocol.WalkAndOrientProtocol` for further reference.
        """

    @walk_and_orient.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.walk_and_orient.synchronously",
    )
    def _legacy_walk_and_orient(
        self,
        frame_t_goal: SE3,
        body_height: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Make the robot walk to an SE(3) pose.

        Args:
            frame_t_goal: pose to walk to, defined w.r.t. the given `base_frame`.
            body_height: body height, in meters, relative to the nominal stand height.
            base_frame: optional base frame, defaults to the VISION frame.

        Returns:
            outcome as command feedback.
        """
        response = self.walk_and_orient.synchronously(
            frame_t_goal, body_height, base_frame=base_frame, timeout_sec=timeout_sec, nothrow=True
        )
        if response is None:
            return None
        return response[-1].result

    def walk_and_orient_async(
        self,
        frame_t_goal: SE3,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[RobotComposeCommandHandle]:
        """Send the command to walk and rotate to a specific SE3 pose

        Args:
            frame_t_goal: pose to walk to, defined w.r.t. the given `base_frame`
            base_frame: optional base frame, defaults to the VISION frame
            result_callback: a callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: a callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: a callback to process/handle when any of actions fail for various reasons

        Returns:
            RobotComposeCommandHandle: A class to handle a list with multiple async RobotCommand (actions). The actions
                are sent sequentially to SpotDriver via an action client
        """
        walk_partial = functools.partial(
            self.walk_to_async,
            frame_t_goal.yaw_SE2(),
            base_frame=base_frame,
            body_height=frame_t_goal.z,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )
        stand_partial = functools.partial(
            self.stand_with_orientation_async,
            SO3(frame_t_goal),
            body_height=frame_t_goal.z,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )
        return RobotComposeCommandHandle([walk_partial, stand_partial], result_callback)

    def walk_to_fiducial(self, fiducial_id: int, fiducial_t_body: SE3) -> None:
        """Send the command to walk to a fiducial

        Args:
            fiducial_id: the fiducial to walk to
            fiducial_t_body: pose of fiducial w.r.t. the vision frame
        """
        vision_t_fiducial = self.lookup_a_tform_b(
            FrameHint.VISION.value, f"filtered_fiducial_{fiducial_id}", wait_for_frames=True
        )
        vision_t_body = vision_t_fiducial * fiducial_t_body
        self.walk_and_orient(frame_t_goal=vision_t_body, base_frame=FrameHint.VISION)

    @generalized_method(spec=GeneralizedMethodLike[SpotMobilityCommandProtocol.BodyGazeProtocol])
    def body_gaze(self, target: R3, frame: Optional[FrameHint] = None, **kwargs: Any) -> Any:
        """Make the robot point its body towards some point in space.

        See `SpotMobilityCommandProtocol.BodyGazeProtocol` for further reference.
        """

    def body_gaze_at(
        self,
        body_t_target: SE3,
        frame_t_body: Optional[SE3] = None,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
    ) -> Optional[RobotCommandFeedback]:
        """Aligns the x-axis of the body frame along the vector from the current body position
        to the target. If you don't provide vision_t_body, you must set the tf listener for spot.

        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            frame_t_body: SE3 transform from the base frame from to the current body position
            timeout_sec: A timeout for waiting on a response/result from the action server. The timeout arg is used
                on each command, so max duration is 2x timeout. Setting to None creates no timeout.
            base_frame: The frame in which the goal is specified. Defaults to VISION frame in the get_set_base_frame

        Returns:
            The feedback from the SpotDriver about the result of the commands if it was accepted, None otherwise
        """
        target = body_t_target.t
        if frame_t_body is not None:
            if base_frame is None:
                base_frame = FrameHint.VISION
            target = frame_t_body * target
            frame = base_frame
        else:
            frame = FrameHint.BODY
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(command_builder.body_gaze_command(np.squeeze(target), frame), goal.command)
        return self.command("gaze", goal, timeout_sec=timeout_sec)

    def body_gaze_at_async(
        self,
        body_t_target: SE3,
        frame_t_body: Optional[SE3] = None,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[RobotComposeCommandHandle]:
        """Aligns the x-axis of the body frame along the vector from the current body position
        to the target. If you don't provide vision_t_body, you must set the tf listener for spot.

        Args:
            body_t_target: SE3 transform from the current body position to the target you want to gaze at
            frame_t_body: SE3 transform from the base_frame from to the current body position
            base_frame: Frame the goal is specified in. Defaults to VISION frame in get_set_base_frame
            result_callback: A callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: A callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: A callback to process/handle when any of actions fail for various reasons
        """
        target = body_t_target.t
        if frame_t_body is not None:
            if base_frame is None:
                base_frame = FrameHint.VISION
            target = frame_t_body * target
            frame = base_frame
        else:
            frame = FrameHint.BODY
        goal = RobotCommandAction.Goal()
        command_builder = SpotMobilityCommandBuilder(self)
        convert(command_builder.body_gaze_command(np.squeeze(target), frame), goal.command)
        return self.command_async(
            "gaze_async",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def walk_with_velocity(
        self, body_velocity: Twist2, duration: Optional[float] = None, rate: Optional[float] = 50.0
    ) -> None:
        """Make the robot walk (i.e. move, turn) with a given body-frame velocity.

        Args:
            body_velocity: the desired body velocity
            (twist2 is a member of se2, which can be viewed as SE2 velocities, see Twist2.skewa)
            duration: the time (in seconds) to maintain the desired velocity
                setting to None (default) means the robot would walk for
                a minimum duration set by the "cmd_duration" parameter for in spot_ros2.
            rate: Rate (hz) at which velocity command is sent for the given duration.
                No effect if duration is None.
        """
        velocity_command = Twist()
        velocity_command.linear.x = body_velocity.v[0]
        velocity_command.linear.y = body_velocity.v[1]
        velocity_command.angular.z = body_velocity.w

        if duration is None:
            self._velocity_command_publisher.publish(velocity_command)
        else:
            clock = self.node.get_clock()
            rate_timer = self.node.create_rate(rate)
            start_time = clock.now()
            elapsed_time = clock.now() - start_time
            while elapsed_time.nanoseconds * 1e-9 < duration:
                self._velocity_command_publisher.publish(velocity_command)
                rate_timer.sleep()
                elapsed_time = clock.now() - start_time

    walk_velocity = walk_with_velocity
