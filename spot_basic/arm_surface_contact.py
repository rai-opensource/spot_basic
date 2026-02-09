# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Optional, Protocol, Union

from bosdyn.api import arm_command_pb2, geometry_pb2, gripper_command_pb2, trajectory_pb2
from bosdyn.api.arm_surface_contact_pb2 import ArmSurfaceContact as ArmSurfaceContactProto
from spatialmath import SE3
from spot_basic.base import SpotBaseProtocol
from spot_basic.command.helpers import se3_trajectory_to_proto
from spot_basic.state import ArmJointMap, FrameHint
from spot_basic.trajectory.se3_trajectory import SE3Trajectory
from spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
    spot_namespaced,
)
from spot_msgs.action import ArmSurfaceContact as ArmSurfaceContactAction
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedGuard, GeneralizedMethodLike, generalized_method

from bdai_ros.utilities.manual_conversions import se3_to_se3pose_proto


class PreferredJointConfigurationMode(IntEnum):
    DEFAULT = 0
    NONE = 1
    CURRENT = 2
    CUSTOM = 3


def _tuple_to_vec3(vec: tuple[float, float, float]) -> geometry_pb2.Vec3:
    return geometry_pb2.Vec3(x=vec[0], y=vec[1], z=vec[2])


def _build_arm_surface_contact_command(
    root_frame: FrameHint,
    task_traj_tool: SE3Trajectory | None,
    x_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
    y_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
    z_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
    press_force_percentage: tuple[float, float, float],
    pos_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
    ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
    wrist_t_tool: SE3 | None = None,
    root_t_task: SE3 | None = None,
    maximum_acceleration: float | None = None,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
    max_pos_tracking_error: float | None = None,
    max_rot_tracking_error: float | None = None,
    # By setting this to custom we ensure that the user won't forget to set this when they set
    # preferred_joint_configuration.  If preferred_joint_configuration isn't set, this will fall back to DEFAULT.
    preferred_joint_configuration_mode: PreferredJointConfigurationMode = PreferredJointConfigurationMode.CUSTOM,
    preferred_joint_configuration: ArmJointMap | None = None,
    xy_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
    z_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
    xy_to_z_cross_term_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
    bias_force_ewrt_body: tuple[float, float, float] | None = None,
    gripper_command: gripper_command_pb2.ClawGripperCommand.Request | None = None,
    follow_hand: bool = False,
) -> ArmSurfaceContactProto.Request:
    cmd = ArmSurfaceContactProto.Request(
        root_frame_name=root_frame.value,
        x_axis=x_axis_mode,
        y_axis=y_axis_mode,
        z_axis=z_axis_mode,
        press_force_percentage=_tuple_to_vec3(press_force_percentage),
        is_robot_following_hand=follow_hand,
    )
    if task_traj_tool is not None:
        cmd.pose_trajectory_in_task.CopyFrom(
            se3_trajectory_to_proto(
                task_traj_tool, pos_interpolation=pos_interpolation, ang_interpolation=ang_interpolation
            )
        )
    if wrist_t_tool is not None:
        cmd.wrist_tform_tool.CopyFrom(se3_to_se3pose_proto(wrist_t_tool))
    if root_t_task is not None:
        cmd.root_tform_task.CopyFrom(se3_to_se3pose_proto(root_t_task))
    if maximum_acceleration is not None:
        cmd.maximum_acceleration = maximum_acceleration
    if max_linear_velocity is not None:
        cmd.max_linear_velocity = max_linear_velocity
    if max_angular_velocity is not None:
        cmd.max_angular_velocity = max_angular_velocity
    if max_pos_tracking_error is not None:
        cmd.max_pos_tracking_error = max_pos_tracking_error
    if max_rot_tracking_error is not None:
        cmd.max_rot_tracking_error = max_rot_tracking_error
    # Confirmed with BD that this is how to set the proto correctly
    if preferred_joint_configuration_mode == PreferredJointConfigurationMode.NONE:
        # Don't do anything - not setting anything in the protobuf will result in Spot not having a preferred
        # configuration
        pass
    elif preferred_joint_configuration_mode == PreferredJointConfigurationMode.CURRENT:
        cmd.force_remain_near_current_joint_configration = True
    elif (
        preferred_joint_configuration_mode == PreferredJointConfigurationMode.CUSTOM
        and preferred_joint_configuration is not None
    ):
        cmd.preferred_joint_configuration.CopyFrom(
            arm_command_pb2.ArmJointPosition(
                sh0=preferred_joint_configuration.sh0,
                sh1=preferred_joint_configuration.sh1,
                el0=preferred_joint_configuration.el0,
                el1=preferred_joint_configuration.el1,
                wr0=preferred_joint_configuration.wr0,
                wr1=preferred_joint_configuration.wr1,
            )
        )
    else:  # This will result in Spot picking a default pose
        cmd.force_remain_near_current_joint_configuration = False
    if xy_admittance is not None:
        cmd.xy_admittance = xy_admittance
    if z_admittance is not None:
        cmd.z_admittance = z_admittance
    if xy_to_z_cross_term_admittance is not None:
        cmd.xy_to_z_cross_term_admittance = xy_to_z_cross_term_admittance
    if bias_force_ewrt_body is not None:
        cmd.bias_force_ewrt_body = _tuple_to_vec3(bias_force_ewrt_body)
    if gripper_command is not None:
        cmd.gripper_command = gripper_command
    return cmd


class SpotArmSurfaceContactProtocol(SpotBaseProtocol, Protocol):
    """Spot manipulation capabilities protocol."""

    class PerformProtocol(
        SpotActionableProtocol[
            ArmSurfaceContactAction.Goal, ArmSurfaceContactAction.Result, ArmSurfaceContactAction.Feedback
        ],
        Protocol,
    ):
        """ArmSurfaceContact request execution protocol."""

        def synchronous(
            self,
            root_frame: FrameHint,
            task_traj_tool: SE3Trajectory | None,
            x_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
            y_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
            z_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
            press_force_percentage: tuple[float, float, float],
            pos_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
            ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
            wrist_t_tool: SE3 | None = None,
            root_t_task: SE3 | None = None,
            maximum_acceleration: float | None = None,
            max_linear_velocity: float | None = None,
            max_angular_velocity: float | None = None,
            max_pos_tracking_error: float | None = None,
            max_rot_tracking_error: float | None = None,
            preferred_joint_configuration_mode: PreferredJointConfigurationMode = PreferredJointConfigurationMode.CUSTOM,  # noqa: E501 (black and ruff are fighting over the right thing to do here)
            preferred_joint_configuration: ArmJointMap | None = None,
            xy_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
            z_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
            xy_to_z_cross_term_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
            bias_force_ewrt_body: tuple[float, float, float] | None = None,
            gripper_command: gripper_command_pb2.ClawGripperCommand.Request | None = None,
            follow_hand: bool = False,
            *,
            feedback_callback: Optional[Callable[[ArmSurfaceContactAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> ArmSurfaceContactAction.Result:
            """ArmSurfaceContact manipulation request synchronously.

            Note: Much of these explanations comes from
            https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/arm_surface_contact.proto
            In cases where arguments are set to None, they will use the Boston Dynamics defaults.

            Args:
                root_frame: The root frame is used to set the optional task frame that all trajectories are specified
                    with respect to. If the optional task frame is left un-specified it defaults to the identity
                    transform and the root frame becomes the task frame.  This allows you to specify a task frame that
                    is not one of the known robot frames.
                task_traj_tool: A 3D pose trajectory for the tool expressed in the task frame, e.g. task_T_tool.
                    This pose trajectory is optional (set to None) if requesting a pure wrench at the end-effector,
                    otherwise required for position or mixed force/position end-effector requests.
                x_axis_mode: Force or position mode for x axis (in the task frame).  If set to position mode, the x
                    desireds will come from the trajectory.  Otherwise they will come from press_force_percentage[0].
                y_axis_mode: Force or position mode for y axis (in the task frame).  If set to position mode, the y
                    desireds will come from the trajectory.  Otherwise they will come from press_force_percentage[1].
                z_axis_mode: Force or position mode for z axis (in the task frame).  If set to position mode, the z
                    desireds will come from the trajectory.  Otherwise they will come from press_force_percentage[2].
                press_force_percentage: Amount of force to use on each axis, from 0 (no force) to 1.0 (maximum force),
                    can also be negative. Full range: [-1.0, 1.0]
                pos_interpolation: How to interpolate position between points on the trajectory
                ang_interpolation: How to interpolate angles between points on the trajectory.  Cubic choice uses
                    Euler method.
                wrist_t_tool: 	The tool pose relative to the wrist.  Defaults to a frame with it's origin slightly in
                    front of the gripper's palm plate aligned with wrist's orientation.
                root_t_task: An optional task frame (defaults to identity) that allows you to express the command in
                    an arbitrary frame.
                maximum_acceleration: Optional maximum acceleration magnitude of the end-effector.  Valid ranges (0, 20]
                max_linear_velocity: Optional maximum linear velocity magnitude of the end-effector. (m/s)
                max_angular_velocity: Optional maximum angular velocity magnitude of the end-effector. (rad/s)
                max_pos_tracking_error: Maximum allowable tracking error in meters of the tool frame from the desired
                    trajectory before the arm will stop moving and cancel the rest of the trajectory. When this limit
                    is exceeded, the hand will stay at the pose it was at when it exceeded the tracking error, and any
                    other part of the trajectory specified will be ignored.
                max_rot_tracking_error: Same as max_pos_tracking_error but in angular space
                preferred_joint_configuration_mode: This controls the robot's behavior if its arm is near singularity.
                    * DEFAULT: At singularity the robot will try to move towards a default pose that it chooses.
                    * NONE: At singularity the robot will try to minimize joint velocity (this is a good choice for
                        doing local moves where large joint motions should be minimized)
                    * CURRENT: At singularity the robot will try to move towards the current configuration (at the time
                        of the initial call)
                    * CUSTOM: At singularity the robot will try to move towards the custom configuration the user should
                        specify in preferred_joint_configuration.  If the mode is set to CUSTOM but no joint
                        configuration is specified, then the mode will act the same as DEFAULT.
                preferred_joint_configuration: When close to singularity the robot will try to move towards this
                    joint configuration ONLY IF preferred_joint_configuration_mode is set to CUSTOM.
                xy_admittance: By default, the robot will stop moving the arm when it encounters resistance.  You can
                    control that reaction to make the robot stiffer or less stiff for the xy axis by adjusting this
                    parameter.  Choices are:
                    * ADMITTANCE_SETTING_OFF: Will not stop moving when it encounters resistance
                    * ADMITTANCE_SETTING_NORMAL: Stops moving when it enconters a "normal" amount of resistance
                      (default)
                    * ADMITTANCE_SETTING_LOOSE: Stops moving when it encounters light resistance
                    * ADMITTANCE_SETTING_STIFF: Stops moving only when it encounters strong resistance
                    * ADMITTANCE_SETTING_VERY_STIFF: Stops moving only when it encounters very strong resistance
                z_admittance: Admittance setting for z axis (see xy_admittance for details on settings)
                xy_to_z_cross_term_admittance: Cross term that causes force in the XY axis cause movement in the z-axis.
                    The possible settings are the same as for xy and z but by default it is OFF.  Setting this value
                    will make the arm move in the negative Z-axis whenever it feels force in the XY axis.  This is
                    helpful for allowing the arm to have a good reaction to a bumpy surface.
                bias_force_ewrt_body: Specifies a force that the body should expect to feel.  This allows the robot to
                    "lean into" an external force.  Be careful using this field, because if you lie to the robot, it
                    can fall over.  Also note that this force is experssed in the body frame NOT the task frame.
                gripper_command: Gripper control while executing this command
                follow_hand: True to have the body follow the hand
                feedback_callback: Optional action feedback callback (note that there is not as of this writing
                    feedback sent out for this action as there is no feedback on the spot side).
                timeout_sec: Pptional action timeout, in seconds. If a timeout is specified and it expires, the
                    request will be cancelled and the call will raise. Note this timeout is local to the caller. It may
                    take some time for the action to be cancelled, if cancellation is not rejected by the action server.
                nothrow: True to return the result instead of throwing an exception if the action does not complete
                    successfully.

            Returns:
                The arm surface contact result (minimal as of this writing).

            Raises:
                ActionTimeout: if the arm surface contact request timed out.
                ActionRejected: if the arm surface contact request was not accepted.
                ActionCancelled: if the arm surface contact request was cancelled.
                ActionAborted: if the arm surface contact request was aborted.
                RuntimeError: if the robot has no arm.
            """

        def asynchronous(
            self,
            root_frame: FrameHint,
            trajectory_in_task: SE3Trajectory,
            x_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
            y_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
            z_axis_mode: ArmSurfaceContactProto.Request.AxisMode,
            press_force_percentage: tuple[float, float, float],
            pos_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
            ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
            wrist_t_tool: SE3 | None = None,
            root_t_task: SE3 | None = None,
            maximum_acceleration: float | None = None,
            max_linear_velocity: float | None = None,
            max_angular_velocity: float | None = None,
            max_pos_tracking_error: float | None = None,
            max_rot_tracking_error: float | None = None,
            force_remain_near_current_joint_configuration: bool | None = None,
            preferred_joint_configuration: ArmJointMap | None = None,
            xy_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
            z_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
            xy_to_z_cross_term_admittance: ArmSurfaceContactProto.Request.AdmittanceSetting | None = None,
            bias_force_ewrt_body: tuple[float, float, float] | None = None,
            gripper_command: gripper_command_pb2.ClawGripperCommand.Request | None = None,
            follow_hand: bool = False,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[ArmSurfaceContactAction.Result, ArmSurfaceContactAction.Feedback]:
            """Perform arm surface contact request asynchronously.

            See "synchronous" for an explanation of the arm surface contact arguments.

            Args:
                track_feedback: whether and how to track action feedback. Other than a boolean to enable or disable
                    tracking, a positive integer may be provided to cap feedback buffer size.

            Returns:
                The future arm surface contact outcome.

            Raises:
                RuntimeError: if the robot has no arm.
            """

    arm_surface_contact: GeneralizedMethodLike[PerformProtocol]


class SpotArmSurfaceContactMixin(SpotBaseProtocol):
    """Mixin offering Spot arm surface contact capabilities.

    Note that the robot must have an arm for any of these APIs to be available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # This creates the action client, which already supports synchronous and asynchronous calls and then tells it
        # to first call the build arm surface contact command.
        action_name = spot_namespaced(self.robot_name, "arm_surface_contact")
        implementation = SpotActionable(ArmSurfaceContactAction, action_name, node=self.node).compose(
            _build_arm_surface_contact_command
        )

        # This sets it up so the arm_surface_contact function here to first check whether the robot has an arm and if so
        # to pass the arguments on to the implementation
        SpotArmSurfaceContactMixin.arm_surface_contact.rebind(
            self,
            GeneralizedGuard(self.has_arm, implementation, f"{self.robot_name} has no arm, cannot manipulate"),
        )

    # This now takes in the same arguments as the protocol and you can call synchronous or asynchronous on it.
    @generalized_method(spec=GeneralizedMethodLike[SpotArmSurfaceContactProtocol.PerformProtocol])
    def arm_surface_contact(self, *args: Any, **kwargs: Any) -> Any:
        """Perform an arm surface contact with the robot.

        See `SpotArmSurfaceContactProtocol.PerformProtocol` for further reference.
        """
