# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import dataclasses
from typing import Any, Callable, Optional, Sequence, Union

from bdai_ros.utilities.manual_conversions import se3_to_se3pose_proto
from bdai_ros2_wrappers.callables import GeneralizedGuard, generalized_method
from bosdyn.api import (
    arm_command_pb2,
    basic_command_pb2,
    geometry_pb2,
    robot_command_pb2,
    trajectory_pb2,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.util import seconds_to_duration
from bosdyn_api_msgs.conversions import convert
from bosdyn_api_msgs.msg import RobotCommandFeedback
from numpy.typing import ArrayLike
from spatialmath import SE3, SpatialForce, Twist3, UnitQuaternion
from spatialmath.base.types import R3
from spatialmath.spatialvector import SpatialVector
from spot_msgs.action import RobotCommand as RobotCommandAction
from spot_utilities.spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_utilities.spot_basic.command.helpers import (
    timestamped_se3_poses_to_se3_trajectory,
    timestamped_spatialforces_to_wrench_protos,
)
from spot_utilities.spot_basic.state import ArmJointMap, FrameHint
from spot_utilities.spot_basic.utilities import SpotActionHandle


@dataclasses.dataclass(kw_only=True)
class AxisModeMap:
    """Stores the type of command for each axis in a Cartesian trajectory command."""

    x: arm_command_pb2.ArmCartesianCommand.Request.AxisMode
    y: arm_command_pb2.ArmCartesianCommand.Request.AxisMode
    z: arm_command_pb2.ArmCartesianCommand.Request.AxisMode
    rx: arm_command_pb2.ArmCartesianCommand.Request.AxisMode
    ry: arm_command_pb2.ArmCartesianCommand.Request.AxisMode
    rz: arm_command_pb2.ArmCartesianCommand.Request.AxisMode

    @classmethod
    def position_mode(cls) -> "AxisModeMap":
        return cls(
            x=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION,
            y=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION,
            z=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION,
            rx=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION,
            ry=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION,
            rz=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION,
        )

    @classmethod
    def force_mode(cls) -> "AxisModeMap":
        return cls(
            x=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE,
            y=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE,
            z=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE,
            rx=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE,
            ry=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE,
            rz=arm_command_pb2.ArmCartesianCommand.Request.AxisMode.AXIS_MODE_FORCE,
        )


class SpotArmCommandBuilder(SpotCommandBuilder):
    """A builder for Spot arm commands."""

    def arm_carry_command(self) -> robot_command_pb2.RobotCommand:
        """Build an arm carrying command.

        The arm will be retracted, useful when carrying objects.
        """
        return RobotCommandBuilder.arm_carry_command()

    def arm_stow_command(self) -> robot_command_pb2.RobotCommand:
        """Build an arm stowing command.

        The arm will be tucked away, easing mobility.
        """
        return RobotCommandBuilder.arm_stow_command()

    def arm_ready_command(self) -> robot_command_pb2.RobotCommand:
        """Build an arm readying (ie. unstowing) command.

        The arm will be deployed, allowing manipulation.
        """
        return RobotCommandBuilder.arm_ready_command()

    # Alias
    arm_unstow_command = arm_ready_command

    def arm_lock_command(self) -> robot_command_pb2.RobotCommand:
        """Build an arm locking command."""
        robot_command = robot_command_pb2.RobotCommand()
        arm_command = robot_command.synchronized_command.arm_command
        arm_command.arm_joint_move_command.trajectory.points.add()
        return robot_command

    def arm_gaze_command(self, point: R3, frame: FrameHint) -> robot_command_pb2.RobotCommand:
        """Build an arm gazing command.

        That is, a command to have the arm/hand camera gaze at a given point.

        Args:
            point: the point to gaze at, relative to `base_frame`.
            frame: the frame with respect to which `point` is defined.
        """
        return RobotCommandBuilder.arm_gaze_command(*point, frame_name=frame.value)

    def arm_joint_command(
        self,
        joint_positions: Union[ArmJointMap, list[float]],
        duration_sec: Optional[float] = None,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Build an arm command, taking each arm joint to a given position.

        Args:
            joint_positions: the arm joint positions, from shoulder to wrist.
            duration_sec: optional duration of the commanded motion.
        """
        if isinstance(joint_positions, list):
            joint_positions = ArmJointMap.from_list(joint_positions)
        trajectory_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0=joint_positions.sh0,
            sh1=joint_positions.sh1,
            el0=joint_positions.el0,
            el1=joint_positions.el1,
            wr0=joint_positions.wr0,
            wr1=joint_positions.wr1,
            time_since_reference_secs=duration_sec,
        )
        robot_command = robot_command_pb2.RobotCommand()
        arm_command = robot_command.synchronized_command.arm_command
        arm_command.arm_joint_move_command.trajectory.points.append(trajectory_point)
        if max_velocity is not None:
            arm_command.arm_joint_move_command.trajectory.maximum_velocity.value = max_velocity
        if max_acceleration is not None:
            arm_command.arm_joint_move_command.trajectory.maximum_acceleration.value = max_acceleration
        return robot_command

    def arm_joint_trajectory_command(
        self,
        joint_positions: Sequence[ArmJointMap],
        times: Sequence[float],
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        reference_time: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        max_velocity: Optional[float] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Build an arm command taking the arm along the given joint space trajectory.

        Note Spot robots impose a hard limit of 10 knots per trajectory.

        Args:
            joint_positions: sequence of arm joint positions to command the arm to move to.
            times: sequence of times, in seconds, to reach each trajectory knot relative to the
            trajectory reference time.
            joint_velocities: optional sequence of of arm joint velocities at each trajectory knot.
            reference_time: optional reference time for the trajectory, in seconds, defined w.r.t. the
            robot clock. Defaults to the robot clock epoch.
            max_acceleration: optional maximum acceleration allowed for any joint along the trajectory (rad/s^2).
            max_velocity: optional maximum velocity allowed for any joint along the trajectory (rad/s).

        Raises:
            ValueError: if the trajectory has more knots than supported by Spot (10 is the hard limit),
                        and if trajectory positions, velocities, and times do not have the same length.

        See https://github.com/boston-dynamics/spot-sdk/blob/b3e41278a7d6b22ad82a2ec4d7df6d36df029470/protos/bosdyn/api/arm_command.proto#L314
        and https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#armjointtrajectory for further reference.
        """
        if len(joint_positions) > 10:
            raise ValueError("An arm trajectory accepts only up to 10 knot points")

        if len(times) != len(joint_positions):
            raise ValueError("Must have the same number of joint positions and times")

        joint_positions_native = [position.to_list() for position in joint_positions]

        joint_velocities_native = None
        if joint_velocities is not None:
            if len(joint_velocities) != len(joint_positions):
                raise ValueError("Must have the same number of joint positions and velocities")
            joint_velocities_native = [velocity.to_list() for velocity in joint_velocities]

        return RobotCommandBuilder.arm_joint_move_helper(
            times=times,
            joint_positions=joint_positions_native,
            joint_velocities=joint_velocities_native,
            ref_time=reference_time,
            max_acc=max_acceleration,
            max_vel=max_velocity,
        )

    def arm_end_effector_cartesian_command(
        self,
        frame_t_goal: SE3,
        duration_sec: float,
        body_follow: bool = False,
        body_hip_assist: bool = False,
        body_yaw_assist: bool = False,
        disable_collisions: bool = False,
        base_frame: Optional[FrameHint] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Build an arm command taking its end effector to a given pose in cartesian space.

        Args:
            frame_t_goal: pose for the end effector to move to relative to the `base_frame`.
            duration_sec: duration of the commanded motion.
            body_follow: have the body follow the arm when it moves.
            body_hip_assist: whether or not to have the robot hips move to assist the end effector movements.
            body_yaw_assist: whether or not to have the robot yaw to assist the end effector movements.
            base_frame: optional base frame, defaults to the VISION frame.
        """
        frame_t_goal_q = UnitQuaternion(frame_t_goal)
        qx, qy, qz, qw = frame_t_goal_q.vec_xyzs
        if base_frame is None:
            base_frame = FrameHint.VISION
        arm_pose_command = RobotCommandBuilder.arm_pose_command(
            x=frame_t_goal.x,
            y=frame_t_goal.y,
            z=frame_t_goal.z,
            qw=qw,
            qx=qx,
            qy=qy,
            qz=qz,
            frame_name=base_frame.value,
            seconds=duration_sec,
        )
        if body_hip_assist or body_yaw_assist:
            body_control_params = spot_command_pb2.MobilityParams(
                body_control=spot_command_pb2.BodyControlParams(
                    body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
                        enable_hip_height_assist=body_hip_assist, enable_body_yaw_assist=body_yaw_assist
                    )
                )
            )
            stand_command = RobotCommandBuilder.synchro_stand_command(params=body_control_params)
        elif body_follow:
            stand_command = RobotCommandBuilder.follow_arm_command()
        else:
            stand_command = RobotCommandBuilder.synchro_stand_command()
        return RobotCommandBuilder.build_synchro_command(stand_command, arm_pose_command)

    def arm_end_effector_cartesian_trajectory_command(
        self,
        root_frame: FrameHint,
        axis_mode_map: AxisModeMap,
        *,
        timestamps_between_poses: Optional[ArrayLike] = None,
        timestamps_between_wrenches: Optional[ArrayLike] = None,
        pose_vecs: Optional[Sequence[SE3]] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        wrench_vecs: Optional[Sequence[SpatialForce]] = None,
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        max_acceleration: Optional[float] = None,
        max_linear_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        max_pos_tracking_error: Optional[float] = None,
        max_rot_tracking_error: Optional[float] = None,
        stay_near_current_joint_configuration: Optional[bool] = None,
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        reference_time: Optional[float] = None,
        positional_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        rotational_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        body_follow_arm: bool = False,
    ) -> robot_command_pb2.RobotCommand:
        """Build an arm command taking its end effector along a given trajectory in Cartesian space.

        This Cartesian space trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be a hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to AXIS_MODE_POSITION
        to avoid undefined behaviour.

        Args:
            root_frame: root frame with respect to which the trajectory is defined.
            axis_mode_map: control mode per axis (either position or force).
            timestamps_between_poses: timestamps, in seconds, to reach each knot pose
            relative to the trajectory reference time.
            timestamps_between_wrenches: timestamps, in seconds, to apply each knot wrench
            relative to the trajectory reference time.
            pose_vecs: sequence of poses for the end effector to pass through along its trajectory.
            vel_vecs: sequence of twists for the end effector to command along its trajectory.
            wrench_vecs: sequence of wrenches for the end effector to apply along its trajectory.
            wrist_t_tool: optional tool pose relative to the arm wrist.
            root_t_task: optional transform between root and task frames.
            max_acceleration: optional maximum acceleration of the end effector (m/s^2).
            max_linear_velocity: optional maximum linear velocity of the end effector (m/s).
            max_angular_velocity: optional maximum angular velocity of the end effector (rad/s).
            max_pos_tracking_error: optional maximum deviation between set and tracked end effector positions.
            max_rot_tracking_error: optional maximum deviation between set and tracked end effector orientations.
            stay_near_current_joint_configuration: forces the system to look for solutions near the current arm
            joint configuration.
            preferred_joint_configuration: optional, preferred joint configuration for the arm to end up in.
            reference_time: optional reference time for the trajectory, in seconds, defined w.r.t. the robot clock.
            Defaults to the robot clock epoch.
            positional_interpolation: optional positional interpolation method. Defaults to cubic.
            rotational_interpolation: optional rotational interpolation method. Defaults to cubic.
            body_follow_arm: have the body follow the arm when it moves.

        See https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        and https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py for further
        reference.
        """
        robot_command = robot_command_pb2.RobotCommand()
        arm_command = robot_command.synchronized_command.arm_command
        arm_cartesian_command = arm_command.arm_cartesian_command

        arm_cartesian_command.root_frame_name = root_frame.value

        if wrist_t_tool is not None:
            arm_cartesian_command.wrist_tform_tool.CopyFrom(se3_to_se3pose_proto(wrist_t_tool))

        if root_t_task is not None:
            arm_cartesian_command.root_tform_task.CopyFrom(se3_to_se3pose_proto(root_t_task))

        if pose_vecs is not None:
            if timestamps_between_poses is None:
                raise ValueError("timestamps between poses must be specified if poses are")
            arm_trajectory = timestamped_se3_poses_to_se3_trajectory(
                timestamps_between_points=timestamps_between_poses,
                pose_vecs=pose_vecs,
                vel_vecs=vel_vecs,
                pose_interpolation=positional_interpolation,
                ang_interpolation=rotational_interpolation,
                reference_time=reference_time,
            )
            arm_cartesian_command.pose_trajectory_in_task.CopyFrom(arm_trajectory)

        if wrench_vecs is not None:
            if timestamps_between_wrenches is None:
                raise ValueError("timestamps between wrenches must be specified if wrenches are")
            wrench_trajectory = timestamped_spatialforces_to_wrench_protos(
                timestamps_between_wrenches=timestamps_between_wrenches,
                wrench_vecs=wrench_vecs,
                reference_time=reference_time,
            )
            arm_cartesian_command.wrench_trajectory_in_task.CopyFrom(wrench_trajectory)

        if body_follow_arm:
            mobility_command = robot_command.synchronized_command.mobility_command
            mobility_command.follow_arm_request.CopyFrom(basic_command_pb2.FollowArmCommand.Request())

        arm_cartesian_command.x_axis = axis_mode_map.x
        arm_cartesian_command.y_axis = axis_mode_map.y
        arm_cartesian_command.z_axis = axis_mode_map.z
        arm_cartesian_command.rz_axis = axis_mode_map.rx
        arm_cartesian_command.ry_axis = axis_mode_map.ry
        arm_cartesian_command.rz_axis = axis_mode_map.rz

        if max_acceleration is not None:
            arm_cartesian_command.maximum_acceleration.value = max_acceleration

        if max_linear_velocity is not None:
            arm_cartesian_command.max_linear_velocity.value = max_linear_velocity

        if max_angular_velocity is not None:
            arm_cartesian_command.max_angular_velocity.value = max_angular_velocity

        if stay_near_current_joint_configuration is not None:
            arm_cartesian_command.force_remain_near_current_joint_configuration = stay_near_current_joint_configuration

        if preferred_joint_configuration is not None:
            arm_cartesian_command.preferred_joint_configuration = preferred_joint_configuration

        return robot_command

    DEFAULT_STIFFNESS_FOR_IMPEDANCE_CONTROL = SpatialVector([500, 500, 500, 60, 60, 60])
    DEFAULT_DAMPING_FOR_IMPEDANCE_CONTROL = SpatialVector([2.5, 2.5, 2.5, 1.0, 1.0, 1.0])

    def arm_end_effector_impedance_command(
        self,
        trajectory: list[tuple[SE3, float]],
        root_frame: Optional[FrameHint] = None,
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Build arm comand taking its end effector along a trajectory using hybrid position/impedance control.

        Args:
            root_frame: root frame with respect to which the trajectory is defined.
            root_t_task: the transform from the root frame to the task frame.
            Note the latter need not be a well known, named frame.
            wrist_t_tool: the tool pose relative to the arm wrist frame.
            It can also be thought of as the "remote center" frame.
            trajectory: sequence of trajectory waypoints, defined as tuples of tool pose
            relative to the task frame and duration to slerp over.
            stiffness: optional position/stiffness gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame.
            damping: optional derivative/damping gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame.
            wrench_nm: optional feedforward wrench for x, y, z, torque_x, torque_y, torque_z in the task frame.

        Note stiffness and damping gains default to their maximum values.
        This is in the task frame, so these were probably empirically obtained.
        See https://github.com/boston-dynamics/spot-sdk/blob/8f781a87b8816168ec59415db853934f41568620/python/examples/arm_impedance_control/arm_impedance_control.py#L139-L145.
        """

        robot_command = robot_command_pb2.RobotCommand()
        arm_impedance_command = robot_command.synchronized_command.arm_command.arm_impedance_command

        if root_frame is None:
            root_frame = FrameHint.VISION
        arm_impedance_command.root_frame_name = root_frame.value

        if wrist_t_tool is not None:
            arm_impedance_command.wrist_tform_tool.CopyFrom(se3_to_se3pose_proto(wrist_t_tool))

        if root_t_task is not None:
            arm_impedance_command.root_tform_task.CopyFrom(se3_to_se3pose_proto(root_t_task))

        if stiffness is None:
            stiffness = self.DEFAULT_STIFFNESS_FOR_IMPEDANCE_CONTROL
        arm_impedance_command.diagonal_stiffness_matrix.CopyFrom(geometry_pb2.Vector(values=stiffness.A.tolist()))

        if damping is None:
            damping = self.DEFAULT_DAMPING_FOR_IMPEDANCE_CONTROL
        arm_impedance_command.diagonal_damping_matrix.CopyFrom(geometry_pb2.Vector(values=damping.A.tolist()))

        if wrench_nm is None:
            wrench_nm = SpatialForce([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        arm_impedance_command.feed_forward_wrench_at_tool_in_desired_tool.force.x = wrench_nm.A[0]
        arm_impedance_command.feed_forward_wrench_at_tool_in_desired_tool.force.y = wrench_nm.A[1]
        arm_impedance_command.feed_forward_wrench_at_tool_in_desired_tool.force.z = wrench_nm.A[2]
        arm_impedance_command.feed_forward_wrench_at_tool_in_desired_tool.torque.x = wrench_nm.A[3]
        arm_impedance_command.feed_forward_wrench_at_tool_in_desired_tool.torque.y = wrench_nm.A[4]
        arm_impedance_command.feed_forward_wrench_at_tool_in_desired_tool.torque.z = wrench_nm.A[5]

        # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
        # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
        commanded_trajectory = arm_impedance_command.task_tform_desired_tool
        for pose, duration in trajectory:
            point = commanded_trajectory.points.add()
            point.time_since_reference.CopyFrom(seconds_to_duration(duration))
            point.pose.CopyFrom(se3_to_se3pose_proto(pose))

        return robot_command

    def arm_zero_g_command(self) -> robot_command_pb2.RobotCommand:
        """Build an arm command that puts it in a gravity compensated mode such that
        the arm 'renders' no apparent weight and low damping.

        Useful for allowing a person to move the arm around freely. Note it does not
        compensate for additional unmodeled mass (e.g. an object held by the gripper).
        """
        stationary_trajectory = [(SE3(), 1.0)]
        return self.arm_end_effector_impedance_command(
            trajectory=stationary_trajectory,
            root_frame=FrameHint.BODY,
            root_t_task=SE3.Tz(0.3),
            wrist_t_tool=SE3(),
            stiffness=SpatialVector([0, 0, 0, 0, 0, 0]),
            damping=SpatialVector([0.25, 0.25, 0.25, 0.10, 0.10, 0.10]),
            wrench_nm=SpatialForce([0, 0, 0, 0, 0, 0]),
        )


class SpotArmCommandMixin(SpotCommandProtocol):
    """Mixin that eases Spot arm command execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotArmCommandBuilder(self)

        do_stow_arm = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_stow_command),
            f"{self.robot_name} has no arm, nothing to stow",
        )
        SpotArmCommandMixin.stow_arm.rebind(self, do_stow_arm)

        do_ready_arm = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_ready_command),
            f"{self.robot_name} has no arm, nothing to unstow",
        )
        SpotArmCommandMixin.ready_arm.rebind(self, do_ready_arm)

        do_carry_arm = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_carry_command),
            f"{self.robot_name} has no arm, nothing to carry",
        )
        SpotArmCommandMixin.carry_arm.rebind(self, do_carry_arm)

        do_lock_arm = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_lock_command),
            f"{self.robot_name} has no arm, nothing to lock",
        )
        SpotArmCommandMixin.lock_arm.rebind(self, do_lock_arm)

        do_arm_gaze = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_gaze_command),
            f"{self.robot_name} has no arm, cannot gaze",
        )
        SpotArmCommandMixin.arm_gaze.rebind(self, do_arm_gaze)

        do_move_arm = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_joint_command),
            f"{self.robot_name} has no arm, nothing to move",
        )
        SpotArmCommandMixin.move_arm.rebind(self, do_move_arm)

        do_move_arm_along_trajectory = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_joint_trajectory_command),
            f"{self.robot_name} has no arm, nothing to move",
        )
        SpotArmCommandMixin.move_arm_along_trajectory.rebind(self, do_move_arm_along_trajectory)

        do_move_end_effector = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_end_effector_cartesian_command),
            f"{self.robot_name} has no arm, no end effector to move",
        )
        SpotArmCommandMixin.move_end_effector.rebind(self, do_move_end_effector)

        do_move_end_effector_along_trajectory = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_end_effector_cartesian_trajectory_command),
            f"{self.robot_name} has no arm, no end effector to move",
        )
        SpotArmCommandMixin.move_end_effector_along_trajectory.rebind(self, do_move_end_effector_along_trajectory)

        do_move_end_effector_with_controlled_impedance = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_end_effector_impedance_command),
            f"{self.robot_name} has no arm, no end effector to move",
        )
        SpotArmCommandMixin.move_end_effector_with_controlled_impedance.rebind(
            self, do_move_end_effector_with_controlled_impedance
        )

        do_arm_zero_g = GeneralizedGuard(
            self.has_arm,
            self.execute.compose(command_builder.arm_zero_g_command),
            f"{self.robot_name} has no arm, cannot emulate zero g motion",
        )
        SpotArmCommandMixin.arm_zero_g.rebind(self, do_arm_zero_g)

    @generalized_method(transitional=True)
    def stow_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Stow the robot arm."""
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_stow_command(), goal.command)
        return self.command("stow_arm", goal, timeout_sec)

    def stow_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to stow the arm asynchronously.

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_stow_command(), goal.command)
        return self.command_async(
            "stow_arm",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def ready_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Ready (ie. unstow) the robot arm."""
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_ready_command(), goal.command)
        return self.command("ready_arm", goal, timeout_sec)

    # Alias
    unstow_arm = ready_arm

    def ready_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Send a command to move the arm to the ready position asynchronously.

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_ready_command(), goal.command)
        return self.command_async(
            "ready_arm",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    # Alias
    unstow_arm_async = ready_arm_async

    @generalized_method(transitional=True)
    def carry_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Carry the robot arm."""
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_carry_command(), goal.command)
        return self.command("carry_arm", goal, timeout_sec)

    def carry_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Move the arm to the carry position asynchronously.

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_carry_command(), goal.command)
        return self.command_async(
            "carry_arm",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def lock_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommandFeedback]:
        """Lock the robot arm joints in place."""
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_lock_command(), goal.command)
        return self.command("lock_arm", goal, timeout_sec)

    def lock_arm_async(
        self,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Locks the arm joint in place asynchronously.

        Args:
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise.
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_lock_command(), goal.command)
        return self.command_async(
            "lock_arm",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def arm_gaze(
        self, point: R3, frame: FrameHint, timeout_sec: Optional[float] = None
    ) -> Optional[RobotCommandFeedback]:
        """Gaze the robot arm (and hand camera) at a point in space.

        Args:
            point: the point to gaze at, relative to `base_frame`.
            frame: the frame with respect to which `point` is defined.
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_gaze_command(point, frame), goal.command)
        return self.command("arm_gaze", goal, timeout_sec)

    arm_gaze_at = arm_gaze

    def arm_gaze_at_async(
        self,
        point: R3,
        frame: FrameHint,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Gaze the arm/hand camera at a point relative to a frame.

        Args:
            point: A 3d vector representing a point, relative to a frame.
            frame: A frame hint representing one of spot's default frames, like vision or body.
                The point is relative to this frame.
            result_callback: A callback to process/handle the resulting feedback from executing each of the commands
            feedback_callback: A callback to process/handle the feedback received during the execution of any of the
                commands
            on_failure_callback: A callback to process/handle when any of actions fail for various reasons
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        convert(command_builder.arm_gaze_command(point, frame), goal.command)
        return self.command_async(
            "arm_gaze",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def move_arm_along_trajectory(
        self,
        joint_angles: Sequence[ArmJointMap],
        times: Sequence[float],
        *,
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        ref_time: Optional[float] = None,
        max_acc: Optional[float] = None,
        max_vel: Optional[float] = None,
        timeout_sec: float = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Move the robot arm along a trajectory in joint (i.e. configuration) space.

        Args:
            joint_angles: List of joint angles to command the arm to move to. Each set of joint
                          angles must be size 6, as there are 6 joints. Max 10 points per BD
            times: List of times we want to reach each joint position relative to the reference time
            joint_velocities: List of joint velocities for the arm to achieve along the trajectory.
            ref_time: reference time to begin trajectory, relative to the robot clock.
            max_acc: maximum acceleration allowed for any joint along the trajectory (rad/s^2).
            max_vel: maximum velocity allowed for any joint along the trajectory (rad/s).
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_joint_trajectory_command(
            times=times,
            joint_positions=joint_angles,
            joint_velocities=joint_velocities,
            reference_time=ref_time,
            max_acceleration=max_acc,
            max_velocity=max_vel,
        )
        convert(command, goal.command)
        return self.command("move_arm_joint_trajectory", goal, timeout_sec)

    move_arm_joint_trajectory = move_arm_along_trajectory

    def move_arm_joint_trajectory_async(
        self,
        joint_angles: Sequence[ArmJointMap],
        times: Sequence[float],
        *,
        joint_velocities: Optional[Sequence[ArmJointMap]] = None,
        ref_time: Optional[float] = None,
        max_acc: Optional[float] = None,
        max_vel: Optional[float] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Move the joint angles of the arm along a specified trajectory.

        Args:
            joint_angles: List of joint angles to command the arm to move to. Each set of joint
                          angles must be size 6, as there are 6 joints. Max 10 points per BD
            times: List of times we want to reach each joint position relative to the reference time
            joint_velocities: List of joint velocities for the arm to achieve along the trajectory.
            ref_time: reference time to begin trajectory, relative to the robot clock.
            max_acc: maximum acceleration allowed for any joint along the trajectory (rad/s^2)
            max_vel: maximum velocity allowed for any joint along the trajectory (rad/s)
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        Raises:
            Exception if too many points

        https://github.com/boston-dynamics/spot-sdk/blob/b3e41278a7d6b22ad82a2ec4d7df6d36df029470/protos/bosdyn/api/arm_command.proto#L314
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#armjointtrajectory
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_joint_trajectory_command(
            times=times,
            joint_positions=joint_angles,
            joint_velocities=joint_velocities,
            reference_time=ref_time,
            max_acceleration=max_acc,
            max_velocity=max_vel,
        )
        convert(command, goal.command)
        return self.command_async(
            "move_arm_joint_trajectory_async",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def move_arm(
        self,
        joint_angles: Optional[ArmJointMap],
        *,
        time_since_reference_secs: Optional[float] = None,
        timeout_sec: float = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Move robot arm to a given joint configuration.

        Args:
            joint_angles: the joint angles of the robot, in order from shoulder to wrist.
                          If the joint angles is None, then the arm is locked in place.
            time_since_reference_secs: duration the command needs to execute with respect to the reference
            timeout_sec: seconds between receive and execution until the command times out

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        if joint_angles is not None:
            command = command_builder.arm_joint_command(
                joint_positions=joint_angles, duration_sec=time_since_reference_secs
            )
        else:
            command = command_builder.arm_lock_command()
        convert(command, goal.command)
        return self.command("move_arm_to_joint_angles", goal, timeout_sec)

    move_arm_to_joint_angles = move_arm

    def move_arm_to_joint_angles_async(
        self,
        joint_angles: Optional[ArmJointMap],
        *,
        time_since_reference_secs: Optional[float] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Moves the arm to a specific set of joint angles

        Args:
            joint_angles: the joint angles of the robot, in order from shoulder to wrist.
                          If the joint angles is None, then the arm is locked in place.
            time_since_reference_secs: duration the command needs to execute with respect to the reference
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        if joint_angles is not None:
            command = command_builder.arm_joint_command(
                joint_positions=joint_angles, duration_sec=time_since_reference_secs
            )
        else:
            command = command_builder.arm_lock_command()
        convert(command, goal.command)
        return self.command_async(
            "move_arm_to_joint_angles_async",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def move_end_effector(
        self,
        frame_t_goal: SE3,
        move_duration_sec: float = 5.0,
        body_follow: bool = False,
        body_hip_assist: bool = False,
        body_yaw_assist: bool = False,
        timeout_sec: Optional[float] = None,
        *,
        base_frame: Optional[FrameHint] = None,
        disable_collisions: bool = False,
    ) -> Optional[RobotCommandFeedback]:
        """Move robot arm end effector to a given pose in Cartesian space.

        Args:
            frame_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            body_follow: If True, the robot's body would automatically follow the arm to achieve the goal
            body_hip_assist: Whether or not to have the robot hips move to assist the end effector movements
            body_yaw_assist: Whether or not to have the robot yaw to assist the end effector movements
            timeout_sec: A timeout for waiting on a response/result from the action server. Setting to None creates no
                timeout
            base_frame: Frame the goal is specified in. Defaults to VISION frame in get_set_base_frame

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        See:
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L1224
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_end_effector_cartesian_command(
            frame_t_goal,
            move_duration_sec,
            base_frame=base_frame,
            body_follow=body_follow,
            body_hip_assist=body_hip_assist,
            body_yaw_assist=body_yaw_assist,
            disable_collisions=disable_collisions,
        )
        convert(command, goal.command)
        return self.command("move_end_effector_cartesian", goal, timeout_sec)

    move_end_effector_cartesian = move_end_effector

    def move_end_effector_cartesian_async(
        self,
        frame_t_goal: SE3,
        move_duration_sec: float = 5.0,
        body_follow: bool = False,
        body_hip_assist: bool = False,
        body_yaw_assist: bool = False,
        *,
        base_frame: Optional[FrameHint] = None,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
        disable_collisions: bool = False,
    ) -> Optional[SpotActionHandle]:
        """Send command to move the end effector to a specific pose asynchronously

        Args:
            frame_t_goal: A pose for the end effector to move to relative to the base frame
            move_duration_sec: The duration of the move
            body_follow: If True, the robot's body would automatically follow the arm to achieve the goal
            body_hip_assist: Whether or not to have the robot hips move to assist the end effector movements
            body_yaw_assist: Whether or not to have the robot yaw to assist the end effector movements
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons
            base_frame: Frame the goal is specified in. Defaults to VISION frame in get_set_base_frame

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_end_effector_cartesian_command(
            frame_t_goal,
            move_duration_sec,
            base_frame=base_frame,
            body_follow=body_follow,
            body_hip_assist=body_hip_assist,
            body_yaw_assist=body_yaw_assist,
            disable_collisions=disable_collisions,
        )
        convert(command, goal.command)
        return self.command_async(
            "move_end_effector_cartesian",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def move_end_effector_along_trajectory(
        self,
        root_frame_name: str,
        axis_mode_map: AxisModeMap,
        *,
        timestamps_between_poses: Optional[ArrayLike] = None,
        timestamps_between_wrenches: Optional[ArrayLike] = None,
        pose_vecs: Optional[Sequence[SE3]] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        wrench_vecs: Optional[Sequence[SpatialForce]] = None,
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        max_acceleration: Optional[float] = None,
        max_linear_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        max_pos_tracking_error: Optional[float] = None,
        max_rot_tracking_error: Optional[float] = None,
        stay_near_current_joint_angles: Optional[bool] = None,
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        reference_time: Optional[float] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        body_follow_arm: bool = False,
        timeout_sec: Optional[float] = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Move robot arm end effector along a trajectory in Cartesian space.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid unexpected behaviour.

        Args:
            root_frame_name: task frame that all trajectories are defined with respect to.
            axis_mode_map: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
            timestamps_between_poses: array of timestamps of when each pose point is traversed (seconds)
            timestamps_between_wrenches: array of timestamps of when each wrench point is traversed (seconds)
            pose_vecs: vector of SE3 poses for the end effector to pass through described by the spatialmath library
            vel_vecs: vector of velocities to pass through at each point
            wrench_vecs: vector of SpatialForces for the end effector to output in the spatialmath library
            wrist_t_tool: tool pose relative to the parent link (wrist)
            root_t_task: transform between root and task frame
            max_acceleration: max acceleration of end effector (m/s^2)
            max_linear_velocity: max linear velocity of end effector (m/s)
            max_angular_velocity: max radial velocity of end effector (rad/s)
            max_pos_tracking_error: max deviation between inputted and tracked end effector position
            max_rot_tracking_error: max deviation between inputted and tracked end effector orientations
            stay_near_current_joint_angles: forces the system to look for solutions near the current joint angles
            preferred_joint_configuration: the preferred joint configuration for the arm to end up in.
            reference_time: time with respect to the robot clock of when to reference when trajectories start and end
            pose_interpolation: positional interpolation method. Defaults to cubic.
            ang_interpolation: angular interpolation method. Defaults to cubic.
            body_follow_arm: have the body follow the arm when it moves
            timeout_sec: seconds it will attempt to run the command before timing out and returning

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_end_effector_cartesian_trajectory_command(
            root_frame=FrameHint(root_frame_name),
            axis_mode_map=axis_mode_map,
            timestamps_between_poses=timestamps_between_poses,
            timestamps_between_wrenches=timestamps_between_wrenches,
            pose_vecs=pose_vecs,
            vel_vecs=vel_vecs,
            wrench_vecs=wrench_vecs,
            wrist_t_tool=wrist_t_tool,
            root_t_task=root_t_task,
            max_acceleration=max_acceleration,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_pos_tracking_error=max_pos_tracking_error,
            max_rot_tracking_error=max_rot_tracking_error,
            stay_near_current_joint_configuration=stay_near_current_joint_angles,
            preferred_joint_configuration=preferred_joint_configuration,
            reference_time=reference_time,
            positional_interpolation=pose_interpolation,
            rotational_interpolation=ang_interpolation,
            body_follow_arm=body_follow_arm,
        )
        convert(command, goal.command)
        return self.command("move_arm_along_cartesian_trajectory", goal, timeout_sec)

    move_end_effector_cartesian_trajectory = move_end_effector_along_trajectory

    def move_end_effector_cartesian_trajectory_async(
        self,
        root_frame_name: str,
        axis_mode_map: AxisModeMap,
        *,
        timestamps_between_poses: Optional[ArrayLike] = None,
        timestamps_between_wrenches: Optional[ArrayLike] = None,
        pose_vecs: Optional[Sequence[SE3]] = None,
        vel_vecs: Optional[Sequence[Twist3]] = None,
        wrench_vecs: Optional[Sequence[SpatialForce]] = None,
        wrist_t_tool: Optional[SE3] = None,
        root_t_task: Optional[SE3] = None,
        max_acceleration: Optional[float] = None,
        max_linear_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        max_pos_tracking_error: Optional[float] = None,
        max_rot_tracking_error: Optional[float] = None,
        stay_near_current_joint_angles: Optional[bool] = None,
        preferred_joint_configuration: Optional[arm_command_pb2.ArmJointPosition] = None,
        reference_time: Optional[float] = None,
        pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
        ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
        body_follow_arm: bool = False,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Return a goal corresponding to the arm following a Cartesian trajectory.

        This Cartesian trajectory can exist in the kinematic SE3 space or be in the force-torque
        wrench space. It can also be hybrid of the two, but any axis can only exist in either the
        pose or wrench mode. Modes that plan on being fixed in space should be set to
        AXIS_MODE_POSITION to avoid unexpected behaviour.

        Args:
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            root_frame_name: optional task frame that all trajectories are defined with respect to.
            axis_mode_map: dataclass in robot_command_helpers.py which maps axis to control modes (position or force).
            timestamps_between_poses: array of timestamps of when each pose point is traversed (seconds)
            timestamps_between_wrenches: array of timestamps of when each wrench point is traversed (seconds)
            pose_vecs: vector of (spatialmath) SE3 poses (wrt the task frame) for the end effector to pass through
            vel_vecs: vector of velocities to pass through at each point
            wrench_vecs: vector of SpatialForces for the end effector to output in the spatialmath library
            wrist_t_tool: tool pose relative to the parent link (wrist)
            root_t_task: transform between root and task frame
            max_acceleration: max acceleration of end effector (m/s^2)
            max_linear_velocity: max linear velocity of end effector (m/s)
            max_angular_velocity: max radial velocity of end effector (rad/s)
            max_pos_tracking_error: max deviation between inputted and tracked end effector position
            max_rot_tracking_error: max deviation between inputted and tracked end effector orientations
            stay_near_current_joint_angles: forces the system to look for solutions near the current joint angles
            preferred_joint_configuration: the preferred joint configuration for the arm to end up in.
            reference_time: time with respect to the robot clock of when to reference when trajectories start and end
            pose_interpolation: positional interpolation method. Defaults to cubic.
            ang_interpolation: angular interpolation method. Defaults to cubic.
            body_follow_arm: have the body follow the arm when it moves
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons
        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise

        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#armcartesiancommand-request
        https://github.com/bdaiinstitute/spatialmath-python/blob/master/spatialmath/pose3d.py
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_end_effector_cartesian_trajectory_command(
            root_frame=FrameHint(root_frame_name),
            axis_mode_map=axis_mode_map,
            timestamps_between_poses=timestamps_between_poses,
            timestamps_between_wrenches=timestamps_between_wrenches,
            pose_vecs=pose_vecs,
            vel_vecs=vel_vecs,
            wrench_vecs=wrench_vecs,
            wrist_t_tool=wrist_t_tool,
            root_t_task=root_t_task,
            max_acceleration=max_acceleration,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_pos_tracking_error=max_pos_tracking_error,
            max_rot_tracking_error=max_rot_tracking_error,
            stay_near_current_joint_configuration=stay_near_current_joint_angles,
            preferred_joint_configuration=preferred_joint_configuration,
            reference_time=reference_time,
            positional_interpolation=pose_interpolation,
            rotational_interpolation=ang_interpolation,
            body_follow_arm=body_follow_arm,
        )
        convert(command, goal.command)
        return self.command_async(
            "move_arm_along_cartesian_trajectory",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    @generalized_method(transitional=True)
    def move_end_effector_with_controlled_impedance(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
        timeout_sec: Optional[float] = 5.0,
    ) -> Optional[RobotCommandFeedback]:
        """Move robot arm end effector along a trajectory while controlling arm impedance.

        This command combines variable stiffness/damping position control with a feedforward force control.

        Args:
            root_frame_name: name of root frame as a string (without 'RobotName/' as the prefix)
            root_t_task: the transform from the root frame to the "task frame" (this doesn't need a name)
            wrist_t_tool: The tool pose relative to the parent link (arm_link_wr1).
                This can also be thought of as the "remote center" frame.
            trajectory: Discrete waypoints of where we want the tool to be relative to the task frame
                and the duration of time to slerp over.
            stiffness: the position/stiffness gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            damping: the derivative/damping gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            wrench_nm: the feedforward wrench for x, y, z, torque_x, torque_y, torque_z in the task frame
            timeout_sec: the timeout

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_end_effector_impedance_command(
            trajectory=trajectory,
            root_frame=root_frame_name,
            root_t_task=root_t_task,
            wrist_t_tool=wrist_t_tool,
            stiffness=stiffness,
            damping=damping,
            wrench_nm=wrench_nm,
        )
        convert(command, goal.command)
        return self.command("move_end_effector_impedance", goal, timeout_sec)

    move_end_effector_impedance = move_end_effector_with_controlled_impedance

    def move_end_effector_impedance_async(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
        *,
        result_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        feedback_callback: Optional[Callable[[RobotCommandFeedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[SpotActionHandle]:
        """Impedance control as variable stiffness/damping position control combined with a
            feed-forward force control as a wrench.

        Args:
            root_frame_name: name of root frame as a string
            root_t_task: the transform from the root frame to the "task frame" (this doesn't need a name)
            wrist_t_tool: The tool pose relative to the parent link (arm_link_wr1).
                This can also be thought of as the "remote center" frame.
            trajectory: Discrete waypoints of where we want the tool to be relative to the task frame
                and the duration of time to slerp over.
            stiffness: the position/stiffness gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            damping: the derivative/damping gains for the x, y, z, rot_x, rot_y, rot_z axes in the task frame
            wrench_nm: the feed-forward wrench for x, y, z, torque_x, torque_y, torque_z in the task frame
            result_callback: A callback to process/handle the resulting feedback from executing the command
            feedback_callback: A callback to process/handle the feedback received during the execution of the command
            on_failure_callback: A callback to process/handle when the action fails for various reasons

        Returns:
            The feedback from the SpotDriver about the result of the command if it was accepted and a result was
                successfully received, None otherwise
        """
        goal = RobotCommandAction.Goal()
        command_builder = SpotArmCommandBuilder(self)
        command = command_builder.arm_end_effector_impedance_command(
            trajectory=trajectory,
            root_frame=root_frame_name,
            root_t_task=root_t_task,
            wrist_t_tool=wrist_t_tool,
            stiffness=stiffness,
            damping=damping,
            wrench_nm=wrench_nm,
        )
        convert(command, goal.command)
        return self.command_async(
            "move_end_effector_impedance",
            goal,
            result_callback=result_callback,
            feedback_callback=feedback_callback,
            on_failure_callback=on_failure_callback,
        )

    def get_arm_impedance_robot_cmd(
        self,
        root_frame_name: FrameHint,
        root_t_task: SE3,
        wrist_t_tool: SE3,
        trajectory: list[tuple[SE3, float]],
        stiffness: Optional[SpatialVector] = None,
        damping: Optional[SpatialVector] = None,
        wrench_nm: Optional[SpatialForce] = None,
    ) -> robot_command_pb2.RobotCommand:
        """Create the robot command proto for moving the end effector to specific
        pose using hybrid position/impedance control.
        The stiffness and damping gains are defaulted to their maximum values.
        Note that this is in the task frame, so these were probably empirically obtained.
        (See code comment in here
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_impedance_control/arm_impedance_control.py.)
        """
        command_builder = SpotArmCommandBuilder(self)
        return command_builder.arm_end_effector_impedance_command(
            trajectory=trajectory,
            root_frame=root_frame_name,
            root_t_task=root_t_task,
            wrist_t_tool=wrist_t_tool,
            damping=damping,
            stiffness=stiffness,
            wrench_nm=wrench_nm,
        )

    @generalized_method(transitional=True)
    def arm_zero_g(self) -> Optional[RobotCommandFeedback]:
        """Sets spot arm into gravity compensated mode such that the arm 'renders'
        no apparent gravitational acceleration and low damping.

        Useful for allowing a person to move the arm around freely. Does not compensate
        for additional unmodeled mass (like putting something in the gripper.)
        """
        response = self.arm_zero_g.synchronously(timeout_sec=2.0, nothrow=True)
        if response is None or not response.success:
            return None
        return response.result

    zero_g = arm_zero_g
