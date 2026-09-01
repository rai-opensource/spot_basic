# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute, LLC. All rights reserved.

"""Helpers for converting spatialmath trajectories into Spot command protobufs."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
from bosdyn.api import (
    basic_command_pb2,
    mobility_command_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    trajectory_pb2,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.trajectory_pb2 import POS_INTERP_CUBIC, PositionalInterpolation
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from google.protobuf import any_pb2, duration_pb2, timestamp_pb2
from spatialmath import SE2, SE3, SpatialForce, Twist3

from spot_basic.manual_conversions import (
    se2_to_se2pose_proto,
    se3_to_se3pose_proto,
    spatialforce_to_wrench_proto,
    twist3_to_se3velocity_proto,
)
from spot_basic.trajectory.se3_trajectory import SE3Trajectory


def timestamped_se2_vecs_to_se2_trajectory_command_proto(
    timestamps_sec: npt.ArrayLike,
    se2_poses: Sequence[SE2] | npt.ArrayLike,
    interpolation: PositionalInterpolation = POS_INTERP_CUBIC,
    *,
    base_frame: str = BODY_FRAME_NAME,
) -> basic_command_pb2.SE2TrajectoryCommand.Request:
    """Convert timestamped SE(2) poses to an SE2 trajectory command protobuf.

    Args:
        timestamps_sec: sequence of times since the command reference time, in seconds, one per pose.
        se2_poses: sequence of spatialmath SE2 poses or an array shaped (_, 3) with x, y, and
            theta values in meters and radians.
        interpolation: positional interpolation mode for the trajectory points.
        base_frame: frame name in which the SE(2) poses are expressed.

    Returns:
        an SE2TrajectoryCommand request containing the timestamped trajectory.

    Raises:
        ValueError: if `se2_poses` is an array whose shape is not (_, 3).
    """
    # See also: RobotCommandBuilder.synchro_se2_trajectory_command,
    # where this is done but with the assumption of only one goal point and no time stamping

    if isinstance(se2_poses, np.ndarray):
        if se2_poses.ndim != 2 or se2_poses.shape[-1] != 3:
            raise ValueError(f"expected se2_poses array shape to be (_, 3), got {se2_poses.shape}")
        se2_poses = [SE2(x=x, y=y, theta=angle) for x, y, angle in se2_poses]

    points = []
    for t, se2 in zip(timestamps_sec, se2_poses, strict=True):
        time_since_reference = duration_pb2.Duration()
        time_since_reference.FromNanoseconds(int(t * 1e9))
        points.append(
            trajectory_pb2.SE2TrajectoryPoint(
                pose=se2_to_se2pose_proto(se2),
                time_since_reference=time_since_reference,
            )
        )

    trajectory = trajectory_pb2.SE2Trajectory(points=points, interpolation=interpolation)
    return basic_command_pb2.SE2TrajectoryCommand.Request(trajectory=trajectory, se2_frame_name=base_frame)


def timestamped_se2_vecs_to_mobility_command_proto(
    *args: Any, mobility_params: Optional[spot_command_pb2.MobilityParams] = None, **kwargs: Any
) -> mobility_command_pb2.MobilityCommand.Request:
    """Convert timestamped SE(2) poses to a mobility command protobuf.

    The forwarded trajectory arguments describe times in seconds and SE(2) poses in meters and radians.

    Args:
        *args: trajectory arguments forwarded to `timestamped_se2_vecs_to_se2_trajectory_command_proto`.
        mobility_params: optional Spot mobility params to pack into the command.
        **kwargs: keyword arguments forwarded to `timestamped_se2_vecs_to_se2_trajectory_command_proto`.

    Returns:
        a MobilityCommand request containing the SE(2) trajectory request and optional mobility params.
    """
    # Build the trajectory proto
    trajectory_command_proto = timestamped_se2_vecs_to_se2_trajectory_command_proto(*args, **kwargs)

    # Build the mobility command request
    any_params = any_pb2.Any()
    if mobility_params is not None:
        any_params.Pack(mobility_params)
    return mobility_command_pb2.MobilityCommand.Request(
        se2_trajectory_request=trajectory_command_proto, params=any_params
    )


def timestamped_se2_vecs_to_robot_command_proto(
    *args: Any, mobility_params: spot_command_pb2.MobilityParams, **kwargs: Any
) -> robot_command_pb2.RobotCommand:
    """Convert timestamped SE(2) poses to a synchronized robot command protobuf.

    The forwarded trajectory arguments describe times in seconds and SE(2) poses in meters and radians.

    Args:
        *args: trajectory arguments forwarded to `timestamped_se2_vecs_to_mobility_command_proto`.
        mobility_params: Spot mobility params to pack into the mobility command.
        **kwargs: keyword arguments forwarded to `timestamped_se2_vecs_to_mobility_command_proto`.

    Returns:
        a RobotCommand containing a synchronized mobility command with the SE(2) trajectory.
    """
    mobility_command = timestamped_se2_vecs_to_mobility_command_proto(*args, mobility_params=mobility_params, **kwargs)
    synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(mobility_command=mobility_command)
    return robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)


def timestamped_se3_poses_to_se3_trajectory(
    timestamps_between_points: np.typing.ArrayLike,
    pose_vecs: Sequence[SE3],
    vel_vecs: Optional[Sequence[Twist3]] = None,
    pose_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
    ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
    reference_time: Optional[float] = None,
) -> trajectory_pb2.SE3Trajectory:
    """Returns SE3Trajectory proto given a trajectory of spatialmath SE3 points.

    Args:
        timestamps_between points: Time from start for each point (this argument is mis-named - #SW-2894).
        pose_vecs: array of values representing the SE3 pose trajectory,
        vel_vecs: array of values representing the SE3 velocity trajectory.
        pose_interpolation: positional interpolation method. Defaults to cubic.
        ang_interpolation: angular interpolation method. Defaults to cubic.
        reference_time: reference time to begin the trajectory at, with respect to the robot clock.

    Returns:
        An SE3Trajectory representing the inputted parameters.
    """
    trajectory = []
    num_points_in_trajectory = len(timestamps_between_points)
    # Ensure the timestamps, quaternion, and pose_vecs have the same lengths
    if num_points_in_trajectory != len(pose_vecs):
        raise IndexError("Number of timestamps and positions must be equal.")

    for pt in range(num_points_in_trajectory):
        pose_proto = se3_to_se3pose_proto(pose_vecs[pt])
        time_since_ref = duration_pb2.Duration()
        time_since_ref.FromNanoseconds(int(timestamps_between_points[pt] * 1e9))
        trajectory_point = trajectory_pb2.SE3TrajectoryPoint(pose=pose_proto, time_since_reference=time_since_ref)
        # if we are supplying velocity as well, we need to add it in
        if vel_vecs is not None:
            spatial_velocity = vel_vecs[pt]
            velocity_proto = twist3_to_se3velocity_proto(spatial_velocity)
            trajectory_point = trajectory_pb2.SE3TrajectoryPoint(
                pose=pose_proto, velocity=velocity_proto, time_since_reference=time_since_ref
            )
        else:
            trajectory_point = trajectory_pb2.SE3TrajectoryPoint(pose=pose_proto, time_since_reference=time_since_ref)
        trajectory.append(trajectory_point)

    if reference_time is not None:
        ref_time = timestamp_pb2.Timestamp()
        ref_time.FromNanoseconds(int(reference_time * 1e9))
        trajectory_proto = trajectory_pb2.SE3Trajectory(
            points=trajectory,
            pos_interpolation=pose_interpolation,
            ang_interpolation=ang_interpolation,
            reference_time=ref_time,
        )
    else:
        trajectory_proto = trajectory_pb2.SE3Trajectory(
            points=trajectory,
            pos_interpolation=pose_interpolation,
            ang_interpolation=ang_interpolation,
        )

    return trajectory_proto


def se3_trajectory_to_proto(
    trajectory: SE3Trajectory,
    pos_interpolation: trajectory_pb2.PositionalInterpolation = trajectory_pb2.POS_INTERP_CUBIC,
    ang_interpolation: trajectory_pb2.AngularInterpolation = trajectory_pb2.ANG_INTERP_CUBIC_EULER,
    reference_time: Optional[float] = None,
) -> trajectory_pb2.SE3Trajectory:
    """Converts an SE3Trajectory object to a SE3Trajectory protobuf message.

    Args:
        trajectory: input trajectory consisting of timestamped SE(3) waypoints.
        pos_interpolation: interpolation approach for position. Defaults to cubic interpolation.
        ang_interpolation: interpolation approach for orientation. Defaults to cubic interpolation of Euler angles.
        reference_time: optional reference time to use as the base for the timestamps. Defaults to None.
    Returns:
        trajectory_pb2.SE3Trajectory: A protobuf message representing the trajectory.
    """
    return timestamped_se3_poses_to_se3_trajectory(
        timestamps_between_points=[pt.time_from_start for pt in trajectory.waypoints],
        pose_vecs=[pt.task_t_waypoint for pt in trajectory.waypoints],
        pose_interpolation=pos_interpolation,
        ang_interpolation=ang_interpolation,
        reference_time=reference_time,
    )


def timestamped_spatialforces_to_wrench_protos(
    timestamps_between_wrenches: np.typing.ArrayLike,
    wrench_vecs: Sequence[SpatialForce],
    reference_time: Optional[float] = None,
) -> trajectory_pb2.WrenchTrajectory:
    """Turns list of wrench vectors into a WrenchTrajectory.

    Args:
        timestamps_between wrenches: array of times of when each point is traversed (seconds).
        wrench_vecs: sequence of spatialforces representing wrenches at the end effector.
        reference_time: reference time to begin the trajectory at, with respect to the robot clock.

    Returns:
        Trajectory of wrenches as defined the Boston Dynamics API.
    """
    trajectory = []
    num_points_in_trajectory = len(timestamps_between_wrenches)
    # Ensure the timestamps, quaternion, and pose_vecs have the same lengths
    if num_points_in_trajectory != len(wrench_vecs):
        raise IndexError("Number of timestamps and positions must be equal.")

    for pt in range(num_points_in_trajectory):
        wrench_proto = spatialforce_to_wrench_proto(wrench_vecs[pt])
        time_since_ref = duration_pb2.Duration()
        time_since_ref.FromNanoseconds(int(timestamps_between_wrenches[pt] * 1e9))
        trajectory_point = trajectory_pb2.WrenchTrajectoryPoint(
            wrench=wrench_proto, time_since_reference=time_since_ref
        )
        trajectory.append(trajectory_point)

    if reference_time is not None:
        ref_time = timestamp_pb2.Timestamp()
        ref_time.FromNanoseconds(int(reference_time * 1e9))
        trajectory_proto = trajectory_pb2.WrenchTrajectory(
            points=trajectory,
            reference_time=ref_time,
        )
    else:
        trajectory_proto = trajectory_pb2.WrenchTrajectory(
            points=trajectory,
        )

    return trajectory_proto
