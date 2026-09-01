#  Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute.  All rights reserved.

"""SE3 trajectory data types and interpolation utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from spatialmath import SE3


@dataclass
class SE3TrajectoryPoint:
    """Pose and time for use in a trajectory

    Attributes:
        task_t_waypoint: Pose for this point on the trajectory
        time_from_start: Cummulative time in seconds from the START of the trajectory (not the previous waypoint).
            The robot's initial pose when the trajectory begins execution is time 0.
    """

    task_t_waypoint: SE3
    time_from_start: float


def get_trajectory_point(prev_pt: SE3TrajectoryPoint, task_t_waypoint: SE3, speed: float) -> SE3TrajectoryPoint:
    """Returns a trajectory point that moves from prev_pt.task_t_waypoint to task_t_waypoint at speed speed.

    This considers only the translational part of the move when calculating time.  Specifically, time from start for
    returned point is:
    prev_pt.time_from_start + (translational distance from prev_pt to task_t_waypoint) / speed

    Args:
        prev_pt: Previous point on the trajectory
        task_t_waypoint: Pose for the next point on the trajectory
        speed: The speed in m/s to travel from prev_pt.task_t_waypoint to task_t_waypoint

    Returns: Trajectory point with task_t_waypoint=task_t_waypoint and appropriate time_from_start.
    """
    return SE3TrajectoryPoint(
        task_t_waypoint=task_t_waypoint,
        time_from_start=np.linalg.norm(task_t_waypoint.t - prev_pt.task_t_waypoint.t) / speed + prev_pt.time_from_start,
    )


@dataclass
class SE3Trajectory:
    """Holds information about a trajectory for a robot hand

    Attributes:
        waypoints: Poses and times along the trajectory
    """

    waypoints: list[SE3TrajectoryPoint]


def interpolate_line(frame_t_start: SE3, frame_t_end: SE3, num_points: int = 10, speed: float = 0.1) -> SE3Trajectory:
    """Interpolate between two SE3 poses (assumed in same frame) in a straight line with smooth orientation transition.

    Args:
        frame_t_start (SE3): Start pose.
        frame_t_end (SE3): End pose.
        num_points (int): Number of interpolation steps, including start and end.
        speed (float): Speed at each step, assumed invariant.

    Returns:
        SE3Trajectory: Interpolated linear trajectory.
    """
    # Set up interpolators (orientation using SLERP and position using LERP)
    rot_start_end = [Rotation.from_matrix(frame_t_start.R), Rotation.from_matrix(frame_t_end.R)]
    slerp_fn = Slerp([0, 1], Rotation.concatenate(rot_start_end))

    slerp = lambda alpha: slerp_fn(alpha).as_matrix()
    lerp = lambda alpha: (1 - alpha) * frame_t_start.t + alpha * frame_t_end.t
    interp = lambda alpha: SE3.Rt(R=slerp(alpha), t=lerp(alpha))

    # Interpolate SE3 poses including start and end poses
    interpolated_poses: list[SE3] = [interp(alpha) for alpha in np.linspace(start=0, stop=1, num=num_points)]

    # Construct as `SE3Trajectory`
    tp_start = SE3TrajectoryPoint(task_t_waypoint=frame_t_start, time_from_start=0.0)
    trajectory = SE3Trajectory(waypoints=[tp_start])

    for k in range(1, len(interpolated_poses)):
        waypoint: SE3TrajectoryPoint = get_trajectory_point(
            prev_pt=trajectory.waypoints[k - 1], task_t_waypoint=interpolated_poses[k], speed=speed
        )
        trajectory.waypoints.append(waypoint)

    return trajectory
