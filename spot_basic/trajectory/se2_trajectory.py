#  Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Tools for handling SE2 reconfigurations"""

from __future__ import annotations

import copy
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from spatialmath import SE2
from spatialmath.base.vectors import angdiff


def pose_change_duration_linear_estimate(
    *,
    pose_change: Optional[SE2] = None,
    pose_change_vec: Optional[npt.ArrayLike] = None,
    se2_speed_vec: npt.ArrayLike,
) -> float:
    """Compute a coarse estimate of how much time is needed to perform a pose change"""

    # NOTE this kind of check is often relevant - maybe there's a clean resource for it?
    # e.g. ensure_exactly_one_kwarg_is_not_none(kwarg1=kwarg1, kwarg2=kwarg2)
    if (pose_change is None) and (pose_change_vec is None):
        raise ValueError("Require exactly one argument for specifying pose change (received zero)")
    if (pose_change is not None) and (pose_change_vec is not None):
        raise ValueError("Require exactly one argument for specifying pose change (received two)")

    if pose_change_vec is None:
        assert pose_change is not None
        pose_change_vec = np.array([pose_change.x, pose_change.y, pose_change.theta()])

    return float(np.linalg.norm(np.array(pose_change_vec) / np.array(se2_speed_vec)))


def pose_to_vec(pose: SE2) -> npt.NDArray:
    return np.array([pose.x, pose.y, pose.theta()])


# Type aliasing - getting rid of some pyright complaints but this probably isn't ideal?
_TrajectoryPoint = Union[npt.ArrayLike, SE2]
_TrajectoryPoints = Union[npt.ArrayLike, list[_TrajectoryPoint]]
_TimestampedTrajectoryPoints = tuple[npt.ArrayLike, _TrajectoryPoints]


class SE2Trajectory:
    def __init__(
        self,
        *,
        timestamps_sec: Optional[npt.ArrayLike] = None,
        duration_sec: Optional[float] = None,  # pylint: disable=invalid-name
        poses: Optional[list[SE2]] = None,
        pose_vecs: Optional[npt.ArrayLike] = None,
        phases: Optional[list[Any]] = None,
        make_relative_to_first_pose: bool = False,
    ):
        if (poses is None) and (pose_vecs is None):
            raise ValueError("Require exactly one argument for specifying poses (received zero)")
        if (poses is not None) and (pose_vecs is not None):
            raise ValueError("Require exactly one argument for specifying poses (received two)")

        if (duration_sec is None) and (timestamps_sec is None):
            raise ValueError("Require exactly one argument for determining timestamps (received zero)")
        if (duration_sec is not None) and (timestamps_sec is not None):
            raise ValueError("Require exactly one argument for determining timestamps (received two)")

        self._poses: list[SE2] = []

        if poses is None:
            self.set_poses([SE2(*pose_vec) for pose_vec in np.array(pose_vecs)])
        else:
            self.set_poses(poses)

        if make_relative_to_first_pose:
            pose_0_inv = self._poses[0].inv()
            self.set_poses([pose_0_inv * pose for pose in self._poses])  # pyright: ignore

        n_poses = len(self._poses)
        self.phases = np.arange(n_poses).tolist() if (phases is None) else phases

        if timestamps_sec is None:
            assert duration_sec is not None
            self.duration_sec = duration_sec  # pylint: disable=invalid-name
            self.timestamps_sec = np.linspace(0, duration_sec, n_poses)
        else:
            timestamps_sec = np.array(timestamps_sec)
            assert len(timestamps_sec) == n_poses
            self.timestamps_sec = timestamps_sec
            self.duration_sec = self.timestamps_sec[-1]

        if np.isnan(self.timestamps_sec).any():
            raise ValueError("Invalid time stamps")

    @property
    def poses(self) -> list[SE2]:
        return self._poses

    # TODO property decorator?
    def set_poses(self, poses: list[SE2]) -> None:
        self._poses = poses
        assert all(isinstance(pose, SE2) for pose in self._poses)

    def copy_with_updated_poses(self, poses: list[SE2]) -> SE2Trajectory:
        # NOTE assuming timestamps_sec and phases immutable
        return SE2Trajectory(timestamps_sec=self.timestamps_sec, poses=poses, phases=self.phases)

    def __mul__(self, other: SE2) -> SE2Trajectory:
        return self.copy_with_updated_poses([pose * other for pose in self._poses])

    def __rmul__(self, other: SE2) -> SE2Trajectory:
        return self.copy_with_updated_poses([other * pose for pose in self._poses])

    def __getitem__(self, query: Union[int, str]) -> SE2:
        if isinstance(query, str):
            if self.phases is None:
                raise ValueError("Cannot query by phase name; phase names are undefined")
            idx = self.phases.index(query)
        else:
            idx = query

        return self._poses[idx]

    def __call__(
        self,
        timestamp_sec: Optional[Union[float, npt.ArrayLike]] = None,
        *,
        timestep_sec: Optional[float] = None,
        n_timesteps: Optional[int] = None,
        as_pose: bool = False,
        with_timestamps: bool = False,
    ) -> Union[_TrajectoryPoint, _TrajectoryPoints, _TimestampedTrajectoryPoints]:
        if timestamp_sec is None:
            if timestep_sec is not None:
                timestamp_sec = np.append(np.arange(0, self.duration_sec, timestep_sec), self.duration_sec)
            elif n_timesteps is not None:
                timestamp_sec = np.linspace(0, self.duration_sec, n_timesteps)
            else:
                timestamp_sec = self.timestamps_sec

        if hasattr(timestamp_sec, "__iter__"):
            timestamps_sec = np.array(timestamp_sec)
            result_list = [self(timestamp_sec, as_pose=as_pose) for timestamp_sec in timestamps_sec]
            if as_pose:
                assert all(isinstance(item, SE2) for item in result_list)
                result = result_list
            else:
                result = np.array(result_list)

            if with_timestamps:  # FIXME type aliasing still didn't fix these pyright complaints...
                return timestamps_sec, result  # pyright: ignore
            return result  # pyright: ignore

        if len(self.timestamps_sec) > 1:
            idx_left = np.nonzero(
                np.logical_and(
                    np.greater_equal(self.timestamps_sec[1:], timestamp_sec),
                    np.less_equal(self.timestamps_sec[:-1], timestamp_sec),
                )
            )[0][0]
            idx_right = idx_left + 1
        else:
            idx_left = 0
            idx_right = 0

        timestamp_left = self.timestamps_sec[idx_left]
        timestamp_right = self.timestamps_sec[idx_right]

        if timestamp_right == timestamp_left:
            interpolation_weight = 0.0
        else:
            interpolation_weight = (timestamp_sec - timestamp_left) / (timestamp_right - timestamp_left)

        pose_vec = (1 - interpolation_weight) * pose_to_vec(self._poses[idx_left]) + interpolation_weight * pose_to_vec(
            self._poses[idx_right]
        )
        if as_pose:
            return SE2(*pose_vec)
        return pose_vec

    @classmethod
    def from_se2_pair(
        cls,
        *,
        target: SE2,
        se2_speed_vec: npt.ArrayLike,
        origin: Optional[SE2] = None,
        skewness: float = 1.0,
    ) -> SE2Trajectory:
        """Map a delta-SE2 command onto a sequence of SE2 points, in a parameterized manner."""

        if origin is None:
            origin = SE2()
        target_relative = origin.inv() * target

        # The angle to turn to face the target
        bearing_angle = np.arctan2(target_relative.y, target_relative.x)
        distance = float(np.linalg.norm([target_relative.x, target_relative.y]))

        # Amount to turn after aligning with bearing angle
        final_angle_delta = angdiff(target.theta(), bearing_angle)

        total_angle_delta_mag = np.abs(bearing_angle) + np.abs(
            final_angle_delta
        )  # accounts for time turning either way
        if not total_angle_delta_mag:
            assert not bearing_angle
            frac_skew1 = 0.0
        else:
            frac_skew1 = (np.abs(bearing_angle) / total_angle_delta_mag) * skewness
        frac_forward = 1 - skewness

        pose_changes = {"initial": origin}
        pose_changes["origin_forward"] = SE2(target.x * frac_skew1, target.y * frac_skew1, bearing_angle)
        pose_changes["forward_skew2"] = SE2(distance * frac_forward, 0, 0)

        # Convert from sequentially relative to cumulative relative
        pose = SE2()
        poses = []
        for pose_change in pose_changes.values():  # NOTE assumes dict is properly ordered.  use ordered keys instead?
            pose *= pose_change  # pyright: ignore  # so much grief from the mult operator typing
            poses.append(copy.deepcopy(pose))

        # Final pose change is prescribed by final pose
        pose_changes["skew2_target"] = poses[-1].inv() * target_relative
        poses.append(target_relative)

        phase_keys = list(pose_changes.keys())

        durations = {
            key: pose_change_duration_linear_estimate(pose_change=pose_change, se2_speed_vec=se2_speed_vec)
            for key, pose_change in pose_changes.items()
        }
        durations["initial"] = 0  # This is just a frame transform not a reconfiguration
        timestamps_sec = np.cumsum([durations[key] for key in phase_keys])

        return cls(timestamps_sec=timestamps_sec, poses=poses, phases=phase_keys)

    @classmethod
    def from_circle(
        cls,
        *,
        radius: float,
        initial_orientation: float = np.pi / 2,
        duration_sec: Optional[float] = None,
        n_poses: Optional[int] = None,
        total_body_rotation: Optional[float] = None,
        total_angular_travel: float = 2 * np.pi,
        forward_speed: Optional[float] = None,
    ) -> SE2Trajectory:
        """Make a circle trajectory with the robot already starting on the circle.

        Args:
            radius: radius of the circle in meters
            initial_orientation: deviation of the robot's initial pose relative to the radial direction.
                    Default of pi/2 orients the robot tangentially in the counterclockwise direction.

            total_body_rotation: net change in orientation that will be linearly accrued
        """

        total_angular_travel_mag = np.abs(total_angular_travel)

        if (duration_sec is None) and (forward_speed is None):
            raise ValueError("Require exactly one argument for determining duration (received zero)")
        if (duration_sec is not None) and (forward_speed is not None):
            raise ValueError("Require exactly one argument for determining duration (received two)")

        if duration_sec is None:
            # Use optimistic assumptions; traveling circumference at rate of forward walk
            duration_sec = total_angular_travel_mag * radius / forward_speed

        if n_poses is None:
            n_poses = int(np.rad2deg(total_angular_travel_mag) // 10)

        angular_positions = np.linspace(0, total_angular_travel, n_poses)
        xs = radius * np.cos(angular_positions)
        ys = radius * np.sin(angular_positions)
        total_body_rotation = total_angular_travel if (total_body_rotation is None) else total_body_rotation
        yaws = initial_orientation + np.linspace(0, total_body_rotation, n_poses)

        pose_vecs = np.vstack((xs, ys, yaws)).T
        return cls(pose_vecs=pose_vecs, duration_sec=duration_sec, make_relative_to_first_pose=True)
