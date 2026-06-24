# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp
from spatialmath import SE3
from spot_basic.trajectory.se3_trajectory import SE3Trajectory, interpolate_line


@pytest.fixture
def setup_interpolate_line() -> tuple[SE3, SE3, int, float, SE3Trajectory]:
    """Setup the `interpolate_line` function's inputs and output for testing."""
    frame_t_start: SE3 = SE3.Rt(R=np.eye(N=3), t=np.array(object=[0, 0, 0]))
    frame_t_end: SE3 = SE3.Rt(
        R=Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix(), t=np.array(object=[1, 1, 1])
    )
    num_points: int = 5
    speed: float = 0.1

    trajectory: SE3Trajectory = interpolate_line(
        frame_t_start=frame_t_start, frame_t_end=frame_t_end, num_points=num_points, speed=speed
    )

    return frame_t_start, frame_t_end, num_points, speed, trajectory


def test_number_of_waypoints(setup_interpolate_line: tuple[SE3, SE3, int, float, SE3Trajectory]) -> None:
    """Test that the trajectory contains the correct number of waypoints."""
    num_points: int = setup_interpolate_line[2]
    trajectory: SE3Trajectory = setup_interpolate_line[4]

    actual_num_waypoints: int = len(trajectory.waypoints)
    expected_num_waypoints: int = num_points

    assert actual_num_waypoints == expected_num_waypoints


def test_start_and_end_points(setup_interpolate_line: tuple[SE3, SE3, int, float, SE3Trajectory]) -> None:
    """Test that the first and last points match the given start and end poses."""
    frame_t_start: SE3 = setup_interpolate_line[0]
    frame_t_end: SE3 = setup_interpolate_line[1]
    trajectory: SE3Trajectory = setup_interpolate_line[4]

    actual_start_point = trajectory.waypoints[0].task_t_waypoint.A
    expected_start_point = frame_t_start.A
    actual_end_point = trajectory.waypoints[-1].task_t_waypoint.A
    expected_end_point = frame_t_end.A

    assert np.allclose(a=actual_start_point, b=expected_start_point, atol=1e-6)
    assert np.allclose(a=actual_end_point, b=expected_end_point, atol=1e-6)


def test_position_interpolation(setup_interpolate_line: tuple[SE3, SE3, int, float, SE3Trajectory]) -> None:
    """Test that the positions are interpolated linearly."""
    frame_t_start: SE3 = setup_interpolate_line[0]
    frame_t_end: SE3 = setup_interpolate_line[1]
    num_points: int = setup_interpolate_line[2]
    trajectory: SE3Trajectory = setup_interpolate_line[4]

    actual_positions = np.array([wp.task_t_waypoint.t for wp in trajectory.waypoints])
    expected_positions = np.linspace(start=frame_t_start.t, stop=frame_t_end.t, num=num_points)

    assert np.allclose(a=actual_positions, b=expected_positions, atol=1e-6)


def test_orientation_interpolation(setup_interpolate_line: tuple[SE3, SE3, int, float, SE3Trajectory]) -> None:
    """Test that the orientations are interpolated using SLERP."""
    frame_t_start: SE3 = setup_interpolate_line[0]
    frame_t_end: SE3 = setup_interpolate_line[1]
    num_points: int = setup_interpolate_line[2]
    trajectory: SE3Trajectory = setup_interpolate_line[4]

    # Define SLERP interpolation to computed expected orientations
    rot_start_end = [Rotation.from_matrix(frame_t_start.R), Rotation.from_matrix(frame_t_end.R)]
    slerp_fn = Slerp([0, 1], Rotation.concatenate(rot_start_end))

    expected_orientations = [slerp_fn(alpha).as_matrix() for alpha in np.linspace(start=0, stop=1, num=num_points)]
    actual_orientations = [wp.task_t_waypoint.R for wp in trajectory.waypoints]

    for expected, actual in zip(expected_orientations, actual_orientations, strict=True):
        assert np.allclose(a=actual, b=expected, atol=1e-6)


def test_time_increase(setup_interpolate_line: tuple[SE3, SE3, int, float, SE3Trajectory]) -> None:
    """Test that the time_from_start values are strictly increasing."""
    trajectory: SE3Trajectory = setup_interpolate_line[4]

    times = np.array(object=[wp.time_from_start for wp in trajectory.waypoints])

    assert times[0] == 0.0
    assert np.all(np.diff(times) > 0)
