# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
from spatialmath import SE2
from spot_basic.trajectory.se2_trajectory import SE2Trajectory
from spot_basic.trajectory.spot_se2_properties import MAX_SPEED_VEC


def test_from_se2_pair() -> None:

    x0 = SE2()
    xf = SE2(2, 1, np.pi / 4)
    origin_t_traj = SE2Trajectory.from_se2_pair(origin=x0, target=xf, skewness=0.5, se2_speed_vec=MAX_SPEED_VEC)

    assert len(origin_t_traj.poses) == 4
    assert len(origin_t_traj.phases) == 4

    assert str(origin_t_traj.phases) == str(["initial", "origin_forward", "forward_skew2", "skew2_target"])

    np.testing.assert_almost_equal(origin_t_traj.timestamps_sec, np.array([0.0, 0.66198458, 1.36075582, 1.79141767]))

    np.testing.assert_almost_equal(
        origin_t_traj.poses[0].A, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )

    np.testing.assert_almost_equal(
        origin_t_traj.poses[1].A,
        np.array([[0.89442719, -0.4472136, 0.59033447], [0.4472136, 0.89442719, 0.29516724], [0.0, 0.0, 1.0]]),
    )

    np.testing.assert_almost_equal(
        origin_t_traj.poses[2].A,
        np.array([[0.89442719, -0.4472136, 1.59033447], [0.4472136, 0.89442719, 0.79516724], [0.0, 0.0, 1.0]]),
    )

    np.testing.assert_almost_equal(
        origin_t_traj.poses[3].A,
        np.array([[0.70710678, -0.70710678, 2.0], [0.70710678, 0.70710678, 1.0], [0.0, 0.0, 1.0]]),
    )
