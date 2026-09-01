#  Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Plotting helpers for SE2 trajectories and robot body outlines."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def se2_homogeneous_transform(points: npt.ArrayLike, se2_tfs: npt.ArrayLike) -> npt.NDArray:
    """Arrayified computation of a sequence of 2d points all transformed by each of a sequence of se2 transformations"""

    points = np.array(points, ndmin=2)  # [i_point, i_xy]
    spatial_dim = points.shape[1]
    if not spatial_dim == 2:
        raise ValueError(f"Function operates on 2D points only. Received spatial dimensionality {spatial_dim}")

    se2_tfs = np.array(se2_tfs, ndmin=2)  # [i_tf, i_xyt]

    ctheta = np.cos(se2_tfs[..., -1])  # [i_tf]
    stheta = np.sin(se2_tfs[..., -1])  # [i_tf]
    rotation_dcm_transpose = np.array([[ctheta, -stheta], [stheta, ctheta]])  # [i_xy_left, i_xy_right, i_tf]

    # Transform each point by each pose
    return points.dot(rotation_dcm_transpose) + se2_tfs.T[None, :2]  # [i_point, i_xy, i_tf]


def wrap_first_dim(arr: npt.NDArray) -> npt.NDArray:
    """Repeats the first entry in an array (slicing on the 0'th axis) at its end, e.g. for drawing a closed loop"""
    return arr[np.r_[: arr.shape[0], 0]]


def draw_shape(ax: plt.Axes, se2_pose_array: npt.NDArray, shape_points: npt.ArrayLike, **plot_kwargs: Any) -> None:
    """Draw a given shape (2d point sequence) at each of a sequence of se2 poses, on a given axis"""

    transformations_of_shape = se2_homogeneous_transform(shape_points, se2_pose_array)  # [i_point, i_xy, i_tf]

    # plt.plot draws a separate axis per column, so we want [i_xy, i_point, i_tf]
    # and we want to wrap the i_point dim before swapping it
    ax.plot(*wrap_first_dim(transformations_of_shape).swapaxes(0, 1), **plot_kwargs)
