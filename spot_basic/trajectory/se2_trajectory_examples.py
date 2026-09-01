#  Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Demonstrate usage of objects pertaining to SE2 trajectories."""
# Rather not use notebooks even though they'd be a natural choice for this

import argparse

import numpy as np
from matplotlib import pyplot as plt
from spatialmath import SE2

from spot_basic.trajectory.se2_trajectory import SE2Trajectory
from spot_basic.trajectory.se2_trajectory_plotting import draw_shape
from spot_basic.trajectory.spot_se2_properties import BODY_SHAPE, MAX_SPEED_VEC


def plot_trajectory(ax: plt.Axes, trajectory: SE2Trajectory) -> None:
    """Draw a Spot body shape along an SE2 trajectory.

    Args:
        ax: matplotlib axes to draw on.
        trajectory: SE2 trajectory samples as ``[x, y, theta]`` rows, with x and y in metres and theta in radians.
    """
    draw_shape(ax, trajectory, BODY_SHAPE)  # type: ignore
    ax.axis("equal")


def example_first() -> None:
    """Demonstrate the effect of the skewness parameter for each of three different target states"""
    skewnesses = [0, 0.5, 1]

    x0 = SE2()
    n_timesteps = 20  # Number of time points to interpolate at

    for yf in [-1, 0, 1]:
        xf = SE2(3, yf, -np.pi / 2)
        trajs = {
            skew: SE2Trajectory.from_se2_pair(origin=x0, target=xf, skewness=skew, se2_speed_vec=MAX_SPEED_VEC)
            for skew in skewnesses
        }

        n_rows = len(skewnesses)
        n_cols = 2
        _, axs = plt.subplots(n_rows, n_cols, figsize=(8, 8), squeeze=False)
        axs_col0 = axs[..., 0]
        axs_col1 = axs[..., 1]

        axs_col0[0].set_title("Traj keypoints")
        axs_col1[0].set_title("Interpolated traj")

        for ax_keypoint, ax_smooth, (skew, traj) in zip(axs_col0, axs_col1, trajs.items(), strict=True):
            ax_keypoint.set_ylabel(f"(skew = {skew})")
            plot_trajectory(ax_keypoint, traj())
            plot_trajectory(ax_smooth, traj(n_timesteps=n_timesteps))

    plt.show()


def example_transform() -> None:
    """Demonstrate applying a transformation to the SE2Trajectory object"""

    x0 = SE2()
    xf = SE2(2, 1, np.pi / 4)
    origin_t_traj = SE2Trajectory.from_se2_pair(origin=x0, target=xf, skewness=0.5, se2_speed_vec=MAX_SPEED_VEC)

    origin_t_newref = SE2(1.68, 4.2, -0.987)
    newref_t_traj = origin_t_newref.inv() * origin_t_traj

    _, axs = plt.subplots(2, 1, figsize=(5, 8), squeeze=True)
    n_timesteps = 20
    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax_origin, ax_newref = axs
    plot_trajectory(ax_origin, origin_t_traj(n_timesteps=n_timesteps))  # type: ignore
    plot_trajectory(ax_newref, newref_t_traj(n_timesteps=n_timesteps))  # type: ignore
    plt.show()


def main() -> None:
    """Run the requested SE2 trajectory example from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name", help="Name of example to run")
    options = parser.parse_args()

    example_name = options.example_name
    if example_name == "first":
        example_first()
    elif example_name == "transform":
        example_transform()
    else:
        print("unrecognized example name: {example_name}")


if __name__ == "__main__":
    main()
