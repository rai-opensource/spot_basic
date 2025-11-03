#!/usr/bin/env python
# Copyright (c) 2023-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import synchros2.process as ros_process
from spatialmath import SE2
from spot_utilities.spot_basic import FrameHint, SpotBasic
from synchros2.futures import wait_for_future

from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai.trajectory.se2_trajectory_plotting import draw_shape
from bdai.trajectory.spot_se2_properties import BODY_SHAPE, MAX_SPEED_FORWARD, MAX_SPEED_VEC


class TrajectoryCommandExample:
    """
    Have Spot walk along a trajectory.

    There are three sample trajectories:
     - a "simple" trajectory, translating and rotating
     - a "basic_circle" trajectory, clockwise
     - a "counterspin_circle" trajectory, counter clockwise

    To use this example:
     1) start spot driver
     2) make sure there isn't anything within a 2 meters of spot
     3) python relative/path/to/this/script/here arguments
     4) spot will walk in a pattern based on the example you picked
    """

    AVAILABLE_TRAJECTORIES = {
        "simple": SE2Trajectory.from_se2_pair(
            target=SE2([1, 2, -np.pi / 2]),
            se2_speed_vec=MAX_SPEED_VEC,
            skewness=0.5,
        ),
        "basic_circle": SE2Trajectory.from_circle(radius=1.5, forward_speed=MAX_SPEED_FORWARD),
        "counterspin_circle": SE2Trajectory.from_circle(radius=1.5, total_body_rotation=-np.pi * 2, forward_speed=0.8),
    }

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._plot_preview = args.plot_preview
        self._asynchronous = args.asynchronous

        traj = TrajectoryCommandExample.AVAILABLE_TRAJECTORIES
        self._body_t_trajectory = traj[args.trajectory_name]

    def start(self) -> int:
        if self._plot_preview:
            self._logger.info("Close trajectory preview to proceed.")
            _, ax = plt.subplots(1, 1)
            trajectory = self._body_t_trajectory(n_timesteps=20)
            draw_shape(ax, trajectory, BODY_SHAPE)  # type: ignore
            ax.axis("equal")
            plt.show()

        with self._robot.leash(timeout_sec=10.0):
            vision_t_body = self._robot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, timeout=10.0).yaw_SE2()
            vision_t_trajectory = vision_t_body * self._body_t_trajectory

            self._logger.info("Walk along trajectory...")
            if self._asynchronous:
                action = self._robot.walk_trajectory.asynchronously(vision_t_trajectory)
                wait_for_future(action.finalization, timeout_sec=20.0)
                if not action.success:
                    self._logger.error("Failed to walk along trajectory")
                    return 1
            else:
                self._robot.walk_trajectory.synchronously(vision_t_trajectory, timeout_sec=30.0)

            self._logger.info("Done walking")

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=TrajectoryCommandExample.__doc__)
    parser.add_argument(
        "-p", "--plot-preview", help="Plot trajectory before executing it", action="store_true", default=False
    )
    parser.add_argument("-a", "--asynchronous", help="Execute trajectory asynchronously", default=False)
    parser.add_argument("robot_name", help="Name of the robot")
    parser.add_argument(
        "trajectory_name",
        choices=TrajectoryCommandExample.AVAILABLE_TRAJECTORIES,
        help="Name of sample trajectory to execute",
    )
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = TrajectoryCommandExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
