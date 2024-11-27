#!/usr/bin/env python
# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import numpy as np
from spatialmath import SE3, SpatialForce
from spatialmath.spatialvector import SpatialVector
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process


class ImpedanceControlExample:
    """
    Make circle-like trajectories using Spot arm impedance control.

    You need a Spot with an arm. The Spot driver must be running too
    before executing this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            root_t_task = SE3.Tx(0.7)
            sweep = np.linspace(0, 2 * np.pi, 50)
            radius = 0.15
            time_scale = 10 / sweep[-1]
            wrist_t_tool = SE3()

            trajectory = [
                (SE3(x=radius * np.cos(angle), y=radius * np.sin(angle), z=0), time_scale * angle) for angle in sweep
            ]

            self._logger.info("Attempting to run trajectory with stiff impedance control")

            self._robot.move_end_effector_impedance.synchronously(
                root_frame=self._robot.FrameHint.BODY,
                root_t_task=root_t_task,
                wrist_t_tool=wrist_t_tool,
                trajectory=trajectory,
            )

            self._logger.info("Trajectory completed")
            self._logger.info("Attempting to run trajectory with soft impedance control and small wrench")

            self._robot.move_end_effector_impedance.synchronously(
                root_frame=self._robot.FrameHint.BODY,
                root_t_task=root_t_task,
                wrist_t_tool=wrist_t_tool,
                trajectory=trajectory,
                stiffness=SpatialVector([100, 100, 0, 20, 20, 20]),
                damping=SpatialVector([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
                wrench_nm=SpatialForce([0, 0, 4, 0, 0, 0]),
            )

            self._logger.info("Trajectory completed")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ImpedanceControlExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ImpedanceControlExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
