#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import bdai_ros2_wrappers.process as ros_process
import numpy as np
from bdai_ros2_wrappers.utilities import fqn
from spatialmath import SE3, SpatialForce
from spatialmath.spatialvector import SpatialVector
from spot_utilities.spot_basic import SpotBasic


class Example:
    """
    Uses arm impedance control to make circle like trajectories.

    Use the following command to run the example:
    `python example_impedance_control.py <robot_name>`
    """

    def __init__(self, robot_name: str) -> None:
        self._robot = SpotBasic(robot_name=robot_name)
        self._logger = logging.getLogger(fqn(self.__class__))

    def start(self) -> int:
        self._logger.info("Attempting to initialize robot")
        self._robot.initialize_robot()
        self._logger.info("Robot is initialized")

        self._logger.info("Attempting to stand robot up")
        self._robot.stand.synchronously()
        self._logger.info("Robot is standing up")

        self._logger.info("Attempting to ready arm")
        self._robot.ready_arm()
        self._logger.info("Arm is ready")

        root_t_task = SE3.Tx(0.7)
        sweep = np.linspace(0, 2 * np.pi, 50)
        radius = 0.15
        time_scale = 10 / sweep[-1]
        circle_trajectory = [
            (SE3(x=radius * np.cos(angle), y=radius * np.sin(angle), z=0), time_scale * angle) for angle in sweep
        ]

        self._logger.info("Attempting to run trajectory with stiff impedance control")

        self._robot.move_end_effector_impedance(
            root_frame_name=self._robot.FrameHint.BODY,
            root_t_task=root_t_task,
            wrist_t_tool=SE3(),
            trajectory=circle_trajectory,
            timeout_sec=circle_trajectory[-1][1] + 0.25,
        )

        self._logger.info("Trajectory completed")

        self._logger.info("Attempting to run trajectory with soft impedance control and small wrench")

        self._robot.move_end_effector_impedance(
            root_frame_name=self._robot.FrameHint.BODY,
            root_t_task=root_t_task,
            wrist_t_tool=SE3(),
            trajectory=circle_trajectory,
            stiffness=SpatialVector([100, 100, 0, 20, 20, 20]),
            damping=SpatialVector([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
            wrench_nm=SpatialForce([0, 0, 4, 0, 0, 0]),
            timeout_sec=circle_trajectory[-1][1] + 0.25,
        )

        self._logger.info("Trajectory completed")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args.robot_name)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
