#!/usr/bin/env python
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import sys

import numpy as np
import synchros2.process as ros_process
from spatialmath import SE2, SE3, SO3
from spot_basic import SpotBasic

from bdai.trajectory.se2_trajectory import SE2Trajectory  # FIXME
from bdai.trajectory.spot_se2_properties import MAX_SPEED_FORWARD  # FIXME


class FrameMovementsExample:
    """
    Spot runs through different frame movements.

    In this example, Spot will move its end effector, walk forward, rotate,
    and walk in a circle. The Spot driver must be started before executing
    this example too.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            self._robot.ready_arm.synchronously()

            self._logger.info("Attempting to move end effector upwards")
            frame_t_goal = SE3.Rt(SO3.Rx(np.pi / 6), t=[1.0, 0.0, 0.5])

            self._robot.move_end_effector_cartesian.synchronously(
                frame_t_goal=frame_t_goal, duration_sec=2.0, body_follow=True, base_frame=self._robot.FrameHint.BODY
            )

            self._logger.info("Moved end effector upwards")
            self._logger.info("Attempting to move end effector downwards")
            self._robot.stow_arm.synchronously()
            self._logger.info("Moved end effector downards")
            self._logger.info("Attempting to make robot body gaze at the origin of the odometry frame")

            frame_t_body = self._robot.lookup_a_tform_b(
                frame_a=self._robot.odom_frame_name, frame_b=self._robot.body_frame_name
            )

            self._robot.body_gaze_at(
                body_t_target=frame_t_body.inv(), frame_t_body=frame_t_body, base_frame=self._robot.FrameHint.ODOM
            )

            self._logger.info("Body gazed at the origin of the odometry frame")
            self._logger.info("Attempting to walk forward 1 meter as specified in the odometry frame")

            odom_t_body = self._robot.lookup_a_tform_b(
                frame_a=self._robot.odom_frame_name, frame_b=self._robot.body_frame_name
            ).yaw_SE2()

            body_t_goal = SE2(1.0, 0.0, 0.0)
            frame_t_goal = odom_t_body * body_t_goal
            self._robot.walk.synchronously(frame_t_goal=frame_t_goal, base_frame=self._robot.FrameHint.ODOM)
            self._logger.info("Walked forward 1 meter as specified in the odometry frame")
            self._logger.info("Attempting to walk forward 1.5 meters and then turn around")

            odom_t_body = self._robot.lookup_a_tform_b(
                frame_a=self._robot.odom_frame_name, frame_b=self._robot.body_frame_name
            )

            body_t_goal = SE3.Rt(SO3.Rz(np.pi), t=[1.5, 0.0, 0.0])
            frame_t_goal = odom_t_body * body_t_goal
            self._robot.walk_and_orient.synchronously(frame_t_goal=frame_t_goal, base_frame=self._robot.FrameHint.ODOM)
            self._logger.info("Walked forward 1.5 meters and then turned around")
            self._logger.info("Attempting to walk in a circle with a radius of 1.25 meters")

            odom_t_body = self._robot.lookup_a_tform_b(
                frame_a=self._robot.odom_frame_name, frame_b=self._robot.body_frame_name
            ).yaw_SE2()

            body_t_traj = SE2Trajectory.from_circle(radius=1.25, forward_speed=MAX_SPEED_FORWARD)
            frame_t_trajectory = odom_t_body * body_t_traj

            self._robot.walk_trajectory.synchronously(
                frame_t_trajectory=frame_t_trajectory, base_frame=self._robot.FrameHint.ODOM
            )

            self._logger.info("Walked in a circle with a radius of 1.25 meters")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=FrameMovementsExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = FrameMovementsExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
