#!/usr/bin/env python3
# Copyright (c) 2023-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import logging
import sys

import synchros2.process as ros_process
from spatialmath import SE3, UnitQuaternion
from spot_utilities.spot_basic import FrameHint, SpotBasic
from synchros2.utilities import ensure


class WalkTo6DPoseExample:
    """
    Have Spot walk to a specific pose.

    The Spot driver must be running before attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._relative = args.relative
        self._goal = args.goal

    def start(self) -> int:
        with self._robot.leash(timeout_sec=10.0):
            if self._relative:
                vision_t_body = self._robot.lookup_a_tform_b(
                    FrameHint.VISION, FrameHint.BODY, wait_for_frames=True, timeout=5.0
                )
                vision_t_goal = vision_t_body * self._goal
            else:
                vision_t_goal = self._goal

            x, y, z = vision_t_goal.t
            rx, ry, rz = vision_t_goal.rpy()
            self._logger.info(
                "\n".join(
                    [
                        "Moving to 6D pose:" f"- x: {x}",
                        f"- y: {y}",
                        f"- z: {z}",
                        f"- rx: {rx}",
                        f"- ry: {ry}",
                        f"- rz: {rz}",
                    ]
                )
            )

            self._robot.walk_and_orient.synchronously(vision_t_goal, timeout_sec=10.0)

            self._logger.info("Done moving!")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=WalkTo6DPoseExample.__doc__)
    parser.add_argument(
        "-r", "--relative", action="store_true", help="goal will be in body frame; Otherwise, goal is in vision frame"
    )
    parser.add_argument("robot_name", type=str, help="name of the robot")
    parser.add_argument(
        "goal",
        type=float,
        nargs="+",
        help=(
            "target SE3 pose in vision frame by default, unless '--relative'."
            "Could be either of length 3 (x,y,z), 6 (x,y,z,roll,pitch,yaw), or 7 (x,y,z,qx,qy,qz,qw)."
            "Note that roll, pitch, yaw are in degrees."
        ),
    )
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    if len(args.goal) == 3:
        x, y, z = args.goal
        quaternion = UnitQuaternion(s=1.0, v=[0.0, 0.0, 0.0])
    elif len(args.goal) == 6:
        x, y, z, thx, thy, thz = args.goal
        quaternion = UnitQuaternion.RPY([thx, thy, thz], unit="deg")
    elif len(args.goal) == 7:
        x, y, z, qx, qy, qz, qw = args.goal
        quaternion = UnitQuaternion(s=qw, v=[qx, qy, qz])
    else:
        ensure(main.cli).error(f"invalid goal: {args.goal}")
    args.goal = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])

    example = WalkTo6DPoseExample(args)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
