#!/urs/bin/env python
# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging

import synchros2.process as ros_process
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from spatialmath import SE3, UnitQuaternion
from synchros2.futures import wait_for_future
from synchros2.utilities import ensure

from spot_basic import FrameHint, SpotBasic


class MoveEEWithBodyFollowExample:
    """
    Move Spot arm end-effector and have the robot follow.

    Provided you have a Spot with arm, you may try running the following
    variations:

    python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25
    python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25 --ac
    python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25 0 15 0 --ac
    python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25 0 15 0

    Note that 5 meters in front of the robot are required to safely execute
    these commands as given. The Spot driver must be running too before
    attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._body_t_hand = args.body_t_hand
        self._manual = args.manual
        self._asynchronous = args.ac

    def start(self) -> int:
        with self._robot.leash(timeout_sec=10.0):
            vision_t_hand = (
                self._robot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True, timeout=5.0)
                * self._body_t_hand
            )

            if self._manual:
                duration_sec = 3.0
                arm_command = RobotCommandBuilder.arm_pose_command(
                    vision_t_hand.position.x,
                    vision_t_hand.position.y,
                    vision_t_hand.position.z,
                    vision_t_hand.rotation.w,
                    vision_t_hand.rotation.x,
                    vision_t_hand.rotation.y,
                    vision_t_hand.rotation.z,
                    VISION_FRAME_NAME,
                    duration_sec,
                )

                follow_arm_command = RobotCommandBuilder.follow_arm_command()
                synchro_command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)
                if self._asynchronous:
                    action = self._robot.execute.asynchronous(synchro_command)
                    wait_for_future(action.finalization, timeout_sec=10.0)
                else:
                    self._robot.execute.synchronous(synchro_command, timeout_sec=10.0)
            else:
                if self._asynchronous:
                    action = self._robot.move_end_effector.asynchronously(
                        vision_t_hand, duration_sec=3.0, body_follow=True
                    )
                    wait_for_future(action.finalization, timeout_sec=10.0)
                else:
                    self._robot.move_end_effector.synchronously(
                        vision_t_hand, duration_sec=3.0, body_follow=True, timeout_sec=10.0
                    )
            self._logger.info("Done moving the end-effector")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=MoveEEWithBodyFollowExample.__doc__)
    parser.add_argument("--manual", action="store_true", help="If True, craft robot command manually")
    parser.add_argument("--ac", "--asynchronous", action="store_true", help="If True, call the async method")
    parser.add_argument("robot_name", help="Name of the robot")
    parser.add_argument(
        "body_t_hand",
        type=float,
        nargs="+",
        help=(
            "target SE3 pose of hand in body frame. "
            "Could be either of length 3 (x,y,z), 6 (x,y,z,roll,pitch,yaw), or 7 (x,y,z,qx,qy,qz,qw). "
            "Note that roll, pitch, yaw are in degrees."
        ),
    )
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    if len(args.body_t_hand) == 3:
        x, y, z = args.body_t_hand
        quaternion = UnitQuaternion(s=1.0, v=[0.0, 0.0, 0.0])
    elif len(args.body_t_hand) == 6:
        x, y, z, thx, thy, thz = args.body_t_hand
        quaternion = UnitQuaternion.RPY([thx, thy, thz], unit="deg")
    elif len(args.body_t_hand) == 7:
        x, y, z, qx, qy, qz, qw = args.body_t_hand
        quaternion = UnitQuaternion(s=qw, v=[qx, qy, qz])
    else:
        ensure(main.cli).error(f"invalid body_t_hand: {args.body_t_hand}")
    args.body_t_hand = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])

    example = MoveEEWithBodyFollowExample(args)
    return example.start()


if __name__ == "__main__":
    main()
