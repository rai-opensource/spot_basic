#!/usr/bin/env python
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from spot_basic import SpotBasic


class SpotBasicExample:
    """
    Show basic SpotBasic use.

    In this example, Spot will stand up, ready its arm, then stow it, then sit.
    The Spot driver must be running before running this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash(weak=True):
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")
            self._logger.info("Attempting to ready arm")
            self._robot.ready_arm.synchronously()
            self._logger.info("Arm is ready")
            self._logger.info("Attempting to stow arm")
            self._robot.stow_arm.synchronously()
            self._logger.info("Arm is stowed")
            self._robot.claw_gripper_open.synchronously()
            self._logger.info("Gripper is open")
            self._robot.claw_gripper_close.synchronously()
            self._logger.info("Gripper is closed")
            self._logger.info("Attempting to sit robot down")
            self._robot.sit.synchronously()
            self._logger.info("Robot is sitting")
            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=SpotBasicExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = SpotBasicExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
