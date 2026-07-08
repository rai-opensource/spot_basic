#!/usr/bin/env python
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process

from spot_basic import SpotBasic


class ZeroGravityExample:
    """
    Put Spot arm in zero-gravity mode (ie. low joint stiffness).

    The Spot driver must be started before executing this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            self._robot.ready_arm.synchronously()
            self._logger.info("Attempting to set arm in zero gravity mode")
            self._robot.arm_zero_g.synchronously()
            self._logger.info("Arm is in zero gravity mode")
            ros_process.wait_for_interrupt()
            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ZeroGravityExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ZeroGravityExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
