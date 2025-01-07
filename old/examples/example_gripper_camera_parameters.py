#!/usr/bin/python3
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import logging
import sys

import synchros2.process as ros_process
from bosdyn.api.gripper_camera_param_pb2 import GripperCameraParamRequest
from spot_utilities.spot_basic import SpotBasic


class GripperCameraParametersExample:
    """
    Query and set Spot gripper camera parameters.

    This example shows how to configure Spot's gripper camera.
    You need a Spot with an arm. The Spot driver must be running
    too before executing this example.
    """

    def __init__(self, args: argparse.Namespace):
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash(timeout_sec=10.0):
            self._logger.info("Reading gamma value...")
            response = self._robot.get_gripper_camera_parameters.synchronously(timeout_sec=10.0)
            self._logger.info(f"Initial gamma value is: {response.response.params.gamma.data}")

            initial_gamma = response.response.params.gamma.data

            request = GripperCameraParamRequest()
            request.params.gamma.value = 0.8
            self._logger.info("Setting gamma value...")
            self._robot.set_gripper_camera_parameters.synchronously(request, timeout_sec=10.0)

            response = self._robot.get_gripper_camera_parameters.synchronously(timeout_sec=10.0)
            self._logger.info(f"Updated gamma value is: {response.response.params.gamma.data}")

            request = GripperCameraParamRequest()
            request.params.gamma.value = initial_gamma
            self._logger.info("Resetting gamma value...")
            self._robot.set_gripper_camera_parameters.synchronously(request, timeout_sec=10.0)

            response = self._robot.get_gripper_camera_parameters.synchronously(timeout_sec=10.0)
            self._logger.info(f"Final gamma value is: {response.response.params.gamma.data}")

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=GripperCameraParametersExample.__doc__)
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = GripperCameraParametersExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
