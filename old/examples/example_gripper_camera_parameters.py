#!/usr/bin/python3

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# We disable Pylint warnings for all Protobuf files which contain objects with
# dynamically added member attributes.
# pylint: disable=no-member

"""
This example shows how to query and set the camera parameters for the gripper.
"""

import argparse
import logging

from bosdyn_msgs.msg import GripperCameraParams
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class SpotGripperCameraParameters:
    """
    This example shows how to query and set the camera parameters for the gripper.
    """

    def __init__(self, args: argparse.Namespace):
        self._robot_name: str = args.robot
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(self._robot_name)

    def test_params(self) -> bool:
        """
        Query the current gripper camera parameters, change the gamma value, then reset the gamma value using
        get_gripper_camera_params and set_gripper_camera_params functions in SpotBasic.

        Returns:
            True the process runs without errors, False otherwise.
        """

        # Claim and power on robot.
        if not self._robot.initialize_robot():
            self._logger.error("Robot could not be initialized")
            return False

        # get gripper camera parameters
        success, msg, response = self._robot.get_gripper_camera_parameters()
        if not success:
            self._logger.error("Failed to get gripper camera parameters")
            return False
        gamma_init = response.gamma.data
        print("Initially gamma value is: {}".format(gamma_init))

        gripper_params = GripperCameraParams()
        # set gripper camera parameters with a new gamma value, leaving all others unchanged
        new_gamma = 0.8
        self._logger.info("Setting a new gamma in the gripper camera")

        gripper_params.gamma.data = new_gamma
        gripper_params.has_field = GripperCameraParams.GAMMA_FIELD_SET
        success = self._robot.set_gripper_camera_parameters(gripper_params)
        if not success:
            self._logger.error("Failed to set gripper camera parameters")
            return False

        # get gripper camera parameters
        self._logger.info("Reading updated camera parameters")
        success, msg, response = self._robot.get_gripper_camera_parameters()
        if not success:
            self._logger.error("Failed to get gripper camera parameters")
            return False
        print("Updated gamma value: {}".format(response.gamma.data))

        # reset camera parameters
        self._logger.info("Resetting gripper camera parameters")
        gripper_params.gamma.data = gamma_init
        success = self._robot.set_gripper_camera_parameters(gripper_params)
        if not success:
            self._logger.error("Failed to set gripper camera parameters")
            return False
        self._logger.info("Set new parameter")

        # read camera parameters one more time to make sure it has reset
        self._logger.info("Reading gripper camera parameters, ensure they are the same as the start!")
        success, msg, response = self._robot.get_gripper_camera_parameters()
        if not success:
            self._logger.error("Failed to get gripper camera parameters")
            return False
        print("Final gamma value: {}".format(response.gamma.data))

        return True


def cli() -> argparse.ArgumentParser:
    """
    Parse all arguments.
    --robot [string]
        The robot name e.g. Opal.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The robot name.")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    """
    Execute the example.
    """
    test_gripper_camera_params = SpotGripperCameraParameters(args)
    test_gripper_camera_params.test_params()


if __name__ == "__main__":
    main()
