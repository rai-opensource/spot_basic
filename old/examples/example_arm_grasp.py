#!/usr/bin/env python
# Copyright (c) 2023-2024 Boston Dynamics AI Institute, Inc. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from bosdyn.api import manipulation_api_pb2
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.image import SpotCameraSubscription
from spot_utilities.spot_basic.viz import prompt_for_click


class ArmGraspExample:
    """
    Spot grabs an object with its arm.

    When running this example, a GUI will pop up showing the chosen Spot camera feed
    for you to click the object that Spot will go pick up.

    Make sure you have a Spot with an arm and that the object you want to grasp is within
    the field of view of one of Spot cameras. The Spot driver must be started before
    executing this example too.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._camera = getattr(self._robot.cameras, args.camera_name)

    def start(self) -> int:
        if not self._robot.has_arm():
            self._logger.error("Robot has no arm")
            return 1

        with self._robot.leash(timeout_sec=10.0):
            self._robot.ready_arm.synchronously(timeout_sec=10.0)
            if self._camera.location == SpotCameraSubscription.Location.HAND:
                self._robot.claw_gripper_open.synchronously(timeout_sec=10.0)

            self._logger.info("Waiting for user input...")
            result = prompt_for_click("Click to grasp", self._camera.rgb)
            if result is None:
                self._logger.info("User aborted the demonstration.")
                return 1
            image_click_xy, image_message, camera_info_message = result

            self._logger.info("Grasping object...")
            self._robot.pick_object_in_image.synchronously(
                image_pixel_xy=image_click_xy,
                image_timestamp=image_message.header.stamp,
                camera_info=camera_info_message,
                grasp_params=manipulation_api_pb2.GraspParams(grasp_palm_to_fingertip=0.5),
                walk_gaze_mode=manipulation_api_pb2.WalkGazeMode.PICK_AUTO_WALK_AND_GAZE,
            )
            self._logger.info("Object grasped")
            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ArmGraspExample.__doc__)
    parser.add_argument(
        "-c",
        "--camera-name",
        choices=SpotBasic.CameraConfiguration._fields,
        help="Camera to use get the images from",
        default="front_left",
    )
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ArmGraspExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
