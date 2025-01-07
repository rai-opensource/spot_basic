#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute, Inc. All rights reserved.
import argparse
import logging
import sys

import synchros2.process as ros_process
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.image import SpotCameraSubscription
from spot_utilities.spot_basic.viz import prompt_for_click


class WalkToObjectExample:
    """
    Have Spot walk to an object.

    When running this example, a GUI will pop up showing the chosen Spot camera feed
    for you to click the object that Spot will walk towards.

    Use:
      1) Have a spot with an object in the field of view of a spot camera
      2) start spot driver
      3) run "python example_walk_to_object_image.py <robot_name> [-c frontleft]"
      4) a GUI will pop up, click the target object, spot will walk to it (distance determined by offset)
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._camera = getattr(self._robot.cameras, args.camera_name)

    def start(self) -> int:
        """Takes an image and attempts to grasp an object based on what the user selected."""
        with self._robot.leash(timeout_sec=10.0):
            self._logger.info("Waiting for user input...")
            if self._camera.location == SpotCameraSubscription.Location.HAND:
                self._robot.claw_gripper_open.synchronously(timeout_sec=10.0)

            result = prompt_for_click("Click to grasp", self._camera.rgb)
            if result is None:
                self._logger.info("User aborted the demonstration.")
                return 1
            image_click_xy, image_message, camera_info_message = result

            self._logger.info("Walking to object...")
            self._robot.walk_to_object_image.synchronously(
                image_pixel_xy=image_click_xy,
                image_timestamp=image_message.header.stamp,
                camera_info=camera_info_message,
                timeout_sec=30.0,
            )
            self._logger.info("Done walking")
            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=WalkToObjectExample.__doc__)
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
    example = WalkToObjectExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
