#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_vision import CameraLocation, SpotImages

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot recieves image and camera data. It then visualizes point cloud, rgb, and depth images from a camera

    Use the following command to run the example:
    `python example_spot_image_viz.py <robot_name> -c <camera_location>`
    """

    def __init__(self, robot_name: str, camera_location: str) -> None:
        self._robot = SpotBasic(robot_name=robot_name)
        self._logger = logging.getLogger(fqn(self.__class__))
        self._camera_location = CameraLocation(camera_location)

    def start(self) -> int:
        self._logger.info("Attempting to initialize robot")
        self._robot.initialize_robot()
        self._logger.info("Robot is initialized")

        self._logger.info("Attempting to stand robot up")
        self._robot.stand.synchronously()
        self._logger.info("Robot is standing up")

        spot_images = SpotImages(use_registered_depth=True)

        self._logger.info("Attempting to display point cloud")
        spot_images.display_pointcloud(self._camera_location, self._robot, 10.0)
        self._logger.info("Point cloud finished displaying")

        rotation_angle = 0
        if self._camera_location == CameraLocation.FRONT_LEFT or self._camera_location == CameraLocation.FRONT_RIGHT:
            rotation_angle = -90
        if self._camera_location == CameraLocation.RIGHT:
            rotation_angle = -180

        [rgb, depth] = spot_images.rbg_and_depth_camera_from_location(self._camera_location)

        self._logger.info("Attempting to display rgb image")
        spot_images.display_image(rgb, self._robot, rotation_angle=rotation_angle, timeout_sec=5.0)
        self._logger.info("rgb image finished displaying")

        self._logger.info("Attempting to display depth image")
        spot_images.display_image(depth, self._robot, rotation_angle=rotation_angle, timeout_sec=5.0)
        self._logger.info("Depth image finished displaying")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    parser.add_argument(
        "-c",
        "--camera-location",
        help=f"Choose a camera to get the image:\n{[e.value for e in CameraLocation]}.",
        default="frontleft",
    )

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args.robot_name, args.camera_location)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
