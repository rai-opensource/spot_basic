#!/usr/bin/env python
#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import open3d
from cv_bridge import CvBridge
from scipy import ndimage
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.image import SpotCameraSubscription

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.futures import unwrap_future


class SpotImageVizExample:
    """
    Visualize RGB frame, depth map, and pointcloud from a Spot camera.

    The Spot driver must be running before attempting to use this example.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._camera_location: SpotCameraSubscription.Location = args.camera_location
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)

    def start(self) -> int:
        with self._robot.leash():
            cv_bridge = CvBridge()

            camera = next(
                camera
                for camera in self._robot.cameras
                if camera is not None and camera.location == self._camera_location
            )

            offset_rotation_angle = 0

            if (
                camera.location == SpotCameraSubscription.Location.FRONT_LEFT
                or camera.location == SpotCameraSubscription.Location.FRONT_RIGHT
            ):
                offset_rotation_angle = -90

            if camera.location == SpotCameraSubscription.Location.RIGHT:
                offset_rotation_angle = 180

            for sensor in (camera.rgb, camera.depth):
                image_message = unwrap_future(sensor.image.update)
                encoding = image_message.encoding
                if encoding.startswith("bgr"):
                    encoding = encoding.replace("bgr", "rgb")
                raw_image = cv_bridge.imgmsg_to_cv2(image_message, encoding)
                rotated_image = ndimage.rotate(raw_image, offset_rotation_angle)
                image = open3d.geometry.Image(rotated_image)
                open3d.visualization.draw_geometries([image], window_name=f"Image with {sensor.name}")
                self._logger.info(f"Drew image with {sensor.name}")

            pointcloud = self._robot.scan_in_place([camera]).voxel_down_sample(voxel_size=0.02)
            self._logger.info("Attempting to draw pointcloud")
            open3d.visualization.draw_geometries([pointcloud], window_name="Pointcloud")
            self._logger.info("Drew pointcloud")

            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=SpotImageVizExample.__doc__)
    parser.add_argument(
        "-c",
        "--camera-location",
        help="Choose a camera to get the image.",
        choices=list(SpotCameraSubscription.Location),
        default=SpotCameraSubscription.Location.FRONT_LEFT,
    )
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = SpotImageVizExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
