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
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot recieves image and camera data. It then visualizes rgb, depth, and pointcloud images from a camera

    Use the following command to run the example:
    `python example_spot_image_viz.py <robot_name>`
    """

    def __init__(self, args: argparse.Namespace) -> None:
        robot_name: str = args.robot_name
        self._camera_location: SpotCameraSubscription.Location = args.camera_location
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(robot_name=robot_name)

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")
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
                raw_image = cv_bridge.imgmsg_to_cv2(image_message)
                rotated_image = ndimage.rotate(raw_image, offset_rotation_angle)
                image = open3d.geometry.Image(rotated_image)
                open3d.visualization.draw_geometries([image], window_name=f"Image with {sensor.name}")
                self._logger.info(f"Drew image with {sensor.name}")

            pointcloud = self._robot.scan_in_place().voxel_down_sample(voxel_size=0.02)
            self._logger.info("Attempting to draw pointcloud")
            open3d.visualization.draw_geometries([pointcloud], window_name="Pointcloud")
            self._logger.info("Drew pointcloud")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    parser.add_argument(
        "-c",
        "--camera-location",
        help="Choose a camera to get the image.",
        choices=list(SpotCameraSubscription.Location),
        default=SpotCameraSubscription.Location.FRONT_LEFT,
    )

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
