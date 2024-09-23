#  Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import logging
import sys

import open3d
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.image import SpotCameraSubscription

import bdai_ros2_wrappers.process as ros_process
from bdai_ros2_wrappers.utilities import fqn


class Example:
    """
    Spot takes point clouds from multiple cameras and stitches them together.

    The resultant point cloud is displayed in an open3d visualizer.

    Spot may move in place to a capture a more comprehensive point cloud.

    Use the following command to run the example:
    `python example_create_point_cloud_in_place.py <robot_name>`
    """

    def __init__(self, args: argparse.Namespace) -> None:
        camera_locations: list[SpotCameraSubscription.Location] = args.cameras
        robot_name: str = args.robot_name
        self._camera_locations = camera_locations
        self._logger = logging.getLogger(fqn(self.__class__))
        self._robot = SpotBasic(robot_name=robot_name)

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Attempting to stand robot up")
            self._robot.stand.synchronously()
            self._logger.info("Robot is standing up")

            cameras = [
                camera
                for camera in self._robot.cameras
                if camera is not None and camera.location in self._camera_locations
            ]

            pointcloud = self._robot.scan_in_place(cameras=cameras)
            self._logger.info("Attempting to draw pointcloud")
            open3d.visualization.draw_geometries([pointcloud], window_name="Pointcloud")
            self._logger.info("Drew pointcloud")

        return 0


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")

    parser.add_argument(
        "-c",
        "--cameras",
        nargs="+",
        help="Choose one or more cameras to create the image",
        choices=list(SpotCameraSubscription.Location),
        default=list(SpotCameraSubscription.Location),
    )

    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = Example(args)

    return example.start()


if __name__ == "__main__":
    sys.exit(main())
