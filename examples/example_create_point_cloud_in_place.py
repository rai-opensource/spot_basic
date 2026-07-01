#!/usr/bin/env python
# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import logging
import sys
from typing import Optional

import open3d
import synchros2.process as ros_process

from spot_basic import SpotBasic


class PointCloudInPlaceExample:
    """
    Spot takes point clouds from multiple cameras and stitches them together.

    The resultant point cloud is displayed in an open3d visualizer.
    Spot may move in place to a capture a more comprehensive point cloud.
    The Spot driver must be started before executing this example too.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._cameras: Optional[list] = None
        if args.cameras:
            self._cameras = []
            for camera_name in args.cameras:
                camera = getattr(self._robot.cameras, camera_name)
                if camera is None:
                    self._logger.warn(f"{camera_name} is not available, skipping")
                    continue
                self._cameras.append(camera)

    def start(self) -> int:
        with self._robot.leash():
            self._logger.info("Scanning the environment...")
            pointcloud = self._robot.scan_in_place(self._cameras)
            self._logger.info("Attempting to render pointcloud")
            open3d.visualization.draw_geometries([pointcloud], window_name="Pointcloud")
            self._logger.info("Done!")
            return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=PointCloudInPlaceExample.__doc__)
    available_cameras = list(SpotBasic.CameraConfiguration._fields)
    parser.add_argument(
        "-c",
        "--cameras",
        nargs="+",
        help="Choose one or more cameras to create the image",
        choices=available_cameras,
        default=[],
    )
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = PointCloudInPlaceExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
