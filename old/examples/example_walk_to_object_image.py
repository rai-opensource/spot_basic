#!/usr/bin/env python
#  Copyright (c) 2023 Boston Dynamics AI Institute, Inc. All rights reserved.
import argparse
import sys
from typing import Any

import cv2
import numpy as np
import rclpy
from bdai_ros.utilities.wait_for_message import wait_for_message
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from spot_utilities.spot_basic import SpotBasic
from tf2_ros import ExtrapolationException

""" This is a test script to demo using the manipulation api.

Use:
    0) Have a spot with an object in the field of view of a spot camera
    1) start spot driver
    2) run "python example_walk_to_object_image.py <robot_name> [-c frontleft]"
    3) a GUI will pop up, click the target object, spot will walk to it (distance determined by offset)
"""

# Assumes the robot with an arm is sitting or standing in an open environment, with an desired object to grasp in the
# field of view of the given camera.
AVAILABLE_CAMERA_NAMES = ["frontleft", "frontright", "left", "right", "back", "hand"]
DISTANCE_OFFSET = None  # None to use default offset

g_image_click, g_image_display = None, np.zeros((460, 640, 3))


def cv_mouse_callback(event: Any, x: float, y: float, flags: Any, param: Any) -> None:
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        color = (30, 30, 30)
        thickness = 2
        image_title = "Movement target"
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


class ObjectWalk(Node):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__("object_walk", namespace=options.robot)

        self._body_frame_name = options.robot + "/" + BODY_FRAME_NAME
        self._vision_frame_name = options.robot + "/" + VISION_FRAME_NAME
        self._tf_listener = TFListenerWrapper(
            "object_walk_tf", cache_time_s=30, wait_for_transform=[self._body_frame_name, self._vision_frame_name]
        )
        self._robot = SpotBasic(robot_name=options.robot, parent_node_name=self.get_name())
        self._robot.set_tf_listener(self._tf_listener)
        self.cv_bridge: CvBridge = CvBridge()

        self._camera_image_topic = f"/{options.robot}/camera/{options.camera_name}/image"
        self._camera_image_info_topic = f"/{options.robot}/camera/{options.camera_name}/camera_info"

    def initialize_robot(self) -> bool:
        return self._robot.initialize_robot() and self._robot.stow_arm() and self._robot.stand()

    def take_image_and_walk(self) -> None:
        """Takes an image and attempts to grasp an object based on what the user selected."""
        self.get_logger().info("Waiting for images...")

        # Get image.
        (image_msg_received, image_message) = wait_for_message(
            msg_type=Image, node=self, topic=self._camera_image_topic
        )

        if image_msg_received:
            self.get_logger().info("Got image.")
        else:
            self.get_logger().error(f"Failed to get message on topic {self._camera_image_topic}")
            return

        # Get camera info.
        (cam_info_msg_received, cam_info_msg) = wait_for_message(
            msg_type=CameraInfo, node=self, topic=self._camera_image_info_topic
        )

        if cam_info_msg_received:
            self.get_logger().info("Got camera info.")
        else:
            self.get_logger().error(f"Failed to get message on topic {self._camera_image_topic}")
            return

        image = self.cv_bridge.imgmsg_to_cv2(image_message, "passthrough")

        # Launch a window to let the user select an object.
        image_title = "Click to set object target"
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_display = image
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                # Quit
                self.get_logger().info("'q' pressed, exiting GUI.")
                cv2.destroyAllWindows()
                return
        cv2.destroyAllWindows()
        # Walk to the object in the image.
        try:
            self._robot.walk_to_object_image(
                pixel_xy=g_image_click,
                image_timestamp=image_message.header.stamp,
                camera_data=cam_info_msg,
                offset=DISTANCE_OFFSET,
            )
        except ExtrapolationException:
            self.get_logger().info("Camera image expired. You waited to long to click on the image.")

    def shutdown(self) -> None:
        self._tf_listener.shutdown()


def main() -> int:
    rclpy.init()
    """Command-line interface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace.")
    parser.add_argument(
        "-c",
        "--camera-name",
        help=f"Choose a camera to get the image {AVAILABLE_CAMERA_NAMES}.",
        default="frontleft",
    )
    options = parser.parse_args()

    if options.camera_name not in AVAILABLE_CAMERA_NAMES:
        raise ValueError(
            f"Invalid camera name '{options.camera_name}'. Select a camera name from {AVAILABLE_CAMERA_NAMES}."
        )

    objw = ObjectWalk(options)
    if objw.initialize_robot():
        objw.take_image_and_walk()
    objw.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
