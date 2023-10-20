#  Copyright (c) 2023 Boston Dynamics AI Institute, Inc. All rights reserved.
import argparse
import logging
import sys
from typing import Any

import bdai_ros2_wrappers.process as ros_process
import cv2
import numpy as np
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.subscription import wait_for_message
from bdai_ros2_wrappers.utilities import fqn, namespace_with
from bosdyn.api import manipulation_api_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from spot_utilities.spot_basic import SpotBasic
from tf2_ros import ExtrapolationException

""" This is a test script to demo using the manipulation api.

Use:
    0) have a spot with an arm, put an object you want to grasp in the field of view of a spot camera
    1) start spot driver
    2) python relative/path/to/this/script/here arguments-here
    3) a GUI will pop up, click the target item, spot will pick it up
"""

# Assumes the robot with an arm is sitting or standing in an open environment, with an desired object to grasp in the
# field of view of the given camera.
AVAILABLE_CAMERA_NAMES = ["frontleft", "frontright", "left", "right", "back"]


def _grasp_params_function(
    grasp_command: manipulation_api_pb2.PickObjectInImage,
) -> manipulation_api_pb2.PickObjectInImage:
    """Sets the auto walk and gaze in a grasp parameter.

    This function is passed into `grasp_in_image` defined in the spot_basic.py file.

    Args:
        grasp_command: The proto to modify.
    Returns:
        A modified grasp command with auto walk and gaze enabled.
    """
    grasp_command.walk_gaze_mode = manipulation_api_pb2.WalkGazeMode.PICK_AUTO_WALK_AND_GAZE
    return grasp_command


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
        image_title = "Click to grasp"
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


class ObjectGrasp:
    def __init__(self, robot_name: str, camera_name: str) -> None:
        self._logger = logging.getLogger(fqn(self.__class__))
        self.cv_bridge: CvBridge = CvBridge()
        self._camera_image_topic = namespace_with(robot_name, "camera", camera_name, "image")
        self._camera_image_info_topic = namespace_with(robot_name, "camera", camera_name, "camera_info")

        self._body_frame_name = namespace_with(robot_name, BODY_FRAME_NAME)
        self._vision_frame_name = namespace_with(robot_name, VISION_FRAME_NAME)
        self._tf_listener = TFListenerWrapper(cache_time_s=30)
        self._tf_listener.wait_for_a_tform_b(self._body_frame_name, self._vision_frame_name)

        self._robot = SpotBasic(robot_name)
        self._robot.set_tf_listener(self._tf_listener)

    def initialize_robot(self) -> bool:
        """Initializes the robot.

        Claims, powers on and stands up spot.
        """
        self._logger.info("Claiming robot.")
        success = self._robot.claim()
        if not success:
            return False
        self._logger.info("Claimed robot.")

        # Check if robot has arm.
        if not self._robot.has_arm():
            self._logger.error("Robot has no arm.")
            return False

        # Stand the robot up and stow the arm.
        self._logger.info("Powering robot on.")
        success = self._robot.power_on()
        if not success:
            return False

        self._logger.info("Opening gripper.")
        success = self._robot.claw_gripper_open()
        if not success:
            return False

        self._logger.info("Stowing arm.")
        success = self._robot.stow_arm()
        if not success:
            return False

        self._logger.info("Standing robot up.")
        success = self._robot.stand()
        if not success:
            return False
        self._logger.info("Successfully stood up.")
        return True

    def take_image_and_grasp(self) -> None:
        """Takes an image and attempts to grasp an object based on what the user selected."""
        self._logger.info("Waiting for images...")

        # Get image.
        image_message = wait_for_message(Image, self._camera_image_topic)

        if image_message is not None:
            self._logger.info("Got image.")
        else:
            self._logger.error(f"Failed to get message on topic {self._camera_image_topic}")
            return

        # Get camera info.
        cam_info_msg = wait_for_message(CameraInfo, self._camera_image_info_topic)

        if cam_info_msg is not None:
            self._logger.info("Got camera info.")
        else:
            self._logger.error(f"Failed to get message on topic {self._camera_image_topic}")
            return

        image = self.cv_bridge.imgmsg_to_cv2(image_message, "passthrough")

        # Launch a window to let the user select an object.
        image_title = "Click to grasp"
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_display = image
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                # Quit
                self._logger.info("'q' pressed, exiting GUI.")
                cv2.destroyAllWindows()
                return
        cv2.destroyAllWindows()
        # Grasp the object in the image.
        try:
            self._robot.grasp_in_image(
                pixel_xy=g_image_click,
                image_timestamp=image_message.header.stamp,
                camera_data=cam_info_msg,
                palm_placement_ratio=0.5,
                apply_grasp_params_func=_grasp_params_function,
            )
        except ExtrapolationException:
            self._logger.info("Camera image expired. You waited to long to click on the image.")


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace.")
    parser.add_argument(
        "-c",
        "--camera-name",
        choices=AVAILABLE_CAMERA_NAMES,
        help=f"Choose a camera to get the image {AVAILABLE_CAMERA_NAMES}.",
        default="frontleft",
    )
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    goto = ObjectGrasp(args.robot, args.camera_name)
    if goto.initialize_robot():
        goto.take_image_and_grasp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
