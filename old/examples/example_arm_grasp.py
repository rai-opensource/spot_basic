#  Copyright (c) 2023 Boston Dynamics AI Institute, Inc. All rights reserved.
import argparse
import sys
from typing import Any

import cv2
import numpy as np
import rclpy
from bdai_ros.utilities.wait_for_message import wait_for_message
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.api import manipulation_api_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from spot_utilities.spot_basic import SpotBasic
from tf2_ros import ExtrapolationException

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


class ObjectGrasp(Node):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__("object_grasp", namespace=options.robot)

        self._body_frame_name = options.robot + "/" + BODY_FRAME_NAME
        self._vision_frame_name = options.robot + "/" + VISION_FRAME_NAME
        self._tf_listener = TFListenerWrapper(
            "object_grasp_tf", cache_time_s=30, wait_for_transform=[self._body_frame_name, self._vision_frame_name]
        )
        self._robot = SpotBasic(robot_name=options.robot, parent_node_name=self.get_name())
        self._robot.set_tf_listener(self._tf_listener)
        self.cv_bridge: CvBridge = CvBridge()

        self._camera_image_topic = f"/{options.robot}/camera/{options.camera_name}/image"
        self._camera_image_info_topic = f"/{options.robot}/camera/{options.camera_name}/camera_info"

    def initialize_robot(self) -> bool:
        """Initializes the robot.

        Claims, powers on and stands up spot.
        """
        self.get_logger().info("Claiming robot.")
        success = self._robot.claim()
        if not success:
            return False
        self.get_logger().info("Claimed robot.")

        # Check if robot has arm.
        if not self._robot.has_arm():
            self._node.get_logger().error("Robot has no arm.")
            return False

        # Stand the robot up and stow the arm.
        self.get_logger().info("Powering robot on.")
        success = self._robot.power_on()
        if not success:
            return False

        self.get_logger().info("Opening gripper.")
        success = self._robot.claw_gripper_open()
        if not success:
            return False

        self.get_logger().info("Stowing arm.")
        success = self._robot.stow_arm()
        if not success:
            return False

        self.get_logger().info("Standing robot up.")
        success = self._robot.stand()
        if not success:
            return False
        self.get_logger().info("Successfully stood up.")
        return True

    def take_image_and_grasp(self) -> None:
        """Takes an image and attempts to grasp an object based on what the user selected."""
        # Take image.
        self.get_logger().info("wait for images.")
        (image_success, image_msg) = wait_for_message(Image, node=self, topic=self._camera_image_topic)
        (camera_info_success, camera_info_msg) = wait_for_message(
            CameraInfo, node=self, topic=self._camera_image_info_topic
        )
        self.get_logger().info("got image.")

        if not (image_success and camera_info_success):
            image_success_str = "succeeded" if image_success else "failed"
            camera_success_str = "succeeded" if camera_info_success else "failed"

            self.get_logger().error(
                f"Failed to get the hand camera messages. On topic {self._camera_image_topic} we {image_success_str},"
                f" and on {self._camera_image_info_topic} we {camera_success_str}"
            )
            return

        image = self.cv_bridge.imgmsg_to_cv2(image_msg, "passthrough")

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
                self.get_logger().info("'q' pressed, exiting GUI.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        # Grasp the object in the image.
        try:
            self._robot.grasp_in_image(
                g_image_click, image_msg.header.stamp, camera_info_msg, 0.5, _grasp_params_function
            )
        except ExtrapolationException:
            self.get_logger().info("Camera image expired. You waited to long to click on the image.")

    def shutdown(self) -> None:
        self._tf_listener.shutdown()


def main() -> int:
    rclpy.init()
    """Command-line interface."""
    import argparse

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

    goto = ObjectGrasp(options)
    success = goto.initialize_robot()
    if success:
        goto.take_image_and_grasp()
    goto.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
