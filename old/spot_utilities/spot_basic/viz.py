# Copyright (c) 2024-2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from contextlib import closing, contextmanager
from typing import Any, Iterator, Optional, Tuple

import cv2
import cv_bridge
from sensor_msgs.msg import CameraInfo, Image
from spot_utilities.spot_basic.image import CameraSubscription


def prompt_for_click(
    name: str, camera: CameraSubscription, rate_ms: int = 10
) -> Optional[Tuple[Tuple[int, int], Image, CameraInfo]]:
    """Prompt the user to click on a camera view.

    Returns:
        Image pixel clicked by the user, as well as the image and the camera info at the time of the click.
    """
    cv2.namedWindow(name)
    try:
        image: Optional[cv2.Mat] = None
        image_click_xy: Optional[Tuple[int, int]] = None
        image_hover_xy: Optional[Tuple[int, int]] = None

        def mouse_callback(event: Any, x: float, y: float, flags: Any, param: Any) -> None:
            nonlocal image, image_click_xy, image_hover_xy
            if event != cv2.EVENT_LBUTTONUP:
                image_hover_xy = (int(x), int(y))
                if image is not None:
                    x, y = image_hover_xy
                    height, width, _ = image.shape
                    rendered_image = image.copy()
                    cv2.line(rendered_image, (0, y), (width, y), line_color, line_thickness)
                    cv2.line(rendered_image, (x, 0), (x, height), line_color, line_thickness)
                    cv2.imshow(name, rendered_image)
            else:
                image_click_xy = (int(x), int(y))

        cv2.setMouseCallback(name, mouse_callback)

        line_thickness = 2
        line_color = (30, 30, 30)
        bridge = cv_bridge.CvBridge()
        with closing(camera.image_with_info.stream()) as imagestream:
            for image_message, camera_info_message in imagestream:
                image = bridge.imgmsg_to_cv2(image_message, "passthrough")

                if image_hover_xy is not None:
                    x, y = image_hover_xy
                    height, width, _ = image.shape
                    rendered_image = image.copy()
                    cv2.line(rendered_image, (0, y), (width, y), line_color, line_thickness)
                    cv2.line(rendered_image, (x, 0), (x, height), line_color, line_thickness)
                else:
                    rendered_image = image
                cv2.imshow(name, rendered_image)

                key = cv2.waitKey(rate_ms) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    break

                if image_click_xy is not None:
                    return image_click_xy, image_message, camera_info_message
            else:
                raise RuntimeError("Image stream stopped")
            return None
    finally:
        cv2.destroyWindow(name)


@contextmanager
def visual_aid(name: str, camera: CameraSubscription) -> Iterator:
    """Render camera view as visual aid."""
    cv2.namedWindow(name)
    try:
        cv2.startWindowThread()
        bridge = cv_bridge.CvBridge()

        def imshow(image: Image) -> None:
            cv2.imshow(name, bridge.imgmsg_to_cv2(image, "passthrough"))

        with closing(camera.image.recall(imshow)):
            yield
    finally:
        cv2.destroyWindow(name)
