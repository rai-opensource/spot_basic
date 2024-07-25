# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from contextlib import contextmanager
from typing import Any, Iterator, Tuple

import cv2
import numpy as np


def prompt_for_click(name: str, image: np.ndarray) -> Tuple[int, int]:
    """Prompt the user to click on a image view.
    Returns:
        Image pixel clicked by the user.
    """
    cv2.namedWindow(name)

    image_pixel_xy = None

    def mouse_callback(event: Any, x: float, y: float, flags: Any, param: Any) -> None:
        nonlocal image_pixel_xy
        if event == cv2.EVENT_LBUTTONUP:
            image_pixel_xy = (int(x), int(y))

    cv2.setMouseCallback(name, mouse_callback)

    cv2.imshow(name, image)
    while image_pixel_xy is None:
        cv2.waitKey(10)
    cv2.destroyWindow(name)
    return image_pixel_xy


@contextmanager
def visual_aid(name: str, image: np.ndarray) -> Iterator:
    """Render image as visual aid."""
    cv2.namedWindow(name)
    try:
        cv2.startWindowThread()
        cv2.imshow(name, image)
        yield
    finally:
        cv2.destroyWindow(name)
