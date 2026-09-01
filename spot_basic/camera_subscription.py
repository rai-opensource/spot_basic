# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Provide ergonomic ROS 2 camera subscriptions and synchronized image feeds."""

from functools import cached_property
from typing import Any, NamedTuple, Optional

import sensor_msgs.msg
from rclpy.node import Node
from synchros2.feeds import AdaptedMessageFeed, MessageFeed, SynchronizedMessageFeed
from synchros2.subscription import Subscription


class ImageWithInfo(NamedTuple):
    """A convenient image and camera info message pair."""

    image: sensor_msgs.msg.Image
    camera_info: sensor_msgs.msg.CameraInfo


class CompressedImageWithInfo(NamedTuple):
    """A convenient compressed image and camera info message pair."""

    compressed_image: sensor_msgs.msg.CompressedImage
    camera_info: sensor_msgs.msg.CameraInfo


class RgbdFrameWithInfo(NamedTuple):
    """A convenient rgb image, depth image, and camera info message tuple."""

    rgb: sensor_msgs.msg.Image
    depth: sensor_msgs.msg.Image
    camera_info: sensor_msgs.msg.CameraInfo


class CameraSubscription:
    """An ergonomic camera subscription in ROS 2.

    Simply a collection for well-known topic subscriptions.
    """

    def __init__(
        self,
        name: str,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the camera subscription.

        Args:
            name: camera name, used for topic namespacing.
            history_length: optional historic data size, defaults to 1.
            node: optional node for subscription, defaults to the current process node.
            kwargs: other keyword arguments for `synchros2.subscription.Subscription` instantiation.
        """
        self._name = name
        self._history_length = history_length
        self._node = node
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Get camera name (ie. camera topics' prefix)."""
        return self._name

    @cached_property
    def image(self) -> Subscription[sensor_msgs.msg.Image]:
        """Get camera image topic subscription."""
        return Subscription(
            sensor_msgs.msg.Image,
            f"{self.name}/image",
            history_length=self._history_length,
            node=self._node,
            **self._kwargs,
        )

    @cached_property
    def compressed_image(self) -> Subscription[sensor_msgs.msg.CompressedImage]:
        """Get camera compressed image topic subscription."""
        return Subscription(
            sensor_msgs.msg.CompressedImage,
            f"{self.name}/compressed",
            history_length=self._history_length,
            node=self._node,
            **self._kwargs,
        )

    @cached_property
    def info(self) -> Subscription[sensor_msgs.msg.CameraInfo]:
        """Get camera info topic subscription."""
        return Subscription(
            sensor_msgs.msg.CameraInfo,
            f"{self.name}/camera_info",
            history_length=self._history_length,
            node=self._node,
            **self._kwargs,
        )

    @cached_property
    def image_with_info(self) -> MessageFeed[ImageWithInfo]:
        """Get synchronized feed of camera image and info message pairs."""
        return AdaptedMessageFeed(
            SynchronizedMessageFeed(
                self.image,
                self.info,
                history_length=0,
                node=self._node,
            ),
            fn=ImageWithInfo,
            history_length=self._history_length,
            node=self._node,
        )

    @cached_property
    def compressed_image_with_info(self) -> MessageFeed[CompressedImageWithInfo]:
        """Get synchronized feed of camera compressed image and info message pairs."""
        return AdaptedMessageFeed(
            SynchronizedMessageFeed(
                self.compressed_image,
                self.info,
                history_length=0,
                node=self._node,
            ),
            fn=CompressedImageWithInfo,
            history_length=self._history_length,
            node=self._node,
        )
