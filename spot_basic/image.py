# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Iterator, NamedTuple, Optional, TypeAlias

import geometry_msgs.msg
import numpy
import open3d
import sensor_msgs.msg
from cv_bridge import CvBridge
from numpy.typing import NDArray
from rclpy.node import Node
from spatialmath import SO3
from spot_basic.command.mobility import SpotMobilityCommandProtocol
from spot_basic.manual_conversions import open3d_pointcloud_to_ros_pointcloud2, to_se3
from spot_basic.state import FrameHint
from spot_basic.utilities import spot_namespaced
from synchros2.feeds import AdaptedMessageFeed, FramedMessageFeed, MessageFeed, SynchronizedMessageFeed
from synchros2.futures import unwrap_future
from synchros2.utilities import namespace_with

from bdai_ros_perception.camera_subscription import CameraSubscription, RgbdFrameWithInfo


def reconstruct_pointcloud(
    rgb_image_message: sensor_msgs.msg.Image,
    depth_image_message: sensor_msgs.msg.Image,
    camera_intrinsics_message: sensor_msgs.msg.CameraInfo,
    camera_extrinsics_message: geometry_msgs.msg.TransformStamped,
    depth_unit_per_meter: int = 1000,
    mask: Optional[NDArray] = None,
) -> open3d.geometry.PointCloud:
    """Reconstruct a pointcloud given RGBD camera messaging.

    Messages are assumed to be synchronized.

    Args:
        rgb_image_message: an rgb frame.
        depth_image_message: a depth map.
        camera_intrinsics_message: camera pinhole model parameters.
        camera_extrinsics_message: camera pose w.r.t. to the frame of interest.
        depth_unit_per_meter: depth map scale factor to express it in meters.
        Defaults to 1000 (for depth maps expressed in mm).
        mask: optional mask to ignore part of the depth map.
    """
    cv_bridge = CvBridge()

    rgb_image = open3d.geometry.Image(cv_bridge.imgmsg_to_cv2(rgb_image_message, "rgb8"))
    depth_image_data = cv_bridge.imgmsg_to_cv2(depth_image_message)
    if mask is not None:
        depth_image_data[numpy.invert(mask)] = 0
    depth_image = open3d.geometry.Image(depth_image_data.astype(depth_image_data.dtype.type))
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, depth_scale=depth_unit_per_meter, convert_rgb_to_intensity=False
    )

    intrinsics_matrix = camera_intrinsics_message.k.reshape((3, 3))
    camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
        width=camera_intrinsics_message.width,
        height=camera_intrinsics_message.height,
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1],
        cx=intrinsics_matrix[0, 2],
        cy=intrinsics_matrix[1, 2],
    )
    pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

    camera_extrinsics = to_se3(camera_extrinsics_message)
    return pointcloud.transform(camera_extrinsics.A)


def reconstruct_pointcloud_message(
    rgb_image_message: sensor_msgs.msg.Image,
    depth_image_message: sensor_msgs.msg.Image,
    camera_intrinsics_message: sensor_msgs.msg.CameraInfo,
    camera_extrinsics_message: geometry_msgs.msg.TransformStamped,
    depth_unit_per_meter: int = 1000,
    mask: Optional[NDArray] = None,
) -> sensor_msgs.msg.PointCloud2:
    """Reconstruct a pointcloud message given RGBD camera messaging.

    See `reconstruct_pointcloud` documentation for further reference.
    """
    return open3d_pointcloud_to_ros_pointcloud2(
        reconstruct_pointcloud(
            rgb_image_message,
            depth_image_message,
            camera_intrinsics_message,
            camera_extrinsics_message,
            depth_unit_per_meter,
            mask,
        ),
        stamp=rgb_image_message.header.stamp,
        frame_id=camera_extrinsics_message.header.frame_id,
    )


def make_pointcloud_reconstruction_feed(
    rgb_image_feed: MessageFeed,
    depth_image_feed: MessageFeed,
    camera_intrinsics_feed: MessageFeed,
    target_frame_id: str,
    reconstruction_method: Callable = reconstruct_pointcloud,
    history_length: Optional[int] = None,
    node: Optional[Node] = None,
) -> MessageFeed:
    """Wraps pointcloud reconstruction logic in a message feed.

    Args:
        rgb_image_feed: an rgb image message feed.
        depth_image_feed: a depth map message feed.
        camera_intrinsics_feed: camera pinhole model message feed.
        target_frame_id: the frame id of the reconstructed pointcloud.
        reconstruction_method: algorithm for pointcloud reconstruction
        as callable, taking rgb image, depth image, camera info, and
        camera transform messages as positional arguments.
        history_length: optional historic data size, defaults to 1.
        node: optional node for the feed, defaults to the current process node.

    Returns:
        a message feed.
    """
    return AdaptedMessageFeed(
        FramedMessageFeed(
            SynchronizedMessageFeed(
                rgb_image_feed, depth_image_feed, camera_intrinsics_feed, history_length=0, node=node
            ),
            target_frame_id=target_frame_id,
            history_length=0,
            node=node,
        ),
        fn=reconstruction_method,
        history_length=history_length,
        node=node,
    )


class SpotCameraSubscription:
    """An ergonomic interface to Spot RGBD cameras in ROS 2."""

    class Location(str, Enum):
        """Camera location on Spot."""

        BACK = "back"
        LEFT = "left"
        RIGHT = "right"
        FRONT_LEFT = "frontleft"
        FRONT_RIGHT = "frontright"
        HAND = "hand"

        def __repr__(self) -> str:
            return self.value

    def __init__(
        self,
        robot_name: str,
        location: Location,
        history_length: Optional[int] = None,
        node: Optional[Node] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Spot camera subscription.

        Args:
            robot_name: the robot name, used for topic namespacing.
            location: the camera location, used for topic namespacing.
            history_length: optional historic data size, defaults to 1.
            node: optional node for subscription, defaults to the current process node.
            kwargs: all other keyword arguments are used for `CameraSubscription` instantiation.
        """
        self._robot_name = robot_name
        self._location = location
        self._history_length = history_length
        self._node = node
        self._kwargs = kwargs

    @property
    def robot_name(self) -> str:
        """Get the name of the Spot the camera belongs to."""
        return self._robot_name

    @property
    def location(self) -> Location:
        """Get the location of the camera on the Spot."""
        return self._location

    @cached_property
    def rgb(self) -> CameraSubscription:
        """Get rgb camera subscription."""
        return CameraSubscription(
            spot_namespaced(self.robot_name, "camera", self.location.value),
            history_length=self._history_length,
            node=self._node,
            **self._kwargs,
        )

    @cached_property
    def depth(self) -> CameraSubscription:
        """Get depth camera subscription."""
        return CameraSubscription(
            spot_namespaced(self.robot_name, "depth", self.location.value),
            history_length=self._history_length,
            node=self._node,
            **self._kwargs,
        )

    @cached_property
    def depth_registered(self) -> CameraSubscription:
        """Get (rgb-)registered depth camera subscription."""
        return CameraSubscription(
            spot_namespaced(self.robot_name, "depth_registered", self.location.value),
            history_length=self._history_length,
            node=self._node,
            **self._kwargs,
        )

    @cached_property
    def rgbd_frame_with_info(self) -> AdaptedMessageFeed[RgbdFrameWithInfo]:
        """Get synchronized rgb image, depth (registered) image, and camera info feed."""
        return AdaptedMessageFeed(
            SynchronizedMessageFeed(
                self.rgb.image, self.depth_registered.image, self.rgb.info, history_length=0, node=self._node
            ),
            fn=RgbdFrameWithInfo,
            history_length=self._history_length,
            node=self._node,
        )

    @cached_property
    def pointcloud_reconstruction(self) -> MessageFeed[open3d.geometry.PointCloud]:
        """Get pointcloud reconstruction feed."""
        return make_pointcloud_reconstruction_feed(
            self.rgb.image,
            self.depth_registered.image,
            self.rgb.info,
            target_frame_id=namespace_with(self.robot_name, FrameHint.VISION.value),
            history_length=self._history_length,
            node=self._node,
        )

    @cached_property
    def pointcloud(self) -> MessageFeed[sensor_msgs.msg.PointCloud2]:
        """Get pointcloud subscription."""
        return make_pointcloud_reconstruction_feed(
            self.rgb.image,
            self.depth_registered.image,
            self.rgb.info,
            target_frame_id=namespace_with(self.robot_name, FrameHint.VISION.value),
            reconstruction_method=reconstruct_pointcloud_message,
            history_length=self._history_length,
            node=self._node,
        )


class SpotImageMixin(SpotMobilityCommandProtocol):
    """Mixin that provides access to Spot cameras."""

    CameraLocation: TypeAlias = SpotCameraSubscription.Location

    class CameraConfiguration(NamedTuple):
        """Collection of Spot camera subscriptions."""

        back: SpotCameraSubscription
        left: SpotCameraSubscription
        right: SpotCameraSubscription
        front_left: SpotCameraSubscription
        front_right: SpotCameraSubscription
        hand: Optional[SpotCameraSubscription]

        def __iter__(self) -> Iterator[SpotCameraSubscription]:
            """Iterate available cameras only."""
            yield from filter(None, tuple.__iter__(self))

    @cached_property
    def cameras(self) -> CameraConfiguration:
        """Get the camera configuration on this robot."""
        Location = SpotCameraSubscription.Location
        return SpotImageMixin.CameraConfiguration(
            back=SpotCameraSubscription(self.robot_name, Location.BACK, node=self.node),
            left=SpotCameraSubscription(self.robot_name, Location.LEFT, node=self.node),
            right=SpotCameraSubscription(self.robot_name, Location.RIGHT, node=self.node),
            front_left=SpotCameraSubscription(self.robot_name, Location.FRONT_LEFT, node=self.node),
            front_right=SpotCameraSubscription(self.robot_name, Location.FRONT_RIGHT, node=self.node),
            hand=SpotCameraSubscription(self.robot_name, Location.HAND, node=self.node) if self.has_arm() else None,
        )

    def scan_in_place(
        self,
        cameras: Optional[Sequence[SpotCameraSubscription]] = None,
        routine: Optional[Sequence[SO3]] = None,
        timeout_sec: Optional[float] = None,
    ) -> open3d.geometry.PointCloud:
        """Scan the robot surroundings by stitching pointclouds from multiple cameras.

        Args:
            cameras: optional subset of camera subscriptions to be used for scanning.
            All available cameras will be used be default if none are provided.
            routine: optional sequence of in-place pan/tilt movements to perform
            during the scan. A full scan will be captured and aggregated in each pose.
            timeout_sec: optional timeout, in seconds, for each blocking operation.

        Returns:
            the scanned point cloud.
        """
        if cameras is not None:
            if any(camera not in self.cameras for camera in cameras):
                raise ValueError(f"Not all cameras are on {self.robot_name}")
        else:
            cameras = list(filter(None, self.cameras))

        combined_pointcloud = open3d.geometry.PointCloud()

        if routine is not None:
            gravbody_t_body_orientation = self.lookup_a_tform_b(
                FrameHint.GRAV_ALIGNED_BODY, FrameHint.BODY, wait_for_frames=True, timeout=timeout_sec
            ).R
            for body_orientation in routine:
                self.stand_with_orientation.synchronously(
                    gravbody_t_body_orientation * body_orientation, timeout_sec=timeout_sec
                )
                for camera in cameras:
                    combined_pointcloud += unwrap_future(
                        camera.pointcloud_reconstruction.update, timeout_sec=timeout_sec
                    )
        else:
            for camera in cameras:
                combined_pointcloud += unwrap_future(camera.pointcloud_reconstruction.update, timeout_sec=timeout_sec)
        return combined_pointcloud
