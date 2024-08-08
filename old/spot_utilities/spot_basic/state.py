# Copyright (c) 2023-2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Union

import bdai_ros2_wrappers.scope as ros_scope
from bdai_ros.utilities.manual_conversions import (
    se3_to_se3pose_proto,
    to_se3,
)
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.subscription import Subscription
from bdai_ros2_wrappers.utilities import namespace_with
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    HAND_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
)
from rclpy.time import Time
from sensor_msgs.msg import JointState
from spatialmath import SE3
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_util import spot_namespaced


@dataclass(kw_only=True)
class ArmJointMap:
    """Stores kinematic state (position or velocity) for each joint in Spot arm."""

    sh0: float
    sh1: float
    el0: float
    el1: float
    wr0: float
    wr1: float

    def to_list(self) -> Sequence[float]:
        """Converts an ArmJointMap to the format expected by the Spot-SDK.

        Returns:
            List representing the correct ordering for the arm joints when commanding a move.
        """
        return [self.sh0, self.sh1, self.el0, self.el1, self.wr0, self.wr1]

    @classmethod
    def from_list(cls, joint_list: Sequence[float]) -> "ArmJointMap":
        """Converts an list of arm joint states to the ArmJointMap used in SpotBasic.

        Returns:
            ArmJointMap representing the correct ordering for the arm joints when commanding a move.
        """
        return ArmJointMap(
            sh0=joint_list[0],
            sh1=joint_list[1],
            el0=joint_list[2],
            el1=joint_list[3],
            wr0=joint_list[4],
            wr1=joint_list[5],
        )


@dataclass(kw_only=True)
class LegJointMap:
    """Stores joint angles or joint velocities for each joint in spot's legs."""

    front_left_hip_x: float
    front_left_hip_y: float
    front_left_knee: float
    front_right_hip_x: float
    front_right_hip_y: float
    front_right_knee: float
    rear_left_hip_x: float
    rear_left_hip_y: float
    rear_left_knee: float
    rear_right_hip_x: float
    rear_right_hip_y: float
    rear_right_knee: float

    def to_list(self) -> Sequence[float]:
        """Converts a LegJointMap to the format expected by the Spot-SDK.

        Returns:
            List representing the leg joints.
        """
        return [
            self.front_left_hip_x,
            self.front_left_hip_y,
            self.front_left_knee,
            self.front_right_hip_x,
            self.front_right_hip_y,
            self.front_right_knee,
            self.rear_left_hip_x,
            self.rear_left_hip_y,
            self.rear_left_knee,
            self.rear_right_hip_x,
            self.rear_right_hip_y,
            self.rear_right_knee,
        ]

    @classmethod
    def from_list(cls, joint_list: Sequence[float]) -> "LegJointMap":
        """Converts an list of arm joint angles to the LegJointMap used in SpotBasic.

        Returns:
            LegJointMap representing the leg joints.
        """
        return LegJointMap(
            front_left_hip_x=joint_list[0],
            front_left_hip_y=joint_list[1],
            front_left_knee=joint_list[2],
            front_right_hip_x=joint_list[3],
            front_right_hip_y=joint_list[4],
            front_right_knee=joint_list[5],
            rear_left_hip_x=joint_list[6],
            rear_left_hip_y=joint_list[7],
            rear_left_knee=joint_list[8],
            rear_right_hip_x=joint_list[9],
            rear_right_hip_y=joint_list[10],
            rear_right_knee=joint_list[11],
        )


class SpotStateMixin(SpotBaseProtocol):
    """Mixin that tracks Spot state."""

    class FrameHint(str, Enum):
        """Well known Spot frames."""

        BODY = BODY_FRAME_NAME
        VISION = VISION_FRAME_NAME
        ODOM = ODOM_FRAME_NAME
        HAND = HAND_FRAME_NAME
        GRAV_ALIGNED_BODY = GRAV_ALIGNED_BODY_FRAME_NAME
        GROUND_PLANE = GROUND_PLANE_FRAME_NAME
        WRIST = "arm_link_wr1"
        HEAD = "head"
        BACK = "back"
        LEFT = "left"
        RIGHT = "right"
        FRONT_LEFT = "frontleft"
        FRONT_RIGHT = "frontright"
        BACK_CAMERA = "back_fisheye"
        FRONT_LEFT_CAMERA = "frontleft_fisheye"
        FRONT_RIGHT_CAMERA = "frontright_fisheye"
        LEFT_CAMERA = "left_fisheye"
        RIGHT_CAMERA = "right_fisheye"
        HAND_CAMERA = "hand_color_image_sensor"
        HAND_DEPTH_CAMERA = "hand_depth_sensor"

    def __init__(self, *args: Any, tf_listener: Optional[TFListenerWrapper] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # A tf listener instance can be connected to the robot.
        self.tf_listener = tf_listener or ros_scope.tf_listener()
        # TODO: are there other frame names to grab?
        self.body_frame_name = namespace_with(self.robot_name, BODY_FRAME_NAME)
        self.hand_frame_name = namespace_with(self.robot_name, HAND_FRAME_NAME)
        self.vision_frame_name = namespace_with(self.robot_name, VISION_FRAME_NAME)
        self.odom_frame_name = namespace_with(self.robot_name, ODOM_FRAME_NAME)

        self.joint_states = Subscription(JointState, spot_namespaced(self.robot_name, "joint_states"), node=self.node)

    def set_tf_listener(self, tf_listener: TFListenerWrapper) -> None:
        """Attach a TFListenerWrapper instance to the robot.

        Args:
            tf_listener: An instance of the TFListenerWrapper.
        """
        self.tf_listener = tf_listener

    def unset_tf_listener(self) -> None:
        self.tf_listener = None

    @property
    def arm_joint_positions(self) -> ArmJointMap:
        """Get current arm joint joint positions.

        Includes shoulder, elbow, and wrist.
        """
        joint_states = self.joint_states.latest
        if joint_states is None:
            return ArmJointMap(sh0=0.0, sh1=0.0, el0=0.0, el1=0.0, wr0=0.0, wr1=0.0)
        joint_positions = dict(zip(joint_states.name, joint_states.position, strict=True))
        return ArmJointMap(
            sh0=joint_positions[namespace_with(self.robot_name, "arm_sh0")],
            sh1=joint_positions[namespace_with(self.robot_name, "arm_sh1")],
            el0=joint_positions[namespace_with(self.robot_name, "arm_el0")],
            el1=joint_positions[namespace_with(self.robot_name, "arm_el1")],
            wr0=joint_positions[namespace_with(self.robot_name, "arm_wr0")],
            wr1=joint_positions[namespace_with(self.robot_name, "arm_wr1")],
        )

    @property
    def leg_joint_positions(self) -> LegJointMap:
        """Get current legs' joint positions.

        Includes left and right hips and knees.
        """
        joint_states = self.joint_states.latest
        if joint_states is None:
            return LegJointMap.from_list([0.0] * 12)
        joint_positions = dict(zip(joint_states.name, joint_states.position, strict=True))
        return LegJointMap(
            front_left_hip_x=joint_positions[namespace_with(self.robot_name, "front_left_hip_x")],
            front_left_hip_y=joint_positions[namespace_with(self.robot_name, "front_left_hip_y")],
            front_left_knee=joint_positions[namespace_with(self.robot_name, "front_left_knee")],
            front_right_hip_x=joint_positions[namespace_with(self.robot_name, "front_right_hip_x")],
            front_right_hip_y=joint_positions[namespace_with(self.robot_name, "front_right_hip_y")],
            front_right_knee=joint_positions[namespace_with(self.robot_name, "front_right_knee")],
            rear_left_hip_x=joint_positions[namespace_with(self.robot_name, "rear_left_hip_x")],
            rear_left_hip_y=joint_positions[namespace_with(self.robot_name, "rear_left_hip_y")],
            rear_left_knee=joint_positions[namespace_with(self.robot_name, "rear_left_knee")],
            rear_right_hip_x=joint_positions[namespace_with(self.robot_name, "rear_right_hip_x")],
            rear_right_hip_y=joint_positions[namespace_with(self.robot_name, "rear_right_hip_y")],
            rear_right_knee=joint_positions[namespace_with(self.robot_name, "rear_right_knee")],
        )

    def wait_for_a_tform_b(
        self,
        frame_a: Union[FrameHint, str],
        frame_b: Union[FrameHint, str],
        transform_time: Optional[Time] = None,
        timeout_sec: Optional[float] = None,
    ) -> bool:
        """Waits for a robot specific transform from `frame_a` to `frame_b`.

        The transform returned will be frame_a_t_frame_b, which is robot robot specific.
        So instead if frame_a is VISION, we internally append it to the robot name
        s.t. "<Robot Name>/VISION", will be called.

        Args:
            frame_a: Base frame for transform.
            frame_b: Tip frame for transform.
            transform_time: optional time for transform lookup. It defaults to the
            latest available transform if none is provided.
            timeout_sec: optional timeout, in seconds, to wait for the transform to become available if
            the requested is beyond the most recent transform in the buffer.
        Returns:
            Whether a transform between frame_a and frame_b is available at the specified time.
        Raises:
            All the possible TransformExceptions. ValueError if no TFListenerWrapper instance is connected.
        """
        if self.tf_listener is None:
            raise ValueError(
                "TFListenerWrapper is not connected. Use set_tf_listener() or ros_process.main(uses_tf=True)."
            )

        if isinstance(frame_a, FrameHint):
            frame_a = frame_a.value

        if isinstance(frame_b, FrameHint):
            frame_b = frame_b.value

        # Check if namespaced and add robot_name if not
        if not frame_a.startswith(self.robot_name):
            frame_a = namespace_with(self.robot_name, frame_a)

        if not frame_b.startswith(self.robot_name):
            frame_b = namespace_with(self.robot_name, frame_b)

        return self.tf_listener.wait_for_a_tform_b(frame_a, frame_b, transform_time, timeout_sec)

    def lookup_a_tform_b(
        self,
        frame_a: Union[FrameHint, str],
        frame_b: Union[FrameHint, str],
        transform_time: Optional[Time] = None,
        timeout: Optional[float] = None,
        wait_for_frames: bool = False,
    ) -> SE3:
        """Looks up a robot specific transform from `frame_a` to `frame_b`.

        The transform returned will be frame_a_t_frame_b, which is robot robot specific.
        So instead if frame_a is VISION, we internally append it to the robot name
        s.t. "<Robot Name>/VISION", will be called.

        Args:
            frame_a: Base frame for transform.
            frame_b: Tip frame for transform.
            transform_time: optional time for transform lookup. It defaults to the latest available
            transform if none is provided.
            timeout: optional timeout, in seconds, to wait for the transform to become available if
            the requested is beyond the most recent transform in the buffer.
            wait_for_frames: if true, this method will wait until timeout for a path from frame_a to
            frame_b to become available in the buffer. If false, this method will return immediately
            if no such path exists, regardless of the given timeout.
        Returns:
            Transform from frame_a to frame_b at the requested time.
        Raises:
            All the possible TransformExceptions. ValueError if no TFListenerWrapper instance is connected.
        """
        if self.tf_listener is None:
            raise ValueError(
                "TFListenerWrapper is not connected. Use set_tf_listener() or ros_process.main(uses_tf=True)."
            )

        if isinstance(frame_a, FrameHint):
            frame_a = frame_a.value

        if isinstance(frame_b, FrameHint):
            frame_b = frame_b.value

        # Check if namespaced and add robot_name if not
        if not frame_a.startswith(self.robot_name):
            frame_a = namespace_with(self.robot_name, frame_a)

        if not frame_b.startswith(self.robot_name):
            frame_b = namespace_with(self.robot_name, frame_b)

        # Return transform as SE3
        return to_se3(self.tf_listener.lookup_a_tform_b(frame_a, frame_b, transform_time, timeout, wait_for_frames))

    def get_transforms_tree_snapshot(
        self,
        transform_time: Optional[Time] = None,
        root_frame: Optional[Union[FrameHint, str]] = None,
        timeout_sec: Optional[float] = None,
    ) -> FrameTreeSnapshot:
        """Takes a full snapshot of all robot-specific transforms in the system.

        Only transforms between frames prefixed by the robot name and available at the requested
        time will be retrieved -- missing transforms are not an error.

        Note this method will return a tree of height 2: implicit world frame to root frame, then
        root frame to every other frame. This is because once broadcasted over tf2, it is no longer
        possible to reconstruct the tree structure as it was at a given point in time.

        Args:
            transform_time: optional time for transform lookup. It defaults to the latest transforms
            (on a pair basis) if none is provided.
            root_frame: optional root frame for the transforms tree. It defaults to the body frame
            if none is provided.
            timeout_sec: optional timeout to wait for transforms to become available. It defaults to
            no waiting if none is provided. Note that this method will only wait for frames it already
            knows about -- it will not wait for new frames to show up.

        Args:
            A tree-based collection of all available transforms.
        """
        if root_frame is None:
            root_frame = FrameHint.BODY

        if isinstance(root_frame, FrameHint):
            root_frame_name = root_frame.value
        else:
            root_frame_name = root_frame

        if not root_frame_name.startswith(self.robot_name):
            root_frame_name = namespace_with(self.robot_name, root_frame_name)

        if timeout_sec is None:
            timeout_sec = 0.0

        child_to_parent_edge_map = {}
        root_frame_basename = root_frame_name[len(self.robot_name) + 1 :]
        child_to_parent_edge_map[root_frame_basename] = FrameTreeSnapshot.ParentEdge(
            parent_tform_child=se3_to_se3pose_proto(SE3())
        )
        for frame_name in self.tf_listener.buffer._getFrameStrings():  # noqa: SLF001
            if not frame_name.startswith(self.robot_name):
                continue
            if root_frame_name == frame_name:
                continue
            frame_transform_available = self.tf_listener.wait_for_a_tform_b(
                root_frame_name, frame_name, transform_time, timeout_sec
            )
            if not frame_transform_available:
                continue
            frame_transform = to_se3(
                self.tf_listener.lookup_a_tform_b(root_frame_name, frame_name, transform_time, timeout_sec)
            )
            frame_basename = frame_name[len(self.robot_name) + 1 :]
            child_to_parent_edge_map[frame_basename] = FrameTreeSnapshot.ParentEdge(
                parent_frame_name=root_frame_basename, parent_tform_child=se3_to_se3pose_proto(frame_transform)
            )
        return FrameTreeSnapshot(child_to_parent_edge_map=child_to_parent_edge_map)

    def get_basic_tform_tree_snapshot(self, transform_time: Optional[Union[Time, float]] = None) -> FrameTreeSnapshot:
        """Take a snapshot of all transforms involving common frames.

        Common frames include transformations to: BODY, VISION, ODOM, HAND, GRAV_ALIGNED_BODY, GROUND_PLANE and WRIST.
        Camera sensor related frames are also included: HEAD, BACK, LEFT, RIGHT, FRONT_LEFT, FRONT_RIGHT, BACK_CAMERA,
        FRONT_LEFT_CAMERA, FRONT_RIGHT_CAMERA, LEFT_CAMERA, RIGHT_CAMERA and HAND_CAMERA.

        Returns:
            A tree-based collection of transformation to common frames.
        """
        if isinstance(transform_time, float):
            transform_time = Time(seconds=transform_time)

        child_to_parent_edge_map = {}

        # Add root frame.
        child_to_parent_edge_map[FrameHint.BODY.value] = FrameTreeSnapshot.ParentEdge(
            parent_tform_child=se3_to_se3pose_proto(SE3())
        )

        # Add head frame.
        child_to_parent_edge_map[FrameHint.HEAD.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.BODY.value, parent_tform_child=se3_to_se3pose_proto(SE3())
        )

        # The ground plane is relative to the odom frame.
        child_to_parent_edge_map[FrameHint.GROUND_PLANE.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.ODOM.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.ODOM, FrameHint.GROUND_PLANE, transform_time)
            ),
        )

        # Add frames relative to the body frame.
        frames_relative_to_body = [
            FrameHint.VISION,
            FrameHint.ODOM,
            FrameHint.GRAV_ALIGNED_BODY,
            FrameHint.BACK,
            FrameHint.LEFT,
            FrameHint.RIGHT,
        ]

        if self.has_arm():
            frames_relative_to_body += [FrameHint.WRIST]

        for target_frame in frames_relative_to_body:
            child_to_parent_edge_map[target_frame.value] = FrameTreeSnapshot.ParentEdge(
                parent_frame_name=FrameHint.BODY.value,
                parent_tform_child=se3_to_se3pose_proto(
                    self.lookup_a_tform_b(FrameHint.BODY, target_frame, transform_time)
                ),
            )

        if self.has_arm():
            # Add hand frame relative to wrist frame.
            child_to_parent_edge_map[FrameHint.HAND.value] = FrameTreeSnapshot.ParentEdge(
                parent_frame_name=FrameHint.WRIST.value,
                parent_tform_child=se3_to_se3pose_proto(
                    self.lookup_a_tform_b(FrameHint.WRIST, FrameHint.HAND, transform_time)
                ),
            )

            # Add hand camera frame.
            child_to_parent_edge_map[FrameHint.HAND_CAMERA.value] = FrameTreeSnapshot.ParentEdge(
                parent_frame_name=FrameHint.WRIST.value,
                parent_tform_child=se3_to_se3pose_proto(
                    self.lookup_a_tform_b(FrameHint.WRIST, FrameHint.HAND_CAMERA, transform_time)
                ),
            )

        # Add frames relative to head frame.
        child_to_parent_edge_map[FrameHint.FRONT_LEFT.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.HEAD.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.HEAD, FrameHint.FRONT_LEFT, transform_time)
            ),
        )

        child_to_parent_edge_map[FrameHint.FRONT_RIGHT.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.HEAD.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.HEAD, FrameHint.FRONT_RIGHT, transform_time)
            ),
        )

        # Add camera sensor frames.
        child_to_parent_edge_map[FrameHint.FRONT_LEFT_CAMERA.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.FRONT_LEFT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.FRONT_LEFT, FrameHint.FRONT_LEFT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[FrameHint.FRONT_RIGHT_CAMERA.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.FRONT_RIGHT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.FRONT_RIGHT, FrameHint.FRONT_RIGHT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[FrameHint.LEFT_CAMERA.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.LEFT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.LEFT, FrameHint.LEFT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[FrameHint.RIGHT_CAMERA.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.RIGHT.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.RIGHT, FrameHint.RIGHT_CAMERA, transform_time)
            ),
        )

        child_to_parent_edge_map[FrameHint.BACK_CAMERA.value] = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=FrameHint.BACK.value,
            parent_tform_child=se3_to_se3pose_proto(
                self.lookup_a_tform_b(FrameHint.BACK, FrameHint.BACK_CAMERA, transform_time)
            ),
        )

        return FrameTreeSnapshot(child_to_parent_edge_map=child_to_parent_edge_map)


# Alias
FrameHint = SpotStateMixin.FrameHint


class SpotStateProtocol(SpotBaseProtocol):

    def lookup_a_tform_b(
        self,
        frame_a: Union[FrameHint, str],
        frame_b: Union[FrameHint, str],
        transform_time: Optional[Time] = None,
        timeout: Optional[float] = None,
        wait_for_frames: bool = False,
    ) -> SE3: ...

    def get_transforms_tree_snapshot(
        self,
        transform_time: Optional[Time] = None,
        root_frame: Optional[Union[FrameHint, str]] = None,
        timeout_sec: Optional[float] = None,
    ) -> FrameTreeSnapshot: ...
