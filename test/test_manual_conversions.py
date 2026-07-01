# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d
import pytest
from bosdyn.api import geometry_pb2
from bosdyn.client.math_helpers import Quat, SE2Pose, SE3Pose
from builtin_interfaces.msg import Time as RosTimestamp
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import PoseStamped as RosPoseStamped
from geometry_msgs.msg import PoseWithCovariance as RosPoseWithCovariance
from geometry_msgs.msg import PoseWithCovarianceStamped as RosPoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion as RosQuaternion
from geometry_msgs.msg import QuaternionStamped as RosQuaternionStamped
from geometry_msgs.msg import Transform as RosTransform
from geometry_msgs.msg import TransformStamped as RosTransformStamped
from geometry_msgs.msg import Vector3
from open3d.geometry import PointCloud as open3dPointCloud
from open3d.t.geometry import PointCloud as open3dTensorPointCloud
from sensor_msgs.msg import PointCloud2 as RosPointCloud2
from sensor_msgs.msg import PointField as RosPointField
from spatialmath import SE2, SE3, UnitQuaternion
from std_msgs.msg import Header

from spot_basic.manual_conversions import (
    get_list_from_wxyz,
    get_list_from_xyz,
    numpy_array_to_ros_pointcloud2,
    open3d_pointcloud_to_numpy_array,
    open3d_pointcloud_to_ros_pointcloud2,
    to_se3,
    to_unit_quaternion,
)


@dataclass
class WXYZ:
    w: Any
    x: Any
    y: Any
    z: Any


def test_get_list_from_xyz() -> None:
    assert get_list_from_xyz(WXYZ(w=1, x=2, y=3, z=4)) == [2, 3, 4]


def test_get_list_from_wxyz() -> None:
    assert get_list_from_wxyz(WXYZ(w=1, x=2, y=3, z=4)) == [1, 2, 3, 4]


def test_open3d_pointcloud_to_numpy_array() -> None:

    expected_no_colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_with_colors = np.array([[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0], [7, 8, 9, 0, 0, 0]])

    # Legacy point cloud
    legacy_cloud = open3dPointCloud(o3d.utility.Vector3dVector(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    # No colors
    result = open3d_pointcloud_to_numpy_array(legacy_cloud)
    assert np.array_equal(result, expected_no_colors)

    # With colors
    legacy_cloud.paint_uniform_color([0, 0, 0])
    result = open3d_pointcloud_to_numpy_array(legacy_cloud)
    assert np.array_equal(result, expected_with_colors)

    # Tensor point cloud
    tensor_cloud = open3dTensorPointCloud(o3d.core.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # No colors
    result = open3d_pointcloud_to_numpy_array(tensor_cloud)
    assert np.array_equal(result, expected_no_colors)

    # With colors
    tensor_cloud.point.colors = o3d.core.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = open3d_pointcloud_to_numpy_array(tensor_cloud)
    assert np.array_equal(result, expected_with_colors)


def test_numpy_array_to_ros_pointcloud2() -> None:

    # Invalid sized array
    cloud_invalid = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        numpy_array_to_ros_pointcloud2(cloud_invalid)

    # No color, dense
    timestamp = RosTimestamp(sec=1234567890)
    frame_id = "test_frame"
    cloud_no_colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Turn off formatting so that `data` items don't end up on each line
    # fmt: off
    expected_no_colors = RosPointCloud2(
        header=Header(stamp=timestamp, frame_id=frame_id),
        height=1,
        width=3,
        is_bigendian=False,
        point_step=12,
        row_step=36,
        is_dense=True,
        fields=[
            RosPointField(name="x", offset=0, datatype=RosPointField.FLOAT32, count=1),
            RosPointField(name="y", offset=4, datatype=RosPointField.FLOAT32, count=1),
            RosPointField(name="z", offset=8, datatype=RosPointField.FLOAT32, count=1),
        ],
        data=[
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64,        # point 1: [1, 2, 3]
            0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,     # point 2: [4, 5, 6]
            0, 0, 224, 64, 0, 0, 0, 65, 0, 0, 16, 65,        # point 3: [7, 8, 9]
        ],
    )
    # fmt: on

    result = numpy_array_to_ros_pointcloud2(cloud_no_colors, frame_id=frame_id, stamp=timestamp)
    print(result)
    assert result == expected_no_colors

    # Non-dense
    cloud_non_dense = np.array([[1, 2, 3], [4, 5, 6], [np.nan, np.nan, np.nan]])
    result = numpy_array_to_ros_pointcloud2(cloud_non_dense, frame_id=frame_id, stamp=timestamp)
    assert not result.is_dense

    # With color
    cloud_color = np.array([[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0], [7, 8, 9, 0, 0, 0]])
    expected_color_field = [
        RosPointField(name="x", offset=0, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="y", offset=4, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="z", offset=8, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="r", offset=12, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="g", offset=16, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="b", offset=20, datatype=RosPointField.FLOAT32, count=1),
    ]
    result = numpy_array_to_ros_pointcloud2(cloud_color, frame_id=frame_id, stamp=timestamp)

    assert result.fields == expected_color_field


def test_open3d_pointcloud_to_ros_pointcloud2() -> None:
    timestamp = RosTimestamp(sec=1234567890)
    frame_id = "test_frame"

    # No colors
    # fmt: off
    expected_no_colors = RosPointCloud2(
        header=Header(stamp=timestamp, frame_id=frame_id),
        height=1,
        width=3,
        is_bigendian=False,
        point_step=12,
        row_step=36,
        is_dense=True,
        fields=[
            RosPointField(name="x", offset=0, datatype=RosPointField.FLOAT32, count=1),
            RosPointField(name="y", offset=4, datatype=RosPointField.FLOAT32, count=1),
            RosPointField(name="z", offset=8, datatype=RosPointField.FLOAT32, count=1),
        ],
        data=[
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64,        # point 1: [1, 2, 3]
            0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,     # point 2: [4, 5, 6]
            0, 0, 224, 64, 0, 0, 0, 65, 0, 0, 16, 65,        # point 3: [7, 8, 9]
        ],
    )
    # fmt: on

    point_cloud = open3dPointCloud(o3d.utility.Vector3dVector(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    result = open3d_pointcloud_to_ros_pointcloud2(point_cloud, frame_id=frame_id, stamp=timestamp)
    assert result == expected_no_colors

    # With colors
    point_cloud.paint_uniform_color([0, 0, 0])
    expected_color_field = [
        RosPointField(name="x", offset=0, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="y", offset=4, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="z", offset=8, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="r", offset=12, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="g", offset=16, datatype=RosPointField.FLOAT32, count=1),
        RosPointField(name="b", offset=20, datatype=RosPointField.FLOAT32, count=1),
    ]
    result = open3d_pointcloud_to_ros_pointcloud2(point_cloud, frame_id=frame_id, stamp=timestamp)
    assert result.fields == expected_color_field


def test_ro_unit_quaternion() -> None:
    # Test all the different types accepted by this function
    expected_q = UnitQuaternion([0.500, 0.577, 0.577, 0.577])

    geometry_pb2_q = geometry_pb2.Quaternion(w=0.500, x=0.577, y=0.577, z=0.577)
    assert to_unit_quaternion(geometry_pb2_q).vec == pytest.approx(expected_q.vec)

    quat_q = Quat(w=0.500, x=0.577, y=0.577, z=0.577)
    assert to_unit_quaternion(quat_q).vec == pytest.approx(expected_q.vec)

    ros_q = RosQuaternion(w=0.500, x=0.577, y=0.577, z=0.577)
    assert to_unit_quaternion(ros_q).vec == pytest.approx(expected_q.vec)

    ros_stamped_q = RosQuaternionStamped(
        header=Header(stamp=RosTimestamp(sec=1234567890), frame_id="test_frame"),
        quaternion=ros_q,
    )
    assert to_unit_quaternion(ros_stamped_q).vec == pytest.approx(expected_q.vec)

    # And one type that isn't accepted
    with pytest.raises(TypeError):
        to_unit_quaternion("not a quaternion")


def test_to_se3() -> None:
    # Test all the different types accepted by this function
    expected_se3 = pytest.approx((SE3(1.0, 2.0, 3.0) * SE3.Rx(np.pi / 2)).A)
    header = Header(stamp=RosTimestamp(sec=1234567890), frame_id="test_frame")

    ros_pose = RosPose(
        position=RosPoint(x=1.0, y=2.0, z=3.0), orientation=RosQuaternion(w=0.7071068, x=0.7071068, y=0.0, z=0.0)
    )
    assert to_se3(ros_pose).A == expected_se3

    ros_pose_with_covariance = RosPoseWithCovariance(
        pose=ros_pose,
        covariance=[0.0] * 36,
    )
    assert to_se3(ros_pose_with_covariance).A == expected_se3

    ros_pose_stamped = RosPoseStamped(
        header=header,
        pose=ros_pose,
    )
    assert to_se3(ros_pose_stamped).A == expected_se3

    ros_pose_with_covariance_stamped = RosPoseWithCovarianceStamped(
        header=header,
        pose=ros_pose_with_covariance,
    )
    assert to_se3(ros_pose_with_covariance_stamped).A == expected_se3

    ros_transform = RosTransform(
        translation=Vector3(x=1.0, y=2.0, z=3.0), rotation=RosQuaternion(w=0.7071068, x=0.7071068, y=0.0, z=0.0)
    )
    assert to_se3(ros_transform).A == expected_se3

    ros_transform_stamped = RosTransformStamped(
        header=header,
        transform=ros_transform,
    )
    assert to_se3(ros_transform_stamped).A == expected_se3

    se3_pose = SE3Pose(1.0, 2.0, 3.0, Quat(0.7071068, 0.7071068, 0.0, 0.0))
    assert to_se3(se3_pose).A == expected_se3

    geometry_pb2_se3_pose = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=1.0, y=2.0, z=3.0),
        rotation=geometry_pb2.Quaternion(w=0.7071068, x=0.7071068, y=0.0, z=0.0),
    )
    assert to_se3(geometry_pb2_se3_pose).A == expected_se3

    se3 = SE3(1.0, 2.0, 3.0) * SE3.Rx(np.pi / 2)
    assert to_se3(se3).A == expected_se3

    # From SE2 - remove z component from expected case and change rotation
    expected_se3_from_se2 = pytest.approx(SE3(1.0, 2.0, 0.0) * SE3.Rz(np.pi / 2).A)

    se2_pose = SE2Pose(1.0, 2.0, np.pi / 2)
    assert to_se3(se2_pose).A == expected_se3_from_se2

    geometry_pb2_se2_pose = geometry_pb2.SE2Pose(position=geometry_pb2.Vec2(x=1.0, y=2.0), angle=np.pi / 2)
    assert to_se3(geometry_pb2_se2_pose).A == expected_se3_from_se2

    se2 = SE2(1.0, 2.0, np.pi / 2)
    assert to_se3(se2).A == expected_se3_from_se2
