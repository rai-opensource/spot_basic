# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import array
import sys
from typing import Any, List, Optional, Protocol

import numpy
import numpy.typing as npt
from bosdyn.api import geometry_pb2
from bosdyn.client.math_helpers import Quat, SE2Pose, SE3Pose, SE3Velocity
from builtin_interfaces.msg import Time as RosTimestamp
from geometry_msgs.msg import (
    Pose as RosPose,
    PoseStamped as RosPoseStamped,
    PoseWithCovariance as RosPoseWithCovariance,
    PoseWithCovarianceStamped as RosPoseWithCovarianceStamped,
    Quaternion as RosQuaternion,
    QuaternionStamped as RosQuaternionStamped,
    Transform as RosTransform,
    TransformStamped as RosTransformStamped,
)
from open3d.geometry import PointCloud as open3dPointCloud
from open3d.t.geometry import PointCloud as open3dTensorPointCloud
from sensor_msgs.msg import (
    PointCloud2 as RosPointCloud2,
    PointField as RosPointField,
)
from spatialmath import (
    SE2,
    SE3,
    SpatialForce,
    Twist3,
    UnitQuaternion,
)

# mappings between PointField types and numpy types
_pfnptype_mappings = [
    (RosPointField.INT8, numpy.dtype("int8")),
    (RosPointField.UINT8, numpy.dtype("uint8")),
    (RosPointField.INT16, numpy.dtype("int16")),
    (RosPointField.UINT16, numpy.dtype("uint16")),
    (RosPointField.INT32, numpy.dtype("int32")),
    (RosPointField.UINT32, numpy.dtype("uint32")),
    (RosPointField.FLOAT32, numpy.dtype("float32")),
    (RosPointField.FLOAT64, numpy.dtype("float64")),
]
_nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in _pfnptype_mappings)


class HasFields_xyz(Protocol):
    x: Any
    y: Any
    z: Any


class HasFields_wxyz(Protocol):
    w: Any
    x: Any
    y: Any
    z: Any


def get_list_from_xyz(o_xyz: HasFields_xyz) -> List:
    """Gets a list of the fields: x,y,z

    Args:
        o_xyz: The source object with named fields.


    Returns:
        A list with the elements x,y,z from the source object.
    """
    o_seq = [
        o_xyz.x,
        o_xyz.y,
        o_xyz.z,
    ]
    return o_seq


def get_list_from_wxyz(o_wxyz: HasFields_wxyz) -> List:
    """Gets a list of the fields: w,x,y,z

    Args:
        o_wxyz: The source object with named fields.


    Returns:
        A list with the elements w,x,y,z from the source object.
    """
    o_seq = [
        o_wxyz.w,
        o_wxyz.x,
        o_wxyz.y,
        o_wxyz.z,
    ]
    return o_seq


def open3d_pointcloud_to_numpy_array(open3d_cloud: open3dPointCloud | open3dTensorPointCloud) -> numpy.ndarray:
    """Given an Open3D PointCloud object, return a numpy array either of
    shape (N,3) or (N,6), depending on whether the cloud is xyz or xyzrgb.
    """
    if isinstance(open3d_cloud, open3dPointCloud):
        return _open3d_legacy_pointcloud_to_numpy_array(open3d_cloud)
    return _open3d_tensor_pointcloud_to_numpy_array(open3d_cloud)


def _open3d_legacy_pointcloud_to_numpy_array(open3d_cloud: open3dPointCloud) -> numpy.ndarray:
    """Given an open3d.geometry.PointCloud object, return a numpy array either of
    shape (N,3) or (N,6), depending on whether the cloud is xyz or xyzrgb.
    """
    xyz = numpy.asarray(open3d_cloud.points)
    cloud_arr = xyz
    if len(open3d_cloud.colors) > 0:
        rgb = numpy.asarray(open3d_cloud.colors)
        cloud_arr = numpy.hstack([xyz, rgb])
    return cloud_arr


def _open3d_tensor_pointcloud_to_numpy_array(open3d_cloud: open3dTensorPointCloud) -> numpy.ndarray:
    """Given an open3d.t.geometry.PointCloud object, return a numpy array either of
    shape (N,3) or (N,6), depending on whether the cloud is xyz or xyzrgb.
    """
    xyz = open3d_cloud.point.positions.cpu().numpy()
    cloud_arr = xyz
    if "colors" in open3d_cloud.point:
        rgb = open3d_cloud.point.colors.cpu().numpy()
        cloud_arr = numpy.hstack([xyz, rgb])
    return cloud_arr


def numpy_array_to_ros_pointcloud2(
    cloud_arr: npt.NDArray,
    stamp: Optional[RosTimestamp] = None,
    frame_id: Optional[str] = None,
) -> RosPointCloud2:
    """Converts an numpy array into a ROS2 PointCloud2 message.
    The point cloud can be either xyz (i.e. no color), or xyzrgb (i.e. with color),
    depending on whether the size is Nx3 or Nx6, respectively.
    Color is represented as floating point RGB values between 0 and 1.

    Args:
        cloud_arr: Numpy array of size Nx3 or Nx6, with rows containing point information
            in the form of [x,y,z] or [x,y,z,r,g,b].
        stamp: ROS Timestamp for PointCloud2 message
        frame_id: Frame ID for PointCloud2 message

    Returns:
        ROS2 PointCloud2 message containing all point data, with header timestamp and frame_id
            if given

    """

    # convert the numpy array to a record array (specify field type for each column)
    if cloud_arr.shape[1] == 3:
        np_dtypes = [(name, numpy.float32) for name in ["x", "y", "z"]]
    elif cloud_arr.shape[1] == 6:
        np_dtypes = [(name, numpy.float32) for name in ["x", "y", "z", "r", "g", "b"]]
    else:
        raise ValueError(f"Unexpected dimensionality {cloud_arr.shape} of cloud array. Expecting (N, 3) or (N, 6)")

    cloud_record_arr = numpy.zeros(len(cloud_arr), dtype=np_dtypes)
    cloud_record_arr["x"] = cloud_arr[:, 0]
    cloud_record_arr["y"] = cloud_arr[:, 1]
    cloud_record_arr["z"] = cloud_arr[:, 2]
    if cloud_arr.shape[1] == 6:
        cloud_record_arr["r"] = cloud_arr[:, 3]
        cloud_record_arr["g"] = cloud_arr[:, 4]
        cloud_record_arr["b"] = cloud_arr[:, 5]

    cloud_msg = _numpy_recarray_to_ros_pointcloud2(cloud_record_arr, stamp=stamp, frame_id=frame_id)
    return cloud_msg


def open3d_pointcloud_to_ros_pointcloud2(
    open3d_cloud: open3dPointCloud | open3dTensorPointCloud,
    stamp: Optional[RosTimestamp] = None,
    frame_id: Optional[str] = None,
) -> RosPointCloud2:
    """Converts an Open3d PointCloud object into a ROS2 PointCloud2 message.
    The point cloud can be either xyz (i.e. no color), or xyzrgb (i.e. with color),
    depending on whether the .colors attribute of the given Open3D PointCloud is empty.
    """
    cloud_arr = open3d_pointcloud_to_numpy_array(open3d_cloud)

    return numpy_array_to_ros_pointcloud2(cloud_arr, stamp=stamp, frame_id=frame_id)


def _numpy_dtype_to_point_fields(dtype: numpy.dtype) -> List:
    """Convert a numpy record datatype into a list of PointFields."""
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = RosPointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(numpy.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = _nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


def _numpy_recarray_to_ros_pointcloud2(
    cloud_arr: numpy.ndarray, stamp: Optional[RosTimestamp] = None, frame_id: Optional[str] = None
) -> RosPointCloud2:
    """Converts a numpy record array to a sensor_msgs.msg.PointCloud2."""
    # make it 2d (even if height will be 1)
    cloud_arr = numpy.atleast_2d(cloud_arr)

    cloud_msg = RosPointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = _numpy_dtype_to_point_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = sys.byteorder != "little"
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * cloud_arr.shape[1]
    cloud_msg.is_dense = all([numpy.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""  # type: ignore
    as_array = array.array("B")  # type: ignore
    as_array.frombytes(array_bytes)
    cloud_msg.data = as_array
    return cloud_msg


def _ros_quaternion_to_unit_quaternion(quaternion: RosQuaternion) -> UnitQuaternion:
    """Internal function to convert an object of ROS's Quaternion msg class to the UnitQuaternion from Spatial Math"""
    return UnitQuaternion(s=quaternion.w, v=get_list_from_xyz(quaternion))


def to_unit_quaternion(
    quaternion: geometry_pb2.Quaternion | Quat | RosQuaternion | RosQuaternionStamped,
) -> UnitQuaternion:
    """Converts an object from a number of quaternion classes to the UnitQuaternion from Spatial Math"""
    if isinstance(quaternion, RosQuaternion):
        return _ros_quaternion_to_unit_quaternion(quaternion)
    elif isinstance(quaternion, RosQuaternionStamped):
        return _ros_quaternion_to_unit_quaternion(quaternion.quaternion)
    elif isinstance(quaternion, geometry_pb2.Quaternion):
        return UnitQuaternion(get_list_from_wxyz(quaternion))
    elif isinstance(quaternion, Quat):
        return UnitQuaternion(get_list_from_wxyz(quaternion))
    else:
        raise TypeError(f"Cannot convert from '{type(quaternion)}' to Quaternion")


def _ros_pose_to_se3(pose: RosPose) -> SE3:
    """Internal function to convert an object of ROS's Pose msg class to the SE3 from Spatial Math"""
    return SE3.Rt(t=get_list_from_xyz(pose.position), R=to_unit_quaternion(pose.orientation).SO3())


def _ros_transform_to_se3(transform: RosTransform) -> SE3:
    """Internal function to convert an object of ROS's Transform msg class to the SE3 class from Spatial Math"""
    return SE3.Rt(
        t=get_list_from_xyz(transform.translation),
        R=to_unit_quaternion(transform.rotation).SO3(),
    )


def _se3pose_to_se3(pose: SE3Pose) -> SE3:
    return SE3.Rt(t=get_list_from_xyz(pose), R=to_unit_quaternion(pose.rotation).SO3())


def to_se3(
    pose: (
        RosPose
        | RosPoseStamped
        | RosPoseWithCovariance
        | RosPoseWithCovarianceStamped
        | RosTransform
        | RosTransformStamped
        | SE2Pose
        | SE3Pose
        | geometry_pb2.SE2Pose
        | geometry_pb2.SE3Pose
        | SE2
        | SE3
    ),
) -> SE3:
    """Converts an object from various classes to the SE3 class from Spatial Math"""
    if isinstance(pose, RosPose):
        return _ros_pose_to_se3(pose)
    elif isinstance(pose, RosPoseWithCovariance) or isinstance(pose, RosPoseStamped):
        return _ros_pose_to_se3(pose.pose)
    elif isinstance(pose, RosPoseWithCovarianceStamped):
        return _ros_pose_to_se3(pose.pose.pose)
    elif isinstance(pose, RosTransform):
        return _ros_transform_to_se3(pose)
    elif isinstance(pose, RosTransformStamped):
        return _ros_transform_to_se3(pose.transform)
    elif isinstance(pose, SE3Pose):
        return _se3pose_to_se3(pose)
    elif isinstance(pose, SE2Pose):
        return _se3pose_to_se3(SE3Pose.from_se2(pose))
    elif isinstance(pose, geometry_pb2.SE3Pose):
        return _se3pose_to_se3(SE3Pose.from_proto(pose))
    elif isinstance(pose, geometry_pb2.SE2Pose):
        return _se3pose_to_se3(SE3Pose.from_se2(SE2Pose.from_proto(pose)))
    elif isinstance(pose, SE2):
        return pose.SE3()
    elif isinstance(pose, SE3):
        return pose
    else:
        raise TypeError(f"Cannot convert from '{type(pose)}' to SE3Pose")


def se2_to_se2pose(se2: SE2) -> SE2Pose:
    return SE2Pose(x=se2.x, y=se2.y, angle=se2.theta())


def se2_to_se2pose_proto(se2: SE2) -> geometry_pb2.SE2Pose:
    return se2_to_se2pose(se2).to_proto()


def se3_to_se3pose(transform: SE3) -> SE3Pose:
    """Converts the SE3 from spatial math to SE3Pose from the spot-sdk"""
    q = UnitQuaternion(transform)
    return SE3Pose(x=transform.x, y=transform.y, z=transform.z, rot=Quat(w=q.s, x=q.v[0], y=q.v[1], z=q.v[2]))


def se3_to_se3pose_proto(transform: SE3) -> geometry_pb2.SE3Pose:
    """Converts the SE3 from spatial math to SE3Pose protobuf from the spot-sdk"""
    return se3_to_se3pose(transform).to_proto()


def twist3_to_se3velocity_proto(velocity: Twist3) -> geometry_pb2.SE3Velocity:
    """Converts a Twist3 from spatial math into a SE3Velocity protobuf from the spot-sdk"""
    return SE3Velocity(
        lin_x=velocity.v[0],
        lin_y=velocity.v[1],
        lin_z=velocity.v[2],
        ang_x=velocity.w[0],
        ang_y=velocity.w[1],
        ang_z=velocity.w[2],
    ).to_proto()


def spatialforce_to_wrench_proto(spatialforce: SpatialForce) -> geometry_pb2.Wrench:
    """Converts a SpatialForce from spatialmath into a Wrench protobuf from the spot-sdk"""
    wrench_vec = spatialforce.data[0]
    force = geometry_pb2.Vec3(x=wrench_vec[0], y=wrench_vec[1], z=wrench_vec[2])
    torque = geometry_pb2.Vec3(x=wrench_vec[3], y=wrench_vec[4], z=wrench_vec[5])
    return geometry_pb2.Wrench(force=force, torque=torque)
