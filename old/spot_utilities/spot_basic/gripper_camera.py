# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional, Protocol, Union, overload

import bosdyn_api_msgs.msg
from bosdyn.api import gripper_camera_param_pb2
from spot_msgs.srv import (
    GetGripperCameraParameters,
    SetGripperCameraParameters,
)
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServicedProtocol
from spot_utilities.spot_util import spot_namespaced

from bdai_ros2_wrappers.callables import GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike


class SpotGripperCameraProtocol(SpotBaseProtocol, Protocol):
    """Spot gripper camera protocol."""

    class GetGripperCameraProtocol(
        SpotServicedProtocol[GetGripperCameraParameters.Request, GetGripperCameraParameters.Response], Protocol
    ):
        """Gripper camera parameters read protocol."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> GetGripperCameraParameters.Response:
            """Get gripper camera parameters synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the gripper camera parameters request.

            Raises:
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[GetGripperCameraParameters.Response]:
            """Get gripper camera parameters synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the gripper camera parameters request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[GetGripperCameraParameters.Response]:
            """Get gripper camera parameters asynchronously.

            Returns:
                a future response to the gripper camera parameters request.
            """

    class SetGripperCameraProtocol(
        SpotServicedProtocol[SetGripperCameraParameters.Request, SetGripperCameraParameters.Response], Protocol
    ):
        """Gripper camera parameters write protocol."""

        @overload
        def synchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.GripperCameraParamRequest, gripper_camera_param_pb2.GripperCameraParamRequest
            ],
            *,
            timeout_sec: Optional[float] = None,
        ) -> SetGripperCameraParameters.Response:
            """Set gripper camera parameters synchronously.

            Args:
                request: gripper camera parameters change request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the gripper camera parameters change request.

            Raises:
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        @overload
        def synchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.GripperCameraParamRequest, gripper_camera_param_pb2.GripperCameraParamRequest
            ],
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> SetGripperCameraParameters.Response:
            """Set gripper camera parameters synchronously.

            Args:
                request: gripper camera parameters change request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note
                this timeout is local to the caller (and it may take a while for the robot
                to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the gripper camera parameters change request or None on timeout or failure.
            """

        def asynchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.GripperCameraParamRequest, gripper_camera_param_pb2.GripperCameraParamRequest
            ],
        ) -> FutureLike[SetGripperCameraParameters.Response]:
            """Set gripper camera parameters asynchronously.

            Args:
                request: gripper camera parameters change request, in its Protobuf or ROS 2 form.

            Returns:
                a future response to the gripper camera parameters change request.
            """

    get_gripper_camera_parameters: GeneralizedMethodLike[GetGripperCameraProtocol]
    set_gripper_camera_parameters: GeneralizedMethodLike[SetGripperCameraProtocol]


class SpotGripperCameraMixin(SpotBaseProtocol):
    """Mixin to manage the Spot gripper camera."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotGripperCameraMixin.get_gripper_camera_parameters.rebind(
            self,
            SpotServiced(
                GetGripperCameraParameters,
                spot_namespaced(self.robot_name, "get_gripper_camera_parameters"),
                node=self.node,
            ),
        )
        SpotGripperCameraMixin.set_gripper_camera_parameters.rebind(
            self,
            SpotServiced(
                SetGripperCameraParameters,
                spot_namespaced(self.robot_name, "set_gripper_camera_parameters"),
                node=self.node,
            ),
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotGripperCameraProtocol.GetGripperCameraProtocol])
    def get_gripper_camera_parameters(self, **kwargs: Any) -> GetGripperCameraParameters.Response:
        """Get the robot gripper camera parameters.

        See `SpotGripperCameraProtocol.GetGripperCameraParameters` for further reference.
        """

    @get_gripper_camera_parameters.legacy
    def _legacy_get_gripper_camera_parameters(self) -> tuple[bool, str, bosdyn_api_msgs.msg.GripperCameraParams]:
        """Get the gripper camera parameters on the SpotDriver

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Returns:
            a tuple containing [success, message, response]
        """
        self.node.get_logger().info(f"Getting gripper camera parameters for {self.robot_name}")
        response = self.get_gripper_camera_parameters.synchronously(nothrow=True)
        if response is None:
            return False, "Call to wait_for_future timed out", None
        if not response.success:
            self.node.get_logger().warning(
                f"Getting gripper camera parameters for {self.robot_name} failed. Msg: {response.message}."
            )
        return response.success, response.message, response.response.params

    @generalized_method(spec=GeneralizedMethodLike[SpotGripperCameraProtocol.SetGripperCameraProtocol])
    def set_gripper_camera_parameters(
        self,
        request: Union[
            bosdyn_api_msgs.msg.GripperCameraParamRequest, gripper_camera_param_pb2.GripperCameraParamRequest
        ],
        **kwargs: Any,
    ) -> SetGripperCameraParameters.Response:
        """Set the robot gripper camera parameters.

        Args:
            request: gripper camera parameters change request, in its Protobuf or ROS 2 form.

        See `SpotGripperCameraProtocol.SetGripperCameraParameters` for further reference.
        """

    @set_gripper_camera_parameters.legacy
    def _legacy_set_gripper_camera_parameters(
        self, gripper_camera_params: bosdyn_api_msgs.msg.GripperCameraParams
    ) -> bool:
        """Set the gripper camera parameters on the SpotDriver

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            gripper_camera_params: the gripper camera parameters and their desired values.
                                    Of type bosdyn_msgs/GripperCameraParams

        Returns: a boolean indicating if the service was successful or not
        """
        self.node.get_logger().info(f"Setting gripper camera parameters for {self.robot_name}")
        request = SetGripperCameraParameters.Request()
        request.request.params = gripper_camera_params
        response = self.set_gripper_camera_parameters.synchronously(request, nothrow=True)
        if response is None:
            return False
        if not response.success:
            self.node.get_logger().warning(
                f"Setting gripper camera parameters for {self.robot_name} failed. Msg: {response.message}."
            )
            return False
        return True
