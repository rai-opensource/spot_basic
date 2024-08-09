# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

from bdai_ros2_wrappers.callables import generalized_method
from bosdyn_api_msgs.msg import GripperCameraParams
from spot_msgs.srv import (
    GetGripperCameraParameters,
    SetGripperCameraParameters,
)
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced


class SpotGripperCameraMixin(SpotBaseProtocol):
    """Mixin to manage Spot gripper camera parameters."""

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

    @generalized_method(transitional=True)
    def get_gripper_camera_parameters(self) -> tuple[bool, str, GripperCameraParams]:
        """Get the gripper camera parameters on the SpotDriver

        Returns: A tuple containing [success, message, response]
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

    @generalized_method(transitional=True)
    def set_gripper_camera_parameters(self, gripper_camera_params: GripperCameraParams) -> bool:
        """Set the gripper camera parameters on the SpotDriver

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
