# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Callable, Optional, Union

from bdai_ros2_wrappers.service import Serviced, ServiceError
from bdai_ros2_wrappers.service_handle import ServiceHandle
from rclpy.task import Future


class SpotServiced(Serviced):
    """Ergonomic ROS 2 service wrapper subclass to support legacy invocation signatures."""

    def __call__(
        self,
        *,
        asynchronous: bool = False,
        result_callback: Optional[Callable[[Future], None]] = None,
        on_failure_callback: Optional[Callable[[None], None]] = None,
    ) -> Union[bool, ServiceHandle]:
        """Invokes Spot ROS 2 driver service

        Args:
            asynchronous: whether or not to perform an asynchronous request
            result_callback: callback executing on the future returned from a service call
            on_failure_callback: callback executing if a service fails

        Returns:
            If synchronous, True if command succeeded, False otherwise
            If asynchronous, service handle containing the updated results
        """
        if asynchronous:
            name = f"{self.robot_name}.{self.service_name}"
            handle = ServiceHandle(name, context=self.client.context)
            handle.set_send_service_future(self.asynchronously())
            if result_callback is not None:
                handle.set_result_callback(result_callback)
            if on_failure_callback is not None:
                handle.set_on_failure_callback(on_failure_callback)
            return handle
        try:
            response = self.synchronously()
            return response.success
        except ServiceError:
            return False
