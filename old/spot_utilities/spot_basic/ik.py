# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

from bdai_ros2_wrappers.callables import generalized_method
from bosdyn.api.spot.inverse_kinematics_pb2 import InverseKinematicsRequest, InverseKinematicsResponse
from bosdyn_api_msgs.conversions import convert
from spot_msgs.srv import GetInverseKinematicSolutions
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced


class SpotIKMixin(SpotBaseProtocol):
    """Mixin that provides access to Spot IK solvers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotIKMixin.get_inverse_kinematic_solutions.rebind(
            self,
            SpotServiced(
                GetInverseKinematicSolutions,
                spot_namespaced(self.robot_name, "get_inverse_kinematic_solutions"),
                node=self.node,
            ),
        )

    @generalized_method(transitional=True)
    def get_inverse_kinematic_solutions(
        self, ik_request: InverseKinematicsRequest
    ) -> Optional[InverseKinematicsResponse]:
        """Query an IK solution for a given pose.
        Returns:
            A solution to the IK query, if any.
        """
        self.node.get_logger().info(f"Query IK solutions on {self.robot_name}")
        request = GetInverseKinematicSolutions.Request()
        convert(ik_request, request.request)
        response = self.get_inverse_kinematic_solutions.synchronously(request, nothrow=True)
        if response is None:
            return None
        proto_response = InverseKinematicsResponse()
        convert(response.response, proto_response)
        return proto_response
