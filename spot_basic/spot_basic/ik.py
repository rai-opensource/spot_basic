# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any, Optional, Protocol, Union, overload

import bosdyn_spot_api_msgs.msg
from bosdyn.api.spot import inverse_kinematics_pb2
from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import SpotServiced, SpotServicedProtocol, spot_namespaced
from spot_msgs.srv import GetInverseKinematicSolutions
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike


class SpotIKProtocol(SpotBaseProtocol, Protocol):
    """Spot inverse kinematics solver protocol."""

    class IKSolutionsProtocol(
        SpotServicedProtocol[GetInverseKinematicSolutions.Request, GetInverseKinematicSolutions.Response], Protocol
    ):
        """Inverse kinematics solutions protocol."""

        @overload
        def synchronously(
            self,
            request: Union[
                bosdyn_spot_api_msgs.msg.InverseKinematicsRequest,
                inverse_kinematics_pb2.InverseKinematicsRequest,
            ],
            *,
            timeout_sec: Optional[float] = None,
        ) -> GetInverseKinematicSolutions.Response:
            """Query an IK solution synchronously.

            Args:
                request: IK solution request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the IK solution query will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to acknowledge it).

            Returns:
                the response to the IK solution query.

            Raises:
                ServiceError: if the IK solution query fails.
                ServiceTimeout: if the IK solution query times out.
            """

        @overload
        def synchronously(
            self,
            request: Union[
                bosdyn_spot_api_msgs.msg.InverseKinematicsRequest,
                inverse_kinematics_pb2.InverseKinematicsRequest,
            ],
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[GetInverseKinematicSolutions.Response]:
            """Query an IK solution synchronously.

            Args:
                request: IK solution request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the IK solution query will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the IK solution query or None on timeout or failure.
            """

        def asynchronously(
            self,
            request: Union[
                bosdyn_spot_api_msgs.msg.InverseKinematicsRequest,
                inverse_kinematics_pb2.InverseKinematicsRequest,
            ],
        ) -> FutureLike[GetInverseKinematicSolutions.Response]:
            """Query an IK solution asynchronously.

            Args:
                request: IK solution request, in its Protobuf or ROS 2 form.

            Returns:
                a future response to the IK solution query.
            """

    get_inverse_kinematic_solutions: GeneralizedMethodLike[IKSolutionsProtocol]


class SpotIKMixin(SpotBaseProtocol):
    """Mixin offering access to the Spot IK solver."""

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

    @generalized_method(spec=GeneralizedMethodLike[SpotIKProtocol.IKSolutionsProtocol])
    def get_inverse_kinematic_solutions(
        self,
        request: Union[
            bosdyn_spot_api_msgs.msg.InverseKinematicsRequest,
            inverse_kinematics_pb2.InverseKinematicsRequest,
        ],
        **kwargs: Any,
    ) -> GetInverseKinematicSolutions.Response:
        """Query the robot for an IK solution.

        See `SpotIKProtocol.IKSolutionsProtocol` for further reference.
        """
