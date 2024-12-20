# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional, Protocol, Union, overload

import bosdyn_api_msgs.msg
from bosdyn.api import world_object_pb2
from spot_msgs.srv import ListWorldObjects
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServicedProtocol
from spot_utilities.spot_util import spot_namespaced

from bdai.deprecation import deprecated
from bdai_ros2_wrappers.callables import GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike


class SpotWorldProtocol(SpotBaseProtocol, Protocol):
    """Spot world represetation protocol."""

    class ListWorldObjectsProtocol(SpotServicedProtocol[ListWorldObjects.Request, ListWorldObjects.Response], Protocol):
        """World representation listing protocol."""

        @overload
        def synchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.ListWorldObjectRequest,
                world_object_pb2.ListWorldObjectRequest,
            ],
            *,
            timeout_sec: Optional[float] = None,
        ) -> ListWorldObjects.Response:
            """List objects in the world synchronously.

            Args:
                request: object listing request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the object listing will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to react).

            Returns:
                the response to the object listing request.

            Raises:
                ServiceError: if the object listing request fails.
                ServiceTimeout: if the object listing request times out.
            """

        @overload
        def synchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.ListWorldObjectRequest,
                world_object_pb2.ListWorldObjectRequest,
            ],
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[ListWorldObjects.Response]:
            """List objects in the world synchronously.

            Args:
                request: object listing request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the object listing will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the object listing request or None on timeout or failure.
            """

        def asynchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.ListWorldObjectRequest,
                world_object_pb2.ListWorldObjectRequest,
            ],
        ) -> FutureLike[ListWorldObjects.Response]:
            """List objects in the world asynchronously.

            Args:
                request: object listing request, in its Protobuf or ROS 2 form.

            Returns:
                a future response to the object listing request.
            """

    list_world_objects: GeneralizedMethodLike[ListWorldObjectsProtocol]


class SpotWorldMixin(SpotBaseProtocol):
    """Mixin that manages Spot world representation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotWorldMixin.list_world_objects.rebind(
            self, SpotServiced(ListWorldObjects, spot_namespaced(self.robot_name, "list_world_objects"), node=self.node)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.ListWorldObjectsProtocol])
    def list_world_objects(self, request: ListWorldObjects.Request, **kwargs: Any) -> ListWorldObjects.Response:
        """List objects in the world known to the robot.

        See `SpotWorldProtocol.ListWorldObjectsProtocol` for further reference.
        """

    @list_world_objects.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_utilities.spot_basic.SpotBasic.list_world_objects.synchronously",
    )
    def _legacy_list_world_objects(
        self, request: bosdyn_api_msgs.msg.ListWorldObjectRequest
    ) -> Optional[bosdyn_api_msgs.msg.ListWorldObjectResponse]:
        """Requests a list of world objects.

        When constructing the calling ListWorldObjectRequest, you must follow the correct typing for
        requested objects, which is somewhat non-intuitive:

            # imports required for below
            from bosdyn_msgs.msg import ListWorldObjectRequest, WorldObjectType
            from bosdyn.api import world_object_pb2

            # must be constructed explicitly, WorldObjectType.WORLD_OBJECT_DOCK is also not the same
            world_object_dock_type = WorldObjectType(value=world_object_pb2.WORLD_OBJECT_DOCK)

            # add any requested objects to the list of object types (or leave it blank for all)
            request_fiducials = ListWorldObjectRequest()
            request_fiducials.object_type.append(world_object_dock_type)

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.

        Args:
            request: object listing request in ROS 2 form.

        Returns:
            the response to the object listing request, if any.
        """
        self.node.get_logger().info(f"List world objects for {self.robot_name}")
        response = self.list_world_objects.synchronously(ListWorldObjects.Request(request=request), nothrow=True)
        if response is None:
            return None
        return response.response
