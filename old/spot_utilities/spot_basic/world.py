# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

from bdai_ros2_wrappers.callables import generalized_method
from bosdyn_api_msgs.msg import (
    ListWorldObjectRequest,
    ListWorldObjectResponse,
)
from spot_msgs.srv import ListWorldObjects
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced


class SpotWorldMixin(SpotBaseProtocol):
    """Mixin that manages Spot world representation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotWorldMixin.list_world_objects.rebind(
            self, SpotServiced(ListWorldObjects, spot_namespaced(self.robot_name, "list_world_objects"), node=self.node)
        )

    @generalized_method(transitional=True)
    def list_world_objects(self, request: ListWorldObjectRequest) -> Optional[ListWorldObjectResponse]:
        """Requests a list of world objects.

        When constructing the calling ListWorldObjectRequest, you must follow the correct typing for
        requested objects, which is somewhat non-intuitive.

        # imports required for below
        from bosdyn_msgs.msg import ListWorldObjectRequest, WorldObjectType
        from bosdyn.api import world_object_pb2

        # must be constructed explicitly, WorldObjectType.WORLD_OBJECT_DOCK is also not the same
        world_object_dock_type = WorldObjectType(value=world_object_pb2.WORLD_OBJECT_DOCK)

        # add any requested objects to the list of object types (or leave it blank for all)
        request_fiducials = ListWorldObjectRequest()
        request_fiducials.object_type.append(world_object_dock_type)

        TODO: make this more robust and user intuitive
        """
        self.node.get_logger().info(f"List world objects for {self.robot_name}")
        response = self.list_world_objects.synchronously(ListWorldObjects.Request(request=request), nothrow=True)
        if response is None:
            return None
        return response.response
