# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from enum import Enum
from functools import cached_property
from typing import Any, List, Optional, TypeAlias, Union

from bdai_ros2_wrappers.callables import generalized_method
from bdai_ros2_wrappers.subscription import Subscription
from geometry_msgs.msg import Pose, PoseStamped
from spot_msgs.srv import (
    GraphNavClearGraph,
    GraphNavGetLocalizationPose,
    GraphNavSetLocalization,
    GraphNavUploadGraph,
    ListGraph,
)
from spot_utilities.spot_basic.base import SpotBaseProtocol
from spot_utilities.spot_basic.utilities import SpotServiced
from spot_utilities.spot_util import spot_namespaced


class SpotGraphNavMixin(SpotBaseProtocol):
    """Mixin that provides access to Spot GraphNav capabilities."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        SpotGraphNavMixin.get_localization_pose.rebind(
            self,
            SpotServiced(
                GraphNavGetLocalizationPose,
                spot_namespaced(self.robot_name, "graph_nav_get_localization_pose"),
                node=self.node,
            ),
        )
        SpotGraphNavMixin.set_localization.rebind(
            self,
            SpotServiced(
                GraphNavSetLocalization, spot_namespaced(self.robot_name, "graph_nav_set_localization"), node=self.node
            ),
        )
        SpotGraphNavMixin.upload_graph.rebind(
            self,
            SpotServiced(
                GraphNavUploadGraph, spot_namespaced(self.robot_name, "graph_nav_upload_graph"), node=self.node
            ),
        )
        SpotGraphNavMixin.list_graph.rebind(
            self, SpotServiced(ListGraph, spot_namespaced(self.robot_name, "list_graph"), node=self.node)
        )
        SpotGraphNavMixin.clear_graph.rebind(
            self,
            SpotServiced(GraphNavClearGraph, spot_namespaced(self.robot_name, "graph_nav_clear_graph"), node=self.node),
        )

    @cached_property
    def localization(self) -> Subscription:
        """Get pose estimates subscription."""
        return Subscription(PoseStamped, spot_namespaced(self.robot_name, "graph_nav/body_pose"), node=self.node)

    class LocalizationMethod(str, Enum):
        FIDUCIAL = "fiducial"
        WAYPOINT = "waypoint"

        def __repr__(self) -> str:
            return str(self)

    @generalized_method(transitional=True)
    def set_localization(self, method: Union[LocalizationMethod, str], waypoint_id: Optional[str] = None) -> bool:
        """Force localization manually.

        If using waypoints, a waypoint id must be specified.
        """
        self.node.get_logger().info("Set localization")
        request = GraphNavSetLocalization.Request()
        request.method = LocalizationMethod(method)
        if request.method == LocalizationMethod.WAYPOINT:
            request.waypoint_id = waypoint_id
        response = self.set_localization.asynchronously(request, nothrow=True)
        return response is not None and response.success

    @generalized_method(transitional=True)
    def get_localization_pose(self) -> Optional[Pose]:
        """Get the estimated pose."""
        self.node.get_logger().info("Get localization pose")
        response = self.get_localization_pose.synchronously(nothrow=True)
        if response is None:
            return None
        if not response.success:
            raise RuntimeError(f"Failed to get localization pose. Reason: {response.message}")
        return response.pose

    get_localization = get_localization_pose

    @generalized_method(transitional=True)
    def upload_graph(self, upload_filepath: str) -> bool:
        """Upload graph structure (waypoints and topology)."""
        self.node.get_logger().info("Upload graph")
        response = self.upload_graph.synchronously(upload_filepath, nothrow=True)
        return response is not None and response.success

    @generalized_method(transitional=True)
    def clear_graph(self) -> bool:
        """Clear graph structure (and localization)."""
        self.node.get_logger().info("Clear graph")
        response = self.clear_graph.synchronously(nothrow=True)
        return response is not None and response.success

    @generalized_method(transitional=True)
    def list_graph(self, upload_filepath: str) -> List[str]:
        """List waypoint IDs in a given graph structure."""
        self.node.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        response = self.list_graph.synchronously(upload_filepath, nothrow=True)
        if response is None:
            return []
        return response.waypoint_ids


# Alias
LocalizationMethod: TypeAlias = SpotGraphNavMixin.LocalizationMethod
