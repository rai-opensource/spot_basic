# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from enum import Enum
from functools import cached_property
from typing import Any, List, Optional, Protocol, TypeAlias, Union, overload

from bdai_ros2_wrappers.callables import GeneralizedMethodLike, generalized_method
from bdai_ros2_wrappers.futures import FutureLike
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
from spot_utilities.spot_basic.utilities import SpotServiced, SpotServicedProtocol
from spot_utilities.spot_util import spot_namespaced


class LocalizationMethod(str, Enum):
    """Avaialble localization resetting methods."""

    FIDUCIAL = "fiducial"
    WAYPOINT = "waypoint"

    def __repr__(self) -> str:
        return str(self)


class SpotGraphNavProtocol(SpotBaseProtocol, Protocol):
    """Spot GraphNav service protocol."""

    class ResetLocalizationProtocol(
        SpotServicedProtocol[GraphNavSetLocalization.Request, GraphNavSetLocalization.Response], Protocol
    ):
        """GraphNav localization reset protocol."""

        @overload
        def synchronously(
            self,
            method: Union[LocalizationMethod, str],
            waypoint_id: Optional[str] = None,
            *,
            timeout_sec: Optional[float] = None,
        ) -> GraphNavSetLocalization.Response:
            """Reset GraphNav localization synchronously.

            Args:
                method: method for re-localization, either waypoint or fiducial based.
                waypoint_id: waypoint ID for waypoint based re-localization.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the reset request will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the robot
                to react).

            Returns:
                the response to the localization reset request.

            Raises:
                ServiceError: if the localization reset request fails.
                ServiceTimeout: if the localization reset request times out.
            """

        @overload
        def synchronously(
            self,
            method: Union[LocalizationMethod, str],
            waypoint_id: Optional[str] = None,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> GraphNavSetLocalization.Response:
            """Reset GraphNav localization synchronously.

            Args:
                method: method for re-localization, either waypoint or fiducial based.
                waypoint_id: waypoint ID for waypoint based re-localization.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the reset request will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the robot
                to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the localization reset request or None on timeout or failure.
            """

        def asynchronously(
            self, method: Union[LocalizationMethod, str], waypoint_id: Optional[str] = None
        ) -> FutureLike[GraphNavSetLocalization.Response]:
            """Reset GraphNav localization asynchronously.

            Args:
                method: method for re-localization, either waypoint or fiducial based.
                waypoint_id: waypoint ID for waypoint based re-localization.

            Returns:
                a future response to the localization reset request.
            """

    class GetLocalizationPoseProtocol(
        SpotServicedProtocol[GraphNavGetLocalizationPose.Request, GraphNavGetLocalizationPose.Response], Protocol
    ):
        """GraphNav localization query protocol."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> GraphNavGetLocalizationPose.Response:
            """Query GraphNav localization estimate synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).

            Returns:
                the response to the localization estimate request.

            Raises:
                ServiceError: if the localization estimate request fails.
                ServiceTimeout: if the localization estimate request times out.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[GraphNavGetLocalizationPose.Response]:
            """Query GraphNav localization estimate synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the request will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the localization estimate request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[GraphNavGetLocalizationPose.Response]:
            """Query GraphNav localization estimate asynchronously.

            Returns:
                a future response to the localization estimate request.
            """

    class UploadLocalizationGraphProtocol(
        SpotServicedProtocol[GraphNavUploadGraph.Request, GraphNavUploadGraph.Response], Protocol
    ):
        """GraphNav graph uploading protocol."""

        @overload
        def synchronously(
            self, upload_filepath: str, *, timeout_sec: Optional[float] = None
        ) -> GraphNavUploadGraph.Response:
            """Upload a new GraphNav graph (ie. waypoints and topology) synchronously.

            Args:
                upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be uploaded.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the upload will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).

            Returns:
                the response to the upload request.

            Raises:
                ServiceError: if the upload request fails.
                ServiceTimeout: if the upload request times out.
            """

        @overload
        def synchronously(
            self, upload_filepath: str, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[GraphNavUploadGraph.Response]:
            """Upload a new GraphNav graph (ie. waypoints and topology) synchronously.

            Args:
                upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be uploaded.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the upload will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the upload request or None on timeout or failure.
            """

        def asynchronously(self, upload_filepath: str) -> FutureLike[GraphNavUploadGraph.Response]:
            """Upload a new GraphNav graph (ie. waypoints and topology) asynchronously.

            Args:
                upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph`
                instance to be uploaded.

            Returns:
                a future response to the upload request.
            """

    class ClearLocalizationGraphProtocol(
        SpotServicedProtocol[GraphNavClearGraph.Request, GraphNavClearGraph.Response], Protocol
    ):
        """GraphNav graph clearing protocol."""

        @overload
        def synchronously(self, *, timeout_sec: Optional[float] = None) -> GraphNavClearGraph.Response:
            """Clear the current GraphNav graph.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, graph clearing will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).

            Returns:
                the response to the graph clearing request.

            Raises:
                ServiceError: if the graph clearing request fails.
                ServiceTimeout: if the graph clearing request times out.
            """

        @overload
        def synchronously(
            self, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[GraphNavClearGraph.Response]:
            """Clear the current GraphNav graph.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, graph clearing will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the graph clearing request or None on timeout or failure.
            """

        def asynchronously(self) -> FutureLike[GraphNavClearGraph.Response]:
            """Clear the current GraphNav graph.

            Returns:
               a future response to the graph clearing request.
            """

    class ListLocalizationGraphProtocol(SpotServicedProtocol[ListGraph.Request, ListGraph.Response], Protocol):
        """GraphNav graph introspection protocol."""

        @overload
        def synchronously(self, upload_filepath: str, *, timeout_sec: Optional[float] = None) -> ListGraph.Response:
            """List a GraphNav graph structure synchronously.

            Args:
                upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be listed.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, graph listing will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).

            Returns:
                the response to the graph listing request.
            """

        @overload
        def synchronously(
            self, upload_filepath: str, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[ListGraph.Response]:
            """List a GraphNav graph structure synchronously.

            Args:
                upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be listed.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, graph listing will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the graph listing request or None on timeout or failure.
            """

        def asynchronously(self, upload_filepath: str) -> FutureLike[ListGraph.Response]:
            """List a GraphNav graph structure asynchronously.

            Args:
                upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be listed.

            Returns:
                a future response to the graph listing request.
            """

    @property
    def localization(self) -> Subscription[PoseStamped]: ...

    reset_localization: GeneralizedMethodLike[ResetLocalizationProtocol]
    set_localization: GeneralizedMethodLike[ResetLocalizationProtocol]

    get_localization: GeneralizedMethodLike[GetLocalizationPoseProtocol]
    get_localization_pose: GeneralizedMethodLike[GetLocalizationPoseProtocol]

    upload_graph: GeneralizedMethodLike[UploadLocalizationGraphProtocol]
    clear_graph: GeneralizedMethodLike[ClearLocalizationGraphProtocol]
    list_graph: GeneralizedMethodLike[ListLocalizationGraphProtocol]


class SpotGraphNavMixin(SpotBaseProtocol):
    """Mixin offering access to the Spot GraphNav service capabilities."""

    LocalizationMethod: TypeAlias = LocalizationMethod

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
    def localization(self) -> Subscription[PoseStamped]:
        """Get latest localization pose estimates subscription."""
        return Subscription(PoseStamped, spot_namespaced(self.robot_name, "graph_nav/body_pose"), node=self.node)

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.ResetLocalizationProtocol])
    def reset_localization(
        self, method: Union[LocalizationMethod, str], waypoint_id: Optional[str] = None, **kwargs: Any
    ) -> GraphNavSetLocalization.Response:
        """Reset GraphNav localization on the robot.

        Args:
            method: method for re-localization, either waypoint or fiducial based.
            waypoint_id: waypoint ID for waypoint based re-localization.

        See `SpotGraphNavProtocol.ResetLocalizationProtocol` for further reference.
        """

    @reset_localization.legacy
    def _legacy_reset_localization(
        self, method: Union[LocalizationMethod, str], waypoint_id: Optional[str] = None
    ) -> bool:
        """Force relocalization manually.

        If using waypoints, a waypoint id must be specified.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.
        """
        self.node.get_logger().info("Set localization")
        request = GraphNavSetLocalization.Request()
        request.method = LocalizationMethod(method)
        if request.method == LocalizationMethod.WAYPOINT:
            request.waypoint_id = waypoint_id
        response = self.reset_localization.synchronously(request, nothrow=True)
        return response is not None and response.success

    set_localization = reset_localization

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.GetLocalizationPoseProtocol])
    def get_localization_pose(self, **kwargs: Any) -> GraphNavGetLocalizationPose.Response:
        """Query GraphNav localization estimate.

        See `SpotGraphNavProtocol.GetLocalizationPoseProtocol` for further reference.
        """

    @get_localization_pose.legacy
    def _legacy_get_localization_pose(self) -> Optional[Pose]:
        """Get the estimated pose.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.
        """
        self.node.get_logger().info("Get localization pose")
        response = self.get_localization_pose.synchronously(nothrow=True)
        if response is None:
            return None
        if not response.success:
            raise RuntimeError(f"Failed to get localization pose. Reason: {response.message}")
        return response.pose

    get_localization = get_localization_pose

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.UploadLocalizationGraphProtocol])
    def upload_graph(self, upload_filepath: str, **kwargs: Any) -> GraphNavUploadGraph.Response:
        """Upload a new GraphNav graph (ie. waypoints and topology).

        Args:
            upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be uploaded.

        See `SpotGraphNavProtocol.UploadGraphProtocol` for further reference.
        """

    @upload_graph.legacy
    def _legacy_upload_graph(self, upload_filepath: str) -> bool:
        """Upload graph structure (waypoints and topology).

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.
        """
        self.node.get_logger().info("Upload graph")
        response = self.upload_graph.synchronously(upload_filepath, nothrow=True)
        return response is not None and response.success

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.ClearLocalizationGraphProtocol])
    def clear_graph(self, **kwargs: Any) -> GraphNavClearGraph.Response:
        """Clear the current GraphNav graph.

        See `SpotGraphNavProtocol.ClearGraphProtocol` for further reference.
        """

    @clear_graph.legacy
    def _legacy_clear_graph(self) -> bool:
        """Clear graph structure (and localization).

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.
        """
        self.node.get_logger().info("Clear graph")
        response = self.clear_graph.synchronously(nothrow=True)
        return response is not None and response.success

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.ListLocalizationGraphProtocol])
    def list_graph(self, upload_filepath: str, **kwargs: Any) -> ListGraph.Response:
        """List a GraphNav graph structure.

        See `SpotGraphNavProtocol.ListGraphProtocol` for further reference.
        """

    @list_graph.legacy
    def _legacy_list_graph(self, upload_filepath: str) -> List[str]:
        """List waypoint IDs in a given graph structure.

        This is a legacy interface and implementation.
        It will eventually be deprecated.
        Its usage is discouraged.
        """
        self.node.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        response = self.list_graph.synchronously(upload_filepath, nothrow=True)
        if response is None:
            return []
        return response.waypoint_ids
