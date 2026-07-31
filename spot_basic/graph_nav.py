# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""GraphNav localization, graph management, and navigation helpers."""

from enum import Enum
from functools import cached_property
from typing import Any, Optional, Protocol, TypeAlias, Union, overload

from geometry_msgs.msg import PoseStamped
from spot_msgs.action import NavigateTo  # type: ignore
from spot_msgs.srv import (
    GraphNavClearGraph,
    GraphNavGetLocalizationPose,
    GraphNavSetLocalization,
    GraphNavUploadGraph,
    ListGraph,
)
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike
from synchros2.subscription import Subscription

from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
    SpotServiced,
    SpotServicedProtocol,
    spot_namespaced,
)


class LocalizationMethod(str, Enum):
    """Avaialble localization resetting methods."""

    FIDUCIAL = "fiducial"
    WAYPOINT = "waypoint"

    def __repr__(self) -> str:
        """Return the string representation of the localization method.

        Returns:
            the localization method as text.
        """
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

    class NavigateToWaypointProtocol(
        SpotActionableProtocol[NavigateTo.Goal, NavigateTo.Result, NavigateTo.Feedback], Protocol
    ):
        """GraphNav graph navigate to a waypoint protocol."""

        @overload
        def synchronously(self, waypoint_id: str, *, timeout_sec: Optional[float] = None) -> NavigateTo.Result:
            """Navigate to a given waypoint

            Args:
                waypoint_id: id of the waypoint to navigate to
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, graph listing will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).

            Returns:
                the result to the navigate to request
            """

        @overload
        def synchronously(
            self, waypoint_id: str, *, timeout_sec: Optional[float] = None, nothrow: bool = False
        ) -> Optional[NavigateTo.Result]:
            """Navigate to a given waypoint synchronously.

            Args:
                waypoint_id: id of the waypoint to navigate to
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, graph listing will be cancelled and the call will raise. Note this timeout
                is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the result to the navigate to request or None on timeout or failure.
            """

        def asynchronously(self, waypoint_id: str) -> ActionFuture[NavigateTo.Result, NavigateTo.Feedback]:
            """Navigate to a given waypoint asynchronously.

            Args:
                waypoint_id: id of the waypoint to navigate to

            Returns:
                a future result to the navigate to request
            """

    @property
    def localization(self) -> Subscription[PoseStamped]:
        """Get latest localization pose estimates subscription.

        Returns:
            a subscription to GraphNav body pose estimates.
        """
        ...

    reset_localization: GeneralizedMethodLike[ResetLocalizationProtocol]
    set_localization: GeneralizedMethodLike[ResetLocalizationProtocol]

    get_localization: GeneralizedMethodLike[GetLocalizationPoseProtocol]
    get_localization_pose: GeneralizedMethodLike[GetLocalizationPoseProtocol]

    upload_graph: GeneralizedMethodLike[UploadLocalizationGraphProtocol]
    clear_graph: GeneralizedMethodLike[ClearLocalizationGraphProtocol]
    list_graph: GeneralizedMethodLike[ListLocalizationGraphProtocol]

    navigate_to_waypoint: GeneralizedMethodLike[NavigateToWaypointProtocol]


class SpotGraphNavMixin(SpotBaseProtocol):
    """Mixin offering access to the Spot GraphNav service capabilities."""

    LocalizationMethod: TypeAlias = LocalizationMethod

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize GraphNav clients and bind generalized methods.

        Creates concrete ROS 2 service and action clients for localization, graph management, and
        waypoint navigation, then rebinds generalized methods to those clients.

        Args:
            args: positional arguments forwarded to the next class in the MRO.
            kwargs: keyword arguments forwarded to the next class in the MRO.
        """
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
        SpotGraphNavMixin.navigate_to_waypoint.rebind(
            self,
            SpotActionable(NavigateTo, spot_namespaced(self.robot_name, "navigate_to_waypoint"), node=self.node),
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

    set_localization = reset_localization

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.GetLocalizationPoseProtocol])
    def get_localization_pose(self, **kwargs: Any) -> GraphNavGetLocalizationPose.Response:
        """Query GraphNav localization estimate.

        See `SpotGraphNavProtocol.GetLocalizationPoseProtocol` for further reference.
        """

    get_localization = get_localization_pose

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.UploadLocalizationGraphProtocol])
    def upload_graph(self, upload_filepath: str, **kwargs: Any) -> GraphNavUploadGraph.Response:
        """Upload a new GraphNav graph (ie. waypoints and topology).

        Args:
            upload_filepath: path to the serialized `bosdyn.api.graph_nav.map_pb2.Graph` instance to be uploaded.

        See `SpotGraphNavProtocol.UploadGraphProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.ClearLocalizationGraphProtocol])
    def clear_graph(self, **kwargs: Any) -> GraphNavClearGraph.Response:
        """Clear the current GraphNav graph.

        See `SpotGraphNavProtocol.ClearGraphProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.ListLocalizationGraphProtocol])
    def list_graph(self, upload_filepath: str, **kwargs: Any) -> ListGraph.Response:
        """List a GraphNav graph structure.

        See `SpotGraphNavProtocol.ListGraphProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotGraphNavProtocol.NavigateToWaypointProtocol])
    def navigate_to_waypoint(self, waypoint_id: str, **kwargs: Any) -> NavigateTo.Result:
        """Navigate to the given waypoint

        See `SpotGraphNavProtocol.NavigateToWaypointProtocol` for further reference.
        """
