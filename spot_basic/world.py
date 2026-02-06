# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any, Iterable, Optional, Protocol, Union, overload

import bosdyn_api_msgs.msg
from bosdyn.api import world_object_pb2
from spatialmath import SE3
from spot_basic.base import SpotBaseProtocol
from spot_basic.state import FrameHint
from spot_basic.utilities import SpotServiced, SpotServicedProtocol, spot_namespaced
from spot_msgs.srv import ListWorldObjects, MutateWorldObject
from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike

from bdai.deprecation import deprecated
from bdai_ros.utilities.manual_conversions import se3_to_se3pose_proto

# There isn't an "infinite lifetime" option so we just go for a year and also getting that song stuck in your head.
INFINITE_LIFETIME_S = 525600 * 60


class WorldObjectCommandBuilder:
    """Creates world object requests for various common list and mutation requests."""

    def __init__(self, lister: SpotServiced):
        """Constructor

        Args:
            lister: A serviced for listing world objects.  Used for deleting specific types of objects.
        """
        self._lister = lister

    def list_user_nogo_regions_request(self) -> world_object_pb2.ListWorldObjectRequest:
        """Return a protobuf that will cause spot to list only user nogo regions."""
        request = world_object_pb2.ListWorldObjectRequest()
        request.object_type.append(world_object_pb2.WorldObjectType.WORLD_OBJECT_USER_NOGO)
        return request

    def _add_user_nogo_basic_request(
        self,
        name: str,
        is_foot_obstacle: bool,
        is_body_obstacle: bool,
        use_foot_obstacle_inflation: bool,
        object_lifetime_s: int,
    ) -> world_object_pb2.MutateWorldObjectRequest:
        request = world_object_pb2.MutateWorldObjectRequest()
        request.mutation.action = world_object_pb2.MutateWorldObjectRequest.Action.ACTION_ADD
        request.mutation.object.name = name
        request.mutation.object.object_lifetime.seconds = object_lifetime_s
        request.mutation.object.nogo_region_properties.disable_foot_obstacle_generation = not is_foot_obstacle
        request.mutation.object.nogo_region_properties.disable_body_obstacle_generation = not is_body_obstacle
        request.mutation.object.nogo_region_properties.disable_foot_obstacle_inflation = not use_foot_obstacle_inflation
        return request

    def add_user_nogo_box_request(
        self,
        box_name: str,
        frame: FrameHint,
        frame_t_origin: SE3,
        min_bounds: tuple[float, float],
        max_bounds: tuple[float, float],
        is_foot_obstacle: bool = True,
        is_body_obstacle: bool = True,
        use_foot_obstacle_inflation: bool = True,
        object_lifetime_s: int = INFINITE_LIFETIME_S,
    ) -> world_object_pb2.MutateWorldObjectRequest:
        """Return a protobuf that will add a nogo region in the shape of a box to Spot.

        See `SpotWorldProtocol.AddUserNoGoBoxProtocol` for further reference.
        """
        request = self._add_user_nogo_basic_request(
            box_name,
            is_foot_obstacle,
            is_body_obstacle,
            use_foot_obstacle_inflation,
            object_lifetime_s,
        )
        # The API only supports box size so shift the box origin to be in the center
        origin_t_center = SE3(x=[(max_bounds[0] + min_bounds[0]) / 2.0, (max_bounds[1] + min_bounds[1]) / 2.0, 0.0])
        frame_t_center = frame_t_origin * origin_t_center
        request.mutation.object.nogo_region_properties.box.box.size.x = max_bounds[0] - min_bounds[0]
        request.mutation.object.nogo_region_properties.box.box.size.y = max_bounds[1] - min_bounds[1]
        request.mutation.object.nogo_region_properties.box.frame_name = frame.value
        request.mutation.object.nogo_region_properties.box.frame_name_tform_box.CopyFrom(
            se3_to_se3pose_proto(frame_t_center)
        )
        return request

    def add_user_nogo_circle_request(
        self,
        circle_name: str,
        frame: FrameHint,
        frame_t_center: SE3,
        radius: float,
        is_foot_obstacle: bool = True,
        is_body_obstacle: bool = True,
        use_foot_obstacle_inflation: bool = True,
        object_lifetime_s: int = INFINITE_LIFETIME_S,
    ) -> world_object_pb2.MutateWorldObjectRequest:
        """Return a request protobuf that will add a nogo region in the shape of a circle.

        See `SpotWorldProtocol.AddUserNoGoCircleProtocol` for further reference.
        """
        request = self._add_user_nogo_basic_request(
            circle_name,
            is_foot_obstacle,
            is_body_obstacle,
            use_foot_obstacle_inflation,
            object_lifetime_s,
        )
        request.mutation.object.nogo_region_properties.circle.circle.center_pt.x = 0.0
        request.mutation.object.nogo_region_properties.circle.circle.center_pt.y = 0.0
        request.mutation.object.nogo_region_properties.circle.circle.radius = radius
        request.mutation.object.nogo_region_properties.circle.frame_name = frame.value
        request.mutation.object.nogo_region_properties.circle.frame_name_tform_circle.CopyFrom(
            se3_to_se3pose_proto(frame_t_center)
        )
        return request

    def delete_user_nogo_region_request(self, nogo_id: int) -> world_object_pb2.MutateWorldObjectRequest:
        """Return a protobuf that will delete a nogo region with a specific ID.

        See `SpotWorldProtocol.DeleteUserNoGoRegionProtocol` for further reference.
        """
        request = world_object_pb2.MutateWorldObjectRequest()
        request.mutation.action = world_object_pb2.MutateWorldObjectRequest.Action.ACTION_DELETE
        request.mutation.object.id = nogo_id
        return request

    def delete_all_user_nogo_regions_request(self) -> Iterable[world_object_pb2.MutateWorldObjectRequest]:
        """Return a series of protobufs that will delete all nogo regions users have added.

        Makes a synchronous call to the list service to identify these regions.

        See `SpotWorldProtocol.DeleteAllUserNoGoRegionsProtocol` for further reference.
        """
        user_nogo_regions = self._lister.synchronously(self.list_user_nogo_regions_request())
        if user_nogo_regions is None:
            return
        for wo in user_nogo_regions.response.world_objects:
            yield self.delete_user_nogo_region_request(wo.id)

    def delete_all_user_nogo_regions_with_name_request(
        self, nogo_name: str
    ) -> Iterable[world_object_pb2.MutateWorldObjectRequest]:
        """Return a series of protobufs that will delete all nogo regions users have added with a specific name.

        Makes a synchronous call to the list service to identify these regions.

        See `SpotWorldProtocol.DeleteAllUserNoGoRegionsWithNameProtocol` for further reference.
        """
        user_nogo_regions = self._lister.synchronously(self.list_user_nogo_regions_request())
        if user_nogo_regions is None:
            return
        for wo in user_nogo_regions.response.world_objects:
            if wo.name == nogo_name:
                yield self.delete_user_nogo_region_request(wo.id)


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

    class ListUserNoGoRegionsProtocol(
        SpotServicedProtocol[ListWorldObjects.Request, ListWorldObjects.Response], Protocol
    ):
        """List the user nogo regions."""

        def synchronously(
            self,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[ListWorldObjects.Response]:
            """List user no go regions synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the object listing will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the object listing request or None on timeout or failure.

            Raises (if nothrow is false):
                ServiceError: if the object listing request fails.
                ServiceTimeout: if the object listing request times out.
            """

        def asynchronously(self) -> FutureLike[ListWorldObjects.Response]:
            """List user no go regions asynchronously.

            Returns:
                a future response to the object listing request.
            """

    list_user_nogo_regions: GeneralizedMethodLike[ListUserNoGoRegionsProtocol]

    class MutateWorldObjectProtocol(
        SpotServicedProtocol[MutateWorldObject.Request, MutateWorldObject.Response], Protocol
    ):
        """World object mutation protocol."""

        def synchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.MutateWorldObjectRequest,
                world_object_pb2.MutateWorldObjectRequest,
            ],
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[MutateWorldObject.Response]:
            """Mutate object in the world synchronously.

            Args:
                request: object mutation request, in its Protobuf or ROS 2 form.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the object mutation will be cancelled and the call will raise.
                Note this timeout is local to the caller (and it may take a while for the
                robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the object mutation request or None on timeout or failure.
            Raises (if nothrow is False):
                ServiceError: if the object mutation request fails.
                ServiceTimeout: if the object mutation request times out.
            """

        def asynchronously(
            self,
            request: Union[
                bosdyn_api_msgs.msg.MutateWorldObjectRequest,
                world_object_pb2.MutateWorldObjectRequest,
            ],
        ) -> FutureLike[MutateWorldObject.Response]:
            """Mutate object in the world asynchronously.

            Args:
                request: object mutation request, in its Protobuf or ROS 2 form.

            Returns:
                a future response to the object mutation request.
            """

    mutate_world_object: GeneralizedMethodLike[MutateWorldObjectProtocol]

    class AddUserNoGoBoxProtocol(SpotServicedProtocol[MutateWorldObject.Request, MutateWorldObject.Response], Protocol):
        """Add a user no go box region."""

        def synchronously(
            self,
            box_name: str,
            frame: FrameHint,
            frame_t_origin: SE3,
            min_bounds: tuple[float, float],
            max_bounds: tuple[float, float],
            is_foot_obstacle: bool = True,
            is_body_obstacle: bool = True,
            use_foot_obstacle_inflation: bool = True,
            object_lifetime_s: int = INFINITE_LIFETIME_S,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[MutateWorldObject.Response]:
            """Add a user nogo box region synchronously.

            Args:
                box_name: A name for the box.  This can't be an existing frame name but multiple no go regions can have
                    the same name.
                frame: A frame known to Spot in which the box origin is defined.
                frame_t_origin: Pose of the box's origin in frame
                min_bounds: Minimum [x, y] bounds in meters for the box defined in the box's frame.
                max_bounds: Maximum [x, y] bounds in meters for the box defined in the box's frame.
                is_foot_obstacle: True to have the robot avoid stepping on this area.
                is_body_obstacle: True to have the robot avoid allowing its body into this area.
                use_foot_obstacle_inflation: True to inflate foot obstacles by half the robot's foot width.
                object_lifetime_s: How long before Spot forgets the object (defaults to a year).
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                   expires, the mutation will be cancelled and the call will raise.
                   Note this timeout is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the no go addition (including a unique ID for the region) or None on timeout or failure.
            Raises (if nothrow is False):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        def asynchronously(
            self,
            box_name: str,
            frame: FrameHint,
            frame_t_origin: SE3,
            min_bounds: tuple[float, float],
            max_bounds: tuple[float, float],
            is_foot_obstacle: bool = True,
            is_body_obstacle: bool = True,
            use_foot_obstacle_inflation: bool = True,
            object_lifetime_s: int = INFINITE_LIFETIME_S,
        ) -> FutureLike[MutateWorldObject.Response]:
            """Add a user nogo box region asynchronously.

            Args:
                box_name: A name for the box.  This can't be an existing frame name but multiple no go regions can have
                    the same name.
                frame: A frame known to Spot in which the box origin is defined.
                frame_t_origin: Pose of the box's origin in frame
                min_bounds: Minimum [x, y] bounds for the box defined in the box's frame.
                max_bounds: Maximum [x, y] bounds for the box defined in the box's frame.
                is_foot_obstacle: True to have the robot avoid stepping on this area.
                is_body_obstacle: True to have the robot avoid allowing its body into this area.
                use_foot_obstacle_inflation: True to inflate foot obstacles by half the robot's foot width.
                object_lifetime_s: How long before Spot forgets the object (defaults to a year).

            Returns:
                a future response to the request.
            """

    add_user_nogo_box: GeneralizedMethodLike[AddUserNoGoBoxProtocol]

    class AddUserNoGoCircleProtocol(
        SpotServicedProtocol[MutateWorldObject.Request, MutateWorldObject.Response], Protocol
    ):
        """Add a user no go circle region."""

        def synchronously(
            self,
            circle_name: str,
            frame: FrameHint,
            frame_t_center: SE3,
            radius: float,
            is_foot_obstacle: bool = True,
            is_body_obstacle: bool = True,
            use_foot_obstacle_inflation: bool = True,
            object_lifetime_s: int = INFINITE_LIFETIME_S,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[MutateWorldObject.Response]:
            """Add a user nogo circle region synchronously.

            Args:
                circle_name: A name for the circle.  This can't be an existing frame name but multiple no go regions
                    can have the same name.
                frame: A frame known to Spot in which the circle origin is defined.
                frame_t_center: The pose of the center of the circle in frame
                radius: Radius of the circle in meters
                is_foot_obstacle: True to have the robot avoid stepping on this area.
                is_body_obstacle: True to have the robot avoid allowing its body into this area.
                use_foot_obstacle_inflation: True to inflate foot obstacles by half the robot's foot width.
                object_lifetime_s: How long before Spot forgets the object (defaults to a year).
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                   expires, the request will be cancelled and the call will raise.
                   Note this timeout is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the no go addition (including a unique ID for the region) or None on timeout or failure.
            Raises (if nothrow is False):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        def asynchronously(
            self,
            circle_name: str,
            frame: FrameHint,
            frame_t_center: SE3,
            radius: float,
            is_foot_obstacle: bool = True,
            is_body_obstacle: bool = True,
            use_foot_obstacle_inflation: bool = True,
            object_lifetime_s: int = INFINITE_LIFETIME_S,
        ) -> FutureLike[MutateWorldObject.Response]:
            """Add a user nogo circle region asynchronously.

            Args:
                circle_name: A name for the circle.  This can't be an existing frame name but multiple no go regions
                    can have the same name.
                frame: A frame known to Spot in which the circle origin is defined.
                frame_t_center: The pose of the center of the circle in frame
                radius: Radius of the circle in meters
                is_foot_obstacle: True to have the robot avoid stepping on this area.
                is_body_obstacle: True to have the robot avoid allowing its body into this area.
                use_foot_obstacle_inflation: True to inflate foot obstacles by half the robot's foot width.
                object_lifetime_s: How long before Spot forgets the object (defaults to a year).

            Returns:
                a future response to the request.
            """

    add_user_nogo_circle: GeneralizedMethodLike[AddUserNoGoCircleProtocol]

    class DeleteUserNoGoRegionProtocol(
        SpotServicedProtocol[MutateWorldObject.Request, MutateWorldObject.Response], Protocol
    ):
        """Delete a previously added user nogo region."""

        def synchronously(
            self,
            nogo_id: int,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[MutateWorldObject.Response]:
            """Delete a user nogo region synchronously.

            Args:
                nogo_id: The unique integer ID for the region to delete.  This was returned when the region was added
                    and can also be found by listing user nogo regions.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                   expires, the request will be cancelled and the call will raise.
                   Note this timeout is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the deletion
            Raises (if nothrow is False):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        def asynchronously(self, nogo_id: int) -> FutureLike[MutateWorldObject.Response]:
            """Delete a user nogo region asynchronously.

            Args:
                nogo_id: The unique integer ID for the region to delete.  This was returned when the region was added
                    and can also be found by listing user nogo regions.

            Returns:
                a future response to the request.
            """

    delete_user_nogo_region: GeneralizedMethodLike[DeleteUserNoGoRegionProtocol]

    class DeleteAllUserNoGoRegionsProtocol(
        SpotServicedProtocol[MutateWorldObject.Request, MutateWorldObject.Response], Protocol
    ):
        """Delete all user nogo regions."""

        def synchronously(
            self,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[MutateWorldObject.Response]:
            """Delete all user nogo regions synchronously.

            Args:
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                   expires, the request will be cancelled and the call will raise.
                   Note this timeout is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the deletion
            Raises (if nothrow is False):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        def asynchronously(self) -> FutureLike[MutateWorldObject.Response]:
            """Delete all user nogo regions asynchronously.

            Note that this does contain a synchronous request to list objects.

            Returns:
                a future response to the request.
            Raises (if list call fails):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

    delete_all_user_nogo_regions: GeneralizedMethodLike[DeleteAllUserNoGoRegionsProtocol]

    class DeleteAllUserNoGoRegionsWithNameProtocol(
        SpotServicedProtocol[MutateWorldObject.Request, MutateWorldObject.Response], Protocol
    ):
        """Delete all user nogo regions with a particular name."""

        def synchronously(
            self,
            nogo_name: str,
            *,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> Optional[MutateWorldObject.Response]:
            """Delete all user nogo regions with a particular name synchronously.

            Args:
                nogo_name: The name of the regions to delete.
                timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                   expires, the request will be cancelled and the call will raise.
                   Note this timeout is local to the caller (and it may take a while for the robot to react).
                nothrow: when set, errors do not raise exceptions.

            Returns:
                the response to the deletion
            Raises (if nothrow is False):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

        def asynchronously(self, nogo_name: str) -> FutureLike[MutateWorldObject.Response]:
            """Delete all user nogo regions with a particular asynchronously.

            Note that this does contain a synchronous request to list objects.

            Args:
                nogo_name: The name of the regions to delete.
            Returns:
                a future response to the request.
            Raises (if list call fails):
                ServiceError: if the request fails.
                ServiceTimeout: if the request times out.
            """

    delete_all_user_nogo_regions_with_name: GeneralizedMethodLike[DeleteAllUserNoGoRegionsWithNameProtocol]


class SpotWorldMixin(SpotBaseProtocol):
    """Mixin that manages Spot world representation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lister = SpotServiced(
            ListWorldObjects, spot_namespaced(self.robot_name, "list_world_objects"), node=self.node
        )
        self._mutater = SpotServiced(
            MutateWorldObject, spot_namespaced(self.robot_name, "mutate_world_objects"), node=self.node
        )
        command_builder = WorldObjectCommandBuilder(self._lister)

        SpotWorldMixin.list_world_objects.rebind(self, self._lister)
        SpotWorldMixin.list_user_nogo_regions.rebind(
            self, self._lister.compose(command_builder.list_user_nogo_regions_request)
        )
        SpotWorldMixin.mutate_world_object.rebind(self, self._mutater)
        SpotWorldMixin.add_user_nogo_box.rebind(self, self._mutater.compose(command_builder.add_user_nogo_box_request))
        SpotWorldMixin.add_user_nogo_circle.rebind(
            self, self._mutater.compose(command_builder.add_user_nogo_circle_request)
        )
        SpotWorldMixin.delete_user_nogo_region.rebind(
            self, self._mutater.compose(command_builder.delete_user_nogo_region_request)
        )
        SpotWorldMixin.delete_all_user_nogo_regions.rebind(
            self, self._mutater.vectorized.compose(command_builder.delete_all_user_nogo_regions_request)
        )
        SpotWorldMixin.delete_all_user_nogo_regions_with_name.rebind(
            self, self._mutater.vectorized.compose(command_builder.delete_all_user_nogo_regions_with_name_request)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.ListWorldObjectsProtocol])
    def list_world_objects(self, request: ListWorldObjects.Request, **kwargs: Any) -> ListWorldObjects.Response:
        """List objects in the world known to the robot.

        See `SpotWorldProtocol.ListWorldObjectsProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.ListUserNoGoRegionsProtocol])
    def list_user_nogo_regions(self, **kwargs: Any) -> ListWorldObjects.Response:
        """List just the user nogo regions that the robot knows about.

        See `SpotWorldProtocol.ListUserNoGoRegionsProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.MutateWorldObjectProtocol])
    def mutate_world_object(self, request: MutateWorldObject.Request, **kwargs: Any) -> MutateWorldObject.Response:
        """Add or delete a world object

        See `SpotWorldProtocol.MutateWorldObjectProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.AddUserNoGoBoxProtocol])
    def add_user_nogo_box(
        self,
        box_name: str,
        frame: FrameHint,
        frame_t_origin: SE3,
        min_bounds: tuple[float, float],
        max_bounds: tuple[float, float],
        is_foot_obstacle: bool = True,
        is_body_obstacle: bool = True,
        use_foot_obstacle_inflation: bool = True,
        object_lifetime_s: int = INFINITE_LIFETIME_S,
        **kwargs: Any,
    ) -> MutateWorldObject.Response:
        """Add a user no go box region

        See `SpotWorldProtocol.AddUserNoGoBoxProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.AddUserNoGoCircleProtocol])
    def add_user_nogo_circle(
        self,
        circle_name: str,
        frame: FrameHint,
        frame_t_center: SE3,
        radius: float,
        is_foot_obstacle: bool = True,
        is_body_obstacle: bool = True,
        use_foot_obstacle_inflation: bool = True,
        object_lifetime_s: int = INFINITE_LIFETIME_S,
        **kwargs: Any,
    ) -> MutateWorldObject.Response:
        """Add a user no go circle region

        See `SpotWorldProtocol.AddUserNoGoCircleProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.DeleteUserNoGoRegionProtocol])
    def delete_user_nogo_region(
        self,
        nogo_id: int,
        **kwargs: Any,
    ) -> MutateWorldObject.Response:
        """Delete a user nogo region by ID.

        See `SpotWorldProtocol.DeleteUserNoGoRegionProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.DeleteAllUserNoGoRegionsProtocol])
    def delete_all_user_nogo_regions(self) -> None:
        """Delete all nogo regions added by a user.

        See `SpotWorldProtocol.DeleteAllUserNoGoRegionsProtocol` for further reference.
        """

    @generalized_method(spec=GeneralizedMethodLike[SpotWorldProtocol.DeleteAllUserNoGoRegionsWithNameProtocol])
    def delete_all_user_nogo_regions_with_name(self, nogo_name: str) -> None:
        """Delete all nogo regions added by a user with a specific name.

        See `SpotWorldProtocol.DeleteAllUserNoGoRegionsWithNameProtocol` for further reference.
        """

    def get_user_nogo_regions_as_string(self) -> str:
        resp = self.list_user_nogo_regions.synchronously()
        if resp is None:
            return ""
        return "[" + ", ".join([f"{wo.name} ({wo.id})" for wo in resp.response.world_objects]) + "]"

    @list_world_objects.legacy
    @deprecated(
        reason="Legacy SpotBasic APIs are deprecated. Please use the new API instead.",
        alternative="spot_basic.SpotBasic.list_world_objects.synchronously",
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
