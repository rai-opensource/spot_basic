# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Provide message construction and ROS 2 service/action wrapper utilities."""

from typing import Any, Callable, Dict, Optional, Protocol, Type, TypeVar, Union

from bosdyn_msgs.conversions import convert
from synchros2.action import Actionable, ActionableProtocol, ActionFuture
from synchros2.futures import FutureLike
from synchros2.service import Serviced, ServicedProtocol
from synchros2.utilities import namespace_with


class MessageProtocol(Protocol):
    """Protocol for ROS 2 message classes exposing field metadata."""

    @classmethod
    def get_fields_and_field_types(self) -> Dict[str, str]:
        """Get ROS 2 message field names and type strings.

        Returns:
            mapping from field names to ROS 2 field type strings.
        """
        ...


MessageT = TypeVar("MessageT", bound=MessageProtocol)


def make_message(cls: Type[MessageT], *args: Any, **kwargs: Any) -> MessageT:
    """Makes a ROS 2 message.

    Note no copies are performed by this function.

    Args:
        cls: message class to instantiate and initialize.
        args: positional arguments for message initialization,
            matched to message fields in order of appearance.
        kwargs: keyword arguments for message initialization, matched to message fields by name.

    Returns:
        a new message instance, unless a message instance is provided as the sole argument to
        `make_message`, in which case it will be forward unmodified.
    """
    if len(args) == 1 and len(kwargs) == 0:
        if isinstance(args[0], cls):
            return args[0]

    if len(args) > 0:
        field_names = list(cls.get_fields_and_field_types().keys())
        if len(args) > len(field_names):
            raise TypeError(
                f"make_message() takes at most {len(field_names)} " f"positional arguments but {len(args)} were given"
            )
        for name, value in zip(field_names[: len(args)], args, strict=False):
            if name in kwargs:
                raise TypeError(f"make_message() got multiple " f"values for argument '{name}'")
            kwargs[name] = value

    message = cls()
    for name, value in kwargs.items():
        if not hasattr(message, name):
            raise TypeError("make_message() got unexpected " f"keyword argument '{name}'")
        default_value = getattr(message, name)
        default_value_type = type(default_value)
        if not isinstance(value, default_value_type):
            if hasattr(value, "to_msg"):
                value = value.to_msg()
            if hasattr(value, "DESCRIPTOR"):
                convert(value, default_value)
                value = default_value
            if hasattr(default_value_type, "value"):
                value = default_value_type(value=value)
        setattr(message, name, value)
    return message


ServiceRequestT = TypeVar("ServiceRequestT", contravariant=True)
ServiceResponseT = TypeVar("ServiceResponseT", covariant=True)

SpotServicedProtocol = ServicedProtocol  # alias for now


class SpotServiced(Serviced[ServiceRequestT, ServiceResponseT]):
    """Ergonomic ROS 2 service wrapper subclass.

    Supports legacy invocation signatures and piecewise request specifications (see `make_message`).
    """

    def synchronous(  # type: ignore[override]
        self, *args: Any, timeout_sec: Optional[float] = None, nothrow: bool = False, **kwargs: Any
    ) -> Optional[ServiceResponseT]:
        """Call the service synchronously with a constructed request.

        Args:
            args: positional values used to populate the service request fields.
            timeout_sec: optional timeout, in seconds. If a timeout is specified and it
                expires, the service request will be cancelled and the call will raise.
            nothrow: when set, errors do not raise exceptions.
            kwargs: keyword values used to populate the service request fields.

        Returns:
            the service response or None on timeout or failure when nothrow is set.

        Raises:
            ServiceError: if the service request fails.
            ServiceTimeout: if the service request times out.
            TypeError: if the request cannot be constructed from the provided arguments.
        """
        request = make_message(self.service_type.Request, *args, **kwargs)
        return super().synchronous(request, timeout_sec=timeout_sec, nothrow=nothrow)

    def asynchronous(self, *args: Any, **kwargs: Any) -> FutureLike[ServiceResponseT]:
        """Call the service asynchronously with a constructed request.

        Args:
            args: positional values used to populate the service request fields.
            kwargs: keyword values used to populate the service request fields.

        Returns:
            a future response to the service request.

        Raises:
            TypeError: if the request cannot be constructed from the provided arguments.
        """
        return super().asynchronous(make_message(self.service_type.Request, *args, **kwargs))


ActionGoalT = TypeVar("ActionGoalT", contravariant=True)
ActionResultT = TypeVar("ActionResultT", covariant=True)
ActionFeedbackT = TypeVar("ActionFeedbackT", covariant=True)

SpotActionableProtocol = ActionableProtocol  # alias for now


class SpotActionable(Actionable[ActionGoalT, ActionResultT, ActionFeedbackT]):
    """Ergonomic ROS 2 action wrapper subclass.

    Supports legacy invocation signatures and piecewise goal specifications (see `make_message`).
    """

    def synchronous(  # type: ignore[override]
        self,
        *args: Any,
        feedback_callback: Optional[Callable[[ActionFeedbackT], None]] = None,
        timeout_sec: Optional[float] = None,
        nothrow: bool = False,
        **kwargs: Any,
    ) -> Optional[ActionResultT]:
        """Send an action goal synchronously with a constructed goal message.

        Args:
            args: positional values used to populate the action goal fields.
            feedback_callback: optional callback invoked for action feedback messages.
            timeout_sec: optional timeout, in seconds. If a timeout is specified and
                it expires, the action goal will be cancelled and the call will raise.
            nothrow: when set, errors do not raise exceptions.
            kwargs: keyword values used to populate the action goal fields.

        Returns:
            the action result or None on timeout or failure when nothrow is set.

        Raises:
            ActionError: if the action goal fails.
            ActionTimeout: if the action goal times out.
            TypeError: if the goal cannot be constructed from the provided arguments.
        """
        return super().synchronous(
            make_message(self.action_type.Goal, *args, **kwargs),
            feedback_callback=feedback_callback,
            timeout_sec=timeout_sec,
            nothrow=nothrow,
        )

    def asynchronous(
        self, *args: Any, track_feedback: Union[int, bool] = False, **kwargs: Any
    ) -> ActionFuture[ActionResultT, ActionFeedbackT]:
        """Send an action goal asynchronously with a constructed goal message.

        Args:
            args: positional values used to populate the action goal fields.
            track_feedback: feedback tracking configuration for the returned action future.
            kwargs: keyword values used to populate the action goal fields.

        Returns:
            a future result to the action goal.

        Raises:
            TypeError: if the goal cannot be constructed from the provided arguments.
        """
        goal = make_message(self.action_type.Goal, *args, **kwargs)
        return super().asynchronous(goal, track_feedback=track_feedback)


def spot_namespaced(robot_name: Optional[str], name: str, *topic_segments: str) -> str:
    """Compute spot namespaced `name`."""
    return namespace_with("/", robot_name, name, *topic_segments)
