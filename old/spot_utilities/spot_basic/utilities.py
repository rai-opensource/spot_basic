# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Callable, Optional, Union

from bdai_ros2_wrappers.action import Actionable
from bdai_ros2_wrappers.action_handle import ActionHandle
from bdai_ros2_wrappers.service import Serviced
from bdai_ros2_wrappers.service_handle import ServiceHandle
from bdai_ros2_wrappers.type_hints import Action, Msg, SrvTypeRequest
from bosdyn_msgs.conversions import convert
from rclpy.action.client import ActionClient
from rclpy.client import Client
from rclpy.context import Context


def make_message(cls: Msg, *args: Any, **kwargs: Any) -> Any:
    """Makes a ROS 2 message.

    Note no copies are performed by this function.

    Args:
        cls: message class to instantiate and initialize.
        args: positional arguments for message initialization,
        matched to message fields in order of appearance.
        kwargs: keyword arguments for message initialization,
        matched to message fields by name.

    Returns:
         a new message instance, unless a message instance is
         provided as the sole argument to `make_message`, in
         which case it will be forward unmodified.
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
        if not isinstance(value, type(default_value)):
            if hasattr(value, "to_msg"):
                value = value.to_msg()
            if hasattr(value, "DESCRIPTOR"):
                convert(value, default_value)
                value = default_value
        setattr(message, name, value)
    return message


class SpotServiced(Serviced):
    """Ergonomic ROS 2 service wrapper subclass.

    Supports legacy invocation signatures and piecewise request specifications (see `make_message`).
    """

    def synchronous(self, *args: Any, timeout_sec: Optional[float] = None, **kwargs: Any) -> Any:
        request = make_message(self.service_type.Request, *args, **kwargs)
        return super().synchronous(request, timeout_sec=timeout_sec)

    def asynchronous(self, *args: Any, **kwargs: Any) -> Any:
        return super().asynchronous(make_message(self.service_type.Request, *args, **kwargs))


class SpotActionable(Actionable):
    """Ergonomic ROS 2 action wrapper subclass.

    Supports legacy invocation signatures and piecewise goal specifications (see `make_message`).
    """

    def synchronous(
        self,
        *args: Any,
        timeout_sec: Optional[float] = None,
        feedback_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        return super().synchronous(
            make_message(self.action_type.Goal, *args, **kwargs),
            feedback_callback=feedback_callback,
            timeout_sec=timeout_sec,
        )

    def asynchronous(self, *args: Any, track_feedback: Union[int, bool] = False, **kwargs: Any) -> Any:
        goal = make_message(self.action_type.Goal, *args, **kwargs)
        return super().asynchronous(goal, track_feedback=track_feedback)


class SpotServiceHandle(ServiceHandle):
    """Extends legacy ServiceHandle for spot specific services"""

    @classmethod
    def call(
        cls,
        robot_name: str,
        service_name: str,
        client: Client,
        request: Optional[SrvTypeRequest] = None,
        result_callback: Optional[Callable] = None,
        on_failure_callback: Optional[Callable] = None,
        context: Optional[Context] = None,
    ) -> "SpotServiceHandle":
        """Call service asynchronously.

        Args:
            robot_name: name of the robot.
            service_name: name of the service.
            client: service client instance.
            request: optional service request, will fallback to
            a default initialized request if none is provided.
            result_callback: optional service result callback.
            on_failure_callback: optional service failure callback.
            context: optional context, will default to the global
            context if none is provided.

        Returns:
            a service handle.
        """
        handle = cls(robot_name, service_name, context=context)
        if result_callback is not None:
            handle.set_result_callback(result_callback)
        if on_failure_callback is not None:
            handle.set_on_failure_callback(on_failure_callback)
        if request is None:
            request = client.srv_type.Request()
        handle.set_send_service_future(client.call_async(request))
        return handle

    def __init__(self, robot_name: str, service_name: str, *args: Any, **kwargs: Any) -> None:
        """ """
        super().__init__(f"{robot_name}.{service_name}", *args, **kwargs)


class SpotActionHandle(ActionHandle):
    """Extends legacy ActionHandle for spot specific actions"""

    @classmethod
    def call(
        cls,
        robot_name: str,
        action_name: str,
        action_client: ActionClient,
        goal: Action.Goal,
        result_callback: Optional[Callable[[Action.Result], None]] = None,
        feedback_callback: Optional[Callable[[Action.Feedback], None]] = None,
        on_failure_callback: Optional[Callable[[], None]] = None,
        context: Optional[Context] = None,
    ) -> "SpotActionHandle":
        """Call action asynchronously.

        Args:
            robot_name: name of the robot.
            action_name: name of the action.
            action_client: action client instance.
            goal: action goal.
            result_callback: optional action result callback.
            feedback_callback: optional action feedback callback.
            on_failure_callback: optional action failure callback.
            context: optional context, will default to the global
            context if none is provided.

        Returns:
            an action handle.
        """
        handle = cls(robot_name, action_name, context=context)
        if result_callback is not None:
            handle.set_result_callback(result_callback)
        if on_failure_callback is not None:
            handle.set_on_failure_callback(on_failure_callback)
        if feedback_callback is not None:
            handle.set_feedback_callback(feedback_callback)
            feedback_callback = handle.get_feedback_callback
        send_goal_future = action_client.send_goal_async(goal, feedback_callback=feedback_callback)
        handle.set_send_goal_future(send_goal_future)
        return handle

    def __init__(self, robot_name: str, action_name: str, *args: Any, **kwargs: Any):
        """Initializes `SpotActionHandle`.

        Args:
            robot_name: name of the robot.
            action_name: name of the action.
            args: positional arguments for base class.
            kwargs: keyword arguments for base class.
        """
        super().__init__(f"{robot_name}.{action_name}", *args, **kwargs)
        self._result_callback = self._result_callback_intermediate
        self._result_callback_internal: Optional[Callable[[Action.Result], None]] = None
        self._result_internal: Optional[Action.Result] = None
        self._feedback_callback = self._feedback_callback_intermediate
        self._feedback_callback_internal: Optional[Callable[[Action.Result], None]] = None

    @property
    def result(self) -> Optional[Action.Result]:
        return self._result_internal

    def set_result_callback(self, result_callback: Callable[[Action.Result], None]) -> None:
        """Sets the callback for processing the result from executing an Action"""
        self._result_callback_internal = result_callback

    def _result_callback_intermediate(self, result: Action.Result) -> None:
        """Callback for pulling the result out of the result and passing it to a user provided callback"""
        if not result.success:
            self._logger.error(f"Action resulted in failure. Msg: {result.message}")
            self._failure()
            return

        self._result_internal = result.result
        if self._result_callback_internal is not None:
            self._result_callback_internal(self._result_internal)

    def set_feedback_callback(self, feedback_callback: Callable[[Action.Feedback], None]) -> None:
        """Sets the callback for processing the feedback during execution of an Action"""
        self._feedback_callback_internal = feedback_callback

    def _feedback_callback_intermediate(self, feedback: Action.Feedback) -> None:
        """Intermediate step to pull the feedback out of the feedback"""
        if self._feedback_callback_internal is not None:
            self._feedback_callback_internal(feedback.feedback)
