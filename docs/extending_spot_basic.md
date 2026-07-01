<!-- Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Extending `SpotBasic`

[← Back to the README](../README.md)

## Table of contents

- [Exposing a robot command](#exposing-a-robot-command)
- [Adding a new action](#adding-a-new-action)
- [Exposing a state stream](#exposing-a-state-stream)
- [Using the Mixed-Level API](#using-the-mixed-level-api)
- [Exposing new SDK functionality](#exposing-new-sdk-functionality)
- [Where to go next](#where-to-go-next)

Under the hood, `SpotBasic` is structured as an aggregate of loosely coupled [mixins](https://en.wikipedia.org/wiki/Mixin). These mixins heavily rely on [factory](https://en.wikipedia.org/wiki/Factory_method_pattern) and [delegation](https://en.wikipedia.org/wiki/Delegation_pattern) patterns to abstract and reuse what is specific to message passing, which is central to `SpotBasic` implementation, the Spot ROS 2 driver being nothing but a bridge to the Spot SDK over ROS 2 topics, services, and actions.

## Exposing a robot command

Any robot command can be constructed and `SpotBasic.execute`d, but some robot commands may be worth adding to `spot_basic` for reuse.

The first step to expose a new robot command is to implement a factory function for it. Command factory functions are typically implemented as methods in `spot_basic.command.SpotCommandBuilder` subclasses under the `spot_basic.command` subpackage. Your factory function may semantically fit in an existing subclass or require another -- use your judgement. In the following we will assume a new subclass is required:

```python
# spot_basic/command/custom.py

from typing import Any
from spot_basic.command import SpotCommandBuilder
from bosdyn.api import robot_command_pb2

class SpotCustomCommandBuilder(SpotCommandBuilder):

    def my_custom_task_command(self, *args: Any, **kwargs: Any) -> robot_command_pb2.RobotCommand:
        # NOTE: args and kwargs are placeholders, the factory method may have any signature.
        # NOTE: The SpotBasic instance that is to execute the command is accessible as self.target,
        # you may use it for e.g. state queries and transform lookups.
        return robot_command_pb2.RobotCommand()
```

This subclass can be used as-is:

```python
spot = SpotBasic("<spot-name>")

command_builder = SpotCustomCommandBuilder(spot)
command = command_builder.my_custom_task_command()
response = spot.execute.synchronous(command)
```

But you may want to further integrate your command with `SpotBasic`. You may want to allow `SpotBasic.do_custom_task()` calls. Typically, `SpotBasic` methods are generalized methods, and that is the case for command execution methods. These generalized command execution methods are implemented in mixin classes under the `spot_basic.command` subpackage. Your command execution method may semantically fit in an existing mixin or require another -- again, use your judgement. In the following we will assume a new mixin is required:

```python
# spot_basic/command/custom.py

from typing import Any

from bosdyn.api import robot_command_pb2
from synchros2.callables import generalized_method
from spot_basic.command import SpotCommandBuilder, SpotCommandProtocol

class SpotCustomCommandBuilder(SpotCommandBuilder):

    def some_custom_command(self, *args: Any, **kwargs: Any) -> robot_command_pb2.RobotCommand:
        # NOTE: args and kwargs are placeholders, the factory method may have any signature.
        # NOTE: The SpotBasic instance that is to execute the command is accessible as self.target,
        # you may use it for e.g. state queries and transform lookups.
        return robot_command_pb2.RobotCommand()

class SpotCustomCommandMixin(SpotCommandProtocol):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotCustomCommandBuilder(self)
        SpotCustomCommandMixin.do_something.rebind(
            self, self.execute.compose(command_builder.some_custom_command)
        )

    @generalized_method
    def do_something(self, *args: Any, **kwargs: Any) -> Any:
        """Perform some custom command."""
```

Once the `SpotCustomCommandMixin` class is appended to the `SpotBasic` base class list, `SpotBasic.do_something` will become available:

```python
from synchros2.futures import wait_for_future

response = spot.do_something.synchronously()

# or alternatively...

action = spot.do_something.asynchronously()
wait_for_future(action.finalization)
```

Exposing a manipulation request is no different, except for the fact that `SpotBasic.perform` must be used in place for `SpotBasic.execute`.

In the above, we glanced over type annotations for our `SpotBasic.do_something` generalized method. Type annotating generalized methods relies on structural subtyping support i.e. [`Protocol`](https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols)s.

```python
# spot_basic/command/custom.py

from typing import Any, Callable, Optional, Protocol, TypeAlias

from bosdyn.api import robot_command_pb2

from synchros2.callables import GeneralizedMethodLike, generalized_method
from synchros2.futures import FutureLike

from spot_basic.command import SpotCommandBuilder, SpotCommandProtocol
from spot_basic.utilities import SpotActionableProtocol
from spot_msgs.action import RobotCommand as RobotCommandAction

class SpotCustomCommandBuilder(SpotCommandBuilder):

    def some_custom_command(self, *args: Any, **kwargs: Any) -> robot_command_pb2.RobotCommand:
        # NOTE: args and kwargs are placeholders, the factory method may have any signature.
        # NOTE: The SpotBasic instance that is to execute the command is accessible as self.target,
        # you may use it for e.g. state queries and transform lookups.
        return robot_command_pb2.RobotCommand()

RobotCommandActionableProtocol: TypeAlias = SpotActionableProtocol[
    RobotCommandAction.Goal, RobotCommandAction.Result, RobotCommandAction.Feedback
]

class SpotCustomCommandProtocol(SpotCommandProtocol, Protocol):

    class DoSomethingProtocol(RobotCommandActionableProtocol, Protocol):

        def __call__(
            self,
            *args: Any,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            **kwargs: Any
        ) -> RobotCommandAction.Result: ...

        def synchronously(
            self,
            *args: Any,
            feedback_callback: Optional[Callable[[RobotCommandAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            **kwargs: Any
        ) -> RobotCommandAction.Result: ...

        def asynchronously(
            self, *args: Any, **kwargs: Any
        ) -> FutureLike[RobotCommandAction.Result]: ...

    do_something: GeneralizedMethodLike[DoSomethingProtocol]

class SpotCustomCommandMixin(SpotCustomCommandProtocol):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        command_builder = SpotCustomCommandBuilder(self)
        SpotCustomCommandMixin.do_something.rebind(
            self, self.execute.compose(command_builder.some_custom_command)
        )

    @generalized_method(spec=GeneralizedMethodLike[SpotCustomCommandProtocol.DoSomethingProtocol])
    def do_something(self, *args: Any, **kwargs: Any) -> Any:
        """Perform some custom command."""
```

Note how the `SpotCustomCommandProtocol.DoSomethingProtocol` stub class makes the method signature explicit for both synchronous and asynchronous invocations. It also shows that the underlying interface is an action, making it clear to a static analyzer that action specific APIs are also available e.g. the [`spot.do_something.wait_for_server()`](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/action.py#L354) method to wait for the remote action server. This added (and entirely optional) complexity is rewarded by improved static analysis reports.

## Adding a new action

The above discusses how to expose a new capability of an existing action. If we actually need to add a new action (corresponding usually to a new Spot gRPC service), a tutorial for that can be found in [Adding an action to `SpotBasic`](adding_an_action.md).

## Exposing a state stream

States streams naturally map to topics in ROS 2. Therefore, a topic subscription is all we need to expose one. Topic subscriptions are typically implemented as `SpotBasic` properties in mixin classes under the `spot_basic` subpackage. Your property may semantically fit in an existing mixin or require another -- use your judgement. In the following we will assume a new mixin is required:

```python
from functools import cached_property

from synchros2.subscription import Subscription
from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import spot_namespaced

import std_msgs.msg

class SpotCustomStateMixin(SpotBaseProtocol):

    @cached_property
    def some_robot_state(self) -> Subscription[std_msgs.msg.Float]:
        return Subscription(std_msgs.msg.Float, spot_namespaced(self.robot_name, "some_state"), node=self.node)
```

Note that:

- our property is a [cached property](https://docs.python.org/3/library/functools.html#functools.cached_property), affording lazy initialization;
- the topic name is namespaced by the robot name, as per Spot ROS 2 conventions.

Once the `SpotCustomStateMixin` class is appended to the `SpotBasic` base class list, `SpotBasic.some_robot_state` will become available:

```python
value = unwrap_future(spot.some_robot_state.update)
print(value.data)
```

## Using the Mixed-Level API

A joint-level API exists within [`spot_ros2`](https://github.com/rai-opensource/spot_ros2/tree/main) using `ros2_control`. A combined API that utilizes both this lower level joint control and the traditional high-level API was created to leverage both of these APIs.

For a full example script, see [`example_controlled_squat.py`](../examples/example_controlled_squat.py). To run this example, first start the driver in mixed-level (controllable) mode by setting the `controllable` launch argument. See the [`spot_ros2_control` Mixed Level API documentation](https://github.com/rai-opensource/spot_ros2/blob/main/spot_ros2_control/README.md#mixed-level-api) for details.

```bash
ros2 launch spot_driver spot_driver.launch.py controllable:=True config_file:=<path/to/spot_ros.yaml> spot_name:=<spot-name>
```

To use this mixed level API, you can use `SpotBasic` as usual with an additional `spot_basic.control` import.

```python
# spot_basic/examples/example_controlled_squat.py
from spot_basic.control import SpotForwardPositionControllerInterface

class ControlledSquatExample:

    SQUAT_JOINT_POSITIONS = { . . . }

    def start(self) -> int:
        with self._robot.leash():
            world_t_start = self._robot.lookup_a_tform_b(
                self._robot.vision_frame_name, self._robot.body_frame_name
            ).yaw_SE2()

            # High level
            robot_t_goal = SE2(0.5, 0.0, 0.0)
            world_t_goal = world_t_start * robot_t_goal
            self._logger.info("Attempting to walk forward")
            self._robot.walk.synchronously(world_t_goal)
            self._logger.info("Robot walked forward")

            clock = self._robot.node.get_clock()
            tf_prefix = self._robot.robot_name + "/"

            # Joint level
            with SpotForwardPositionControllerInterface(self._robot) as ifc:
                # Get initial joint positions
                joint_states = unwrap_future(ifc.joint_states_feedback.latest_update, timeout_sec=5.0)
                initial_joint_positions = dict(
                    zip(
                        map(lambda name: name.removeprefix(tf_prefix), joint_states.name),
                        joint_states.position,
                        strict=True,
                    )
                )

                start_time = clock.now()
                last_log_time: Optional[Time] = None
                half_duration_nsec = self._duration.nanoseconds / 2.0
                self._logger.info("Attempting controlled squat")
                with contextlib.closing(ifc.joint_states_feedback.stream()) as joint_states_stream:
                    for joint_states in joint_states_stream:
                        now = clock.now()

                        elapsed_time = now - start_time
                        if elapsed_time >= self._duration:
                            break

                        # Interpolate joint position setpoints along the joint position trajectory
                        alpha = float(elapsed_time.nanoseconds) / half_duration_nsec
                        if elapsed_time.nanoseconds > half_duration_nsec:
                            alpha = 2 - alpha
                        alpha = max(0, min(1, alpha))

                        # Update joint setpoints
                        joint_setpoints = {
                            name: (
                                alpha * self.SQUAT_JOINT_POSITIONS[name] + (1 - alpha) * initial_joint_positions[name]
                            )
                            for name in initial_joint_positions
                        }

                        if last_log_time is None or now - last_log_time > self._throttle_duration:

                            joint_positions = {
                                name: position
                                for name, position in zip(
                                    map(lambda name: name.removeprefix(tf_prefix), joint_states.name),
                                    joint_states.position,
                                    strict=True,
                                )
                            }
                            self._logger.info(f"Currently at {joint_positions}")
                            self._logger.info(f"Commanding {joint_setpoints}")
                            last_log_time = now

                        ifc.update_joint_position_command(**joint_setpoints)

                self._logger.info("Done squatting")

            # High level
            self._logger.info("Attempting to walk back")
            self._robot.walk.synchronously(world_t_start)
            self._logger.info("Robot walked back")

        return 0
```

Note the use of `with SpotForwardPositionControllerInterface(self._robot) as ifc:` to transition into the joint level control and expose the state and command streamers while the high level driver is still accessible through the usual interfaces like `walk`. This example specifically makes use of a provided forward position controller for Spot. A forward state controller is also provided in `SpotForwardStateControllerInterface` inside `spot_basic/control/forward_state.py`. New controllers added to the joint-level API will also need a similar interface to access.

## Exposing new SDK functionality

If the Spot ROS 2 driver does not already expose some Spot SDK functionality that is of interest to us, things get slightly more involved. The Spot SDK sits deep down the [`spot_driver`](https://github.com/rai-opensource/spot_ros2/tree/main/spot_driver), potentially within the [`spot_wrapper`](https://github.com/rai-opensource/spot_wrapper/tree/main). A suitable ROS 2 interface must be chosen to expose it. [`spot_msgs`](https://github.com/rai-opensource/spot_ros2/tree/main/spot_msgs) hosts service and action definitions, [`bosdyn_msgs`](https://github.com/rai-opensource/bosdyn_msgs) hosts equivalent ROS 2 message for every Spot SDK Protobuf message. Only then a `SpotBasic` mixin class can be put together.

## Where to go next

- Take a look at existing mixins in the [`spot_basic`](../spot_basic) subpackage.
- Explore available functionality in the [`synchros2`](https://github.com/rai-opensource/synchros2/tree/main/synchros2) package.
