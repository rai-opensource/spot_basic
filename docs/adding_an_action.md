<!-- Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Adding a new action to `SpotBasic`

[← Back to the README](../README.md)

In this tutorial we will add an action to `SpotBasic` to show how the underlying concepts -- and especially [generalized callables](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/callables.py) -- work.

The complete code for this tutorial is shown inline below.

## Create an action definition and server

Normally the server will have been written already (usually as a part of spot ros2 and eventually communicating via a protobuf to the Spot robot).  However, for example purposes we’re going to create a custom server that just prints numbers every second for N seconds.  We will arbitrarily and for no reason other than pedagogical assume that this action can only be called when the Spot robot has an arm.  This tutorial will not focus on how this action works - for an introduction to actions in ROS2 see [here](https://docs.ros.org/en/humble/Tutorials/Intermediate/Creating-an-Action.html).

1. Create the action definition:
    1. Add the following to a new file `spot_msgs/action/Count.action` (in the `spot_msgs` interfaces package, which is part of spot_ros2):

        ```bash
        # Goal
        int32 n_seconds
        ---
        # Result
        ---
        # Feedback

        ```

    2. Add the new action to `spot_msgs/CMakeLists.txt`:

        ```makefile
        ...
        set(msg_files
          "action/Count.action"  # ADD THIS HERE
          "action/SpotGraspingPipeline.action"
          ...
        ```

2. Create an action server that prints out the second every second.  Add the following to a new file `scripts/count_server.py`:

    ```python
    #!/usr/bin/env python
    # Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

    import argparse
    import time

    import synchros2.process as ros_process
    from rclpy.action.server import ServerGoalHandle
    from spot_basic.utilities import spot_namespaced
    from spot_msgs.action import Count
    from synchros2.single_goal_action_server import SingleGoalActionServer

    def _count_callback(goal_handle: ServerGoalHandle) -> Count.Result:
        start = time.time()
        while time.time() - start < goal_handle.request.n_seconds:
            print(f"It's been {time.time() - start} seconds")
            time.sleep(1.0)
        goal_handle.succeed()
        return Count.Result()

    def cli() -> argparse.ArgumentParser:
        """Command-line interface."""
        parser = argparse.ArgumentParser(description="Action server for count action")
        parser.add_argument("robot_name", help="Name of robot to use")
        return parser

    @ros_process.main(cli())
    def main(args: argparse.Namespace) -> None:
        server = SingleGoalActionServer(  # noqa: F841
            main.node, Count, spot_namespaced(args.robot_name, "count"), _count_callback
        )
        print("Ready for counting goals!")
        ros_process.wait_for_shutdown()

    if __name__ == "__main__":
        main()
    ```

3. Mark it executable:

    ```bash
    chmod +x scripts/count_server.py
    ```

4. Add this to the list of runnable scripts in `CMakeLists.txt`:

    ```makefile
    ...
    install(
      PROGRAMS
      scripts/count_server.py  # ADD HERE
      scripts/estop_gui.sh
      ...
    ```


## SpotBasic code and running

Let’s first write and run the code that calls this action server from Spot Basic and then we’ll analyze it.

### Code

Copy the following into `spot_basic/count.py`:

```python
# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Union

from spot_basic.base import SpotBaseProtocol
from spot_basic.utilities import (
    SpotActionable,
    SpotActionableProtocol,
)
from spot_basic.utilities import spot_namespaced
from spot_msgs.action import Count as CountAction
from synchros2.action import ActionFuture
from synchros2.callables import GeneralizedGuard, GeneralizedMethodLike, generalized_method

def _build_count_command(n_seconds: int) -> CountAction.Goal:
    return CountAction.Goal(n_seconds=n_seconds)

class SpotCountProtocol(SpotBaseProtocol, Protocol):
    """Counting protocol."""

    class PerformProtocol(
        SpotActionableProtocol[CountAction.Goal, CountAction.Result, CountAction.Feedback],
        Protocol,
    ):
        """Counting request execution protocol."""

        def synchronous(
            self,
            n_seconds: int,
            *,
            feedback_callback: Optional[Callable[[CountAction.Feedback], None]] = None,
            timeout_sec: Optional[float] = None,
            nothrow: bool = False,
        ) -> CountAction.Result:
            """Counting request synchronously.

            Args:
                n_seconds: The number of seconds the action should run for
                feedback_callback: Optional action feedback callback (note that there is not as of this writing
                    feedback sent out for this action as there is no feedback on the spot side).
                timeout_sec: Optional action timeout, in seconds. If a timeout is specified and it expires, the
                    request will be cancelled and the call will raise. Note this timeout is local to the caller. It may
                    take some time for the action to be cancelled, if cancellation is not rejected by the action server.
                nothrow: True to return the result instead of throwing an exception if the action does not complete
                    successfully.

            Returns:
                The counting result.

            Raises:
                ActionTimeout: if the count request timed out.
                ActionRejected: if the count request was not accepted.
                ActionCancelled: if the count request was cancelled.
                ActionAborted: if the count request was aborted.
                RuntimeError: if the robot has no arm.
            """

        def asynchronous(
            self,
            n_seconds: int,
            *,
            track_feedback: Union[int, bool] = False,
        ) -> ActionFuture[CountAction.Result, CountAction.Feedback]:
            """Perform count request asynchronously.

            See "synchronous" for an explanation of the count arguments.

            Args:
                track_feedback: whether and how to track action feedback. Other than a boolean to enable or disable
                    tracking, a positive integer may be provided to cap feedback buffer size.

            Returns:
                The future counting outcome.

            Raises:
                RuntimeError: if the robot has no arm.
            """

    count: GeneralizedMethodLike[PerformProtocol]

class SpotCountMixin(SpotBaseProtocol):
    """Mixin offering Spot counting capabilities.

    Note that the robot must have an arm for any of these APIs to be available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # This creates the action client, which already supports synchronous and asynchronous calls and then tells it
        # to first call the build count command.
        implementation = SpotActionable(CountAction, spot_namespaced(self.robot_name, "count"), node=self.node).compose(
            _build_count_command
        )

        # This sets it up so the count function here to first check whether the robot has an arm and if so
        # to pass the arguments on to the implementation
        SpotCountMixin.count.rebind(
            self,
            GeneralizedGuard(self.has_arm, implementation, f"{self.robot_name} has no arm, cannot count"),
        )

    # This now takes in the same arguments as the protocol and you can call synchronous or asynchronous on it.
    @generalized_method(spec=GeneralizedMethodLike[SpotCountProtocol.PerformProtocol])
    def count(self, *args: Any, **kwargs: Any) -> Any:
        """Perform counting with the robot.

        See `SpotCountProtocol.PerformProtocol` for further reference.
        """
```

We also need to add this to the Spot Basic in `spot_basic/__init__.py` (note that TWO additions are needed):

```python
...
from spot_basic.command.mobility import SpotMobilityCommandMixin
from spot_basic.count import SpotCountMixin  # ADD THIS HERE
from spot_basic.docking import SpotDockingMixin
...

class SpotBasic(
    SpotBase,
    ...
    SpotCountMixin,  #  ADD THIS HERE
    SpotImageMixin,
):
```

Now create a script that calls these functions in `examples/example_count.py` :

```python
# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import time

import synchros2.process as ros_process
from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Count for a number of seconds using SpotBasic")
    parser.add_argument("robot_name", help="Name of robot to use")
    parser.add_argument(
        "-n",
        "--n-seconds",
        type=int,
        default=10,
        help="Number of seconds for which to run the action",
    )
    return parser

@ros_process.main(cli(), interruptible=True)
def main(args: argparse.Namespace) -> None:
    spot = SpotBasic(args.robot_name)

    print(f"Calling action synchronously (this will take {args.n_seconds} seconds)")
    spot.count.synchronously(args.n_seconds)
    print("Calling action asynchronously")
    future = spot.count.asynchronously(args.n_seconds)
    while not future.finalized:
        print("Waiting for action")
        time.sleep(1)
    print("Action returned")

if __name__ == "__main__":
    main()

```

### Running

Everything should be done from within your ROS 2 workspace (with the workspace sourced):

1. First build the necessary packages (note: this may take a while):

    ```bash
    colcon build --packages-select spot_msgs spot_basic
    ```

2. Start the Spot ROS 2 driver, since it needs to run for `SpotBasic` to be created (see the [`spot_ros2`](https://github.com/rai-opensource/spot_ros2) documentation for driver launch instructions):

    ```bash
    ros2 launch spot_driver spot_driver.launch.py
    ```

3. Now start the count action server:

    ```bash
    ros2 run spot_basic count_server.py <Robot name>
    ```

4. Finally run the example script:

    ```bash
    cd examples
    python example_count.py <Robot name>
    ```


You should see the prints from the script and also prints in the count server terminal.

## Analyzing the code

The Spot Basic code is a little complicated because we want to easily be able to create these synchronous and asynchronous calls without having to rewrite that logic every time.  You could imagine the following class:

```python
class SpotCountMixin:
    def __init__(self):
        self._action = SpotActionable(
            CountAction, spot_namespaced(self.robot_name, "count"), node=self.node
        )

     def count_synchronous(
        self,
        n_seconds: int,
        *,
        feedback_callback: Optional[Callable[[CountAction.Feedback], None]] = None,
        timeout_sec: Optional[float] = None,
        nothrow: bool = False,
    ) -> CountAction.Result:
        goal = CountAction.Goal(n_seconds=n_seconds)
        return self._action.synchronous(goal, feedback_callback, timeout_sec, nothrow)


     def count_asynchronous(
        n_seconds: int,
        *,
        track_feedback: Union[int, bool] = False,
     ) -> ActionFuture[CountAction.Result, CountAction.Feedback]:
        goal = CountAction.Goal(n_seconds=n_seconds)
        return self._action.asynchronous(goal, feedback_callback, timeout_sec, nothrow)
```

and in fact this is how Spot Basic was written for a long time.  But it led to a lot of different versions of calling asynchronous and synchronous protocols and way too much living in one file and general chaos and confusion.  So the [`GeneralizedCallable`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/callables.py#L61) class of objects were born.

All `GeneralizedCallable` objects have `synchronous` and `asynchronous` functions.  This allows us to write a bunch of stuff without having to write two functions for synchronous and asynchronous.  Specifically, we make `count` a [`GeneralizedMethod`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/callables.py#L250):

```python
    # This now takes in the same arguments as the protocol and you can call synchronous or asynchronous on it.
    @generalized_method(spec=GeneralizedMethodLike[SpotCountProtocol.PerformProtocol])
    def count(self, *args: Any, **kwargs: Any) -> Any:
        """Perform counting with the robot.

        See `SpotCountProtocol.PerformProtocol` for further reference.
        """
```

A generalized method is just a class that takes another class with `synchronous` and `asynchronous` defined and forwards calls to those functions.  We have this in the form of a [`SpotActionable`](../spot_basic/utilities.py):

```python
        # This creates the action client, which already supports synchronous and asynchronous calls and then tells it
        # to first call the build count command.
        implementation = SpotActionable(CountAction, spot_namespaced(self.robot_name, "count"), node=self.node)
```

A [`SpotActionable`](../spot_basic/utilities.py) is a wrapper around synchros2’s [`Actionable`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/action.py#L451) class, which handles synchronous and asynchronous calls to ROS actions in a way that won’t lead to deadlocks.  The Spot wrapper allows us to pass in either an action goal or the corresponding protobuf (not necessary for this example but usually necessary when dealing with Spot).

However, `SpotActionable` requires that we pass it an action goal (or the equivalent protobuf) but we don’t want people to have to pass in the full action goal - the point of Spot Basic is to allow users to use nice function calls instead.  That’s what the `compose` call does - it tells the `SpotActionable` (which inherits from [`Actionable`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/action.py#L451) which inherits from [`ComposedCallable`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/callables.py#L107) which allows composition) to first call `_build_count`, which returns the action goal, and then call the `SpotActionable` functions:

```python
        # This creates the action client, which already supports synchronous and asynchronous calls and then tells it
        # to first call the build count command.
        implementation = SpotActionable(CountAction, spot_namespaced(self.robot_name, "count"), node=self.node).compose(
            _build_count_command
        )
```

Now `implementation` is a class with `synchronous` and `asynchronous` calls that does what we want.  So we bind it to our `GeneralizedMethod` `count` :

```python
        SpotCountMixin.count.rebind(self, implementation)
```

Except we don’t quite do this.  Instead purely for instructive purposes we’ve decided you can only call the count action if the robot has an arm and as such have added in a [`GeneralizedGuard`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/callables.py#L184):

```python
        # This sets it up so the count function here to first check whether the robot has an arm and if so
        # to pass the arguments on to the implementation
        SpotCountMixin.count.rebind(
            self,
            GeneralizedGuard(self.has_arm, implementation, f"{self.robot_name} has no arm, cannot count"),
        )

```

A [`GeneralizedGuard`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/callables.py#L184) takes in a boolean function (`self.has_arm`), a class with `synchronous` and `asynchronous` functions (`implementation`) and an error message.  `GeneralizedGuard` itself then has `asynchronous` and `synchronous` methods that check the boolean function (`self.has_arm`) and throw a `RuntimeError` with the corresponding error message if it’s false.  Otherwise they call the corresponding function of the underlying class (`implementation`).  So this makes it so that if we call `count` with a robot without an arm, there will be a runtime error.

And finally we need to make our [mixin](https://www.residentmar.io/2019/07/07/python-mixins.html) available inside the SpotBasic class.  This is done by having `SpotBasic` inherit from the mixin:

```python
class SpotBasic(
    SpotBase,
    ...
    SpotCountMixin,  #  ADD THIS HERE
    SpotImageMixin,
):
```

Let’s put all this together by tracing what happens when we call `spot.count.synchronous(n_seconds)`:

1. `spot.count` is bound to `GeneralizedGuard(self.has_arm, implementation, f"{self.robot_name} has no arm, cannot count")` .  So `spot.count.synchronous` calls `GeneralizedGuard.synchronous`
2. `GeneralizedGuard.synchronous`  calls `self.has_arm`  because that’s the boolean guard function we passed.  If the robot doesn’t have an arm, this raises a runtime error.  Otherwise it calls `implementation.synchronous`
3. `implementation` is a `SpotActionable` composed with `_build_count_command` .  So, because of the composition, it first calls `_build_count_command(n_seconds)` and then sends the output of that (a `CountAction.Goal`) to `SpotActionable.synchronous`
4. `SpotActionable.synchronous` calls the underlying `Actionable.synchronous` which sends the goal to the action and waits for it to return.

You may have noticed that at no point in the above analysis did we ever reference the big long `SpotCountProtocol` class shown in the `count.py` snippet above. That class isn’t functional! It exists purely for type hinting and documentation -- it makes the `n_seconds` argument and the `synchronous`/`asynchronous` signatures explicit to both static analyzers and human readers.

The other thing you might have noticed is that we call `asynchronously` and `synchronously` instead of `asynchronous` and `synchronous`.  These are aliased at the very top of [`GeneralizedCallable`](https://github.com/rai-opensource/synchros2/blob/bbe365dfafef15e9a6379246f801aca4d90a932b/synchros2/synchros2/callables.py#L61) so that `asynchronously` calls `asynchronous` and `synchronously` calls `synchronous`.  We tend to prefer the adverb for actions because it’s grammatically correct.
