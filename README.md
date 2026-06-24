<!-- Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# `spot_basic`

## Overview

This package provides libraries and tools that make it easier to work with Boston Dynamics' Spot robots from ROS 2.

### Features

#### `SpotBasic`

`spot_basic.SpotBasic` is to the Spot ROS 2 driver what the Spot Python SDK is to Spot gRPC servers: a simpler interface to Spot capabilities. Under the hood, `SpotBasic` takes all ROS 2 topics, services, and actions that the [`spot_ros2` driver](https://github.com/rai-opensource/spot_ros2/blob/main/spot_driver/launch/spot_driver.launch.py) exposes and repackages them as a set of consistent, intuitive APIs, suitable for both synchronous and asynchronous programming.

> [!IMPORTANT]
> Integration with [process-wide `synchros2`](https://github.com/rai-opensource/synchros2/wiki/Process%E2%80%90wide-APIs) is tight and it is usually easier (though not strictly required) to use them when working with `spot_basic`.


#### Semi-automated HIL tests

A complete, interactive hardware-in-the-loop testing suite can be run with:

``` python
pytest --manual -v -s path/to/spot_basic/test/system
```

> [!TIP]
> Wire your workstation to a Spot robot to avoid network issues.

## Architecture

It might be overwhelming and confusing when you first start to work with Spots in ROS 2.

Here is a quick rundown of all packages that functionally contribute to a `bdai spot driver` run:

| Repository      | Package          | Purpose                                                                                           |
|-----------------|------------------|---------------------------------------------------------------------------------------------------|
| `spot_ros2`     | `spot_driver`    | The driver nodes for Spot in ROS 2 (though conceptually it is closer to a bridge).                |
| `spot_ros2`     | `spot_msgs`      | ROS 2 interfaces specific to Spot (mainly service and action definitions).                        |
| `spot_wrapper`  | `spot_wrapper`   | A thin, convenience layer atop the Spot Python SDK.                                               |
| `bosdyn_msgs`   | `bosdyn_msgs`    | ROS 2 equivalents to Spot SDK Protobuf messages (plus bidirectional conversion APIs).             |

Once brought up, the Spot ROS 2 driver (graph of) node(s) acts as a ROS 2 bridge to the Spot SDK. When a service request or action goal is sent, using `SpotBasic` or ROS 2 APIs, the Spot ROS 2 driver node serves it by forwarding the request(s) to Spot wrapper, that forwards it to Spot SDK APIs , which in turn hit the Spot robot over gRPC. Responses follow the same path in reverse. State and sensor streams are handled separately, placing data requests via Spot SDK APIs at fixed rates, publishing Spot specific messages on topics and broadcasting `/tf` transforms.

## Guides

### Using `SpotBasic`

`SpotBasic` user facing APIs are the result of repeatedly applying two abstractions: [generalized callables](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/callables.py), to put out requests and commands over ROS 2 services and actions, and [message feeds](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/feeds.py), to fetch and iterate over stateful ROS 2 topic streams. In combination, these abstractions afford an idiomatic approach to Spot robot programming.

```python
import argparse
import cv2
import cv_bridge
import sys

from contextlib import closing
from spatialmath import SE2

from spot_basic import SpotBasic
import synchros2.process as ros_process

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    """
    Make Spot stand up, walk in straight line, and sit down.

    While it walks, the front right camera feed will be shown.
    """
    bridge = cv_bridge.CvBridge()
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)
        spot.stand.synchronously(timeout_sec=10.0)
        action = spot.walk.asynchronously(SE2.Trans([1., 0.]))
        with closing(spot.cameras.front_right.rgb.image.stream()) as imagestream:
            for image in map(bridge.imgmsg_to_cv2, imagestream):
                if action.finalized:
                    break
                cv2.imshow("Spot View", image)
                cv2.waitKey(10)
            cv2.destroyAllWindows()
        spot.sit.synchronously(timeout_sec=10.0)
        spot.power_off.synchronously(timeout_sec=10.0)
        return 0 if action.succeeded else 1

if __name__ == "__main__":
    sys.exit(main())
```

Most `SpotBasic` methods are generalized methods, which in practice means they can be blocked on thru completion, with an optional timeout, or managed using futures, and even cancelled, once started. Most `SpotBasic` properties, on the other hand, are message feeds enabling simple iteration as well as more complex update notification schemes, using callbacks or using futures. There is ample room for a variety of programming patterns and idioms.

#### Executing commands

`SpotBasic` can execute arbitrary `RobotCommand`s on Spot. These commands can be built using `bosdyn_api_msgs.msg.RobotCommand` ROS 2 (equivalent) messages, or `bosdyn.api.robot_command_pb2.RobotCommand` Protobuf messages as Spot SDK code would. Either way, ROS 2 (equivalent) messages will be returned, as these convey at least the same amount of information (e.g. as to what was the outcome of the command), often more.

```python
import argparse
import sys

from bosdyn.client.robot_command import RobotCommandBuilder
from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)
        command = RobotCommandBuilder.freeze_command()
        if args.asynchronous:
            action = spot.execute.asynchronous(command)
            result = unwrap_future(action, timeout_sec=10.0)
        else:
            result = spot.execute.synchronous(command, timeout_sec=10.0)
        spot.power_off.synchronously(timeout_sec=10.0)
        return 0 if action.succeeded else 1

if __name__ == "__main__":
    sys.exit(main())
```

Synchronous commands are blocking. On completion, either a successful result is returned or an exception is raised. Asynchronous commands, on the other hand, return a future-like object immediately without blocking. This object allows the underlying ROS 2 action lifecycle to be monitored, awaiting for specific transitions, registering callbacks, querying status and result, and more.

> [!TIP]
> When prototyping, it is usually simpler to develop for the happy path. With synchronous APIs you can defer error handling
> to the caller, which may be none other than the user once the exception reaches the interpreter, greatly simplifying the code.

Note that in both cases commands will be followed through completion. This may not always be desirable e.g. for piecewise trajectory control, the extra overhead that command tracking requires can significantly degrade the responsiveness of the system. Instead, for true fire-and-forget behavior, one can simply start executing a command:

```python
import argparse
import sys

from bosdyn.client.robot_command import RobotCommandBuilder
from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)
        command = RobotCommandBuilder.freeze_command()
        if args.asynchronous:
            action = spot.start_executing.asynchronous(command)
            result = unwrap_future(action, timeout_sec=10.0)
        else:
            result = spot.start_executing.synchronous(command, timeout_sec=10.0)
        spot.power_off.synchronously(timeout_sec=10.0)
        return 0 if action.succeeded else 1

if __name__ == "__main__":
    sys.exit(main())
```

To complement the Spot SDK, the `spot_basic.command` subpackage includes a number of command builders to simplify command construction across the full range of Spot capabilities e.g.:

```python
import argparse
import sys

from spatialmath import SE3

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic
from spot_basic.state import ArmJointMap
from spot_basic.command.full_body import SpotFullBodyCommandBuilder

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)
        command_builder = SpotFullBodyCommandBuilder(spot)
        command = command_builder.whole_body_command(
            SE3.Trans([1.0, 0.0, 0.0]), ArmJointMap(
                sh0=0.25, sh1=0.25, el0=-0.25, el1=-0.25, wr0=0.25, wr1=0.25
            )
        )
        if args.asynchronous:
            action = spot.execute.asynchronous(command)
            result = unwrap_future(action, timeout_sec=20.0)
        else:
            result = spot.execute.synchronous(command, timeout_sec=20.0)
        spot.power_off.synchronously(timeout_sec=10.0)
        return 0 if action.succeeded else 1

if __name__ == "__main__":
    sys.exit(main())
```

These command builders can craft simple commands and complex maneuvers. These maneuvers are sequences of commands, potentially lazy evaluated so that later commands can react to robot state. Maneuvers can be executed sequentially by means of vectorized execution. Vectorized execution will return all commands results or raise on error.

``` python
import argparse
import sys

from spatialmath import SE3, SO3

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic
from spot_basic.command.mobility import SpotMobilityCommandBuilder

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)
        command_builder = SpotMobilityCommandBuilder(spot)
        maneuver = command_builder.walk_and_orient_maneuver(
            SE3.Rt(SO3.Rz(90.0, unit="deg"), [1.0, 0.0, 0.0])
        )
        if args.asynchronous:
            future = spot.execute.vectorized.asynchronous(maneuver)
            results = unwrap_future(future, timeout_sec=20.0)
        else:
            results = spot.execute.vectorized.synchronous(maneuver, timeout_sec=20.0)
        spot.power_off.synchronously(timeout_sec=10.0)
        return 0 if all(r.success for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())
```

Note that this specific maneuver is twice instantiated, as it is a consumable iterable.

> [!TIP]
> Do not reuse commands or maneuvers. Many are robot state-dependent.

When working on Spot robots, some commands are used over and over: standing, sitting, walking. These commands are so common that `SpotBasic` already exposes direct interfaces to them.

``` python
import argparse
import sys

from spatialmath import SE3, SO3

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic
from spot_basic.command.mobility import SpotMobilityCommandBuilder

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)
        if args.asynchronous:
            future = spot.stand.asynchronously()
            result = unwrap_future(future, timeout_sec=20.0)
        else:
            result = spot.stand.synchronously(timeout_sec=20.0)
        spot.power_off.synchronously(timeout_sec=10.0)
        return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())
```

A command may be `stop`d during execution too e.g. on `Ctrl+C`:

```python
try:
    spot.walk.synchronously(SE3.Trans([1.0, 0.0, 0.0]), timeout_sec=20.0)
except KeyboardInterrupt:
    spot.stop.synchronously(timeout_sec=20.0)
```

Explore the [`spot_basic.command`](./spot_basic/command) subpackage and its documentation for a complete list of all available commands. For manipulation commands, see the [`spot_basic.manipulation`](./spot_basic/manipulation.py) submodule instead. Note `SpotBasic.perform` semantics are very similar to those of `SpotBasic.execute`, yet not the same, to accommodate the distinction that the Spot SDK makes between regular commands and manipulation commands (or requests).

#### Querying state

`SpotBasic` exposes Spot state via stateful feeds and explicit queries. Such is the case for joint states and world object listings, for example.

``` python
import argparse
import sys

from bosdyn.api.world_object_pb2 import ListWorldObjectRequest, WorldObjectType,

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)

        joint_states = unwrap_future(spot.joint_states.update, timeout_sec=10.0)
        print("Joints:", joint_states.names)

        request = ListWorldObjectRequest()
        request.object_type.append(WorldObjectType.WORLD_OBJECT_APRILTAG)
        response = spot.list_world_objects.synchronously(request, timeout_sec=10.0)
        print("Tags:", response.response.world_objects)

        spot.power_off.synchronously(timeout_sec=10.0)
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

However, by far the most common interface is the transform tree. `SpotBasic` API for transform lookup mimics that of the [`TransformListener` wrapper](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/tf_listener_wrapper.py) in `synchros2`.

```python
import argparse
import sys

from bosdyn.api.world_object_pb2 import ListWorldObjectRequest, WorldObjectType,

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(args.robot_name)
    with spot.lease(timeout_sec=10.0):
        spot.power_on.synchronously(timeout_sec=10.0)

        assert spot.wait_for_a_tform_b("odom", "head", timeout_sec=10.0, wait_for_frames=True)
        odom_t_head = spot.lookup_a_tform_b("odom", "head", timeout_sec=10.0)
        print("Odom to Head:", odom_t_head)

        spot.power_off.synchronously(timeout_sec=10.0)
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

Explore the [`spot_basic.state`](./spot_basic/state.py) submodule and its documentation for a complete list of all APIs.

#### Capturing video

`SpotBasic` can access all RGBD cameras on Spot. Each RGBD camera streams RGB images, depth maps, camera intrinsics (usually but not necessarily static), and (locally reconstructed) pointclouds. Subscriptions to these streams are exposed as a single arrangement (or configuration) of [`spot.cameras`](./spot_basic/image.py). You can work with these message feeds in a number of ways:

- fetching the latest frame if available:

```python
spot.cameras.front_left.rgb.image_with_info.latest  # may be None
spot.cameras.front_left.depth.image_with_info.latest  # may be None
spot.cameras.front_left.pointcloud.latest  # may be None
```

- waiting for the next incoming frame:

```python
from synchros2.futures import unwrap_future

rgb_image, info = unwrap_future(spot.cameras.front_left.rgb.image_with_info.update)
depth_map, info = unwrap_future(spot.cameras.front_left.depth.image_with_info.update)
pointcloud = unwrap_future(spot.cameras.front_left.pointcloud.update)
```

- registering a callback for frames:

```python
def process_frame(image, info):
    # do something

connection = spot.cameras.front_left.rgb.image_with_info.recall(process_frame)
connection.close()  # to unregister
```

- iterating over the stream:

```python
from contextlib import closing

rgb_camera = spot.cameras.front_left.rgb
with closing(rgb_camera.image_with_info.stream()) as imagestream:
    for image, info in imagestream:
        # do something
```

> [!IMPORTANT]
> Note the use of `contextlib.closing` on the image stream. This ensures the underlying connection is closed as soon
> as the code exits the scope, preventing unnecessary buffering (wasting memory and CPU cycles).

- synchronizing multiple streams:

```python
from synchros2.feeds import SynchronizedMessageFeed

rgb_image_feeds = [camera.rgb.image for camera in spot.cameras]
sync_rgb_image_feeds = SynchronizedMessageFeed(*rgb_image_feeds)
with closing(sync_rgb_image_feeds.stream()) as imagestream:
    for rgb_images in imagestream:
        for camera, rgb_image in zip(
            spot.cameras, rgb_images
        ):
            # do something
```

And more. Explore the [`synchros2.feeds`](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/feeds.py) submodule for an overview of what message feeds can do.

#### Where to go next

- Explore the [`spot_basic`](./spot_basic) subpackage and its documentation for a complete list of all available Spot APIs.
- Explore [`spot_basic` examples](./examples) and [tests](./test/spot_basic) for a practical overview of what you can do with it.

### Migrating away from `SpotBasic` legacy APIs

As of Q3 2024, `SpotBasic` offers more functionality than earlier (now legacy) APIs once used to and it is ready to move on.

When migrating your code to the latest APIs, while nothing trumps a good grasp of available features and supported idioms, the following rules of thumb may be of use:

- `SpotBasic.<command>()` API calls should be replaced by `SpotBasic.<command>.synchronously()` API calls. Calling code will now get full responses to inspect. It will also no longer get `None` on error but a suitable exception -- manage it accordingly or let it propagate.
- `SpotBasic.<command>_async()` API calls should be replaced by `SpotBasic.<command>.asynchronously()` API calls. Calling code will no longer get a handle but a suitable future i.e. a plain future for ROS 2 service-based APIs and an [action future](https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/action.py#L54) for ROS 2 action-based APIs.
- Use `SpotBasic.lookup_*()` and `SpotBasic.wait_*()` APIs over direct `SpotBasic.tf_listener` use whenever possible.
- Use `SpotBasic.cameras` property over the `SpotVision` class whenever possible.

For example, take the following body gazing sample script:

``` python
import argparse
import time

import synchros2.process as ros_process
from synchros2.tf_listener_wrapper import TFListenerWrapper
from spot_basic import SpotBasic
from spot_basic.manual_conversions import to_se3

def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    spot = SpotBasic(args.robot_name)
    spot.initialize_robot()

    tf = TFListenerWrapper(cache_time_s=30)
    tf.wait_for_a_tform_b(spot.body_frame_name, spot.vision_frame_name)
    time.sleep(1.0)
    spot.set_tf_listener(tf)
    vision_to_body = to_se3(tf.lookup_a_tform_b(spot.vision_frame_name, spot.body_frame_name, wait_for_frames=True))

    if args.asynchronous:
        spot.body_gaze_at_async(
            body_t_target=vision_to_body.inv(), result_callback=(lambda x: print("Body gaze at complete!"))
        )
    else:
        spot.body_gaze_at(vision_to_body.inv(), vision_to_body)
    time.sleep(5.0)
    print("TADA!")


if __name__ == "__main__":
    main()
```

Using latest APIs, this sample script could look like this:

``` python
import argparse

import synchros2.process as ros_process
from synchros2.futures import wait_for_future
from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", "--asynchronous", action="store_true")
    parser.add_argument("robot_name")
    return parser

@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> None:
    spot = SpotBasic(args.robot_name)

    with spot.lease(timeout_sec=5.0):
        spot.power_on.synchronously(timeout_sec=5.0)
        try:
            assert spot.wait_for_a_tform_b(FrameHint.BODY, FrameHint.VISION, timeout_sec=10.0, wait_for_frames=True)
            body_t_vision = spot.lookup_a_tform_b(FrameHint.BODY, FrameHint.VISION)
            if args.asynchronous:
                action = spot.body_gaze.asynchronously(body_t_vision.t, FrameHint.BODY)
                assert wait_for_future(action.finalization, timeout_sec=5.0)
                assert action.succeeded
            else:
                spot.body_gaze.synchronously(body_t_vision.t, FrameHint.BODY, timeout_sec=5.0)
            print("Body gaze at complete!")
        except KeyboardInterrupt:
            spot.stop.synchronously(timeout_sec=5.0)
        finally:
            spot.power_off.synchronously(timeout_sec=5.0)

if __name__ == "__main__":
    main()
```

Note that:

- Spot initialization is explicit. We hold the lease for as long as the script needs to. We power the robot on at the beginning, and power it off at the end, always, regardless of the script outcome (as the `finally` clause guarantees it).
- No transform listener needs to be set explicitly. The process-wide node provides one if `uses_tf` is set to `True`.
- Every potentially blocking operation has a timeout to avoid indefinite hangs.
- For asynchronous body gazing, we apply a synchronous wait on command finalization, then check for successful execution. Note that synchronous body gazing needs not perform the same check, as it does so implicitly already.
- If the sample script is explicitly interrupted, any Spot motion will be stopped before proceeding.
- For the most, there is little if any error checking. That is a choice that makes sense for a sample script -- we will just let any exceptions raised by `SpotBasic` on error propagate to the user.

Exploring [`spot_basic` examples](./examples) diffs and checking the [`spot_basic` system test suite](./test/system) may also prove instructive if you looking forward to migrate.

### Extending `SpotBasic`

Under the hood, `SpotBasic` is structured as an aggregate of loosely coupled [mixins](https://en.wikipedia.org/wiki/Mixin). These mixins heavily rely on [factory](https://en.wikipedia.org/wiki/Factory_method_pattern) and [delegation](https://en.wikipedia.org/wiki/Delegation_pattern) patterns to abstract and reuse what is specific to message passing, which is central to `SpotBasic` implementation, the Spot ROS 2 driver being nothing but a bridge to the Spot SDK over ROS 2 topics, services, and actions.

#### Exposing a robot command

Any robot command can be constructed and `SpotBasic.execute`d, but some robot commands may be worth adding to `spot_basic` for reuse.

The first step to expose a new robot command is to implement a factory function for it. Command factory functions are typically implemented as methods in `spot_basic.command.SpotCommandBuilder` subclasses under the `spot_basic.command` subpackage. Your factory function may semantically fit in an existing subclass or require another -- use your judgement. In the following we will assume a new subclass is required:

``` python
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

``` python
spot = SpotBasic("<spot-name>")

command_builder = SpotCustomCommandBuilder(spot)
command = command_builder.my_custom_task_command()
response = spot.execute.synchronous(command)
```

But you may want to further integrate your command with `SpotBasic`. You may want to allow `SpotBasic.do_custom_task()` calls. Typically, `SpotBasic` methods are generalized methods, and that is the case for command execution methods. These generalized command execution methods are implemented in mixin classes under the `spot_basic.command` subpackage. Your command execution method may semantically fit in an existing mixin or require another -- again, use your judgement. In the following we will assume a new mixin is required:

``` python
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

``` python
from synchros2.futures import wait_for_future

response = spot.do_something.synchronously()

# or alternatively...

action = spot.do_something.asynchronously()
wait_for_future(action.finalization)
```

Exposing a manipulation request is no different, except for the fact that `SpotBasic.perform` must be used in place for `SpotBasic.execute`.

In the above, we glanced over type annotations for our `SpotBasic.do_something` generalized method. Type annotating generalized methods relies on structural subtyping support i.e. [`Protocol`](https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols)s.

``` python
# spot_basic/command/custom.py

from typing import Any, Callable, Protocol, TypeAlias

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

#### Adding a new action

The above discusses how to expose a new capability of an existing action.  If we actually need to add a new action (corresponding usually to a new Spot grpc service), a tutorial for that can be found [here](spot_basic/AddingActionToSpotBasicTutorial.md).

#### Exposing a state stream

States streams naturally map to topics in ROS 2. Therefore, a topic subscription is all we need to expose one. Topic subscriptions are typically implemented as `SpotBasic` properties in mixin classes under the `spot_basic` subpackage. Your property may semantically fit in an existing mixin or require another -- use your judgement. In the following we will assume a new mixin is required:

``` python
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

``` python
value = unwrap_future(spot.some_robot_state.update)
print(value.data)
```

### Using the Mixed-Level API

A joint-level API exists within [spot_ros2](https://github.com/rai-opensource/spot_ros2/tree/main) using `ros2_control`. A combined API that utilizes both this lower level joint control and the traditional high-level API was created to leverage both of these APIs.

For a full example script, see [`example_controlled_squat.py`](examples/example_controlled_squat.py). To run this example, first start the driver in full controllable mode using the `--controllable` flag.

```bash
bdai spot driver <spot-name> --controllable
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

#### Exposing new SDK functionality

If the Spot ROS 2 driver does not already expose some Spot SDK functionality that is of interest to us, things get slightly more involved. The Spot SDK sits deep down the [`spot_driver`](https://github.com/rai-opensource/spot_ros2/tree/main/spot_driver), potentially within the [`spot_wrapper`](https://github.com/rai-opensource/spot_wrapper/tree/main). A suitable ROS 2 interface must be chosen to expose it. [`spot_msgs`](https://github.com/rai-opensource/spot_ros2/tree/main/spot_msgs) hosts service and action definitions, [`bosdyn_msgs`](https://github.com/rai-opensource/bosdyn_msgs) hosts equivalent ROS 2 message for every Spot SDK Protobuf message. Only then a `SpotBasic` mixin class can be put together.

#### Where to go next

- Take a look at existing mixins in the [`spot_basic`](./spot_basic) subpackage.
- Explore available functionality in the  [`synchros2`](https://github.com/rai-opensource/synchros2/tree/main/synchros2) package.
