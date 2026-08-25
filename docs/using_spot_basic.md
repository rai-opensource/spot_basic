<!-- Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Using `SpotBasic`

[← Back to the README](../README.md)

## Table of contents

- [Executing commands](#executing-commands)
- [Querying state](#querying-state)
- [Capturing video](#capturing-video)
- [Where to go next](#where-to-go-next)

`SpotBasic` user facing APIs are the result of repeatedly applying two abstractions:

- [generalized callables][1], to put out requests and commands over ROS 2 services and actions, and
- [message feeds][2], to fetch and iterate over stateful ROS 2 topic streams.

In combination, these abstractions afford an idiomatic approach to Spot robot programming. The
example below makes Spot stand up, walk in a straight line, and sit down, showing the front right
camera feed while it walks:

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

Most `SpotBasic` methods are generalized methods, which in practice means they can be blocked on
thru completion, with an optional timeout, or managed using futures, and even cancelled, once
started. Most `SpotBasic` properties, on the other hand, are message feeds enabling simple iteration
as well as more complex update notification schemes, using callbacks or using futures. There is
ample room for a variety of programming patterns and idioms.

## Executing commands

`SpotBasic` can execute arbitrary `RobotCommand`s on Spot. These commands can be built using
`bosdyn_api_msgs.msg.RobotCommand` ROS 2 (equivalent) messages, or
`bosdyn.api.robot_command_pb2.RobotCommand` Protobuf messages as Spot SDK code would. Either way,
ROS 2 (equivalent) messages will be returned, as these convey at least the same amount of
information (e.g. as to what was the outcome of the command), often more.

```python
import argparse
import sys

from bosdyn.client.robot_command import RobotCommandBuilder
from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asynchronous", "--async", action="store_true")
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
        return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())
```

Synchronous commands are blocking. On completion, either a successful result is returned or an
exception is raised. Asynchronous commands, on the other hand, return a future-like object
immediately without blocking. This object allows the underlying ROS 2 action lifecycle to be
monitored, awaiting for specific transitions, registering callbacks, querying status and result, and
more.

> [!TIP]
> When prototyping, it is usually simpler to develop for the happy path. With synchronous APIs you
> can defer error handling to the caller, which may be none other than the user once the exception
> reaches the interpreter, greatly simplifying the code.

Note that in both cases commands will be followed through completion. This may not always be
desirable e.g. for piecewise trajectory control, the extra overhead that command tracking requires
can significantly degrade the responsiveness of the system. Instead, for true fire-and-forget
behavior, one can simply start executing a command:

```python
import argparse
import sys

from bosdyn.client.robot_command import RobotCommandBuilder
from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asynchronous", "--async", action="store_true")
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
        return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())
```

To complement the Spot SDK, the `spot_basic.command` subpackage includes a number of command
builders to simplify command construction across the full range of Spot capabilities e.g.:

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
    parser.add_argument("--asynchronous", "--async", action="store_true")
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
        return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())
```

These command builders can craft simple commands and complex maneuvers. These maneuvers are
sequences of commands, potentially lazy evaluated so that later commands can react to robot state.
Maneuvers can be executed sequentially by means of vectorized execution. Vectorized execution will
return all commands results or raise on error.

```python
import argparse
import sys

from spatialmath import SE3, SO3

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic
from spot_basic.command.mobility import SpotMobilityCommandBuilder

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asynchronous", "--async", action="store_true")
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

When working on Spot robots, some commands are used over and over: standing, sitting, walking. These
commands are so common that `SpotBasic` already exposes direct interfaces to them.

```python
import argparse
import sys

from spatialmath import SE3, SO3

from synchros2.futures import unwrap_future
import synchros2.process as ros_process

from spot_basic import SpotBasic
from spot_basic.command.mobility import SpotMobilityCommandBuilder

def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asynchronous", "--async", action="store_true")
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

Explore the [`spot_basic.command`](../spot_basic/command) subpackage and its documentation for a
complete list of all available commands. For manipulation commands, see the
[`spot_basic.manipulation`](../spot_basic/manipulation.py) submodule instead. Note
`SpotBasic.perform` semantics are very similar to those of `SpotBasic.execute`, yet not the same, to
accommodate the distinction that the Spot SDK makes between regular commands and manipulation
commands (or requests).

## Querying state

`SpotBasic` exposes Spot state via stateful feeds and explicit queries. Such is the case for joint
states and world object listings, for example.

```python
import argparse
import sys

from bosdyn.api.world_object_pb2 import ListWorldObjectRequest, WorldObjectType

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

However, by far the most common interface is the transform tree. `SpotBasic` API for transform
lookup mimics that of the [`TransformListener` wrapper][3] in `synchros2`.

```python
import argparse
import sys

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

Explore the [`spot_basic.state`](../spot_basic/state.py) submodule and its documentation for a
complete list of all APIs.

## Capturing video

`SpotBasic` can access all RGBD cameras on Spot. Each RGBD camera streams RGB images, depth maps,
camera intrinsics (usually but not necessarily static), and (locally reconstructed) pointclouds.
Subscriptions to these streams are exposed as a single arrangement (or configuration) of
[`spot.cameras`](../spot_basic/image.py). You can work with these message feeds in a number of ways:

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
> Note the use of `contextlib.closing` on the image stream. This ensures the underlying connection
> is closed as soon as the code exits the scope, preventing unnecessary buffering (wasting memory
> and CPU cycles).

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

And more. Explore the [`synchros2.feeds`][2] submodule for an overview of what message feeds can do.

## Where to go next

- Explore the [`spot_basic`](../spot_basic) subpackage and its documentation for a complete list of
  all available Spot APIs.
- Explore [`spot_basic` examples](../examples) and [tests](../test/spot_basic) for a practical
  overview of what you can do with it.
- Learn how to [extend `SpotBasic`](extending_spot_basic.md) with new commands, state streams, and
  joint-level control.

<!-- References -->

[1]: https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/callables.py
[2]: https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/feeds.py
[3]: https://github.com/rai-opensource/synchros2/blob/main/synchros2/synchros2/tf_listener_wrapper.py
