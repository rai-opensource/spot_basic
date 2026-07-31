<!-- Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# `spot_basic`

A simpler, more intuitive interface to Boston Dynamics' Spot robots with ROS 2.

## Table of contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Architecture](#architecture)
- [Installation](#installation)
- [Running the Spot driver](#running-the-spot-driver)
- [A first example](#a-first-example)
- [Documentation](#documentation)
- [Testing](#testing)
- [Limitations](#limitations)
- [Versioning and releases](#versioning-and-releases)
- [Maintenance](#maintenance)
- [Contributing](#contributing)
- [Help](#help)
- [License](#license)

## Overview

`spot_basic.SpotBasic` is to the Spot ROS 2 driver what the Spot Python SDK is to Spot gRPC servers: a simpler interface to Spot capabilities. Under the hood, `SpotBasic` takes all ROS 2 topics, services, and actions that the [`spot_ros2` driver](https://github.com/rai-opensource/spot_ros2/blob/main/spot_driver/launch/spot_driver.launch.py) exposes and repackages them as a set of consistent, intuitive APIs, suitable for both synchronous and asynchronous programming.

> [!IMPORTANT]
> Integration with [process-wide `synchros2`](https://github.com/rai-opensource/synchros2/wiki/Process%E2%80%90wide-APIs) is tight and it is usually easier (though not strictly required) to use it when working with `spot_basic`.

## Requirements

`spot_basic` is developed and tested against the following environment. Other combinations may work but are not exercised in CI.

| Component | Supported version |
|-----------|-------------------|
| Operating system | Ubuntu 22.04 (Jammy) |
| ROS 2 distribution | Humble Hawksbill |
| Python | 3.10 |
| CPU architecture | AMD64 (x86_64) and ARM64 (aarch64) |
| Spot SDK (`bosdyn-client`, `bosdyn-api`, `bosdyn-core`) | 5.0.1 |
| NumPy | `< 2` (see [`constraints.txt`](constraints.txt)) |

You will also need:

- A Boston Dynamics Spot robot, reachable over the network, with firmware compatible with Spot SDK 5.0.1. Arm-specific APIs additionally require a Spot equipped with an arm.
- The [`spot_ros2`](https://github.com/rai-opensource/spot_ros2) driver and [`bosdyn_msgs`](https://github.com/rai-opensource/bosdyn_msgs), both built from source alongside `spot_basic` (see [Installation](#installation)).

> [!NOTE]
> The `numpy < 2` constraint is not arbitrary: ROS 2 Humble binaries and the `spot_wrapper` stack are built against the apt-provided NumPy 1.x, and a NumPy 2.x installed into `~/.local` will silently shadow it and break those extension modules. See [`constraints.txt`](constraints.txt) for details.

## Architecture

It might be overwhelming and confusing when you first start to work with Spots in ROS 2. Here is a quick rundown of all packages that functionally contribute to a Spot ROS 2 driver (`spot_driver`) run:

| Repository      | Package          | Purpose                                                                                |
|-----------------|------------------|----------------------------------------------------------------------------------------|
| `spot_ros2`     | `spot_driver`    | The driver nodes for Spot in ROS 2 (though conceptually it is closer to a bridge).      |
| `spot_ros2`     | `spot_msgs`      | ROS 2 interfaces specific to Spot (mainly service and action definitions).             |
| `spot_wrapper`  | `spot_wrapper`   | A thin, convenience layer atop the Spot Python SDK.                                     |
| `bosdyn_msgs`   | `bosdyn_api_msgs`, `bosdyn_msgs` | ROS 2 equivalents of Spot SDK Protobuf messages (`bosdyn_api_msgs`) plus bidirectional conversion APIs (`bosdyn_msgs`). |

Once brought up, the Spot ROS 2 driver node(s) act as a ROS 2 bridge to the Spot SDK. When a service request or action goal is sent -- using `SpotBasic` or ROS 2 APIs directly -- the driver forwards it to Spot wrapper, which forwards it to the Spot SDK APIs, which in turn hit the Spot robot over gRPC. Responses follow the same path in reverse. State and sensor streams are handled separately, requesting data via Spot SDK APIs at fixed rates and publishing Spot-specific messages on topics while broadcasting `/tf` transforms.

## Installation

`spot_basic` is a ROS 2 (`ament_cmake`) package, built from source inside a colcon workspace. It depends on several open-source packages that are also built from source; these are listed in [`spot_basic.repos`](spot_basic.repos) and fetched with [vcstool](https://github.com/dirk-thomas/vcstool). Note that `spot_ros2` vendors `synchros2` and `spot_wrapper` as git submodules, so the import is run with `--recursive` to pull them in.

### Quick start (install script)

For most users, the [`install_spot_basic.sh`](install_spot_basic.sh) script handles the whole setup -- fetching dependencies, installing the Spot SDK and `bosdyn_msgs`, resolving rosdep keys, and building the workspace:

```bash
# Create a workspace and clone spot_basic into src/
mkdir -p ~/spot_ws/src && cd ~/spot_ws
git clone https://github.com/rai-opensource/spot_basic.git src/spot_basic

# Install everything and build (use --arm64 on ARM hosts; --help for options)
./src/spot_basic/install_spot_basic.sh

# Source the workspace
source install/setup.bash
```

### Manual installation

If you prefer to run the steps yourself:

```bash
# Create a workspace and clone spot_basic into src/
mkdir -p ~/spot_ws/src && cd ~/spot_ws
git clone https://github.com/rai-opensource/spot_basic.git src/spot_basic

# Fetch source dependencies (spot_ros2 with its synchros2/spot_wrapper submodules, bosdyn_msgs)
vcs import --recursive src < src/spot_basic/spot_basic.repos

# Install system / ROS dependencies declared in package.xml
rosdep install --from-paths src --ignore-src -r -y

# Install PyPI dependencies that have no rosdep key (Spot SDK, spatialmath)
pip install -r src/spot_basic/requirements.txt

# Build and source
colcon build
source install/setup.bash
```

Note that you will also need to follow the `spot_ros2` installation instructions as well if you choose manual installation (the build will most likely fail before you do this).

## Running the Spot driver

`spot_basic` talks to a robot through the `spot_ros2` driver, so a `spot_driver` instance must be running first. Bring it up with:

```bash
ros2 launch spot_driver spot_driver.launch.py config_file:=<path/to/spot_ros.yaml> spot_name:=<spot-name>
```

The `config_file` holds your robot's connection details (`hostname`, `username`, `password`). See the [`spot_driver` documentation](https://github.com/rai-opensource/spot_ros2/blob/main/spot_driver/README.md) for the full list of launch arguments and configuration options.

> [!TIP]
> For joint-level (mixed-level) control, launch the driver with `controllable:=True`. See [Using the Mixed-Level API](docs/extending_spot_basic.md#using-the-mixed-level-api).

## A first example

With the driver running, this script makes Spot stand up, walk forward, and sit down while streaming its front-right camera:

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

More runnable scripts live in [`examples/`](examples).

## Documentation

| Guide | Contents |
|-------|----------|
| [Using `SpotBasic`](docs/using_spot_basic.md) | Executing commands, querying state, and capturing video. |
| [Extending `SpotBasic`](docs/extending_spot_basic.md) | Adding commands, actions, state streams, and joint-level control. |
| [Adding an action](docs/adding_an_action.md) | Step-by-step tutorial for wiring up a brand-new action. |

You can also explore the [`spot_basic`](spot_basic) package, the [`examples/`](examples), and the [tests](test/spot_basic) directly.

## Testing

Unit tests run with `colcon test`. A complete, interactive hardware-in-the-loop (HIL) testing suite is also available:

```bash
pytest --manual -v -s path/to/spot_basic/test/system
```

> [!TIP]
> Wire your workstation to a Spot robot to avoid network issues.

## Limitations

Please read these before adopting `spot_basic`.

- **A running driver is required.** `spot_basic` is a client of the `spot_ros2` driver, not a standalone Spot SDK application. Every API call ultimately travels through a running `spot_driver` instance; there is no direct-to-robot mode.
- **Tight coupling to `synchros2`.** The API is designed around [process-wide `synchros2`](https://github.com/rai-opensource/synchros2/wiki/Process%E2%80%90wide-APIs) APIs. Using `spot_basic` outside a `synchros2` process is possible but noticeably harder.
- **Arm APIs require an arm.** Arm, gripper, manipulation, and surface-contact APIs raise a `RuntimeError` on robots without an arm rather than degrading gracefully.
- **Joint-level control requires driver opt-in.** Mixed-level (joint) control only works when the driver is launched with `controllable:=True`.
- **Partial Spot CAM coverage.** Only Spot CAM audio is wrapped; other Spot CAM subsystems (for example PTZ, lighting, and media streaming) are not exposed.
- **Single platform.** Only Ubuntu 22.04 with ROS 2 Humble is tested. Other ROS 2 distributions are not supported.
- **Source dependencies track `main`.** [`spot_basic.repos`](spot_basic.repos) pins `spot_ros2` and `bosdyn_msgs` to their `main` branches rather than fixed commits, so an upstream change can affect a fresh build.
- **Automated tests do not touch hardware.** The unit tests that run in CI use mocked ROS 2 interfaces. Real robot behavior is only covered by the interactive hardware-in-the-loop suite, which must be run manually (see [Testing](#testing)).

## Versioning and releases

`spot_basic` is versioned through the `<version>` field of its [`package.xml`](package.xml), following the usual ROS 2 package convention of `MAJOR.MINOR.PATCH`.

- Releases are published as [GitHub Releases](https://github.com/rai-opensource/spot_basic/releases), each backed by a matching git tag.
- Every release states the Spot SDK version and ROS 2 distribution it was tested against, since compatibility with the Spot SDK is the dominant compatibility concern for this package.
- Because the project is under [light maintenance](#maintenance), breaking API changes may be introduced in a **minor** version bump, not just a major one. Pin to an exact version if you need stability.

## Maintenance

> [!IMPORTANT]
> **`spot_basic` is under light maintenance.** No feature development is guaranteed. If you file a bug report, or open a pull request that fixes a bug, expect an RAI maintainer to respond within a few weeks. Requests for new features may be closed without implementation, and breaking changes may be introduced between releases (see [Versioning and releases](#versioning-and-releases)).

If you need behavior that this project does not provide or guarantee, forking is an entirely reasonable option and the MIT license permits it.

## Contributing

Bug reports and bug-fix pull requests are welcome, within the maintenance expectations described above.

**Reporting an issue.** Open an issue on the [issue tracker](https://github.com/rai-opensource/spot_basic/issues). The most useful reports include your Ubuntu, ROS 2, and Spot SDK versions, whether your robot has an arm, the driver launch arguments you used, a minimal script that reproduces the problem, and the full error output.

**Submitting a change.** Fork the repository, follow the [Installation](#installation) steps, and install the pre-commit hooks before making changes:

```bash
cd <ROS workspace>/src/spot_basic
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then open a pull request describing what changed, why it changed, and how you tested it. All changes require review and approval from a project maintainer before merging, and CI (lint, build, and unit tests) must pass. Note that CI runs for first-time contributors require explicit maintainer approval, so there may be a delay before checks start on your pull request.

## Help

If something is not working, first check [Requirements](#requirements) and [Limitations](#limitations) — most problems come from a version mismatch or from expecting functionality that is intentionally out of scope.

A quick way to confirm your Spot SDK installation is correct (all versions should read `5.0.1`):

```bash
pip list | grep bosdyn
```

If that looks right and the problem persists, open an [issue](https://github.com/rai-opensource/spot_basic/issues).

## License

`spot_basic` is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

Copyright (c) Robotics and AI Institute LLC dba RAI Institute.
