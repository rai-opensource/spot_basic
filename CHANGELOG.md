<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Changelog

All notable changes to `spot_basic` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with the caveat noted in the
README's [Versioning and releases](README.md#versioning-and-releases) section: because the project
is under light maintenance, breaking changes may be introduced in a minor version bump.

## [Unreleased][unreleased]

### Added

- `SpotBasic`, a single entry point to Spot capabilities over ROS 2, composed of focused mixins and
  usable both synchronously and asynchronously.
- Lease, power, E-Stop, and docking control.
- Robot state access, including TF lookups and frame tree snapshots.
- Mobility commands, including SE2 trajectory following and velocity commands.
- Arm and gripper commands, arm surface contact, inverse kinematics, and manipulation and grasping
  helpers.
- Full body commands, world object and no-go region management, and GraphNav localization and
  navigation.
- Image and camera access, including gripper camera parameters and image streaming.
- Spot CAM audio support.
- Joint-level (mixed-level) control built on `ros2_control`, for use with the driver launched with
  `controllable:=True`.
- SE2 and SE3 trajectory types built on `spatialmath`, with plotting helpers.
- 26 runnable examples under [`examples/`](examples), and guides under [`docs/`](docs).
- Unit test suite plus an interactive hardware-in-the-loop suite under `test/system`.
- Secret scanning in the pre-commit configuration, via the `gitleaks` hook.
- Unit test coverage measurement in CI, configured by `.coveragerc` and published as a
  `coverage.xml` workflow artifact.
- A four minute timeout on unit tests, via `pytest-timeout`, so that a hung test fails instead of
  stalling CI. The hardware-in-the-loop tests under `test/system` are exempt, because they block on
  interactive prompts and so wall clock time says nothing about whether they are hung.

### Changed

- Constrained `spatialmath-python` to `>=1.1,<2` and `open3d` to `>=0.17,<0.19`, the latter to stay
  compatible with the existing `numpy < 2` constraint.
- Repointed Dependabot at the `pip` and `github-actions` ecosystems, replacing a stale
  `gitsubmodule` configuration that matched nothing, and documented in `SECURITY.md` which
  dependencies it covers, which are updated by hand, and how updates are triaged.

### Fixed

- Declared dependencies that the package imports but never listed, so that a clean `rosdep install`
  and `pip install -r requirements.txt` now provide everything needed to import `spot_basic`:
  `open3d`, `matplotlib`, `prompt_toolkit`, `std_srvs`, `controller_manager_msgs`, and
  `lifecycle_msgs`.

[unreleased]: https://github.com/rai-opensource/spot_basic/commits/main
