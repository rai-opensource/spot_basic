<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Tested versions

This file records the exact versions `spot_basic` was last verified against, so that a build can be reproduced when a floating dependency changes upstream.

It is a **reference, not a lockfile**. Version constraints in [`package.xml`](../package.xml), [`requirements.txt`](../requirements.txt), and [`constraints.txt`](../constraints.txt) intentionally allow ranges; only known incompatibilities are constrained, and applications are left to determine the precise versions they use. Consult this file when you need a known-good combination.

Please update this file whenever the tested combination changes.

## Last verified

- **Date:** 2026-07-31
- **Result:** 87 unit tests passing (`pytest test --ignore=test/system`)

## Platform

| Component | Version |
|-----------|---------|
| Operating system | Ubuntu 22.04.5 LTS (Jammy) |
| ROS 2 distribution | Humble Hawksbill |
| Python | 3.10.12 |
| CPU architecture | x86_64 |

> [!NOTE]
> ARM64 (aarch64) is supported by the installer (`install_spot_basic.sh --arm64`) but the versions above were captured on x86_64.

## Source dependencies

[`spot_basic.repos`](../spot_basic.repos) tracks the `main` branch of each repository below rather than a fixed commit, so a fresh `vcs import` may pick up newer revisions. These are the commits last tested:

| Repository | Commit |
|------------|--------|
| [`spot_ros2`](https://github.com/rai-opensource/spot_ros2) | `b344640b172afeda72e0c47a8bee4a4a44ba9a99` |
| [`bosdyn_msgs`](https://github.com/rai-opensource/bosdyn_msgs) | `567565882fbf1ab31c1d93662a2c71b8e9683014` |

To reproduce exactly, check out these commits after importing:

```bash
vcs import --recursive src < src/spot_basic/spot_basic.repos
git -C src/spot_ros2 checkout b344640b172afeda72e0c47a8bee4a4a44ba9a99
git -C src/bosdyn_msgs checkout 567565882fbf1ab31c1d93662a2c71b8e9683014
```

Note that `spot_ros2` vendors `synchros2` and `spot_wrapper` as git submodules; run `git submodule update --init --recursive` inside it after checking out a specific commit.

## Python dependencies

| Package | Version | Declared in |
|---------|---------|-------------|
| `bosdyn-api` | 5.0.1 | `requirements.txt` (pinned) |
| `bosdyn-client` | 5.0.1 | `requirements.txt` (pinned) |
| `bosdyn-core` | 5.0.1 | `requirements.txt` (pinned) |
| `spatialmath-python` | 1.1.14 | `requirements.txt` (`>=1.1,<2`) |
| `open3d` | 0.18.0 | `requirements.txt` (`>=0.17,<0.19`) |
| `numpy` | 1.26.4 | `constraints.txt` (`<2`) |
| `scipy` | 1.15.3 | `package.xml` (`python3-scipy`) |
| `matplotlib` | 3.5.1 | `package.xml` (`python3-matplotlib`) |
| `prompt_toolkit` | 3.0.52 | `package.xml` (`python3-prompt-toolkit`) |
| `cv-bridge` | 3.2.1 | `package.xml` (`cv_bridge`) |

ROS 2 interface dependencies (`rclpy`, `sensor_msgs`, `std_msgs`, `std_srvs`, `geometry_msgs`, `builtin_interfaces`, `rcl_interfaces`, `lifecycle_msgs`, `controller_manager_msgs`, `spot_msgs`, `bosdyn_msgs`, `synchros2`) are resolved by `rosdep` against the ROS 2 Humble distribution and are not pinned independently.

## Regenerating this record

After a verified test run, capture the current environment with:

```bash
pip list | grep -iE "bosdyn|spatialmath|open3d|numpy|scipy|matplotlib|prompt_toolkit|cv-bridge"
git -C src/spot_ros2 rev-parse HEAD
git -C src/bosdyn_msgs rev-parse HEAD
```
