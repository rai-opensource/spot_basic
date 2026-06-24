#!/bin/bash
# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Installs spot_basic and all of its source and system dependencies into a colcon
# workspace, then builds it.
#
# Assumes this repository lives at <workspace>/src/spot_basic, i.e.:
#
#   mkdir -p ~/spot_ws/src && cd ~/spot_ws
#   git clone https://github.com/rai-opensource/spot_basic.git src/spot_basic
#   ./src/spot_basic/install_spot_basic.sh
#
# Steps performed:
#   1. Install host tooling (rosdep, vcstool, pip, git, wget).
#   2. Fetch source dependencies declared in spot_basic.repos (spot_ros2 with its
#      synchros2/spot_wrapper submodules, bosdyn_msgs) via vcstool.
#   3. Run the spot_ros2 installer (spot_wrapper requirements, bosdyn_msgs,
#      spot-cpp-sdk, and its own rosdep keys).
#   4. Install spot_basic PyPI dependencies (requirements.txt).
#   5. Resolve remaining rosdep dependencies and colcon build the workspace.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# spot_basic is expected at <workspace>/src/spot_basic.
SRC_DIR=$(dirname "$SCRIPT_DIR")
WS_DIR=$(dirname "$SRC_DIR")

ROS_DISTRO="${ROS_DISTRO:-humble}"
ARCH="amd64"
DO_BUILD=1

HELP=$'Usage: install_spot_basic.sh [options]

Options:
  --arm64, --aarch64   Install the ARM64 builds of binary dependencies.
  --ros-distro <name>  ROS 2 distribution to target (default: humble, or $ROS_DISTRO).
  --no-build           Skip the final colcon build step.
  -h, --help           Show this help message and exit.'

while [ $# -gt 0 ]; do
  case "$1" in
    --arm64 | --aarch64 ) ARCH="arm64"; shift ;;
    --ros-distro ) ROS_DISTRO="$2"; shift 2 ;;
    --no-build ) DO_BUILD=0; shift ;;
    -h | --help ) echo "$HELP"; exit 0 ;;
    -- ) shift; break ;;
    * ) echo "Unknown option: $1" >&2; echo "$HELP" >&2; exit 1 ;;
  esac
done

ARCH_FLAG=""
if [ "$ARCH" = "arm64" ]; then
  ARCH_FLAG="--arm64"
fi

echo "==> Installing host tooling"
sudo apt-get update
sudo apt-get install -y python3-rosdep python3-vcstool python3-pip git wget

echo "==> Fetching source dependencies into $SRC_DIR"
vcs import --recursive "$SRC_DIR" < "$SCRIPT_DIR/spot_basic.repos"

echo "==> Running the spot_ros2 installer"
SPOT_ROS2_INSTALLER="$SRC_DIR/spot_ros2/install_spot_ros2.sh"
if [ -f "$SPOT_ROS2_INSTALLER" ]; then
  ( cd "$SRC_DIR/spot_ros2" && ./install_spot_ros2.sh $ARCH_FLAG )
else
  echo "ERROR: $SPOT_ROS2_INSTALLER not found. Did 'vcs import' fail?" >&2
  exit 1
fi

echo "==> Installing spot_basic PyPI dependencies"
pip3 install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

echo "==> Resolving rosdep dependencies"
# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO/setup.bash"
if ! [ -d /etc/ros/rosdep/sources.list.d ] || ! ls /etc/ros/rosdep/sources.list.d/*default.list > /dev/null 2>&1; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths "$SRC_DIR" --ignore-src -y -r --rosdistro="$ROS_DISTRO"

if [ "$DO_BUILD" -eq 1 ]; then
  echo "==> Building the workspace at $WS_DIR"
  ( cd "$WS_DIR" && colcon build --symlink-install )
  echo "==> Done. Source the workspace with: source $WS_DIR/install/setup.bash"
else
  echo "==> Skipping build (--no-build). To build later:"
  echo "    cd $WS_DIR && colcon build --symlink-install"
fi
