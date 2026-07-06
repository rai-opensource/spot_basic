# `spot_basic` examples

Runnable scripts demonstrating the [`spot_basic`](../spot_basic) API. Each script takes the
robot name (the ROS namespace the driver runs under) as its first argument.

## Prerequisites

- The Spot ROS 2 driver must be running for the target robot.
- Source your workspace in the terminal you run examples from:

  ```bash
  cd ~/ws
  source install/setup.bash
  ```

## Running an example

From this directory, run any script with `python3` (they are also directly executable via
their shebang, e.g. `./example_walk_forward.py <robot_name>`):

```bash
cd ~/ws/src/spot_basic/examples
python3 example_walk_forward.py <robot_name>
```

Replace `<robot_name>` with your robot's namespace (e.g. `Nyla`). Pass `-h`/`--help` to any
script to see its full set of options.

## Getting started

| Example | Description | Run |
|---------|-------------|-----|
| [example_spot_basic_util.py](example_spot_basic_util.py) | Show basic `SpotBasic` use. | `python3 example_spot_basic_util.py <robot_name>` |

## Mobility

| Example | Description | Run |
|---------|-------------|-----|
| [example_walk_forward.py](example_walk_forward.py) | Have Spot walk forward synchronously. | `python3 example_walk_forward.py <robot_name>` |
| [example_walk_forward_async.py](example_walk_forward_async.py) | Have Spot walk forward asynchronously. | `python3 example_walk_forward_async.py <robot_name>` |
| [example_walk_velocity_with_duration.py](example_walk_velocity_with_duration.py) | Make Spot walk forward at a specified velocity. | `python3 example_walk_velocity_with_duration.py <robot_name> <vx> <vy> <vyaw>` |
| [example_walk_to_6dpose.py](example_walk_to_6dpose.py) | Have Spot walk to a specific pose. | `python3 example_walk_to_6dpose.py <robot_name> <x> <y> <z>` |
| [example_walk_to_object_image.py](example_walk_to_object_image.py) | Have Spot walk to an object. | `python3 example_walk_to_object_image.py <robot_name>` |
| [example_walk_to_object_ray_in_world.py](example_walk_to_object_ray_in_world.py) | Have Spot walk forward about 1.25 m. | `python3 example_walk_to_object_ray_in_world.py <robot_name>` |
| [example_trajectory_command.py](example_trajectory_command.py) | Have Spot walk along a trajectory. | `python3 example_trajectory_command.py <robot_name> {simple,basic_circle,counterspin_circle}` |
| [example_different_frame_movements.py](example_different_frame_movements.py) | Spot runs through different frame movements. | `python3 example_different_frame_movements.py <robot_name>` |

## Body posture

| Example | Description | Run |
|---------|-------------|-----|
| [example_stand_with_orientation.py](example_stand_with_orientation.py) | Have Spot rotate its body in place. | `python3 example_stand_with_orientation.py <robot_name>` |
| [example_body_gaze_at.py](example_body_gaze_at.py) | Spot gazes at a point in space with its body. | `python3 example_body_gaze_at.py <robot_name>` |

## Arm and manipulation

| Example | Description | Run |
|---------|-------------|-----|
| [example_arm_cartesian_trajectory.py](example_arm_cartesian_trajectory.py) | Spot moves its hand around (Cartesian trajectory). | `python3 example_arm_cartesian_trajectory.py <robot_name>` |
| [example_arm_joint_trajectory.py](example_arm_joint_trajectory.py) | Spot moves its arm around (joint trajectory). | `python3 example_arm_joint_trajectory.py <robot_name>` |
| [example_arm_force_trajectory.py](example_arm_force_trajectory.py) | Spot moves its arm around with controlled forces. | `python3 example_arm_force_trajectory.py <robot_name>` |
| [example_arm_gaze_at.py](example_arm_gaze_at.py) | Spot gazes at a point in space with its arm. | `python3 example_arm_gaze_at.py <robot_name>` |
| [example_arm_grasp.py](example_arm_grasp.py) | Spot grabs an object with its arm. | `python3 example_arm_grasp.py <robot_name>` |
| [example_arm_surface_contact.py](example_arm_surface_contact.py) | Use arm surface contact to draw a square. | `python3 example_arm_surface_contact.py <robot_name>` |
| [example_impedance_control.py](example_impedance_control.py) | Make circle-like trajectories using Spot arm impedance control. | `python3 example_impedance_control.py <robot_name>` |
| [example_move_ee_with_body_follow.py](example_move_ee_with_body_follow.py) | Move Spot arm end-effector and have the robot follow. | `python3 example_move_ee_with_body_follow.py <robot_name> <x> <y> <z>` |
| [example_ik_solutions.py](example_ik_solutions.py) | Query and apply IK solutions to Spot arm. | `python3 example_ik_solutions.py <robot_name>` |
| [example_zerog.py](example_zerog.py) | Put Spot arm in zero-gravity mode (i.e. low joint stiffness). | `python3 example_zerog.py <robot_name>` |

## Perception and cameras

| Example | Description | Run |
|---------|-------------|-----|
| [example_spot_image_viz.py](example_spot_image_viz.py) | Visualize RGB frame, depth map, and pointcloud from a Spot camera. | `python3 example_spot_image_viz.py <robot_name>` |
| [example_gripper_camera_parameters.py](example_gripper_camera_parameters.py) | Query and set Spot gripper camera parameters. | `python3 example_gripper_camera_parameters.py <robot_name>` |
| [example_create_point_cloud_in_place.py](example_create_point_cloud_in_place.py) | Spot takes point clouds from multiple cameras and stitches them together. | `python3 example_create_point_cloud_in_place.py <robot_name>` |
| [example_nogo_regions.py](example_nogo_regions.py) | Add and remove user nogo regions. | `python3 example_nogo_regions.py <robot_name>` |

## Joint-level control

These examples drive Spot through the joint-level (mixed) control API. Start the driver in
mixed-level (controllable) mode first — see the
[`spot_ros2_control` Mixed Level API documentation](https://github.com/rai-opensource/spot_ros2/blob/main/spot_ros2_control/README.md#mixed-level-api).

| Example | Description | Run |
|---------|-------------|-----|
| [example_controlled_squat.py](example_controlled_squat.py) | Have Spot squat in a controlled manner (forward position controller). | `python3 example_controlled_squat.py <robot_name>` |
| [example_joint_control_squat.py](example_joint_control_squat.py) | Have Spot squat in a controlled manner (joint-level control). | `python3 example_joint_control_squat.py <robot_name>` |

> **Note:** Arm examples require a Spot equipped with an arm. Some scripts accept extra
> options (poses, durations, trajectory names); run with `--help` for details.
