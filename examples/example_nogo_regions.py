#!/usr/bin/env python
# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse

import synchros2.process as ros_process
from spatialmath import SE3

from spot_basic import FrameHint, SpotBasic


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Add and remove user nogo regions")
    parser.add_argument("robot_name", help="Name of robot to use")

    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> None:
    """Main"""
    spot = SpotBasic(args.robot_name)

    # Do this first so we start clean
    spot.delete_all_user_nogo_regions.synchronously()

    input(
        "Pull up the top down view on the tablet so you can watch the nogo regions being added and removed.  "
        "Note that regions get merged into each other so things may not look like perfect boxes and circles."
    )

    # Add a box 1m in front of the robot that extends for 0.5m in x and to 1m in either side in y
    spot.add_user_nogo_box.synchronously(
        box_name="nogo_box",
        frame=FrameHint.VISION,
        frame_t_origin=spot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True)
        * SE3(x=[1.0, 0.0, 0.0]),
        min_bounds=[0.0, -1.0],
        max_bounds=[0.5, 1.0],
    )

    # Add a box 2m behind the robot that extends 0.5m towards the robot in x (note the box orientation is the same as
    # the robot's body) and to 1m in either side in y
    spot.add_user_nogo_box.synchronously(
        box_name="nogo_box2",
        frame=FrameHint.VISION,
        frame_t_origin=spot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True)
        * SE3(x=[-2.0, 0.0, 0.0]),
        min_bounds=[0.0, -1.0],
        max_bounds=[0.5, 1.0],
    )

    input("There should now be nogo box regions in front of and behind spot.")

    # Add a circle 1m to the left of Spot with a radius of 0.5m
    resp = spot.add_user_nogo_circle.synchronously(
        circle_name="nogo_circle",
        frame=FrameHint.VISION,
        frame_t_center=spot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True)
        * SE3(x=[0.0, 1.0, 0.0]),
        radius=0.5,
    )
    circle_id = resp.response.mutated_object_id
    input("Now you should also see a circle to the left of spot.")

    print(f"User nogo regions are: {spot.get_user_nogo_regions_as_string()}")

    spot.delete_user_nogo_region.synchronously(circle_id)

    input("The circle should now be deleted")

    spot.delete_all_user_nogo_regions.synchronously()

    input("All user nogo regions should be deleted")

    # Re-add the box in front of spot
    spot.add_user_nogo_box.synchronously(
        box_name="nogo_box",
        frame=FrameHint.VISION,
        frame_t_origin=spot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True)
        * SE3(x=[1.0, 0.0, 0.0]),
        min_bounds=[0.0, -1.0],
        max_bounds=[0.5, 1.0],
    )

    # Re-add the box behind spot
    spot.add_user_nogo_box.synchronously(
        box_name="nogo_box",
        frame=FrameHint.VISION,
        frame_t_origin=spot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True)
        * SE3(x=[-2.0, 0.0, 0.0]),
        min_bounds=[0.0, -1.0],
        max_bounds=[0.5, 1.0],
    )

    # Re-add the circle to the left of spot
    spot.add_user_nogo_circle.synchronously(
        circle_name="nogo_circle",
        frame=FrameHint.VISION,
        frame_t_center=spot.lookup_a_tform_b(FrameHint.VISION, FrameHint.BODY, wait_for_frames=True)
        * SE3(x=[0.0, 1.0, 0.0]),
        radius=0.5,
    )

    input("Now they're all back!")

    spot.delete_all_user_nogo_regions_with_name.synchronously("nogo_box")

    input("Both boxes should be gone")

    spot.delete_all_user_nogo_regions.synchronously()

    print("And now everything is gone.  All remaining world objects are:")

    resp = spot.list_world_objects.synchronously()
    for wo in resp.response.world_objects:
        print(wo.name, wo.id)


if __name__ == "__main__":
    main()
