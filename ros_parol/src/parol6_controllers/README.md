# parol6_controllers

Control helper nodes for PAROL6.

## Purpose

This package includes controller-side utilities that complement ros2_control and MoveIt.

## Main executable

- `virtual_gripper_node.py`: virtual grasp logic based on finger contacts and gripper joint state.

Run:

```bash
ros2 run parol6_controllers virtual_gripper_node.py
```

## Integration notes

- Works with Gazebo contact topics and entity state services.
- Uses TF to keep attached object aligned to `gripper_base_link`.
- Intended for simulation workflows when physical grasping is not available.
