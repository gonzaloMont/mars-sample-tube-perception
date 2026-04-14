# parol6_apps

User-facing scripts and demos for PAROL6.

## Purpose

This package contains utility nodes, motion examples, and pick-and-place orchestration scripts.

## Main executables

- `get_pose.py`: reads and prints end-effector pose from TF.
- `pose_ik_subscriber.py`: receives pose targets and executes IK + MoveIt action.
- `pick_and_place_orchestrator.py`: full state-machine cycle for pick and place.
- `test_moveit.py`: simple MoveIt motion test.
- `test_moveit_cartesian.py`: Cartesian path test.

Run examples:

```bash
ros2 run parol6_apps get_pose.py
ros2 run parol6_apps pose_ik_subscriber.py
ros2 run parol6_apps pick_and_place_orchestrator.py
```

## Extra scripts

Under `scripts/demos`:

- `spawn_object.sh`
- `spawn_all_objects.sh`
- `test_tubo.sh`

## Inputs and outputs

Key channels used by this package include:

- Subscribes to pose targets (`/pose_target_rpy`, `/pose_target_cuaterniones`)
- Publishes robot motion status (`/robot_motion_status`)
- Consumes vision detections and mask topics in the orchestrator
