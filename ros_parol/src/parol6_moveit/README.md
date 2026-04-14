# parol6_moveit

MoveIt 2 configuration package for PAROL6.

## Purpose

Contains planning, kinematics, controllers, and launch configuration required for MoveIt.

## Main launch files

- `demo.launch.py`: standard MoveIt demo flow.
- `move_group.launch.py`: launches only `move_group`.
- `moveit_rviz.launch.py`: MoveIt RViz setup.
- `rsp.launch.py`: robot state publisher entry.
- `spawn_controllers.launch.py`: controller spawning.

Run example:

```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch parol6_moveit demo.launch.py
```

## Main config files

- `config/parol6.srdf`
- `config/kinematics.yaml`
- `config/joint_limits.yaml`
- `config/moveit_controllers.yaml`
- `config/moveit.rviz`
