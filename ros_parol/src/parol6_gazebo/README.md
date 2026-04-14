# parol6_gazebo

Gazebo simulation package for PAROL6.

## Purpose

Provides simulation worlds, object models, controller config, and launch files for Gazebo-based execution.

## Main launch files

- `gazebo.launch.py`: pure Gazebo simulation with robot and camera.
- `gazebo_moveit.launch.py`: Gazebo + MoveIt integrated flow.
- `gazebo_moveit_mars.launch.py`: integrated flow for Mars world with configurable spawn positions.

Run examples:

```bash
ros2 launch parol6_gazebo gazebo.launch.py
ros2 launch parol6_gazebo gazebo_moveit.launch.py rviz:=true gripper:=true ik:=true
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py rviz:=true robot_x:=10.311 robot_y:=5.066 robot_z:=0.8
```

## Main folders

- `worlds/`: Gazebo worlds.
- `models/`: SDF objects used in demos.
- `config/`: ros2_control and controller settings.
- `launch/`: simulation entry points.
