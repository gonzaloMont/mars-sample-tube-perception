# ROS packages index

This directory contains all ROS 2 packages for the PAROL6 workspace.

## Packages

- [parol6_apps](parol6_apps/README.md): user scripts, demos, and orchestration.
- [parol6_controllers](parol6_controllers/README.md): controller-side helpers.
- [parol6_description](parol6_description/README.md): robot URDF/Xacro and meshes.
- [parol6_gazebo](parol6_gazebo/README.md): simulation launch files, worlds, and models.
- [parol6_interfaces](parol6_interfaces/README.md): custom ROS messages.
- [parol6_moveit](parol6_moveit/README.md): MoveIt configuration and launch files.
- [parol6_vision](parol6_vision/README.md): camera, detection, and pose estimation nodes.

## Typical workflow

1. Launch simulation with `parol6_gazebo`.
2. Launch planning with `parol6_moveit` or use the integrated Gazebo + MoveIt flow.
3. Run application workflows with `parol6_apps`.
4. Enable the vision pipeline with `parol6_vision` when required.
