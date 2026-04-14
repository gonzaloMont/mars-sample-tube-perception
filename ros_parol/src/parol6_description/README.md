# parol6_description

Robot description package for PAROL6.

## Purpose

Contains robot geometry, Xacro/URDF files, and mesh resources used by Gazebo and MoveIt.

## Main contents

- `urdf/parol6_gazebo.xacro`: robot model configured for Gazebo simulation.
- `urdf/parol6.urdf`: generated URDF artifact.
- `urdf/overhead_camera.urdf.xacro`: overhead camera model.
- `urdf/gripper.urdf.xacro`: gripper description.
- `meshes/`: visual and collision geometry.

## Notes

- Keep names lowercase for robust ROS and MoveIt behavior.
- Any structural robot change should begin in this package.
