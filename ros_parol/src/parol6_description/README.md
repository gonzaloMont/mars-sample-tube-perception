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

## NavCam stereo model

The rover description now includes a stereo navigation camera pair that follows the documented SFR/ExoMars-style requirements:

- monocular output in mono8
- 1024 x 1024 pixels per camera
- 65° horizontal FOV
- 5.0 Hz update rate
- 0.15 m baseline along Y between left and right optical centers

Gazebo publishes:

- /navcam/left/image_raw
- /navcam/left/camera_info
- /navcam/right/image_raw
- /navcam/right/camera_info

To keep the original overhead camera as the default, the NavCam rig is enabled from launch files with `use_navcam:=true`.

## Notes

- Keep names lowercase for robust ROS and MoveIt behavior.
- Any structural robot change should begin in this package.
