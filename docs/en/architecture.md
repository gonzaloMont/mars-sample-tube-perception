# Architecture overview

The workspace is split into small packages so each responsibility is easy to find and maintain.

## Package roles

### parol6_description
Contains the robot model, Xacro files, meshes, and other geometry assets. This package is the source of truth for how the robot is described.

### parol6_gazebo
Contains the Gazebo worlds, model files, controller parameters, and launch files used to simulate the robot.

### parol6_moveit
Contains the MoveIt 2 configuration, RViz settings, and launch files that bring up planning and visualization.

### parol6_controllers
Contains control-related nodes, including the virtual gripper behavior.

### parol6_apps
Contains user-facing demos and orchestration scripts used to exercise the robot in simulation or through ROS nodes.

### parol6_vision
Contains the camera and object-detection pipeline used to estimate target poses.

### parol6_interfaces
Contains custom ROS message definitions shared across the rest of the workspace.

## Data flow

The most common runtime path is:

1. `parol6_description` defines the robot model.
2. `parol6_gazebo` or `parol6_moveit` launches the simulation or planning environment.
3. `parol6_controllers` manages robot-specific control behavior.
4. `parol6_vision` estimates target poses from the camera.
5. `parol6_apps` coordinates demos, testing, and pick-and-place workflows.

## Folder conventions

- Launch files live in each package's `launch/` directory.
- Python ROS nodes live in `scripts/` or `nodos/` depending on the package.
- Static assets such as meshes, worlds, and models are stored next to the package that uses them.
