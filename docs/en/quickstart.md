# Quickstart

## 1. Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- MoveIt 2

## 2. Build the workspace

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 3. Run the main demos

### MoveIt RViz demo

```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch parol6_moveit demo.launch.py
```

### Gazebo simulation

```bash
ros2 launch parol6_gazebo gazebo.launch.py
```

### Full simulation: Gazebo + MoveIt

```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py
```

### Mars world variant

```bash
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py
```
