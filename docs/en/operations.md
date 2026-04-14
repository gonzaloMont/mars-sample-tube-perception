# Operations guide

This document is a practical command reference for running and debugging the PAROL6 workspace.

## Environment setup

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Main launch flows

### 1) MoveIt only (RViz)

```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch parol6_moveit demo.launch.py
```

### 2) Gazebo only

```bash
ros2 launch parol6_gazebo gazebo.launch.py
```

### 3) Gazebo + MoveIt

```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py
```

Optional launch arguments:

```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py rviz:=true gripper:=true ik:=true
```

### 4) Gazebo + MoveIt in Mars world

```bash
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py rviz:=true
```

## Node cookbook

### Apps package

```bash
ros2 run parol6_apps get_pose.py
ros2 run parol6_apps pose_ik_subscriber.py
ros2 run parol6_apps pick_and_place_orchestrator.py
```

### Vision package

```bash
ros2 run parol6_vision yolo_camera_subscriber.py
ros2 run parol6_vision pose_estimator_node.py
```

### Controllers package

```bash
ros2 run parol6_controllers virtual_gripper_node.py
```
