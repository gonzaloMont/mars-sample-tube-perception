# PAROL6-ROS2-MOVEIT

This repository provides ROS 2 MoveIt 2 integration for the [PAROL6](https://github.com/PCrnjak/PAROL6-Desktop-robot-arm) 6-DOF open-source robotic arm. It includes a simulation demo, URDF models, and launch files for testing and planning.

![PAROL6 Demo](MEDIA/PAROL6_moveit_gif.gif)


---

## Package structure

| Package | Type | Description |
|---|---|---|
| `parol6_description` | ament_cmake | URDF, xacro files and STL meshes |
| `parol6_gazebo` | ament_cmake | Gazebo worlds, SDF models, controller config and simulation launch files |
| `parol6_moveit` | ament_cmake | MoveIt 2 configuration (SRDF, kinematics, planners, launch files) |
| `parol6_controllers` | ament_cmake | Control nodes (virtual gripper with contact detection) |
| `parol6_apps` | ament_cmake | User scripts: IK subscriber, MoveIt tests, pose getter |
| `parol6_vision` | ament_cmake | Vision processing and camera integration | 


---

## 🛠️ Installation & Setup

**Tested on:**
- Ubuntu 22.04
- ROS 2 Humble

### 1. Install ROS 2 Humble and MoveIt 2

You can follow [this guide](https://aleksandarhaber.com/how-to-properly-install-moveit2-in-ros2-humble-and-fix-tutorial-errors/) or install via APT:

```bash
sudo apt update
sudo apt install ros-humble-desktop ros-humble-moveit
```

### 1. Create and build a workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/PCrnjak/PAROL6-ROS2-MOVEIT.git 
## OR Copy this repo manually to ros2_ws/src
cd ~/ros2_ws
colcon build
```

### 1.  Source the environment

```bash
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 🚀 Run the Simulation

### MoveIt 2 demo (RViz, no Gazebo)
```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch parol6_moveit demo.launch.py
```

### Gazebo simulation only
```bash
ros2 launch parol6_gazebo gazebo.launch.py
```

### Gazebo + MoveIt 2 (complete simulation)
```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py
```


### 📚 Helpful Resources
- [Link1](https://www.youtube.com/watch?v=M2yiVbJmzKY&list=PLeEzO_sX5H6TNMBiworxO8RpQlcYfl54y&index=10)
- [Link2](https://www.youtube.com/watch?v=nZqTdzGAfYs)


### 🛠️ Troubleshoot

Fix common MoveIt 2 issues : 

```bash
sudo apt install --reinstall ros-humble-geometric-shapes
```

MoveIt Setup Assistant (Optional):
```bash
ros2 run moveit_setup_assistant moveit_setup_assistant
ros2 launch moveit_robot_arm_sim demo.launch.py
```

⚠️ Note: All URDF/XACRO names must be lowercase to avoid issues!
⚠️ Note: - [Common_errors](https://robotics.stackexchange.com/questions/103300/error-loading-custom-robotic-manipulator-model-in-moveit2-on-ros2-humbleerror-l)


