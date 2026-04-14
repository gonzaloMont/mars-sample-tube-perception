# Guía de operaciones

Documento de referencia práctica para ejecutar y depurar el workspace PAROL6.

## Preparación de entorno

Desde la raíz del workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Flujos principales de lanzamiento

### 1) Solo MoveIt (RViz)

```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch parol6_moveit demo.launch.py
```

### 2) Solo Gazebo

```bash
ros2 launch parol6_gazebo gazebo.launch.py
```

### 3) Gazebo + MoveIt

```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py
```

Argumentos opcionales:

```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py rviz:=true gripper:=true ik:=true
```

### 4) Gazebo + MoveIt en Mars

```bash
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py rviz:=true
```

## Recetario de nodos

### Paquete apps

```bash
ros2 run parol6_apps get_pose.py
ros2 run parol6_apps pose_ik_subscriber.py
ros2 run parol6_apps pick_and_place_orchestrator.py
```

### Paquete vision

```bash
ros2 run parol6_vision yolo_camera_subscriber.py
ros2 run parol6_vision pose_estimator_node.py
```

### Paquete controllers

```bash
ros2 run parol6_controllers virtual_gripper_node.py
```
