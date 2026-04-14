# Inicio rápido

## 1. Requisitos

- Ubuntu 22.04
- ROS 2 Humble
- MoveIt 2

## 2. Compilar el workspace

Desde la raíz del workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 3. Ejecutar demos principales

### Demo MoveIt con RViz

```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch parol6_moveit demo.launch.py
```

### Simulación Gazebo

```bash
ros2 launch parol6_gazebo gazebo.launch.py
```

### Simulación completa: Gazebo + MoveIt

```bash
ros2 launch parol6_gazebo gazebo_moveit.launch.py
```

### Variante mundo Mars

```bash
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py
```
