#!/usr/bin/env bash
set -euo pipefail

# Script para ejecutar la secuencia de prueba del robot.
# Ejecutar: ./test_tubo.sh


# Obtener la ruta correcta al tube.sdf desde el paquete instalado
TUBE_SDF="$(ros2 pkg prefix --share parol6_gazebo)/models/tube.sdf"

# Spawnear el tubo en Gazebo
ros2 run gazebo_ros spawn_entity.py -file "$TUBE_SDF" -entity tube -x 0.32 -y 0.0 -z 0.03 -P 1.5708

sleep 1


# Mueve el brazo a donde está el tubo 
ros2 topic pub --once /arm_group_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{joint_names: ['L1','L2','L3','L4','L5','L6'], points: [{positions: [-0.0075, 0.8158, 0.3848, -0.0019, -1.1980, 0.0112], time_from_start: {sec: 3}}]}"

sleep 2


# Cerrar el gripper
ros2 topic pub --once /gripper_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{joint_names: ['gripper_joint','gripper_right_joint'], points: [{positions: [0.01, 0.01], time_from_start: {sec: 1}}]}"

sleep 2

# Volver a postura home
ros2 topic pub --once /arm_group_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{joint_names: ['L1','L2','L3','L4','L5','L6'], points: [{positions: [0, 0.0, 0.0, -0.0, -0, 0.0], time_from_start: {sec: 3}}]}"
