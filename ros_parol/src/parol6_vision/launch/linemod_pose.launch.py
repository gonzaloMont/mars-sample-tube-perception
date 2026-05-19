#!/usr/bin/env python3
"""
linemod_pose.launch.py — Lanza el pipeline de visión con estimación 6D LINEMOD.

Arranca:
  • yolo_camera_subscriber.py        -> /vision/detections, /vision/mask_image
  • linemod_pose_estimator_node.py   -> /vision/pose_estimates,
                                        /pose_target_cuaterniones

Uso:
  ros2 launch parol6_vision linemod_pose.launch.py

  # con una base de datos LINEMOD en otra ruta:
  ros2 launch parol6_vision linemod_pose.launch.py linemod_db_dir:=/ruta/db

  # sin arrancar YOLO (si ya corre por otro lado):
  ros2 launch parol6_vision linemod_pose.launch.py enable_yolo:=false

NOTA: este launch asume que Gazebo y la cámara cenital ya están corriendo
(p.ej. gazebo_moveit_mars.launch.py). La base de datos LINEMOD debe haberse
generado antes con scripts/train_linemod_templates.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("parol6_vision")
    default_params = os.path.join(pkg_share, "config", "linemod_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params,
        description="YAML de parámetros del nodo LINEMOD.")
    db_dir_arg = DeclareLaunchArgument(
        "linemod_db_dir", default_value="",
        description="Directorio de la base de datos LINEMOD "
                    "(vacío => share/parol6_vision/models/linemod).")
    enable_yolo_arg = DeclareLaunchArgument(
        "enable_yolo", default_value="true",
        description="Arrancar también el detector YOLO.")

    yolo_node = Node(
        package="parol6_vision",
        executable="yolo_camera_subscriber.py",
        name="yolo_detector_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_yolo")),
    )

    linemod_node = Node(
        package="parol6_vision",
        executable="linemod_pose_estimator_node.py",
        name="linemod_pose_estimator_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"linemod_db_dir": LaunchConfiguration("linemod_db_dir")},
        ],
    )

    return LaunchDescription([
        params_file_arg,
        db_dir_arg,
        enable_yolo_arg,
        yolo_node,
        linemod_node,
    ])
