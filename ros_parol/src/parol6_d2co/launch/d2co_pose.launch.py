#!/usr/bin/env python3
"""
d2co_pose.launch.py — Lanza el pipeline de visión con estimación de pose 6D
mediante D2CO (Direct Directional Chamfer Optimization) sobre la NavCam.

Arranca:
  • yolo26n_camera_subscriber.py  -> /vision/detections   (bbox 2D del tubo)
  • d2co_pose_estimator_node      -> /vision/tube_pose    (PoseStamped 6D)

Es el equivalente edge-based de parol6_vision/icp_pose.launch.py: publica el
MISMO contrato (/vision/tube_pose + TF tube_link_estimated), por lo que
tube_grasp_pipeline.launch.py funciona sin cambios aguas abajo. A diferencia
del ICP, NO necesita nube de puntos estéreo: solo la imagen izquierda.

Uso:
  ros2 launch parol6_d2co d2co_pose.launch.py

  # sin arrancar YOLO (si ya corre por otro lado):
  ros2 launch parol6_d2co d2co_pose.launch.py enable_yolo:=false

  # publicando la pose directamente en base_link:
  ros2 launch parol6_d2co d2co_pose.launch.py output_frame:=base_link

Recuerda activar YOLO la primera vez:
  ros2 topic pub --once /orchestrator/vision_activate std_msgs/String '{data: activate}'
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("parol6_d2co")
    default_params = os.path.join(pkg_share, "config", "d2co_pose_params.yaml")

    # ── Argumentos del launch ───────────────────────────────────────────────
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params,
        description="YAML de parametros del nodo D2CO.")
    enable_yolo_arg = DeclareLaunchArgument(
        "enable_yolo", default_value="true",
        description="Arrancar tambien el detector YOLO.")
    output_frame_arg = DeclareLaunchArgument(
        "output_frame", default_value="world",
        description="Frame de salida de la pose (vacio => frame optico).")
    stl_path_arg = DeclareLaunchArgument(
        "stl_path", default_value="",
        description="Ruta al STL PRE-ESCALADO (vacio => models/ del paquete).")
    yolo_image_topic_arg = DeclareLaunchArgument(
        "yolo_image_topic",
        default_value="/navcam/left/navcam_left/image_raw",
        description="Topico de imagen sobre el que corre YOLO.")
    yolo_executable_arg = DeclareLaunchArgument(
        "yolo_executable",
        default_value="yolo26n_camera_subscriber.py",
        description="Detector YOLO a usar (yolo26n entrenado para el tubo).")

    # ── Nodo YOLO (reutiliza el detector existente de parol6_vision) ────────
    yolo_node = Node(
        package="parol6_vision",
        executable=LaunchConfiguration("yolo_executable"),
        name="yolo_detector_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_yolo")),
        parameters=[{
            "image_topic": LaunchConfiguration("yolo_image_topic"),
        }],
    )

    # ── Nodo D2CO de estimación de pose 6D ──────────────────────────────────
    d2co_node = Node(
        package="parol6_d2co",
        executable="d2co_pose_estimator_node",
        name="d2co_pose_estimator_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "output_frame": LaunchConfiguration("output_frame"),
                "stl_path":     LaunchConfiguration("stl_path"),
                "image_topic":  LaunchConfiguration("yolo_image_topic"),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        enable_yolo_arg,
        output_frame_arg,
        stl_path_arg,
        yolo_image_topic_arg,
        yolo_executable_arg,
        yolo_node,
        d2co_node,
    ])
