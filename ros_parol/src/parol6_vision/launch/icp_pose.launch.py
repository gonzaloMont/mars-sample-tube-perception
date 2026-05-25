#!/usr/bin/env python3
"""
icp_pose.launch.py — Lanza el pipeline de visión con estimación de pose 6D
mediante la arquitectura híbrida YOLOv8 + ICP (Open3D).

Arranca:
  • yolo_camera_subscriber.py     -> /vision/detections   (bbox 2D del tubo)
  • icp_pose_estimator_node.py    -> /vision/tube_pose     (PoseStamped 6D)
                                     /pose_target_cuaterniones (compat. IK)

Uso:
  ros2 launch parol6_vision icp_pose.launch.py

  # con otro fichero de parámetros:
  ros2 launch parol6_vision icp_pose.launch.py params_file:=/ruta/mis_params.yaml

  # sin arrancar YOLO (si ya corre por otro lado):
  ros2 launch parol6_vision icp_pose.launch.py enable_yolo:=false

  # usando una imagen de profundidad en vez de PointCloud2:
  ros2 launch parol6_vision icp_pose.launch.py cloud_source:=depth

  # publicando la pose directamente en base_link:
  ros2 launch parol6_vision icp_pose.launch.py output_frame:=base_link

──────────────────────────────────────────────────────────────────────────────
COHERENCIA DE FRAMES
  La nube 3D, su camera_info y el bbox de YOLO deben venir de la MISMA cámara.
  En este repositorio la nube la genera stereo_pointcloud_node.py a partir de
  la NavCam estéreo, así que para este pipeline conviene:
    • lanzar Gazebo con  camera:=navcam
    • arrancar stereo_pointcloud_node.py (publica /navcam/pointcloud)
    • ejecutar YOLO sobre la imagen IZQUIERDA de la NavCam

  Este launch deja el tópico de imagen de YOLO como argumento `yolo_image_topic`
  para que apunte a la cámara correcta.
──────────────────────────────────────────────────────────────────────────────
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
    default_params = os.path.join(pkg_share, "config", "icp_pose_params.yaml")

    # ── Argumentos del launch ───────────────────────────────────────────────
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params,
        description="YAML de parametros del nodo ICP.")
    enable_yolo_arg = DeclareLaunchArgument(
        "enable_yolo", default_value="true",
        description="Arrancar tambien el detector YOLO.")
    enable_stereo_arg = DeclareLaunchArgument(
        "enable_stereo", default_value="false",
        description="Arrancar tambien stereo_pointcloud_node.py (genera la nube).")
    cloud_source_arg = DeclareLaunchArgument(
        "cloud_source", default_value="pointcloud",
        description="Fuente de la nube: 'pointcloud' o 'depth'.")
    output_frame_arg = DeclareLaunchArgument(
        "output_frame", default_value="",
        description="Frame de salida de la pose (vacio => frame de la nube).")
    stl_path_arg = DeclareLaunchArgument(
        "stl_path", default_value="",
        description="Ruta al .stl del tubo (vacio => parol6_gazebo/meshes).")
    yolo_image_topic_arg = DeclareLaunchArgument(
        "yolo_image_topic",
        default_value="/navcam/left/navcam_left/image_raw",
        description="Topico de imagen sobre el que corre YOLO.")
    yolo_executable_arg = DeclareLaunchArgument(
        "yolo_executable",
        default_value="yolo26n_camera_subscriber.py",
        description="Detector YOLO a usar. Por defecto: yolo26n (entrenado "
                    "para el tubo). Alternativa: yolo_camera_subscriber.py "
                    "(YOLOE open-vocab, requiere 'red tube').")

    # ── Nodo YOLO (reutiliza el detector existente) ─────────────────────────
    # El detector declara `image_topic` como parámetro ROS, así que en lugar
    # de un remap se le pasa directamente la cámara a usar (más robusto:
    # `ros2 node info` mostrará la suscripción real).
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

    # ── Nodo de nube estéreo (opcional) ─────────────────────────────────────
    stereo_node = Node(
        package="parol6_vision",
        executable="stereo_pointcloud_node.py",
        name="stereo_pointcloud",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_stereo")),
        parameters=[os.path.join(pkg_share, "config",
                                 "stereo_pointcloud_params.yaml")],
    )

    # ── Nodo ICP de estimación de pose 6D ───────────────────────────────────
    icp_node = Node(
        package="parol6_vision",
        executable="icp_pose_estimator_node.py",
        name="icp_pose_estimator_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "cloud_source": LaunchConfiguration("cloud_source"),
                "output_frame": LaunchConfiguration("output_frame"),
                "stl_path":     LaunchConfiguration("stl_path"),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        enable_yolo_arg,
        enable_stereo_arg,
        cloud_source_arg,
        output_frame_arg,
        stl_path_arg,
        yolo_image_topic_arg,
        yolo_executable_arg,
        yolo_node,
        stereo_node,
        icp_node,
    ])
