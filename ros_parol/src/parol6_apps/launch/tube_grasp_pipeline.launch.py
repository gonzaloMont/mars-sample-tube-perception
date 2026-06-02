#!/usr/bin/env python3
"""
tube_grasp_pipeline.launch.py — pipeline END-TO-END visión → IK del tubo.

Encadena:
  1. parol6_vision/icp_pose.launch.py  → publica `/vision/tube_pose` (PoseStamped 6D
                                          del tubo en `navcam_left_optical_frame`).
  2. tube_grasp_pose_node.py           → convierte la pose del tubo en una pose
                                          de agarre del gripper en `base_link`
                                          y la publica como
                                            `/grasp/pick_pose`     (PoseStamped)
                                            `/grasp/approach_pose` (PoseStamped)
                                          y, opcionalmente,
                                            `/pose_target_cuaterniones` (Pose).
  3. (opcional) pose_ik_subscriber.py  → cuando `enable_ik:=true`, consume
                                          `/pose_target_cuaterniones` y mueve
                                          el brazo con MoveIt.

PRE-REQUISITOS

  • Gazebo + MoveIt arrancados con la NavCam estéreo:
      ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py camera:=navcam

  • Nube estéreo:
      ros2 run parol6_vision stereo_pointcloud_node.py

  • YOLO activo (este launch lo arranca dentro de icp_pose.launch.py por defecto).
    Recuerda publicar 'activate' en /orchestrator/vision_activate la primera vez:
      ros2 topic pub --once /orchestrator/vision_activate std_msgs/String '{data: activate}'

USO

  # Solo visión + nodo de agarre (no toca el robot):
  ros2 launch parol6_apps tube_grasp_pipeline.launch.py

  # Visión + agarre + IK + MoveIt (mueve el brazo). Asegúrate de que
  # pose_ik_subscriber acepta el `legacy_tip_link` que pases (por defecto
  # 'L6_link' es compatible con el `pose_ik_subscriber.py` actual del repo).
  ros2 launch parol6_apps tube_grasp_pipeline.launch.py \
      enable_ik:=true publish_legacy_pose:=true

  # Sin re-arrancar el detector YOLO si ya está corriendo en otro launch:
  ros2 launch parol6_apps tube_grasp_pipeline.launch.py enable_yolo:=false

NOTA

  Por defecto, `icp_pose.launch.py` arranca con `output_frame=""`, lo que hace
  que el ICP publique la pose del tubo en `navcam_left_optical_frame`.
  `tube_grasp_pose_node` se encarga de transformarla a `base_link` con TF2.
  Si prefieres que ya venga en `base_link`, lanza este launch con
  `output_frame:=base_link` (se reenvía a icp_pose.launch.py).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_vision = get_package_share_directory('parol6_vision')

    # ── Argumentos ────────────────────────────────────────────────────────────
    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision', default_value='true',
        description='Arrancar el pipeline ICP de visión (icp_pose.launch.py).')
    enable_yolo_arg = DeclareLaunchArgument(
        'enable_yolo', default_value='true',
        description='Si enable_vision=true, arrancar también el detector YOLO.')
    enable_stereo_arg = DeclareLaunchArgument(
        'enable_stereo', default_value='false',
        description='Arrancar también stereo_pointcloud_node.py.')
    enable_ik_arg = DeclareLaunchArgument(
        'enable_ik', default_value='false',
        description='Arrancar pose_ik_subscriber.py para mover el brazo.')
    enable_grasp_executor_arg = DeclareLaunchArgument(
        'enable_grasp_executor', default_value='false',
        description='Arrancar grasp_executor.py, que ejecuta approach→pick '
                    'con fallback automático a la variante alt cuando la IK '
                    'rechaza la primary. Expone /grasp/execute (std_srvs/Trigger).')
    grasp_executor_try_alt_arg = DeclareLaunchArgument(
        'grasp_executor_try_alt', default_value='true',
        description='Si enable_grasp_executor=true, hacer fallback a la '
                    'variante alt cuando la primary falla la IK.')
    output_frame_arg = DeclareLaunchArgument(
        'output_frame', default_value='',
        description='Frame en el que el nodo ICP publica /vision/tube_pose '
                    '(vacío => navcam_left_optical_frame).')

    # ── Parámetros del nodo de agarre ────────────────────────────────────────
    base_frame_arg = DeclareLaunchArgument(
        'base_frame', default_value='base_link',
        description='Frame del robot al que se transforma la pose del tubo.')
    grasp_tip_link_arg = DeclareLaunchArgument(
        'grasp_tip_link', default_value='gripper_base_link',
        description='Tip al que se refieren /grasp/pick_pose y /grasp/approach_pose.')
    approach_height_arg = DeclareLaunchArgument(
        'approach_height', default_value='0.10',
        description='Altura de la pose de aproximación sobre el pick (m, Z_world).')
    finger_offset_z_arg = DeclareLaunchArgument(
        'finger_offset_z', default_value='0.035',
        description='Offset gripper_base_link → centro entre dedos (m).')
    tilt_threshold_arg = DeclareLaunchArgument(
        'tilt_threshold_deg', default_value='70.0',
        description='Inclinación máxima admitida entre z_g y +Z_world (deg).')
    publish_legacy_arg = DeclareLaunchArgument(
        'publish_legacy_pose', default_value='false',
        description='Publicar /pose_target_cuaterniones para pose_ik_subscriber.py.')
    legacy_tip_arg = DeclareLaunchArgument(
        'legacy_tip_link', default_value='L6_link',
        description="Tip que asume el consumidor legacy ('L6_link' o "
                    "'gripper_base_link').")

    # ── Visión (pipeline ICP existente) ──────────────────────────────────────
    # Importante: pasamos `publish_legacy_pose=false` al ICP para que ESTE
    # nodo (tube_grasp_pose_node) sea el ÚNICO publicador en
    # /pose_target_cuaterniones. Si el ICP también publicase ahí, el
    # pose_ik_subscriber recibiría dos poses por ciclo: la del tubo (frame
    # óptico, sin conversión a grasp) y la del agarre — y resolvería IK
    # sobre ambas alternándose.
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_vision, 'launch', 'icp_pose.launch.py')),
        launch_arguments={
            'enable_yolo':             LaunchConfiguration('enable_yolo'),
            'enable_stereo':           LaunchConfiguration('enable_stereo'),
            'output_frame':            LaunchConfiguration('output_frame'),
            # Distinto nombre que el del padre para no contaminar el contexto:
            'icp_publish_legacy_pose': 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_vision')),
    )

    # ── Nodo nuevo de conversión pose-del-tubo → pose-de-agarre ──────────────
    # ParameterValue(value_type=…) convierte el `LaunchConfiguration` (string)
    # al tipo declarado por el nodo. Sin esto, ROS 2 ignora el override y se
    # queda con el default declarado en declare_parameter().
    grasp_node = Node(
        package='parol6_apps',
        executable='tube_grasp_pose_node.py',
        name='tube_grasp_pose_node',
        output='screen',
        parameters=[{
            'tube_pose_topic':     '/vision/tube_pose',
            'pick_pose_topic':     '/grasp/pick_pose',
            'approach_pose_topic': '/grasp/approach_pose',
            'legacy_pose_topic':   '/pose_target_cuaterniones',
            'base_frame':           LaunchConfiguration('base_frame'),
            'grasp_tip_link':       LaunchConfiguration('grasp_tip_link'),
            'approach_height':      ParameterValue(
                LaunchConfiguration('approach_height'), value_type=float),
            'finger_offset_z':      ParameterValue(
                LaunchConfiguration('finger_offset_z'), value_type=float),
            'tilt_threshold_deg':   ParameterValue(
                LaunchConfiguration('tilt_threshold_deg'), value_type=float),
            'publish_legacy_pose':  ParameterValue(
                LaunchConfiguration('publish_legacy_pose'), value_type=bool),
            'legacy_tip_link':      LaunchConfiguration('legacy_tip_link'),
        }],
    )

    # ── Consumidor del topic legacy: pose_ik_subscriber.py (opcional) ────────
    # Solo ejecuta UNA pose (la principal) sin fallback. Adecuado para
    # debugging manual o integraciones simples.
    ik_node = Node(
        package='parol6_apps',
        executable='pose_ik_subscriber.py',
        name='pose_ik_subscriber',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_ik')),
    )

    # ── Ejecutor con fallback primary→alt: grasp_executor.py (opcional) ──────
    # Consume las 4 poses publicadas por tube_grasp_pose_node y ejecuta
    # approach→pick con fallback automático. Más robusto que pose_ik_subscriber
    # para usos productivos; se dispara vía servicio /grasp/execute (Trigger).
    grasp_executor_node = Node(
        package='parol6_apps',
        executable='grasp_executor.py',
        name='grasp_executor',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_grasp_executor')),
        parameters=[{
            'arm_group_name':    'arm_group',
            'tip_link':           LaunchConfiguration('grasp_tip_link'),
            'try_alt_on_failure': ParameterValue(
                LaunchConfiguration('grasp_executor_try_alt'), value_type=bool),
        }],
    )

    return LaunchDescription([
        enable_vision_arg,
        enable_yolo_arg,
        enable_stereo_arg,
        enable_ik_arg,
        enable_grasp_executor_arg,
        grasp_executor_try_alt_arg,
        output_frame_arg,
        base_frame_arg,
        grasp_tip_link_arg,
        approach_height_arg,
        finger_offset_z_arg,
        tilt_threshold_arg,
        publish_legacy_arg,
        legacy_tip_arg,
        vision_launch,
        grasp_node,
        ik_node,
        grasp_executor_node,
    ])
