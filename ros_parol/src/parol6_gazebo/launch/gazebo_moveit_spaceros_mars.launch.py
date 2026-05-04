"""
gazebo_moveit_spaceros_mars.launch.py
=====================================
Lanza la simulación completa de PAROL6 en el entorno marciano de SpaceROS,
usando **Gazebo Classic** (gazebo_ros) + MoveIt 2.

Integración SpaceROS → Gazebo Classic:
  · El mundo `spaceros_mars.world` usa la malla `martian_surface.glb` del repo
    spaceros_gz_demos, adaptada para Gazebo Classic (URI "model://" en vez de
    "models://", plugins libgazebo_ros_*.so en vez de gz-sim-*).
  · GAZEBO_MODEL_PATH se extiende con la carpeta de modelos de parol6_gazebo
    para que Gazebo encuentre el modelo martian_surface.
  · El resto del stack (robot_state_publisher, controllers, MoveIt, RViz,
    cámara cenital) es idéntico a gazebo_moveit_mars.launch.py.

Uso:
  # Mínimo (sin RViz, sin gripper virtual, sin IK)
  ros2 launch parol6_gazebo gazebo_moveit_spaceros_mars.launch.py

  # Con RViz y gripper virtual
  ros2 launch parol6_gazebo gazebo_moveit_spaceros_mars.launch.py rviz:=true gripper:=true

  # Con posición personalizada del robot sobre el terreno
  ros2 launch parol6_gazebo gazebo_moveit_spaceros_mars.launch.py \\
      robot_x:=10.0 robot_y:=5.0 robot_z:=0.5

Nota sobre robot_z:
  El terreno marciano no es plano. Ajusta robot_z hasta que el robot quede
  apoyado sobre la superficie (típicamente entre 0.3 y 1.5 m dependiendo de
  la zona del terreno elegida). Puedes usar el spawn_object.sh para primero
  soltar un cubo y medir a qué z queda apoyado.
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # ── Paquetes ────────────────────────────────────────────────────────────
    pkg_parol6_desc    = get_package_share_directory('parol6_description')
    pkg_parol6_gazebo  = get_package_share_directory('parol6_gazebo')
    pkg_gazebo_ros     = get_package_share_directory('gazebo_ros')
    pkg_parol6_moveit  = get_package_share_directory('parol6_moveit')

    # ── Mundo marciano SpaceROS ─────────────────────────────────────────────
    world_file = os.path.join(pkg_parol6_gazebo, 'worlds', 'spaceros_mars.world')

    # ── Ruta de modelos Gazebo ──────────────────────────────────────────────
    # Necesitamos que Gazebo encuentre:
    #   1. parol6_description (mallas del robot)
    #   2. parol6_gazebo/models (martian_surface y otros objetos de la escena)
    gazebo_models_path = ':'.join([
        os.path.join(pkg_parol6_gazebo, 'models'),       # martian_surface, tubes, etc.
        os.path.join(pkg_parol6_desc, '..'),              # parol6_description
    ])

    # SetEnvironmentVariable actúa en tiempo de launch, antes de que Gazebo arranque
    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=gazebo_models_path + ':' + os.environ.get('GAZEBO_MODEL_PATH', ''),
    )

    # También actualizamos el entorno del proceso Python actual para que xacro
    # pueda resolver las URIs si fuera necesario
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = (
            gazebo_models_path + ':' + os.environ['GAZEBO_MODEL_PATH']
        )
    else:
        os.environ['GAZEBO_MODEL_PATH'] = gazebo_models_path

    # ── Argumentos de lanzamiento ────────────────────────────────────────────
    rviz_arg     = DeclareLaunchArgument(
        'rviz',    default_value='false',
        description='Lanzar RViz con la configuración de MoveIt'
    )
    gripper_arg  = DeclareLaunchArgument(
        'gripper', default_value='false',
        description='Lanzar el nodo de gripper virtual'
    )
    ik_arg       = DeclareLaunchArgument(
        'ik',      default_value='false',
        description='Lanzar el nodo de cinemática inversa'
    )
    robot_x_arg  = DeclareLaunchArgument(
        'robot_x', default_value='0.0',
        description='Posición X del robot sobre el terreno marciano'
    )
    robot_y_arg  = DeclareLaunchArgument(
        'robot_y', default_value='0.0',
        description='Posición Y del robot sobre el terreno marciano'
    )
    robot_z_arg  = DeclareLaunchArgument(
        'robot_z', default_value='0.5',
        description=(
            'Posición Z del robot (altura sobre el terreno). '
            'Ajustar según la topografía del punto de spawn.'
        )
    )
    camera_x_arg = DeclareLaunchArgument(
        'camera_x', default_value='0.2',
        description='Posición X de la cámara cenital'
    )
    camera_y_arg = DeclareLaunchArgument(
        'camera_y', default_value='0.0',
        description='Posición Y de la cámara cenital'
    )
    camera_z_arg = DeclareLaunchArgument(
        'camera_z', default_value='0.75',
        description='Altura de la cámara cenital sobre el workspace del robot'
    )

    use_rviz    = LaunchConfiguration('rviz')
    use_gripper = LaunchConfiguration('gripper')
    use_ik      = LaunchConfiguration('ik')
    robot_x     = LaunchConfiguration('robot_x')
    robot_y     = LaunchConfiguration('robot_y')
    robot_z     = LaunchConfiguration('robot_z')
    camera_x    = LaunchConfiguration('camera_x')
    camera_y    = LaunchConfiguration('camera_y')
    camera_z    = LaunchConfiguration('camera_z')

    # ── URDF / Xacro ────────────────────────────────────────────────────────
    # Robot (URDF + GazeboSystem + gazebo_ros2_control plugin)
    xacro_file = os.path.join(pkg_parol6_desc, 'urdf', 'parol6_gazebo.xacro')
    robot_description_content = xacro.process_file(xacro_file).toxml()

    # Cámara cenital (modelo independiente del robot)
    camera_xacro = os.path.join(pkg_parol6_desc, 'urdf', 'overhead_camera.urdf.xacro')
    camera_description = xacro.process_file(camera_xacro).toxml()

    # Escribir URDF de cámara a archivo temporal (evita problemas de shell con XML largo)
    camera_urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', prefix='overhead_camera_', delete=False
    )
    camera_urdf_tmp.write(camera_description)
    camera_urdf_tmp.close()
    camera_urdf_path = camera_urdf_tmp.name

    use_sim_time = {'use_sim_time': True}

    # ── Nodos de simulación ─────────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}, use_sim_time],
        output='screen',
    )

    # Gazebo Classic con el mundo marciano SpaceROS
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world':  world_file,
            'paused': 'false',
            # Habilitar verbose para ver errores de carga de mallas en el log
            # 'verbose': 'true',
        }.items(),
    )

    # Spawn del robot PAROL6
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity',  'parol6',
            '-topic',   'robot_description',
            '-x',       robot_x,
            '-y',       robot_y,
            '-z',       robot_z,
            '-unpause',
            '-timeout', '120',
        ],
        output='screen',
    )

    # Controladores (secuenciales)
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_group_controller', '--controller-manager', '/controller_manager'],
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
    )

    # ── Cámara cenital ───────────────────────────────────────────────────────
    spawn_camera = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity',  'overhead_camera',
            '-file',    camera_urdf_path,
            '-x',       camera_x,
            '-y',       camera_y,
            '-z',       camera_z,
            '-R',       '0.0',
            '-P',       '1.5707963267949',
            '-Y',       '0.0',
            '-timeout', '120',
        ],
        output='screen',
    )

    # TF estática world → overhead_camera_link
    camera_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x',        camera_x,
            '--y',        camera_y,
            '--z',        camera_z,
            '--roll',     '0.0',
            '--pitch',    '1.5707963267949',
            '--yaw',      '0.0',
            '--frame-id', 'world',
            '--child-frame-id', 'overhead_camera_link',
        ],
        output='screen',
    )

    # ── MoveIt 2 ─────────────────────────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder('parol6', package_name='parol6_moveit')
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    # Sobrescribir robot_description con la versión Gazebo (GazeboSystem)
    moveit_config.robot_description = {'robot_description': robot_description_content}

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), use_sim_time],
    )

    # ── RViz (opcional) ──────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
        arguments=['-d', os.path.join(pkg_parol6_moveit, 'config', 'moveit.rviz')],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            use_sim_time,
        ],
    )

    # ── Gripper virtual (opcional) ────────────────────────────────────────────
    gripper_node = Node(
        package='parol6_controllers',
        executable='virtual_gripper_node.py',
        output='screen',
        condition=IfCondition(use_gripper),
    )

    # ── Cinemática inversa (opcional) ─────────────────────────────────────────
    ik_node = Node(
        package='parol6_apps',
        executable='pose_ik_subscriber.py',
        output='screen',
        condition=IfCondition(use_ik),
        parameters=[use_sim_time],
    )

    # ── Descripción del grafo de lanzamiento ──────────────────────────────────
    return LaunchDescription([
        # Argumentos
        rviz_arg,
        gripper_arg,
        ik_arg,
        robot_x_arg,
        robot_y_arg,
        robot_z_arg,
        camera_x_arg,
        camera_y_arg,
        camera_z_arg,

        # Variables de entorno (deben ir antes de Gazebo)
        set_gazebo_model_path,

        # Simulación
        gazebo,
        robot_state_publisher,
        spawn_entity,
        spawn_camera,
        camera_static_tf,

        # ── Secuencia de arranque de controladores ───────────────────────────
        # 1. joint_state_broadcaster  (requiere que el robot esté spawneado)
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )),
        # 2. arm_group_controller  (requiere joint_state_broadcaster activo)
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )),
        # 3. gripper_controller  (requiere arm_group_controller activo)
        RegisterEventHandler(OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner],
        )),
        # 4. MoveIt, RViz, gripper virtual, IK  (requieren todos los controllers)
        RegisterEventHandler(OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[move_group_node, rviz_node, gripper_node, ik_node],
        )),
    ])
