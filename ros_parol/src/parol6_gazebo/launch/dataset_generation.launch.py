"""
dataset_generation.launch.py — Lanzamiento LIGERO para generar el dataset
sintético (synthetic_dataset_generator_v2.py). Sin MoveIt y sin RViz.

Levanta:
  * Gazebo Classic con el mundo marciano (por defecto `mars_optimized`, que
    tiene el parche de terreno centrado en el origen; en `mars/mars.world`
    el terreno está desplazado a ~(20,20,-2.4) y NO hay suelo bajo la cámara).
  * El robot PAROL6 + controller_manager (joint_state_broadcaster,
    arm_group_controller y gripper_controller: el controlador SOSTIENE el
    brazo en las configuraciones aleatorias; sin él se desplomaría).
  * La NavCam estéreo (con los sensores depth GT aditivos del xacro) y sus
    static_transform_publisher.
  * El nodo generador con use_sim_time:=True (obligatorio).

El tubo y las rocas distractoras los spawnea el PROPIO generador (pools de
entidades reutilizables), no este launch.

Uso:
  ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=20
  ros2 launch parol6_gazebo dataset_generation.launch.py \
      num_samples:=1000 output_dir:=$HOME/parol6_synth_dataset seed:=0
  ros2 launch parol6_gazebo dataset_generation.launch.py gui:=true   # con gzclient
"""

import os
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _write_urdf_tmp(xacro_path, mappings=None):
    """Procesa un xacro y escribe el URDF en un fichero temporal."""
    xml = xacro.process_file(xacro_path, mappings=mappings or {}).toxml()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(xml)
    tmp.close()
    return tmp.name, xml


def launch_setup(context, *args, **kwargs):
    pkg_parol6_desc = get_package_share_directory('parol6_description')
    pkg_parol6_gazebo = get_package_share_directory('parol6_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    world_arg = LaunchConfiguration('world').perform(context)
    gui = LaunchConfiguration('gui').perform(context)
    num_samples = LaunchConfiguration('num_samples').perform(context)
    output_dir = LaunchConfiguration('output_dir').perform(context)
    seed = LaunchConfiguration('seed').perform(context)
    start_index = LaunchConfiguration('start_index').perform(context)
    shutdown_when_done = LaunchConfiguration('shutdown_when_done').perform(context)
    robot_x = LaunchConfiguration('robot_x').perform(context)
    robot_y = LaunchConfiguration('robot_y').perform(context)
    robot_z = LaunchConfiguration('robot_z').perform(context)
    mast_x = LaunchConfiguration('mast_x').perform(context)
    mast_y = LaunchConfiguration('mast_y').perform(context)
    mast_z = LaunchConfiguration('mast_z').perform(context)
    mast_roll = LaunchConfiguration('mast_roll').perform(context)
    mast_pitch = LaunchConfiguration('mast_pitch').perform(context)
    mast_yaw = LaunchConfiguration('mast_yaw').perform(context)

    # ── Parámetros del generador expuestos como argumentos (v2.1) ─────────
    # Los defaults coinciden con los del nodo, así que el dataset YOLO original
    # se regenera igual. Para el set de EVALUACIÓN DE POSE:
    #   stereo_capture:=true placement_mode:=footprint frac_floating:=0.0
    #   negative_fraction:=0.0 split_ratios:=0.5,0.2,0.3
    def _f(name):
        return float(LaunchConfiguration(name).perform(context))

    def _b(name):
        return LaunchConfiguration(name).perform(context).strip().lower() in (
            'true', '1', 'yes')

    split_ratios = [float(v) for v in
                    LaunchConfiguration('split_ratios').perform(context).split(',')]
    if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
        raise RuntimeError(f'split_ratios debe ser 3 fracciones que sumen 1: '
                           f'{split_ratios}')
    generator_extra = {
        'stereo_capture': _b('stereo_capture'),
        'placement_mode': LaunchConfiguration('placement_mode').perform(context),
        'footprint_cells_file': LaunchConfiguration('footprint_cells_file').perform(context),
        'split_ratios': split_ratios,
        'frac_grounded': _f('frac_grounded'),
        'frac_floating': _f('frac_floating'),
        'frac_buried': _f('frac_buried'),
        'negative_fraction': _f('negative_fraction'),
        'multi_tube_prob': _f('multi_tube_prob'),
        'occlusion_fraction': _f('occlusion_fraction'),
        'heavy_occlusion_fraction': _f('heavy_occlusion_fraction'),
        'arm_clearance_m': _f('arm_clearance_m'),
        'arm_home_fraction': _f('arm_home_fraction'),
        'save_hard_samples': _b('save_hard_samples'),
        'randomize_lighting': _b('randomize_lighting'),
        'light_intensity_min': _f('light_intensity_min'),
        'light_intensity_max': _f('light_intensity_max'),
    }

    # Mundo: nombre corto conocido o ruta absoluta a un .world
    known_worlds = {
        'mars_optimized': os.path.join(pkg_parol6_gazebo, 'worlds',
                                       'mars_optimized', 'mars_optimized.world'),
        'mars': os.path.join(pkg_parol6_gazebo, 'worlds', 'mars', 'mars.world'),
        'mars_bumpy': os.path.join(pkg_parol6_gazebo, 'worlds', 'mars_bumpy',
                                   'mars_bumpy.world'),
    }
    world_file = known_worlds.get(world_arg, world_arg)
    if not os.path.isfile(world_file):
        raise RuntimeError(f'Mundo no encontrado: {world_arg!r} → {world_file}')
    # Nombre corto del mundo para trazabilidad en meta.jsonl
    world_name = (world_arg if world_arg in known_worlds
                  else os.path.splitext(os.path.basename(world_file))[0])

    use_sim_time = {'use_sim_time': True}

    # Robot PAROL6 (URDF largo: se publica por topic, como en los demás launch)
    xacro_file = os.path.join(pkg_parol6_desc, 'urdf', 'parol6_gazebo.xacro')
    _, robot_description_content = _write_urdf_tmp(xacro_file)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content},
                    use_sim_time],
        output='screen',
    )

    # Nota: el argumento top-level `seed` se hereda también dentro de
    # gazebo.launch.py (gzserver --seed=N): determiniza además el RNG del
    # propio Gazebo. Es deliberado y armless.
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file, 'pause': 'false',
                          'gui': gui, 'verbose': 'false'}.items(),
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'parol6',
            '-topic', 'robot_description',
            '-x', robot_x, '-y', robot_y, '-z', robot_z,
            '-unpause',
            '-timeout', '120',
        ],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'],
    )
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_group_controller',
                   '--controller-manager', '/controller_manager'],
    )
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller',
                   '--controller-manager', '/controller_manager'],
    )

    # ── NavCam estéreo en el mástil (misma pose que el pipeline real) ──────
    # El xacro incluye ahora, ADEMÁS de las RGB, los sensores depth GT
    # (namespace gt_depth) que usa el generador para resolver oclusión.
    spawn_x = str(float(robot_x) + float(mast_x))
    spawn_y = str(float(robot_y) + float(mast_y))
    spawn_z = str(float(robot_z) + float(mast_z))
    cam_urdf_path, _ = _write_urdf_tmp(
        os.path.join(pkg_parol6_desc, 'urdf', 'navcam_stereo.urdf.xacro'))

    spawn_navcam = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'navcam_stereo',
            '-file', cam_urdf_path,
            '-x', spawn_x, '-y', spawn_y, '-z', spawn_z,
            '-R', mast_roll, '-P', mast_pitch, '-Y', mast_yaw,
            '-timeout', '120',
        ],
        output='screen',
    )
    static_tfs = [
        # base_link → navcam_rig_link (pose del rig respecto a la base del brazo)
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--x', mast_x, '--y', mast_y, '--z', mast_z,
                        '--roll', mast_roll, '--pitch', mast_pitch,
                        '--yaw', mast_yaw,
                        '--frame-id', 'base_link',
                        '--child-frame-id', 'navcam_rig_link']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--x', '0', '--y', '0.075', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'navcam_rig_link',
                        '--child-frame-id', 'navcam_left_link']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--x', '0', '--y', '0', '--z', '0',
                        '--roll', '-1.5707963267949', '--pitch', '0',
                        '--yaw', '-1.5707963267949',
                        '--frame-id', 'navcam_left_link',
                        '--child-frame-id', 'navcam_left_optical_frame']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--x', '0', '--y', '-0.075', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'navcam_rig_link',
                        '--child-frame-id', 'navcam_right_link']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--x', '0', '--y', '0', '--z', '0',
                        '--roll', '-1.5707963267949', '--pitch', '0',
                        '--yaw', '-1.5707963267949',
                        '--frame-id', 'navcam_right_link',
                        '--child-frame-id', 'navcam_right_optical_frame']),
    ]

    # ── Nodo generador (use_sim_time OBLIGATORIO) ──────────────────────────
    generator = Node(
        package='parol6_vision',
        executable='synthetic_dataset_generator_v2.py',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'num_samples': int(num_samples),
            'output_dir': output_dir,
            'seed': int(seed),
            'start_index': int(start_index),
            'world_name': world_name,
            **generator_extra,
        }],
    )

    # Al terminar el generador (bien o mal), apagar todo el launch. Esencial
    # para encadenar mundos con generate_dataset_multiworld.py.
    extra_handlers = []
    if shutdown_when_done.strip().lower() in ('true', '1', 'yes'):
        extra_handlers.append(RegisterEventHandler(OnProcessExit(
            target_action=generator,
            on_exit=[Shutdown(reason='generación de dataset terminada')],
        )))

    return [
        *extra_handlers,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        spawn_navcam,
        *static_tfs,
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[generator],
        )),
    ]


def generate_launch_description():
    # model://<pkg>/... necesita el 'share' de <pkg> en GAZEBO_MODEL_PATH
    # (igual que en gazebo_moveit_mars_optimized.launch.py).
    extra_model_paths = [
        os.path.join(get_package_share_directory('parol6_description'), '..'),
        os.path.join(get_package_share_directory('parol6_gazebo'), '..'),
    ]
    prev = os.environ.get('GAZEBO_MODEL_PATH', '')
    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(
        p for p in [prev, *extra_model_paths] if p)

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='mars_optimized',
            description="Mundo: 'mars_optimized' (defecto), 'mars', "
                        "'mars_bumpy' o ruta absoluta a un .world"),
        DeclareLaunchArgument('gui', default_value='false',
                              description='Lanzar gzclient (innecesario para generar)'),
        DeclareLaunchArgument('num_samples', default_value='1000',
                              description='Número de muestras VÁLIDAS a generar'),
        DeclareLaunchArgument('output_dir', default_value='~/parol6_synth_dataset',
                              description='Directorio de salida del dataset'),
        DeclareLaunchArgument('seed', default_value='0',
                              description='Semilla de reproducibilidad'),
        DeclareLaunchArgument(
            'start_index', default_value='0',
            description='Primer id global de esta tanda (modo append '
                        'multi-mundo; ver generate_dataset_multiworld.py)'),
        DeclareLaunchArgument(
            'shutdown_when_done', default_value='true',
            description='Apagar el launch (y Gazebo) al terminar el generador'),
        DeclareLaunchArgument('robot_x', default_value='0.0'),
        DeclareLaunchArgument('robot_y', default_value='0.0'),
        DeclareLaunchArgument('robot_z', default_value='0.0'),
        # Pose del mástil de la NavCam respecto a base_link (REP-103: pitch
        # positivo inclina la vista hacia abajo). Idéntica al pipeline real.
        DeclareLaunchArgument('mast_x', default_value='-0.3'),
        DeclareLaunchArgument('mast_y', default_value='-0.3'),
        DeclareLaunchArgument('mast_z', default_value='1.11'),
        DeclareLaunchArgument('mast_roll', default_value='0.0'),
        DeclareLaunchArgument('mast_pitch', default_value='0.91'),
        DeclareLaunchArgument('mast_yaw', default_value='0.3'),
        # ── Modo evaluación de pose (v2.1); defaults = comportamiento original ─
        DeclareLaunchArgument(
            'stereo_capture', default_value='false',
            description='Guardar par estéreo + depth GT izquierdo; GT en el '
                        'frame óptico izquierdo (evaluación de ICP/D2CO)'),
        DeclareLaunchArgument(
            'placement_mode', default_value='image_bins',
            description="'image_bins' (original) o 'footprint' (celdas "
                        "agarrables por el brazo: config/grasp_footprint_cells.json)"),
        DeclareLaunchArgument(
            'footprint_cells_file', default_value='',
            description='JSON de celdas de la huella (vacío => el del paquete)'),
        DeclareLaunchArgument('split_ratios', default_value='0.70,0.20,0.10',
                              description='Fracciones train,val,test'),
        DeclareLaunchArgument('frac_grounded', default_value='0.70'),
        DeclareLaunchArgument('frac_floating', default_value='0.15'),
        DeclareLaunchArgument('frac_buried', default_value='0.15'),
        DeclareLaunchArgument('negative_fraction', default_value='0.15'),
        DeclareLaunchArgument('multi_tube_prob', default_value='0.15'),
        DeclareLaunchArgument('occlusion_fraction', default_value='0.50'),
        DeclareLaunchArgument('heavy_occlusion_fraction', default_value='0.12'),
        DeclareLaunchArgument('arm_clearance_m', default_value='0.05',
                              description='Holgura mínima brazo↔tubo por FK '
                                          '(modo footprint; 0 = sin comprobar)'),
        DeclareLaunchArgument('arm_home_fraction', default_value='0.3',
                              description='Fracción de muestras con el brazo en '
                                          'home (modo footprint)'),
        DeclareLaunchArgument(
            'save_hard_samples', default_value='true',
            description='true: muestras con visib del tubo principal < '
                        'min_visibility van a test sin etiqueta; false: se '
                        'descartan y se regeneran (splits exactos)'),
        DeclareLaunchArgument(
            'randomize_lighting', default_value='false',
            description='Intensidad difusa del sol aleatoria por escena '
                        '(requiere libgazebo_ros_properties.so en el mundo)'),
        DeclareLaunchArgument('light_intensity_min', default_value='0.55'),
        DeclareLaunchArgument('light_intensity_max', default_value='1.0'),
        OpaqueFunction(function=launch_setup),
    ])
