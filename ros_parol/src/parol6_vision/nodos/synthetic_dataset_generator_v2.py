#!/usr/bin/env python3
"""
synthetic_dataset_generator_v2.py — Generador de dataset sintético Sim2Real
para el robot PAROL6 con NavCam estéreo en Gazebo Classic (ROS 2).

Objeto de interés: tubo de muestras Mars 2020 (`mars2020tube`).

El dataset tiene DOBLE PROPÓSITO y ambas salidas se generan en la misma pasada:

  1. Detección 2D (YOLO): imágenes + etiquetas `0 cx cy w h` normalizadas
     (bbox de la parte VISIBLE) para entrenar `best.pt`, que consume
     `yolo26n_camera_subscriber.py` sobre la NavCam izquierda.
  2. Ground truth de pose 6D estilo BOP (scene_camera / scene_gt /
     scene_gt_info / models_info) para EVALUAR los estimadores existentes
     (ICP, LINEMOD) estratificando por nivel de oclusión (`visib_fract`),
     y para un posible futuro estimador aprendido.

Decisiones de diseño importantes (ver docs/README_SYNTH_DATASET_V2.md):

  * UNIDADES: todo el GT 6D está en METROS (deliberadamente distinto del
    convenio BOP en milímetros) para que sea directamente comparable con la
    salida de ICP/Open3D/ROS sin conversiones.
  * SIMETRÍA: el tubo es (casi) un sólido de revolución; `models_info.json`
    declara `symmetries_continuous` alrededor de su eje Z. Métrica
    recomendada: ADD-S. El extremo +Z (cabeza compleja) es el único rasgo
    que rompe el flip: su visibilidad se reporta en `keypoints_visib`.
  * OCLUSIÓN: solo se hornea oclusión por GEOMETRÍA REAL (terreno al
    enterrar, rocas procedurales, brazo/gripper). La profundidad ground
    truth (`gt_depth`, sensores aditivos de navcam_stereo.urdf.xacro)
    resuelve el bbox visible y `visib_fract` automáticamente. El "polvo" en
    espacio de imagen va en el AUMENTO de entrenamiento, no aquí
    (docs/train_yolo_navcam.md).
  * VISIBILIDAD: para cada instancia se raycastea la malla escalada del
    tubo (Open3D RaycastingScene, CPU) en el frame óptico → z-buffer
    "solo-objeto" (amodal). Un píxel es visible si
    z_obj <= z_escena + epsilon, con z_escena el depth GT de Gazebo.
    Esto capta autooclusión, rocas, enterrado y el propio brazo.
  * CÁMARAS MONOCROMAS (L8): no se aplica ninguna variación de color. Las
    imágenes se guardan en gris replicado a 3 canales (gris→BGR idéntico)
    para que Ultralytics y los pesos preentrenados funcionen sin cambios.
  * BRAZO: se comanda `arm_group_controller` (JointTrajectory) y se espera
    asentamiento por ESTABILIDAD DE POSICIÓN en /joint_states. Con la
    interfaz de posición de gazebo_ros2_control las velocidades reportadas
    son ruidosas (~10 rad/s) e inservibles como criterio. Nunca se usa
    set_model_configuration (colapso gravitatorio al reanudar).
  * COLOCACIÓN DEL TUBO: por retro-proyección de un mapa de profundidad del
    TERRENO cacheado al arranque (cámara y terreno son estáticos). Esto
    garantiza que el tubo cae dentro del frustum, sobre/bajo la superficie
    real, y permite estratificar distancia y posición en imagen.
  * El tubo y las rocas se spawnean ESTÁTICOS y se reutilizan entre escenas
    (pool de entidades aparcadas detrás de la cámara): no hay física que
    asentar y el coste por escena es mínimo.

Este nodo NO toca el pipeline real: los topics `gt_depth` son nuevos y el
viejo `synthetic_dataset_generator.py` queda intacto.

Uso:
    ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=1000

Requiere `use_sim_time=True` (el launch lo fija; el nodo lo fuerza además
por código): las TF y los stamps de imagen viven en tiempo de simulación.
"""

import json
import math
import os
import random
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np
import open3d as o3d
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import DeleteEntity, GetEntityState, SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from gazebo_msgs.srv import SetLightProperties
    _HAS_SET_LIGHT = True
except ImportError:  # versiones de gazebo_msgs sin el servicio
    _HAS_SET_LIGHT = False


# ═══════════════════════════════════════════════════════════════════════════
# Utilidades de álgebra (sin scipy: la del sistema es incompatible con numpy)
# ═══════════════════════════════════════════════════════════════════════════

def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Cuaternión (x,y,z,w) → matriz de rotación 3x3."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray):
    """Matriz de rotación 3x3 → cuaternión (x,y,z,w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def random_rotation(rng: random.Random) -> np.ndarray:
    """Rotación uniforme en SO(3) (cuaternión aleatorio, método de Shoemake)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = (
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    )
    return quat_to_mat(*q)


def rotation_with_axis(axis_world: np.ndarray, roll: float) -> np.ndarray:
    """Rotación cuya columna Z es `axis_world`, con giro `roll` sobre ese eje."""
    z = axis_world / np.linalg.norm(axis_world)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]) @ rotz(roll)


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def pose_msg_to_T(p: Pose) -> np.ndarray:
    R = quat_to_mat(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    return make_T(R, [p.position.x, p.position.y, p.position.z])


def T_to_pose_msg(T: np.ndarray) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    x, y, z, w = mat_to_quat(T[:3, :3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = x, y, z, w
    return p


def mask_to_xywh(mask: np.ndarray, u_off: int, v_off: int):
    """Máscara booleana (crop) → bbox [x, y, w, h] en píxeles de imagen completa."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return None
    x0, x1 = int(us.min()), int(us.max())
    y0, y1 = int(vs.min()), int(vs.max())
    return [x0 + u_off, y0 + v_off, x1 - x0 + 1, y1 - y0 + 1]


# ═══════════════════════════════════════════════════════════════════════════
# Nodo generador
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticDatasetGeneratorV2(Node):

    # ──────────────────────────────────────────────────────────────────────
    # Inicialización
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__(
            'synthetic_dataset_generator_v2',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )
        self._declare_params()
        self._read_params()

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = {}          # clave → (stamp_ns, np.ndarray)
        self._cam_info = {}        # 'left'/'right' → dict(K, w, h)
        self._gt_depth_info = {}   # para verificar que K coincide con el RGB
        self._joint_state = None   # (stamp_ns, dict nombre→posición)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Suscripciones (QoS de sensor: compatible con publishers reliable)
        sub = self.create_subscription
        sub(Image, self.p['left_image_topic'],
            lambda m: self._on_image(m, 'left_rgb'), qos_profile_sensor_data)
        sub(Image, self.p['right_image_topic'],
            lambda m: self._on_image(m, 'right_rgb'), qos_profile_sensor_data)
        sub(Image, self.p['left_gt_depth_topic'],
            lambda m: self._on_depth(m, 'left_depth'), qos_profile_sensor_data)
        sub(Image, self.p['right_gt_depth_topic'],
            lambda m: self._on_depth(m, 'right_depth'), qos_profile_sensor_data)
        sub(CameraInfo, self.p['left_info_topic'],
            lambda m: self._on_info(m, 'left'), qos_profile_sensor_data)
        sub(CameraInfo, self.p['right_info_topic'],
            lambda m: self._on_info(m, 'right'), qos_profile_sensor_data)
        sub(CameraInfo, self.p['left_gt_info_topic'],
            lambda m: self._on_gt_info(m, 'left'), qos_profile_sensor_data)
        sub(CameraInfo, self.p['right_gt_info_topic'],
            lambda m: self._on_gt_info(m, 'right'), qos_profile_sensor_data)
        sub(JointState, '/joint_states', self._on_joint_state, 10)

        self.arm_pub = self.create_publisher(
            JointTrajectory, self.p['arm_controller_topic'], 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, self.p['gripper_controller_topic'], 10)

        # Aleatoriedad reproducible
        self.rng = random.Random(self.p['seed'])
        self.nprng = np.random.default_rng(self.p['seed'])

        # Malla del tubo (escala NO uniforme + offset del visual, frame tube_link)
        self._load_tube_mesh()

        # Estado del generador
        self.out_dir = os.path.expanduser(self.p['output_dir'])
        self.bop = {'scene_camera': {}, 'scene_gt': {}, 'scene_gt_info': {}}
        self.split_of = {}
        self.discard_reasons = {}
        self.active_tubes = []     # nombres de entidades activas en la escena
        self.active_rocks = []
        self.tube_pool = []        # [(nombre, albedo)]
        self.rock_pool = []        # [dict(name, size, kind)]
        self._light_warned = False

    def _declare_params(self):
        d = self.declare_parameter
        d('num_samples', 1000)
        d('output_dir', '~/parol6_synth_dataset')
        d('split_ratios', [0.70, 0.20, 0.10])
        d('seed', 0)
        # Topics y frames (sección 2 del diseño)
        d('left_image_topic', '/navcam/left/navcam_left/image_raw')
        d('right_image_topic', '/navcam/right/navcam_right/image_raw')
        d('left_info_topic', '/navcam/left/navcam_left/camera_info')
        d('right_info_topic', '/navcam/right/navcam_right/camera_info')
        d('left_gt_depth_topic', '/navcam/left/navcam_left/gt_depth/depth/image_raw')
        d('right_gt_depth_topic', '/navcam/right/navcam_right/gt_depth/depth/image_raw')
        d('left_gt_info_topic', '/navcam/left/navcam_left/gt_depth/camera_info')
        d('right_gt_info_topic', '/navcam/right/navcam_right/gt_depth/camera_info')
        d('left_optical_frame', 'navcam_left_optical_frame')
        d('right_optical_frame', 'navcam_right_optical_frame')
        d('world_frame', 'world')
        # Tubo
        d('tube_entity_name', 'mars2020tube')
        d('tube_scale', [0.022939, 0.022939, 0.033212])
        d('tube_visual_offset', [-0.010497, -0.010489, -0.093])
        d('tube_mesh_package', 'parol6_gazebo')
        d('tube_mesh_relpath', 'meshes/mars2020tube.stl')
        d('tube_pool_size', 6)
        d('tube_albedo_min', 0.30)   # apariencia "polvorienta" (gris apagado)
        d('tube_albedo_max', 0.90)   # apariencia limpia (gris claro)
        # Brazo
        d('arm_joint_names', ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
        d('arm_controller_topic', '/arm_group_controller/joint_trajectory')
        d('gripper_controller_topic', '/gripper_controller/joint_trajectory')
        d('gripper_joint_names', ['gripper_joint', 'gripper_right_joint'])
        d('randomize_gripper', True)
        d('gripper_max_opening', 0.025)
        # Límites reales del URDF parol6.urdf (verificados)
        d('arm_limits_lower', [-1.7, -0.98, -2.0, -2.0, -2.1, -3.1])
        d('arm_limits_upper', [1.7, 1.0, 1.3, 2.0, 2.1, 3.1])
        d('arm_home', [0.0, -0.9599771561327288, 1.28999018740207,
                       0.0, 0.2581663987092836, 0.017957190239044962])
        d('arm_settle_pos_threshold', 0.005)   # rad entre lecturas consecutivas
        d('arm_settle_pos_tolerance', 0.08)    # rad de error al objetivo
        d('arm_motion_time', 1.2)              # s (time_from_start)
        d('arm_min_link_z', 0.02)              # m: ningún link por debajo (anti-suelo)
        d('arm_links_check', ['L3_link', 'L4_link', 'L5_link', 'L6_link',
                              'gripper_base_link', 'left_finger_link',
                              'right_finger_link'])
        d('settle_timeout_sec', 15.0)          # wall-clock
        d('capture_timeout_sec', 20.0)         # wall-clock
        # Colocación / estratificación
        d('camera_distance_min', 0.8)
        d('camera_distance_max', 3.5)
        d('n_distance_bins', 4)
        d('n_image_bins', 3)                   # rejilla n×n de posición en imagen
        d('min_dist_from_base', 0.55)          # m: que el brazo no atraviese el tubo
        d('terrain_max_z', 0.35)               # m: por encima = brazo, no terreno
        d('terrain_radius', 5.5)               # m: radio útil del parche de terreno
        d('frac_grounded', 0.70)
        d('frac_floating', 0.15)
        d('frac_buried', 0.15)
        d('float_height_min', 0.05)
        d('float_height_max', 0.50)
        d('grounded_max_tilt_deg', 35.0)
        # Oclusión / visibilidad
        d('occlusion_fraction', 0.50)
        d('heavy_occlusion_fraction', 0.12)
        d('min_visibility', 0.10)
        d('save_hard_samples', True)           # visib<min → split test marcado hard
        d('bury_min_fraction', 0.15)
        d('bury_max_fraction', 0.60)
        d('depth_visibility_epsilon', 0.005)
        # Negativos / multi-instancia / rocas
        d('negative_fraction', 0.15)
        d('multi_tube_prob', 0.15)
        d('max_tubes', 3)
        d('num_rocks_min', 0)
        d('num_rocks_max', 8)
        d('rock_pool_size', 14)
        # Varios
        d('randomize_lighting', False)
        d('max_attempts', 5)
        d('flush_every', 25)
        d('blocked_depth_m', 0.45)             # detección de brazo tapando la lente
        d('blocked_frac', 0.50)

    def _read_params(self):
        names = [
            'num_samples', 'output_dir', 'split_ratios', 'seed',
            'left_image_topic', 'right_image_topic', 'left_info_topic',
            'right_info_topic', 'left_gt_depth_topic', 'right_gt_depth_topic',
            'left_gt_info_topic', 'right_gt_info_topic',
            'left_optical_frame', 'right_optical_frame', 'world_frame',
            'tube_entity_name', 'tube_scale', 'tube_visual_offset',
            'tube_mesh_package', 'tube_mesh_relpath', 'tube_pool_size',
            'tube_albedo_min', 'tube_albedo_max',
            'arm_joint_names', 'arm_controller_topic', 'gripper_controller_topic',
            'gripper_joint_names', 'randomize_gripper', 'gripper_max_opening',
            'arm_limits_lower', 'arm_limits_upper', 'arm_home',
            'arm_settle_pos_threshold', 'arm_settle_pos_tolerance',
            'arm_motion_time', 'arm_min_link_z', 'arm_links_check',
            'settle_timeout_sec', 'capture_timeout_sec',
            'camera_distance_min', 'camera_distance_max', 'n_distance_bins',
            'n_image_bins', 'min_dist_from_base', 'terrain_max_z',
            'terrain_radius', 'frac_grounded', 'frac_floating', 'frac_buried',
            'float_height_min', 'float_height_max', 'grounded_max_tilt_deg',
            'occlusion_fraction', 'heavy_occlusion_fraction', 'min_visibility',
            'save_hard_samples', 'bury_min_fraction', 'bury_max_fraction',
            'depth_visibility_epsilon', 'negative_fraction', 'multi_tube_prob',
            'max_tubes', 'num_rocks_min', 'num_rocks_max', 'rock_pool_size',
            'randomize_lighting', 'max_attempts', 'flush_every',
            'blocked_depth_m', 'blocked_frac',
        ]
        self.p = {n: self.get_parameter(n).value for n in names}

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks de suscripción
    # ──────────────────────────────────────────────────────────────────────

    def _stamp_ns(self, msg) -> int:
        return Time.from_msg(msg.header.stamp).nanoseconds

    def _on_image(self, msg, key):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if img.ndim == 3:  # por si la cámara no fuera mono
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with self._lock:
            self._latest[key] = (self._stamp_ns(msg), img)

    def _on_depth(self, msg, key):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        d = np.asarray(d, dtype=np.float32)
        with self._lock:
            self._latest[key] = (self._stamp_ns(msg), d)

    def _on_info(self, msg, side):
        if side not in self._cam_info:
            with self._lock:
                self._cam_info[side] = {
                    'K': np.array(msg.k, dtype=float).reshape(3, 3),
                    'w': int(msg.width), 'h': int(msg.height),
                }

    def _on_gt_info(self, msg, side):
        if side not in self._gt_depth_info:
            with self._lock:
                self._gt_depth_info[side] = np.array(msg.k, dtype=float).reshape(3, 3)

    def _on_joint_state(self, msg):
        with self._lock:
            self._joint_state = (self._stamp_ns(msg),
                                 dict(zip(msg.name, msg.position)))

    # ──────────────────────────────────────────────────────────────────────
    # Esperas y servicios
    # ──────────────────────────────────────────────────────────────────────

    def _wait_fresh(self, key, min_stamp_ns, timeout_wall):
        """Espera (reloj de pared) un mensaje con stamp de sim >= min_stamp_ns."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_wall and rclpy.ok():
            with self._lock:
                item = self._latest.get(key)
            if item is not None and item[0] >= min_stamp_ns:
                return item
            time.sleep(0.02)
        return None

    def _make_client(self, srv_type, names, timeout=180.0):
        """Crea cliente probando nombres con y sin prefijo /gazebo."""
        clients = [self.create_client(srv_type, n) for n in names]
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and rclpy.ok():
            for c in clients:
                if c.service_is_ready():
                    self.get_logger().info(f'Servicio disponible: {c.srv_name}')
                    return c
            time.sleep(0.25)
        raise RuntimeError(f'Servicio no disponible: {names}')

    def _call(self, client, request, timeout=20.0):
        fut = client.call_async(request)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout or not rclpy.ok():
                raise TimeoutError(f'Timeout llamando a {client.srv_name}')
            time.sleep(0.01)
        return fut.result()

    def _setup_services(self):
        self.cli_spawn = self._make_client(
            SpawnEntity, ['/spawn_entity', '/gazebo/spawn_entity'])
        self.cli_delete = self._make_client(
            DeleteEntity, ['/delete_entity', '/gazebo/delete_entity'])
        self.cli_set_state = self._make_client(
            SetEntityState, ['/set_entity_state', '/gazebo/set_entity_state'])
        self.cli_get_state = self._make_client(
            GetEntityState, ['/get_entity_state', '/gazebo/get_entity_state'])
        self.cli_pause = self._make_client(
            Empty, ['/pause_physics', '/gazebo/pause_physics'])
        self.cli_unpause = self._make_client(
            Empty, ['/unpause_physics', '/gazebo/unpause_physics'])
        self.cli_set_light = None
        if self.p['randomize_lighting'] and _HAS_SET_LIGHT:
            try:
                self.cli_set_light = self._make_client(
                    SetLightProperties,
                    ['/set_light_properties', '/gazebo/set_light_properties'],
                    timeout=10.0)
            except RuntimeError:
                self.get_logger().warning(
                    'set_light_properties no disponible; iluminación fija.')

    def _pause(self):
        self._call(self.cli_pause, Empty.Request())

    def _unpause(self):
        self._call(self.cli_unpause, Empty.Request())

    def _set_entity_pose(self, name, T):
        st = EntityState()
        st.name = name
        st.pose = T_to_pose_msg(T)
        st.reference_frame = self.p['world_frame']
        req = SetEntityState.Request()
        req.state = st
        res = self._call(self.cli_set_state, req)
        if not res.success:
            self.get_logger().warning(f'set_entity_state falló para {name}')
        return res.success

    def _get_entity_T(self, name):
        """Lee de vuelta la pose de una entidad (robustez del GT)."""
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = self.p['world_frame']
        try:
            res = self._call(self.cli_get_state, req)
            if res.success:
                return pose_msg_to_T(res.state.pose)
        except TimeoutError:
            pass
        return None

    def _spawn(self, name, xml, T):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose = T_to_pose_msg(T)
        req.reference_frame = self.p['world_frame']
        res = self._call(self.cli_spawn, req, timeout=30.0)
        if not res.success:
            # Si ya existe (relanzamiento del nodo con el sim vivo), reutilizar
            if 'exist' in (res.status_message or '').lower():
                self.get_logger().info(f'{name} ya existía: se reutiliza.')
                return self._set_entity_pose(name, T)
            self.get_logger().error(f'spawn {name}: {res.status_message}')
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Malla del tubo, keypoints y models_info
    # ──────────────────────────────────────────────────────────────────────

    def _load_tube_mesh(self):
        share = get_package_share_directory(self.p['tube_mesh_package'])
        self.tube_mesh_path = os.path.join(share, self.p['tube_mesh_relpath'])
        mesh = o3d.io.read_triangle_mesh(self.tube_mesh_path)
        if not mesh.has_triangles():
            raise RuntimeError(f'No pude leer la malla {self.tube_mesh_path}')
        scale = np.asarray(self.p['tube_scale'], dtype=float)
        off = np.asarray(self.p['tube_visual_offset'], dtype=float)
        verts = np.asarray(mesh.vertices) * scale + off  # frame tube_link
        tris = np.asarray(mesh.triangles)
        # Si la malla fuera enorme, simplificar SOLO para el raycasting
        if len(tris) > 120000:
            m2 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
            m2 = m2.simplify_quadric_decimation(60000)
            verts, tris = np.asarray(m2.vertices), np.asarray(m2.triangles)
        self.tube_verts_link = verts.astype(np.float32)
        self.tube_tris = tris.astype(np.uint32)
        # Submuestreo de vértices para proyecciones rápidas (bbox del crop)
        step = max(1, len(verts) // 4000)
        self.tube_verts_sub = self.tube_verts_link[::step]

        # models_info (en METROS): extents y diámetro del modelo escalado
        vmin, vmax = verts.min(axis=0), verts.max(axis=0)
        hull = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(tris)).compute_convex_hull()[0]
        hv = np.asarray(hull.vertices)
        diam = 0.0
        for i in range(len(hv)):
            d = np.linalg.norm(hv[i + 1:] - hv[i], axis=1)
            if len(d):
                diam = max(diam, float(d.max()))
        self.models_info = {
            '1': {
                'diameter': diam,
                'min_x': float(vmin[0]), 'min_y': float(vmin[1]), 'min_z': float(vmin[2]),
                'size_x': float(vmax[0] - vmin[0]),
                'size_y': float(vmax[1] - vmin[1]),
                'size_z': float(vmax[2] - vmin[2]),
                # Simetría rotacional CONTINUA sobre el eje Z del tubo: el roll
                # es libre → evaluar con ADD-S. (Unidades: METROS, no mm BOP.)
                'symmetries_continuous': [{'axis': [0, 0, 1], 'offset': [0, 0, 0]}],
            }
        }

        # Keypoints 3D en el frame tube_link (cilindro de colisión):
        #   0 centro, 1 cabeza (+Z, extremo complejo), 2 cola (−Z, tapa simple),
        #   3-5 borde de la cabeza a 0/120/240° (marcan orientación; con la
        #   simetría continua el roll es arbitrario, pero se proyectan igual).
        r, half = 0.01016, 0.093
        self.kp_names = ['center', 'head_+z', 'tail_-z',
                         'head_rim_0', 'head_rim_120', 'head_rim_240']
        self.kp_link = np.array([
            [0, 0, 0], [0, 0, half], [0, 0, -half],
            [r, 0, half],
            [r * math.cos(2 * math.pi / 3), r * math.sin(2 * math.pi / 3), half],
            [r * math.cos(4 * math.pi / 3), r * math.sin(4 * math.pi / 3), half],
        ], dtype=float)
        self.tube_radius = r
        self.tube_half = half

    # ──────────────────────────────────────────────────────────────────────
    # SDFs y pools de entidades (tubos + rocas procedurales)
    # ──────────────────────────────────────────────────────────────────────

    def _tube_sdf(self, name, albedo):
        a = round(albedo * 0.88, 3)
        dcol = round(albedo, 3)
        spec = round(self.rng.uniform(0.02, 0.15), 3)
        sx, sy, sz = self.p['tube_scale']
        ox, oy, oz = self.p['tube_visual_offset']
        # Estático y sin colisión: solo lo ven las cámaras; la pose la fija el
        # generador. La malla se resuelve vía share del paquete (portable).
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="tube_link">
      <visual name="visual">
        <pose>{ox} {oy} {oz} 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>file://{self.tube_mesh_path}</uri>
            <scale>{sx} {sy} {sz}</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>{a} {a} {a} 1</ambient>
          <diffuse>{dcol} {dcol} {dcol} 1</diffuse>
          <specular>{spec} {spec} {spec} 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def _rock_sdf(self, name, mesh_path, albedo):
        a = round(albedo * 0.85, 3)
        dcol = round(albedo, 3)
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh><uri>file://{mesh_path}</uri></mesh>
        </geometry>
        <material>
          <ambient>{a} {a} {a} 1</ambient>
          <diffuse>{dcol} {dcol} {dcol} 1</diffuse>
          <specular>0.03 0.03 0.03 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def _make_rock_mesh(self, path, half_sizes):
        """Icoesfera subdividida + ruido radial suave → roca irregular (STL)."""
        m = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
        m = m.subdivide_midpoint(number_of_iterations=3)
        v = np.asarray(m.vertices)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        disp = np.zeros(len(v))
        for _ in range(4):  # bultos de baja frecuencia (tipo Perlin barato)
            d = self.nprng.normal(size=3)
            d /= np.linalg.norm(d)
            freq = self.rng.uniform(0.8, 2.6)
            phase = self.rng.uniform(0, 2 * math.pi)
            amp = self.rng.uniform(0.05, 0.16)
            disp += amp * np.sin(freq * math.pi * (v @ d) + phase)
        disp += self.nprng.normal(0.0, 0.02, len(v))  # rugosidad fina
        v = v * (1.0 + disp)[:, None]
        v = v * np.asarray(half_sizes)  # anisotropía (rocas alargadas = hard neg.)
        m.vertices = o3d.utility.Vector3dVector(v)
        m.compute_vertex_normals()
        o3d.io.write_triangle_mesh(path, m)
        ext = v.max(axis=0) - v.min(axis=0)
        return ext

    def _build_pools(self):
        """Spawnea pools reutilizables de tubos y rocas, aparcados fuera de vista."""
        base = self.p['tube_entity_name']
        for i in range(int(self.p['tube_pool_size'])):
            albedo = self.rng.uniform(self.p['tube_albedo_min'],
                                      self.p['tube_albedo_max'])
            name = f'{base}_{i}'
            if self._spawn(name, self._tube_sdf(name, albedo), self._park_T(i)):
                self.tube_pool.append({'name': name, 'albedo': albedo})
        if not self.tube_pool:
            raise RuntimeError('No se pudo spawnear ningún tubo')

        self.rock_dir = tempfile.mkdtemp(prefix='parol6_synth_rocks_')
        n = int(self.p['rock_pool_size'])
        # Reparto: pequeñas / medianas / grandes / alargadas (hard negatives,
        # comparten silueta con el tubo: ~2-5 cm de grosor y 14-24 cm de largo)
        n_small = max(1, round(n * 0.36))
        n_med = max(1, round(n * 0.28))
        n_large = max(1, round(n * 0.14))
        n_elong = max(2, n - n_small - n_med - n_large)
        kinds = (['small'] * n_small + ['medium'] * n_med
                 + ['large'] * n_large + ['elongated'] * n_elong)
        for i, kind in enumerate(kinds):
            if kind == 'small':
                s = self.rng.uniform(0.05, 0.12)
                hs = [s / 2 * self.rng.uniform(0.8, 1.3) for _ in range(3)]
            elif kind == 'medium':
                s = self.rng.uniform(0.12, 0.28)
                hs = [s / 2 * self.rng.uniform(0.8, 1.3) for _ in range(3)]
            elif kind == 'large':
                s = self.rng.uniform(0.28, 0.50)
                hs = [s / 2 * self.rng.uniform(0.8, 1.3) for _ in range(3)]
            else:  # elongated
                length = self.rng.uniform(0.14, 0.24)
                thick = self.rng.uniform(0.02, 0.05)
                hs = [length / 2, thick / 2 * self.rng.uniform(0.9, 1.4),
                      thick / 2 * self.rng.uniform(0.9, 1.4)]
            path = os.path.join(self.rock_dir, f'rock_{i:02d}.stl')
            ext = self._make_rock_mesh(path, hs)
            albedo = self.rng.uniform(0.15, 0.50)  # regolito gris apagado
            name = f'synth_rock_{i}'
            if self._spawn(name, self._rock_sdf(name, path, albedo),
                           self._park_T(100 + i)):
                self.rock_pool.append({'name': name, 'size': ext,
                                       'kind': kind, 'albedo': albedo})
        self.get_logger().info(
            f'Pools listos: {len(self.tube_pool)} tubos, '
            f'{len(self.rock_pool)} rocas en {self.rock_dir}')

    def _park_T(self, i):
        """Pose de aparcamiento detrás de la cámara, fuera del frustum."""
        return make_T(np.eye(3), [-25.0 - 1.5 * (i % 40), -25.0, -8.0])

    def _park_all(self):
        for e in self.active_tubes + self.active_rocks:
            self._set_entity_pose(e, self._park_T(hash(e) % 40))
        self.active_tubes, self.active_rocks = [], []

    # ──────────────────────────────────────────────────────────────────────
    # Brazo
    # ──────────────────────────────────────────────────────────────────────

    def _command_arm(self, q, motion_time=None):
        jt = JointTrajectory()
        jt.joint_names = list(self.p['arm_joint_names'])
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        t = motion_time or self.p['arm_motion_time']
        pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1) * 1e9))
        jt.points = [pt]
        self.arm_pub.publish(jt)

    def _command_gripper(self, opening):
        jt = JointTrajectory()
        jt.joint_names = list(self.p['gripper_joint_names'])
        pt = JointTrajectoryPoint()
        pt.positions = [float(opening)] * len(jt.joint_names)
        pt.time_from_start = DurationMsg(sec=0, nanosec=int(0.8e9))
        jt.points = [pt]
        self.gripper_pub.publish(jt)

    def _wait_arm_settled(self, q_target):
        """Asentamiento por ESTABILIDAD DE POSICIÓN (no velocidades: con la
        interfaz de posición de gazebo_ros2_control son ruido ~10 rad/s)."""
        names = list(self.p['arm_joint_names'])
        tol = self.p['arm_settle_pos_tolerance']
        thr = self.p['arm_settle_pos_threshold']
        deadline = time.monotonic() + self.p['settle_timeout_sec']
        prev, stable = None, 0
        while time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.10)
            with self._lock:
                js = self._joint_state
            if js is None:
                continue
            pos = js[1]
            if any(n not in pos for n in names):
                continue
            q = np.array([pos[n] for n in names])
            err = float(np.max(np.abs(q - np.asarray(q_target))))
            delta = float(np.max(np.abs(q - prev))) if prev is not None else 1.0
            prev = q
            if err < tol and delta < thr:
                stable += 1
                if stable >= 3:
                    return True
            else:
                stable = 0
        return False

    def _arm_links_above_ground(self):
        """True si ningún link comprobado está bajo el margen anti-suelo."""
        for link in self.p['arm_links_check']:
            try:
                tr = self.tf_buffer.lookup_transform(
                    self.p['world_frame'], link, Time())
                if tr.transform.translation.z < self.p['arm_min_link_z']:
                    return False
            except Exception:
                continue  # link sin TF aún: no bloquear por esto
        return True

    def _random_arm_config(self):
        lo = np.asarray(self.p['arm_limits_lower'])
        hi = np.asarray(self.p['arm_limits_upper'])
        margin = 0.05 * (hi - lo)
        return [self.rng.uniform(l, h) for l, h in zip(lo + margin, hi - margin)]

    def _randomize_arm(self):
        """Comanda una configuración válida y espera asentamiento. El
        controlador mantiene la pose (nada de set_model_configuration, que
        colapsa por gravedad al reanudar)."""
        for _ in range(4):
            q = self._random_arm_config()
            self._command_arm(q)
            if self.p['randomize_gripper']:
                self._command_gripper(
                    self.rng.uniform(0.0, self.p['gripper_max_opening']))
            if not self._wait_arm_settled(q):
                self._note_discard('arm_settle_timeout')
                continue
            if self._arm_links_above_ground():
                return q
            self._note_discard('arm_below_ground')
        # Último recurso: home (siempre válida)
        q = list(self.p['arm_home'])
        self._command_arm(q)
        self._wait_arm_settled(q)
        return q

    # ──────────────────────────────────────────────────────────────────────
    # Caché de terreno por retro-proyección
    # ──────────────────────────────────────────────────────────────────────

    def _lookup_T_world_optical(self, side):
        frame = self.p[f'{side}_optical_frame']
        tr = self.tf_buffer.lookup_transform(self.p['world_frame'], frame, Time())
        q = tr.transform.rotation
        t = tr.transform.translation
        return make_T(quat_to_mat(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])

    def _build_terrain_cache(self):
        """Backproyecta el depth GT (escena vacía: solo terreno + brazo en home)
        a puntos 3D world por píxel. Las zonas de brazo quedan excluidas por el
        filtro de altura (terrain_max_z)."""
        self.terrain = {}
        t_ready = self.get_clock().now().nanoseconds
        for side in ('left', 'right'):
            item = self._wait_fresh(f'{side}_depth', t_ready,
                                    self.p['capture_timeout_sec'])
            if item is None:
                raise RuntimeError(f'Sin depth GT de {side} para el caché de terreno')
            depth = item[1]
            info = self._cam_info[side]
            K, w, h = info['K'], info['w'], info['h']
            T = self._lookup_T_world_optical(side)
            us, vs = np.meshgrid(np.arange(w), np.arange(h))
            z = depth.copy()
            z[~np.isfinite(z)] = np.nan
            x = (us - K[0, 2]) / K[0, 0] * z
            y = (vs - K[1, 2]) / K[1, 1] * z
            pts_o = np.stack([x, y, z], axis=-1)
            P = pts_o.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
            P = P.reshape(h, w, 3).astype(np.float32)
            cam = T[:3, 3]
            dist = np.linalg.norm(P - cam, axis=-1)
            r_base = np.hypot(P[..., 0], P[..., 1])
            valid = (np.isfinite(z)
                     & (P[..., 2] < self.p['terrain_max_z'])
                     & (r_base < self.p['terrain_radius'])
                     & (r_base > self.p['min_dist_from_base'])
                     & (dist >= self.p['camera_distance_min'])
                     & (dist <= self.p['camera_distance_max']))
            self.terrain[side] = {'P': P, 'dist': dist.astype(np.float32),
                                  'valid': valid, 'T_w_o': T, 'cam': cam,
                                  'w': w, 'h': h, 'K': K}
            frac = float(valid.mean())
            self.get_logger().info(
                f'Caché de terreno {side}: {frac * 100:.1f}% de píxeles útiles')
            if frac < 0.02:
                raise RuntimeError(
                    f'Caché de terreno {side} casi vacío ({frac * 100:.2f}%). '
                    '¿Hay suelo bajo la cámara en este mundo?')

    def _pick_terrain_pixel(self, side, dist_bin, region_bin, rng):
        """Píxel válido estratificado por bin de distancia y celda de imagen.
        Relaja la estratificación si el cruce de bins queda vacío."""
        c = self.terrain[side]
        nb = int(self.p['n_distance_bins'])
        edges = np.linspace(self.p['camera_distance_min'],
                            self.p['camera_distance_max'], nb + 1)
        m_dist = (c['dist'] >= edges[dist_bin]) & (c['dist'] < edges[dist_bin + 1])
        ni = int(self.p['n_image_bins'])
        bi, bj = divmod(region_bin, ni)
        h, w = c['h'], c['w']
        m_reg = np.zeros((h, w), bool)
        m_reg[bi * h // ni:(bi + 1) * h // ni, bj * w // ni:(bj + 1) * w // ni] = True
        for mask in (c['valid'] & m_dist & m_reg,
                     c['valid'] & m_dist,
                     c['valid']):
            idx = np.flatnonzero(mask)
            if len(idx):
                k = idx[rng.randrange(len(idx))]
                return k // w, k % w  # (v, u)
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Construcción de escena
    # ──────────────────────────────────────────────────────────────────────

    def _sample_tube_T(self, side, mode, dist_bin, region_bin, rng):
        """Pose world del tubo según modo (grounded/floating/buried).
        Devuelve (T, info) o None si no se pudo (frustum, sin píxeles...)."""
        c = self.terrain[side]
        K = c['K']
        for _ in range(25):
            px = self._pick_terrain_pixel(side, dist_bin, region_bin, rng)
            if px is None:
                return None
            v, u = px
            P = c['P'][v, u].astype(float)
            P[0] += rng.uniform(-0.02, 0.02)
            P[1] += rng.uniform(-0.02, 0.02)

            if mode == 'floating':
                R = random_rotation(rng)
            else:
                max_t = math.radians(self.p['grounded_max_tilt_deg'])
                elev = rng.uniform(-max_t, max_t)
                yaw = rng.uniform(0, 2 * math.pi)
                axis = np.array([math.cos(yaw) * math.cos(elev),
                                 math.sin(yaw) * math.cos(elev),
                                 math.sin(elev)])
                R = rotation_with_axis(axis, rng.uniform(0, 2 * math.pi))

            a_z = abs(R[2, 2])
            lowest = (self.tube_half * a_z
                      + self.tube_radius * math.sqrt(max(0.0, 1 - a_z * a_z)))
            z = P[2] + lowest
            bury = 0.0
            if mode == 'grounded':
                z -= rng.uniform(0.0005, 0.002)  # contacto sin z-fighting
            elif mode == 'buried':
                bury = rng.uniform(self.p['bury_min_fraction'],
                                   self.p['bury_max_fraction']) * 2 * self.tube_radius
                z -= bury
            elif mode == 'floating':
                z += rng.uniform(self.p['float_height_min'],
                                 self.p['float_height_max'])
            T = make_T(R, [P[0], P[1], z])

            # Chequeo de frustum: el centro debe proyectar dentro con margen
            T_o_t = inv_T(c['T_w_o']) @ T
            ctr = T_o_t[:3, 3]
            if ctr[2] < 0.25:
                continue
            uc = K[0, 0] * ctr[0] / ctr[2] + K[0, 2]
            vc = K[1, 1] * ctr[1] / ctr[2] + K[1, 2]
            m = 24
            if m <= uc < c['w'] - m and m <= vc < c['h'] - m:
                dist = float(np.linalg.norm(T[:3, 3] - c['cam']))
                return T, {'mode': mode, 'bury_depth': bury, 'dist': dist,
                           'terrain_z': float(P[2])}
        return None

    def _place_occluder_rocks(self, side, tube_T, level, rng, used):
        """Coloca rocas cruzando la línea de visión cámara→tubo para una
        oclusión CONTROLADA (la real la mide luego el depth GT)."""
        c = self.terrain[side]
        C = c['cam']
        P = tube_T[:3, 3]
        n_occ = 1 if level == 'light' else 2
        placed = []
        candidates = [r for r in self.rock_pool if r['name'] not in used]
        rng.shuffle(candidates)
        for rock in candidates:
            if len(placed) >= n_occ:
                break
            hr = float(np.max(rock['size']))
            if level == 'light' and hr > 0.30:
                continue
            if level == 'heavy' and hr < 0.10:
                continue
            drop = max(0.2, C[2] - P[2])
            f = 1.0 - min(0.45, max(0.05, 0.55 * hr / drop))
            f = min(0.96, max(0.55, f + rng.uniform(-0.05, 0.05)))
            Q = C + f * (P - C)
            los = P - C
            perp = np.cross([0.0, 0.0, 1.0], los)
            n = np.linalg.norm(perp)
            perp = perp / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
            lat = (rng.uniform(0.15, 0.55) if level == 'light'
                   else rng.uniform(0.0, 0.30))
            off = perp * lat * 0.186 * (1 if rng.random() < 0.5 else -1)
            # Altura: apoyada en el terreno bajo Q (aprox: z del tubo - bury)
            zg = P[2] - (self.tube_half + self.tube_radius)
            pos = [Q[0] + off[0], Q[1] + off[1],
                   zg + rock['size'][2] * rng.uniform(0.25, 0.45)]
            R = rotz(rng.uniform(0, 2 * math.pi))
            self._set_entity_pose(rock['name'], make_T(R, pos))
            placed.append(rock['name'])
            used.add(rock['name'])
        return placed

    def _scatter_rocks(self, side, n, rng, used):
        """Distractores: rocas en píxeles de terreno aleatorios del frustum."""
        placed = []
        candidates = [r for r in self.rock_pool if r['name'] not in used]
        rng.shuffle(candidates)
        for rock in candidates[:n]:
            px = self._pick_terrain_pixel(
                side, rng.randrange(int(self.p['n_distance_bins'])),
                rng.randrange(int(self.p['n_image_bins']) ** 2), rng)
            if px is None:
                break
            v, u = px
            P = self.terrain[side]['P'][v, u].astype(float)
            z = P[2] + rock['size'][2] * rng.uniform(0.20, 0.45)
            tilt = rng.uniform(0, math.radians(25))
            R = rotz(rng.uniform(0, 2 * math.pi)) @ rotation_with_axis(
                np.array([math.sin(tilt), 0, math.cos(tilt)]), 0.0)
            self._set_entity_pose(rock['name'], make_T(R, [P[0], P[1], z]))
            placed.append(rock['name'])
            used.add(rock['name'])
        return placed

    def _randomize_light(self):
        if self.cli_set_light is None:
            return
        try:
            req = SetLightProperties.Request()
            req.name = 'sun'
            g = self.rng.uniform(0.55, 1.0)
            req.diffuse.r = g
            req.diffuse.g = g * self.rng.uniform(0.92, 1.0)
            req.diffuse.b = g * self.rng.uniform(0.82, 0.95)
            req.diffuse.a = 1.0
            req.attenuation_constant = self.rng.uniform(0.6, 1.0)
            req.attenuation_linear = 0.01
            req.attenuation_quadratic = 0.0
            self._call(self.cli_set_light, req, timeout=5.0)
        except Exception as e:
            if not self._light_warned:
                self.get_logger().warning(f'Iluminación no randomizada: {e}')
                self._light_warned = True

    # ──────────────────────────────────────────────────────────────────────
    # Ground truth por raycasting + depth GT
    # ──────────────────────────────────────────────────────────────────────

    def _instance_gt(self, T_o_t, K, scene_depth, w, h):
        """GT de una instancia: bboxes consciente de oclusión, visib_fract y
        keypoints. z-buffer solo-objeto (raycast) vs depth de escena completa."""
        R, t = T_o_t[:3, :3], T_o_t[:3, 3]
        verts = self.tube_verts_sub @ R.T + t
        zs = verts[:, 2]
        front = zs > 0.01
        if front.sum() < 3:
            return None
        us = K[0, 0] * verts[front, 0] / zs[front] + K[0, 2]
        vs = K[1, 1] * verts[front, 1] / zs[front] + K[1, 2]
        mrg = 4
        u0 = max(0, int(math.floor(us.min())) - mrg)
        u1 = min(w, int(math.ceil(us.max())) + mrg + 1)
        v0 = max(0, int(math.floor(vs.min())) - mrg)
        v1 = min(h, int(math.ceil(vs.max())) + mrg + 1)
        if u1 <= u0 or v1 <= v0:
            return None

        # Raycast del tubo SOLO (amodal) en el crop, en el frame óptico.
        # Rayos con dir_z = 1 → t_hit es directamente la Z (comparable con el
        # z-buffer del depth GT de Gazebo).
        full = self.tube_verts_link @ R.T + t
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(full.astype(np.float32)),
            o3d.core.Tensor(self.tube_tris))
        gu, gv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
        dirs = np.stack([(gu - K[0, 2]) / K[0, 0],
                         (gv - K[1, 2]) / K[1, 1],
                         np.ones_like(gu, dtype=float)], axis=-1)
        rays = np.concatenate(
            [np.zeros_like(dirs), dirs], axis=-1).reshape(-1, 6).astype(np.float32)
        zb = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        zb = zb.reshape(v1 - v0, u1 - u0)

        amodal = np.isfinite(zb)
        n_all = int(amodal.sum())
        if n_all == 0:
            return None
        Dc = scene_depth[v0:v1, u0:u1].copy()
        Dc[~np.isfinite(Dc)] = np.inf  # sin retorno (cielo) = no ocluye
        eps = self.p['depth_visibility_epsilon']
        visible = amodal & (zb <= Dc + eps)
        n_vis = int(visible.sum())

        bbox_obj = mask_to_xywh(amodal, u0, v0)
        bbox_vis = mask_to_xywh(visible, u0, v0) if n_vis else None

        # Keypoints proyectados + visibilidad (vecindad 3×3 en la máscara)
        kp = self.kp_link @ R.T + t
        kp2d, kpvis = [], []
        for p in kp:
            if p[2] <= 0.01:
                kp2d.append([-1.0, -1.0])
                kpvis.append(False)
                continue
            uu = K[0, 0] * p[0] / p[2] + K[0, 2]
            vv = K[1, 1] * p[1] / p[2] + K[1, 2]
            kp2d.append([round(float(uu), 2), round(float(vv), 2)])
            iu, iv = int(round(uu)) - u0, int(round(vv)) - v0
            vis = False
            if 0 <= iu < visible.shape[1] and 0 <= iv < visible.shape[0]:
                sl = visible[max(0, iv - 1):iv + 2, max(0, iu - 1):iu + 2]
                vis = bool(sl.any())
            kpvis.append(vis)

        return {
            'bbox_obj': bbox_obj, 'bbox_visib': bbox_vis,
            'px_count_all': n_all, 'px_count_visib': n_vis,
            'visib_fract': n_vis / n_all,
            'keypoints_2d': kp2d, 'keypoints_visib': kpvis,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Plan de muestreo estratificado
    # ──────────────────────────────────────────────────────────────────────

    def _build_plans(self, n):
        rng = random.Random(self.p['seed'] + 12345)
        n_neg = round(self.p['negative_fraction'] * n)
        neg_set = set(rng.sample(range(n), n_neg)) if n_neg else set()
        nb = int(self.p['n_distance_bins'])
        ncell = int(self.p['n_image_bins']) ** 2
        plans = []
        p_heavy = self.p['heavy_occlusion_fraction']
        p_light = max(0.0, self.p['occlusion_fraction'] - p_heavy)
        for i in range(n):
            cam = rng.choice(['left', 'right'])
            n_rocks = rng.randint(int(self.p['num_rocks_min']),
                                  int(self.p['num_rocks_max']))
            if i in neg_set:
                plans.append({'kind': 'neg', 'cam': cam, 'n_rocks': n_rocks})
                continue
            r = rng.random()
            if r < self.p['frac_grounded']:
                mode = 'grounded'
            elif r < self.p['frac_grounded'] + self.p['frac_floating']:
                mode = 'floating'
            else:
                mode = 'buried'
            r = rng.random()
            occl = 'none'
            if mode != 'floating':  # flotante: oclusión por roca no aplica
                occl = 'heavy' if r < p_heavy else (
                    'light' if r < p_heavy + p_light else 'none')
            n_tubes = 1
            if rng.random() < self.p['multi_tube_prob']:
                n_tubes = rng.randint(2, int(self.p['max_tubes']))
            plans.append({
                'kind': 'pos', 'cam': cam, 'mode': mode, 'occl': occl,
                'n_tubes': n_tubes, 'n_rocks': n_rocks,
                'dist_bin': i % nb,             # ciclos → cobertura uniforme
                'region_bin': (i * 5) % ncell,  # 5 coprimo con 9: barre celdas
            })
        return plans

    def _build_split_map(self, n):
        rng = random.Random(self.p['seed'] + 777)
        ids = list(range(n))
        rng.shuffle(ids)
        r = self.p['split_ratios']
        n_tr = round(r[0] * n)
        n_va = round(r[1] * n)
        split = {}
        for k, gid in enumerate(ids):
            split[gid] = ('train' if k < n_tr else
                          'val' if k < n_tr + n_va else 'test')
        return split

    # ──────────────────────────────────────────────────────────────────────
    # Guardado
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_output(self):
        for split in ('train', 'val', 'test'):
            os.makedirs(os.path.join(self.out_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'bop'), exist_ok=True)
        self._write_data_yaml()
        with open(os.path.join(self.out_dir, 'bop', 'models_info.json'), 'w') as f:
            json.dump(self.models_info, f, indent=1)
        readme = os.path.join(self.out_dir, 'README.md')
        if not os.path.exists(readme):
            with open(readme, 'w') as f:
                f.write(_DATASET_README)
        self.meta_path = os.path.join(self.out_dir, 'meta.jsonl')

    def _write_data_yaml(self):
        data = {
            'path': os.path.abspath(self.out_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['mars2020tube'],
        }
        with open(os.path.join(self.out_dir, 'data.yaml'), 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _flush_bop(self):
        b = os.path.join(self.out_dir, 'bop')
        for name in ('scene_camera', 'scene_gt', 'scene_gt_info'):
            with open(os.path.join(b, f'{name}.json'), 'w') as f:
                json.dump(self.bop[name], f, indent=1)
        with open(os.path.join(self.out_dir, 'split_map.json'), 'w') as f:
            json.dump({str(k): v for k, v in sorted(self.split_of.items())}, f,
                      indent=1)

    def _save_sample(self, gid, split, gray, label_lines, cam_entry, gt_entries,
                     info_entries, meta):
        name = f'img_{gid:06d}'
        # Gris replicado a 3 canales (gris→BGR idéntico): Ultralytics y los
        # pesos preentrenados esperan 3 canales.
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(self.out_dir, 'images', split, name + '.png'), bgr)
        with open(os.path.join(self.out_dir, 'labels', split, name + '.txt'),
                  'w') as f:
            f.write('\n'.join(label_lines) + ('\n' if label_lines else ''))
        k = str(gid)
        self.bop['scene_camera'][k] = cam_entry
        self.bop['scene_gt'][k] = gt_entries
        self.bop['scene_gt_info'][k] = info_entries
        self.split_of[gid] = split
        meta.update({'im_id': gid, 'file': f'images/{split}/{name}.png',
                     'split': split})
        with open(self.meta_path, 'a') as f:
            f.write(json.dumps(meta) + '\n')
        if (gid + 1) % int(self.p['flush_every']) == 0:
            self._flush_bop()

    def _note_discard(self, reason):
        self.discard_reasons[reason] = self.discard_reasons.get(reason, 0) + 1

    # ──────────────────────────────────────────────────────────────────────
    # Bucle principal
    # ──────────────────────────────────────────────────────────────────────

    def _wait_startup(self):
        log = self.get_logger()
        log.info('Esperando cámaras, TF y joint_states…')
        t0 = time.monotonic()
        need = ['left_rgb', 'right_rgb', 'left_depth', 'right_depth']
        while rclpy.ok():
            with self._lock:
                have = all(k in self._latest for k in need)
                have_info = 'left' in self._cam_info and 'right' in self._cam_info
                have_js = self._joint_state is not None
            tf_ok = True
            for side in ('left', 'right'):
                try:
                    self._lookup_T_world_optical(side)
                except Exception:
                    tf_ok = False
            if have and have_info and have_js and tf_ok:
                break
            if time.monotonic() - t0 > 300:
                raise RuntimeError(
                    'Timeout esperando sensores/TF. ¿Está dataset_generation.'
                    'launch.py corriendo con la NavCam (con gt_depth) spawneada?')
            time.sleep(0.5)
        # Verificación: la K del gt_depth debe coincidir con la del RGB
        for side in ('left', 'right'):
            Kd = self._gt_depth_info.get(side)
            if Kd is not None and not np.allclose(
                    Kd, self._cam_info[side]['K'], atol=1e-3):
                log.error(f'K de gt_depth {side} difiere del RGB: el GT de '
                          'oclusión sería inválido. Revisa navcam_stereo.urdf.xacro')
        log.info('Sensores y TF listos.')

    def run(self):
        log = self.get_logger()
        self._wait_startup()
        self._setup_services()
        self._prepare_output()

        # Brazo a home y pools aparcados ANTES del caché de terreno
        self._command_arm(self.p['arm_home'])
        self._wait_arm_settled(self.p['arm_home'])
        self._build_pools()
        time.sleep(0.5)
        self._build_terrain_cache()

        n = int(self.p['num_samples'])
        plans = self._build_plans(n)
        split_map = self._build_split_map(n)
        min_vis = self.p['min_visibility']

        generated = 0
        t_start = time.monotonic()
        log.info(f'Generando {n} muestras válidas en {self.out_dir} …')

        while generated < n and rclpy.ok():
            plan = plans[generated]
            ok = False
            for attempt in range(int(self.p['max_attempts'])):
                # Tras la mitad de intentos, relajar la oclusión planificada
                eff = dict(plan)
                if attempt >= int(self.p['max_attempts']) // 2 + 1 and \
                        eff.get('occl') not in (None, 'none'):
                    eff['occl'] = 'none'
                try:
                    result = self._generate_one(generated, eff, split_map)
                except Exception as e:
                    log.error(f'[{generated + 1}/{n}] excepción: {e}\n'
                              f'{traceback.format_exc(limit=4)}')
                    self._note_discard('exception')
                    result = None
                if result is not None:
                    generated += 1
                    el = time.monotonic() - t_start
                    log.info(
                        f"[{generated}/{n}] {result['summary']} "
                        f"(intento {attempt + 1}, {el / generated:.1f} s/muestra)")
                    ok = True
                    break
            if not ok:
                log.warning(
                    f'Muestra {generated} descartada tras '
                    f"{self.p['max_attempts']} intentos; se replanifica.")
                plans[generated] = {'kind': 'pos', 'cam': plan['cam'],
                                    'mode': 'grounded', 'occl': 'none',
                                    'n_tubes': 1, 'n_rocks': 2,
                                    'dist_bin': 0, 'region_bin': 4}

        self._flush_bop()
        self._write_data_yaml()
        el = time.monotonic() - t_start
        log.info(
            f'FIN: {generated} muestras válidas en {el / 60:.1f} min '
            f'({el / max(1, generated):.1f} s/muestra). Descartes: '
            f'{self.discard_reasons or "ninguno"}')
        counts = {}
        for s in self.split_of.values():
            counts[s] = counts.get(s, 0) + 1
        log.info(f'Reparto por split: {counts}')

    def _generate_one(self, gid, plan, split_map):
        """Genera UNA muestra válida. Devuelve dict con resumen o None."""
        rng = self.rng
        side = plan['cam']
        cache = self.terrain[side]
        K, w, h = cache['K'], cache['w'], cache['h']

        # 1) Física en pausa mientras se recoloca la escena
        self._pause()
        try:
            self._park_all()
            used_rocks = set()
            tubes = []  # [(entity_name, T_world, info)]
            if plan['kind'] == 'pos':
                pool = self.tube_pool[:]
                rng.shuffle(pool)
                modes = [plan['mode']] + [
                    rng.choice(['grounded', 'buried'])
                    for _ in range(plan['n_tubes'] - 1)]
                for k, mode in enumerate(modes[:len(pool)]):
                    if k == 0:
                        s = self._sample_tube_T(side, mode, plan['dist_bin'],
                                                plan['region_bin'], rng)
                    else:
                        s = self._sample_tube_T(
                            side, mode,
                            rng.randrange(int(self.p['n_distance_bins'])),
                            rng.randrange(int(self.p['n_image_bins']) ** 2), rng)
                    if s is None:
                        if k == 0:
                            self._note_discard('placement_failed')
                            return None
                        continue
                    T, info = s
                    ent = pool[k]['name']
                    self._set_entity_pose(ent, T)
                    tubes.append((ent, T, info))
                    self.active_tubes.append(ent)
                if not tubes:
                    self._note_discard('placement_failed')
                    return None
                # Oclusores dirigidos sobre el tubo principal
                if plan['occl'] != 'none':
                    occ = self._place_occluder_rocks(
                        side, tubes[0][1], plan['occl'], rng, used_rocks)
                    self.active_rocks += occ
            # Distractores sueltos
            sc = self._scatter_rocks(side, plan['n_rocks'], rng, used_rocks)
            self.active_rocks += sc
            if self.p['randomize_lighting']:
                self._randomize_light()
        finally:
            self._unpause()

        # 2) Brazo aleatorio sostenido por el controlador
        arm_q = self._randomize_arm()

        # 3) Captura RGB + depth GT posteriores al asentamiento
        t_ready = self.get_clock().now().nanoseconds + int(0.02e9)
        rgb_item = self._wait_fresh(f'{side}_rgb', t_ready,
                                    self.p['capture_timeout_sec'])
        dep_item = self._wait_fresh(f'{side}_depth', t_ready,
                                    self.p['capture_timeout_sec'])
        if rgb_item is None or dep_item is None:
            self._note_discard('capture_timeout')
            return None
        gray, depth = rgb_item[1], dep_item[1]
        if abs(rgb_item[0] - dep_item[0]) > 0.3e9:
            self.get_logger().warning(
                'RGB y depth GT separados >0.3 s (escena estática: GT válido)',
                throttle_duration_sec=10.0)

        # Brazo tapando la lente → descartar intento
        close = np.isfinite(depth) & (depth < self.p['blocked_depth_m'])
        if float(close.mean()) > self.p['blocked_frac']:
            self._note_discard('camera_blocked')
            return None

        # 4) GT por instancia
        T_w_o = self._lookup_T_world_optical(side)
        T_o_w = inv_T(T_w_o)
        gt_entries, info_entries, label_lines = [], [], []
        vis_list, modes_meta, bury_meta = [], [], []
        for ent, T_cmd, info in tubes:
            T_w_t = self._get_entity_T(ent)
            if T_w_t is None:
                T_w_t = T_cmd  # fallback: pose comandada (entidad estática)
            T_o_t = T_o_w @ T_w_t
            g = self._instance_gt(T_o_t, K, depth, w, h)
            if g is None or g['px_count_all'] == 0:
                if ent == tubes[0][0]:
                    self._note_discard('primary_out_of_view')
                    return None
                continue
            R = T_o_t[:3, :3]
            t = T_o_t[:3, 3]
            gt_entries.append({
                'obj_id': 1,
                'cam_R_m2c': [round(float(x), 8) for x in R.reshape(-1)],
                'cam_t_m2c': [round(float(x), 8) for x in t],  # METROS
                'inst_name': ent,
            })
            info_entries.append({
                'bbox_obj': g['bbox_obj'],
                'bbox_visib': g['bbox_visib'] or [0, 0, 0, 0],
                'px_count_all': g['px_count_all'],
                'px_count_visib': g['px_count_visib'],
                'visib_fract': round(g['visib_fract'], 5),
                'keypoints_2d': g['keypoints_2d'],
                'keypoints_visib': g['keypoints_visib'],
                'mode': info['mode'],
                'bury_depth': round(info['bury_depth'], 5),
            })
            vis_list.append(round(g['visib_fract'], 4))
            modes_meta.append(info['mode'])
            bury_meta.append(round(info['bury_depth'], 5))
            # Etiqueta YOLO: SOLO la parte visible y solo si supera el umbral
            if g['visib_fract'] >= self.p['min_visibility'] and g['bbox_visib']:
                x, y, bw, bh = g['bbox_visib']
                cx = min(max((x + bw / 2) / w, 0.0), 1.0)
                cy = min(max((y + bh / 2) / h, 0.0), 1.0)
                label_lines.append(
                    f'0 {cx:.6f} {cy:.6f} {min(bw / w, 1.0):.6f} '
                    f'{min(bh / h, 1.0):.6f}')

        # 5) Política de visibilidad sobre el tubo PRINCIPAL
        hard = False
        split = split_map[gid]
        if plan['kind'] == 'pos':
            if not vis_list or vis_list[0] <= 0.0:
                self._note_discard('visib_zero')
                return None
            if vis_list[0] < self.p['min_visibility']:
                if not self.p['save_hard_samples']:
                    self._note_discard('visib_below_min')
                    return None
                hard = True
                split = 'test'  # las "hard" nunca van a train/val

        cam_entry = {
            'cam_K': [round(float(x), 6) for x in K.reshape(-1)],
            'depth_scale': 1.0,  # depth GT ya en metros
            'cam_R_w2c': [round(float(x), 8) for x in T_o_w[:3, :3].reshape(-1)],
            'cam_t_w2c': [round(float(x), 8) for x in T_o_w[:3, 3]],
        }
        dist0 = tubes[0][2]['dist'] if tubes else None
        meta = {
            'cam': side, 'hard': hard, 'kind': plan['kind'],
            'n_tubes': len(gt_entries), 'visib': vis_list, 'modes': modes_meta,
            'bury_depths': bury_meta, 'dist': dist0,
            'n_rocks': len(self.active_rocks),
            'occl_plan': plan.get('occl', 'none'),
            'arm_q': [round(float(v), 4) for v in arm_q],
            'stamp_ns': rgb_item[0],
        }
        self._save_sample(gid, split, gray, label_lines, cam_entry,
                          gt_entries, info_entries, meta)
        if plan['kind'] == 'neg':
            summary = f'NEGATIVA cam={side} rocas={len(self.active_rocks)}'
        else:
            summary = (f"cam={side} modo={modes_meta[0]} "
                       f"dist={dist0:.2f}m visib={vis_list[0]:.2f} "
                       f"tubos={len(gt_entries)} rocas={len(self.active_rocks)}"
                       + (' HARD→test' if hard else ''))
        return {'summary': summary}


_DATASET_README = """# Dataset sintético PAROL6 / NavCam / mars2020tube (v2)

Generado por `synthetic_dataset_generator_v2.py`. Documentación completa:
`parol6_vision/docs/README_SYNTH_DATASET_V2.md` y
`parol6_vision/docs/train_yolo_navcam.md` en el repo PAROL6.

Claves rápidas:
- Imágenes monocromas guardadas como PNG de 3 canales replicados.
- `labels/` en formato YOLO (bbox de la parte VISIBLE); `.txt` vacío = negativo.
- `bop/` con GT 6D estilo BOP **en METROS** (no mm): `cam_R_m2c`, `cam_t_m2c`.
- `models_info.json` declara `symmetries_continuous` (eje Z) → usar **ADD-S**.
- `visib_fract` por instancia permite estratificar la evaluación por oclusión.
- Muestras con visib < 0.10 van SOLO a test, marcadas `hard` en `meta.jsonl`.
- `split_map.json` y `meta.jsonl` mapean imagen → split y metadatos por muestra.
- Si en el futuro se capturan ambos ojos de una misma escena, ambos deben ir
  al MISMO split (aquí no aplica: una sola cámara por escena).
"""


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticDatasetGeneratorV2()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por el usuario.')
    except Exception as e:
        node.get_logger().error(f'Error fatal: {e}\n{traceback.format_exc()}')
    finally:
        try:
            node._flush_bop()
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
