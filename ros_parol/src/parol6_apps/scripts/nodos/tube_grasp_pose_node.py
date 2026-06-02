#!/usr/bin/env python3
"""
tube_grasp_pose_node.py

Convierte la pose 6D del tubo (publicada por `icp_pose_estimator_node.py` en
`/vision/tube_pose`) en una pose 6D de AGARRE para el gripper del PAROL6,
lista para que el pipeline de IK + MoveIt mueva el brazo.

╔══════════════════════════════════════════════════════════════════════════╗
║  POR QUÉ EXISTE ESTE NODO                                                  ║
║                                                                            ║
║  `icp_pose_estimator_node.py` estima la pose del CENTRO geométrico del      ║
║  tubo, con el eje largo del cilindro alineado con +Z del frame del         ║
║  objeto. Ese mensaje describe DÓNDE está el tubo, pero NO es una pose      ║
║  válida para el end-effector: el IK del brazo necesita la pose deseada     ║
║  de `gripper_base_link` (el tip de `arm_group` según el SRDF), con la      ║
║  orientación adecuada para que los dedos rodeen al tubo y el cierre        ║
║  (eje Y del gripper) sea perpendicular al eje longitudinal.                ║
║                                                                            ║
║  Este nodo cierra ese hueco: escucha la pose 6D del tubo, la transforma    ║
║  a `base_link`, calcula la pose de agarre top-down adecuada y la publica   ║
║  como `PoseStamped` (target tip = `gripper_base_link`, frame =             ║
║  `base_link`). Opcionalmente la republica en el topic legacy plano         ║
║  `/pose_target_cuaterniones` consumido por `pose_ik_subscriber.py`.        ║
╚══════════════════════════════════════════════════════════════════════════╝

CONVENCIÓN DE AGARRE (top-down sobre cilindro horizontal o ligeramente tilteado)

  Dado un eje longitudinal del tubo `axis_world` (vector unitario en
  `base_link`), construimos la base ortonormal del gripper así:

      x_g = axis_world                       (eje largo del cilindro)
      z_g = proj_perp(x_g, +Z_world)         (lo más vertical posible)
      y_g = z_g × x_g                        (right-handed; dir. de cierre)

  El gripper_base_link está por encima del centro del tubo a lo largo de
  +z_g por un offset configurable `finger_offset_z` (≈ 0.035 m por defecto,
  derivado del URDF del gripper: dedos en z=-0.02 con caja de 0.04 m de
  longitud → el centroide del agarre está a -0.035 m del gripper_base_link).
  Así, cuando los dedos cierran, lo hacen alrededor del centro del tubo.

  Por encima del pick, una pose de APPROACH desplazada
  `approach_height` metros en +Z_world (configurable) permite descender
  vertical y limpiamente al pick.

CASO DEGENERADO

  Si el tubo está casi vertical (|axis · Z_world| > cos(tilt_threshold_deg)),
  no existe un agarre top-down: z_g no puede ser vertical sin colisionar
  con la mesa. El nodo lo detecta y omite la publicación, con un warning
  al log. Ajustable con `tilt_threshold_deg` (default 70°). Para ampliar
  a side-grasps, súbelo o cambia la heurística de z_g.

ENTRADAS / SALIDAS

  Entradas:
    `tube_pose_topic`  (PoseStamped, default `/vision/tube_pose`)
        Pose 6D del tubo. Header.frame_id se respeta y se hace lookup TF2
        al frame objetivo (`base_frame`). Si la transformada no está
        disponible se descarta el mensaje con warning.

  Salidas:
    `pick_pose_topic`     (PoseStamped, default `/grasp/pick_pose`)
        Pose final de agarre (gripper_base_link) en `base_frame`.
    `approach_pose_topic` (PoseStamped, default `/grasp/approach_pose`)
        Pose de aproximación (gripper_base_link) en `base_frame`,
        `approach_height` metros por encima del pick (en +Z_world).
    `legacy_pose_topic`   (Pose, opcional, default `/pose_target_cuaterniones`)
        Pose plana para `pose_ik_subscriber.py`. Solo se publica si
        `publish_legacy_pose=true`. Si `legacy_tip_link` ≠ `grasp_tip_link`,
        se aplica la transformación rígida del URDF (gripper_base_joint)
        para devolver la pose en el frame del tip indicado (típicamente
        `L6_link` para compat. con el `pose_ik_subscriber.py` actual).
    TF: `base_frame` → `grasp_tf_child_frame` (default `tube_grasp_target`).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

import tf2_ros
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped


# ── QoS coherente con el publisher del ICP (RELIABLE / VOLATILE / depth 10) ──
TUBE_POSE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)


# ════════════════════════════════════════════════════════════════════════════
# Utilidades de álgebra (sin dependencias externas más allá de NumPy)
# ════════════════════════════════════════════════════════════════════════════

def quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Cuaternión (x, y, z, w) → matriz de rotación 3x3 (numpy float64)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def mat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Matriz de rotación 3x3 → cuaternión (x, y, z, w). Algoritmo de Shepperd."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return tuple(q / np.linalg.norm(q))


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY estilo URDF (extrínseca XYZ = Rz · Ry · Rx) → matriz 3x3."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Empaqueta R (3x3) y p (3,) en una matriz homogénea 4x4."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inversa de una transformación rígida 4x4 (R, p) → (Rᵀ, -Rᵀ p)."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


# ════════════════════════════════════════════════════════════════════════════
# Lógica geométrica del agarre (testeable sin ROS)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GraspGeometry:
    """Resultado puro del cálculo geométrico: pose del gripper en base_link."""
    pick_R: np.ndarray         # 3x3 rotación de gripper_base_link en base_link
    pick_p: np.ndarray         # (3,) posición del gripper_base_link en base_link
    approach_p: np.ndarray     # (3,) posición de aproximación (misma R)
    tube_axis_world: np.ndarray  # (3,) eje largo del tubo en base_link (unitario)


def compute_top_down_grasp(
    R_tube: np.ndarray,
    p_tube: np.ndarray,
    *,
    finger_offset_z: float,
    approach_height: float,
    tilt_threshold_deg: float,
) -> Optional[GraspGeometry]:
    """Calcula la pose top-down de agarre dado el tubo en `base_link`.

    Parámetros
    ----------
    R_tube : matriz 3x3 — rotación del frame del tubo en base_link.
    p_tube : (3,) — posición del centro del tubo en base_link.
    finger_offset_z : separación entre `gripper_base_link` y el centro entre
        los dedos (a lo largo del +Z del gripper). Para el gripper del PAROL6
        ≈ 0.035 m (URDF: dedos en z=-0.02 con caja de 0.04 m → centro -0.035).
    approach_height : metros que el approach se eleva respecto al pick en
        +Z del mundo (no del gripper).
    tilt_threshold_deg : ángulo máximo permitido entre +Z_world y z_g. Si el
        tubo está tan vertical que z_g exigiría inclinarse más, devolvemos
        None (no hay agarre top-down razonable).

    Devuelve
    --------
    GraspGeometry o None si el caso es degenerado.
    """
    # Eje +Z del frame del tubo (convención ICP: eje largo del cilindro).
    axis_world = R_tube @ np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(axis_world))
    if n < 1e-9:
        return None
    axis_world = axis_world / n

    # Resolver la ambigüedad ±180° del cilindro: preferimos que la componente
    # X del eje sea no negativa (si X≈0, que la componente Y sea no negativa).
    # Esto evita que el yaw del gripper "flipe" por el ruido del ICP.
    if axis_world[0] < -1e-6 or (abs(axis_world[0]) < 1e-6 and axis_world[1] < 0.0):
        axis_world = -axis_world

    x_g = axis_world

    # z_g = proyección de +Z_world sobre el plano perpendicular a x_g.
    z_world = np.array([0.0, 0.0, 1.0])
    z_g = z_world - float(np.dot(z_world, x_g)) * x_g
    norm_zg = float(np.linalg.norm(z_g))
    if norm_zg < 1e-6:
        # x_g paralelo a z_world: no hay top-down posible.
        return None
    z_g = z_g / norm_zg

    # Comprobar el tilt máximo (z_g respecto a Z_world).
    cos_tilt = float(np.clip(np.dot(z_g, z_world), -1.0, 1.0))
    tilt_deg = math.degrees(math.acos(cos_tilt))
    if tilt_deg > tilt_threshold_deg:
        return None

    # y_g = z_g × x_g  (right-handed). Garantizamos columnas ortonormales:
    # como x_g y z_g son ya ortogonales por construcción, y_g es unitario.
    y_g = np.cross(z_g, x_g)
    y_g = y_g / (np.linalg.norm(y_g) + 1e-12)

    # Reconstruir z_g final como x_g × y_g para asegurar ortonormalidad exacta.
    z_g = np.cross(x_g, y_g)
    z_g = z_g / (np.linalg.norm(z_g) + 1e-12)

    R_grasp = np.column_stack([x_g, y_g, z_g])

    # Posición de pick: el gripper_base_link se desplaza +finger_offset_z a
    # lo largo de su propio eje Z (que apunta arriba en top-down) respecto
    # al centro del tubo, de modo que el centro entre los dedos coincide
    # con el centro del tubo.
    pick_p = p_tube + finger_offset_z * z_g

    # Aproximación: desplazamiento puro en +Z del mundo (descenso vertical).
    approach_p = pick_p + approach_height * z_world

    return GraspGeometry(
        pick_R=R_grasp,
        pick_p=pick_p,
        approach_p=approach_p,
        tube_axis_world=axis_world,
    )


def gripper_base_to_L6(R_grasp: np.ndarray, p_grasp: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Traduce una pose de `gripper_base_link` a la pose equivalente de `L6_link`.

    Necesario cuando el consumidor (p.ej. `pose_ik_subscriber.py` actual) usa
    `ik_link_name = 'L6_link'` y no `gripper_base_link`. El joint fijo del
    URDF (`gripper_base_joint`) define:

        T_L6→gripper = trans(0.025, 0, -0.065) · rot(rpy=0, -π/2, 0)

    Para que `gripper_base_link` quede en (R_grasp, p_grasp), `L6_link` debe
    estar en T_base_L6 = T_base_grasp · inv(T_L6→gripper).
    """
    # Constantes del URDF (gripper.urdf.xacro, joint gripper_base_joint).
    R_L6_to_gripper = rpy_to_mat(0.0, -math.pi / 2.0, 0.0)
    p_L6_to_gripper = np.array([0.025, 0.0, -0.065], dtype=np.float64)
    T_L6_to_gripper = make_T(R_L6_to_gripper, p_L6_to_gripper)
    T_gripper_to_L6 = inv_T(T_L6_to_gripper)

    T_base_grasp = make_T(R_grasp, p_grasp)
    T_base_L6 = T_base_grasp @ T_gripper_to_L6
    return T_base_L6[:3, :3], T_base_L6[:3, 3]


# ════════════════════════════════════════════════════════════════════════════
# Nodo ROS 2
# ════════════════════════════════════════════════════════════════════════════

class TubeGraspPoseNode(Node):

    def __init__(self) -> None:
        super().__init__('tube_grasp_pose_node')

        # ── Topics ────────────────────────────────────────────────────────────
        self.declare_parameter('tube_pose_topic',     '/vision/tube_pose')
        self.declare_parameter('pick_pose_topic',     '/grasp/pick_pose')
        self.declare_parameter('approach_pose_topic', '/grasp/approach_pose')
        self.declare_parameter('legacy_pose_topic',   '/pose_target_cuaterniones')

        # ── Frames ────────────────────────────────────────────────────────────
        self.declare_parameter('base_frame',           'base_link')
        # `grasp_tip_link` define el tip al que se refieren `pick_pose` y
        # `approach_pose`. Por defecto, `gripper_base_link` (tip de `arm_group`
        # según el SRDF) — el consumidor (orchestrator o un wrapper de IK) debe
        # llamar a /compute_ik con ik_link_name = `gripper_base_link`.
        self.declare_parameter('grasp_tip_link',       'gripper_base_link')
        self.declare_parameter('grasp_tf_child_frame', 'tube_grasp_target')
        self.declare_parameter('broadcast_tf',         True)

        # ── Geometría del agarre ──────────────────────────────────────────────
        # Offset gripper_base_link → punto medio entre los dedos (URDF: -0.035 m
        # en Z del gripper). El gripper_base_link se eleva esa cantidad en +z_g.
        self.declare_parameter('finger_offset_z',    0.035)
        # Altura de la pose de aproximación sobre el pick, en +Z_world.
        self.declare_parameter('approach_height',    0.10)
        # Inclinación máxima admitida entre z_g y +Z_world. Si el tubo es más
        # vertical, no hay agarre top-down razonable y descartamos el mensaje.
        self.declare_parameter('tilt_threshold_deg', 70.0)
        # Si la diferencia de tiempo entre la pose recibida y "ahora" supera
        # `stale_pose_timeout_s`, no se publica nada. 0 desactiva el chequeo.
        self.declare_parameter('stale_pose_timeout_s', 0.0)

        # ── Compatibilidad con pose_ik_subscriber.py (topic legacy plano) ─────
        self.declare_parameter('publish_legacy_pose', False)
        # `legacy_tip_link` indica el tip que asume el consumidor del topic
        # legacy. Si es distinto de `grasp_tip_link`, el nodo aplica la
        # transformación rígida del URDF para devolver una pose coherente
        # (solo soportado: 'gripper_base_link' o 'L6_link').
        self.declare_parameter('legacy_tip_link',     'L6_link')

        # ── Lectura de parámetros ─────────────────────────────────────────────
        self._tube_topic     = self.get_parameter('tube_pose_topic').value
        self._pick_topic     = self.get_parameter('pick_pose_topic').value
        self._approach_topic = self.get_parameter('approach_pose_topic').value
        self._legacy_topic   = self.get_parameter('legacy_pose_topic').value
        self._base_frame     = self.get_parameter('base_frame').value
        self._grasp_tip      = self.get_parameter('grasp_tip_link').value
        self._tf_child       = self.get_parameter('grasp_tf_child_frame').value
        self._broadcast_tf   = bool(self.get_parameter('broadcast_tf').value)
        self._finger_off_z   = float(self.get_parameter('finger_offset_z').value)
        self._approach_h     = float(self.get_parameter('approach_height').value)
        self._tilt_max_deg   = float(self.get_parameter('tilt_threshold_deg').value)
        self._stale_to       = float(self.get_parameter('stale_pose_timeout_s').value)
        self._publish_legacy = bool(self.get_parameter('publish_legacy_pose').value)
        self._legacy_tip     = self.get_parameter('legacy_tip_link').value

        if self._legacy_tip not in ('gripper_base_link', 'L6_link'):
            self.get_logger().warn(
                f"legacy_tip_link='{self._legacy_tip}' no soportado; "
                f"solo 'gripper_base_link' o 'L6_link'. Se asume 'L6_link'.")
            self._legacy_tip = 'L6_link'

        # ── TF2 ──────────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Publicadores ─────────────────────────────────────────────────────
        self._pub_pick = self.create_publisher(PoseStamped, self._pick_topic, 10)
        self._pub_approach = self.create_publisher(
            PoseStamped, self._approach_topic, 10)
        # El publisher del topic legacy se crea SIEMPRE para soportar el cambio
        # dinámico de `publish_legacy_pose` (callback más abajo). Si el flag
        # está a False, simplemente no se llama a `.publish()`.
        self._pub_legacy: Publisher = self.create_publisher(
            Pose, self._legacy_topic, 10)

        # ── Callback para parámetros dinámicos ──────────────────────────────
        # Permite cambiar `publish_legacy_pose` en caliente con `ros2 param set`
        # o con el servicio /set_parameters. Útil para tests que quieren
        # alternar entre "solo calcular" y "mover el brazo".
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Suscriptor ───────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self.create_subscription(
            PoseStamped, self._tube_topic, self._tube_pose_cb, TUBE_POSE_QOS)

        self.get_logger().info(
            f"TubeGraspPoseNode iniciado.\n"
            f"  Entrada: {self._tube_topic}\n"
            f"  Pick    : {self._pick_topic} (tip={self._grasp_tip}, frame={self._base_frame})\n"
            f"  Approach: {self._approach_topic} (+{self._approach_h:.3f} m Z_world)\n"
            f"  Legacy  : {'ON  ' + self._legacy_topic if self._publish_legacy else 'OFF'} "
            f"(tip={self._legacy_tip})\n"
            f"  finger_offset_z={self._finger_off_z:.3f} m  "
            f"tilt_max={self._tilt_max_deg:.1f}°")

    # ─────────────────────────────────────────────────────────────────────────

    def _tube_pose_cb(self, msg: PoseStamped) -> None:
        # 1) Chequeo opcional de staleness.
        if self._stale_to > 0.0:
            stamp = msg.header.stamp
            now = self.get_clock().now()
            msg_t = stamp.sec + stamp.nanosec * 1e-9
            now_t = now.nanoseconds * 1e-9
            if now_t - msg_t > self._stale_to:
                self.get_logger().warn(
                    f"Pose del tubo demasiado antigua "
                    f"(Δt={now_t - msg_t:.2f}s > {self._stale_to:.2f}s); ignorando.")
                return

        # 2) Transformar al frame del robot si hace falta.
        if msg.header.frame_id and msg.header.frame_id != self._base_frame:
            transformed = self._transform_pose(msg, self._base_frame)
            if transformed is None:
                return
            msg = transformed

        # 3) Construir la pose del tubo en base_link como (R_tube, p_tube).
        p = msg.pose.position
        o = msg.pose.orientation
        R_tube = quat_to_mat(o.x, o.y, o.z, o.w)
        p_tube = np.array([p.x, p.y, p.z], dtype=np.float64)

        # 4) Calcular pose de agarre.
        geom = compute_top_down_grasp(
            R_tube, p_tube,
            finger_offset_z=self._finger_off_z,
            approach_height=self._approach_h,
            tilt_threshold_deg=self._tilt_max_deg,
        )
        if geom is None:
            self.get_logger().warn(
                "Pose del tubo no admite agarre top-down "
                "(tubo casi vertical o eje degenerado); ignorando.",
                throttle_duration_sec=2.0)
            return

        # 5) Publicar pick + approach (PoseStamped) y TF de debug.
        stamp = msg.header.stamp
        qx, qy, qz, qw = mat_to_quat(geom.pick_R)

        pick_ps = PoseStamped()
        pick_ps.header.stamp = stamp
        pick_ps.header.frame_id = self._base_frame
        pick_ps.pose.position.x = float(geom.pick_p[0])
        pick_ps.pose.position.y = float(geom.pick_p[1])
        pick_ps.pose.position.z = float(geom.pick_p[2])
        pick_ps.pose.orientation.x = float(qx)
        pick_ps.pose.orientation.y = float(qy)
        pick_ps.pose.orientation.z = float(qz)
        pick_ps.pose.orientation.w = float(qw)
        self._pub_pick.publish(pick_ps)

        approach_ps = PoseStamped()
        approach_ps.header = pick_ps.header
        approach_ps.pose.position.x = float(geom.approach_p[0])
        approach_ps.pose.position.y = float(geom.approach_p[1])
        approach_ps.pose.position.z = float(geom.approach_p[2])
        approach_ps.pose.orientation = pick_ps.pose.orientation
        self._pub_approach.publish(approach_ps)

        if self._broadcast_tf:
            tf_msg = TransformStamped()
            tf_msg.header = pick_ps.header
            tf_msg.child_frame_id = self._tf_child
            tf_msg.transform.translation.x = pick_ps.pose.position.x
            tf_msg.transform.translation.y = pick_ps.pose.position.y
            tf_msg.transform.translation.z = pick_ps.pose.position.z
            tf_msg.transform.rotation = pick_ps.pose.orientation
            self._tf_broadcaster.sendTransform(tf_msg)

        # 6) Topic legacy plano (Pose) — opcional, controlado por el flag
        # dinámico `publish_legacy_pose`.
        if self._publish_legacy:
            if self._legacy_tip == self._grasp_tip:
                R_out, p_out = geom.pick_R, geom.pick_p
            else:  # legacy_tip == 'L6_link' (asegurado en __init__)
                R_out, p_out = gripper_base_to_L6(geom.pick_R, geom.pick_p)

            qxl, qyl, qzl, qwl = mat_to_quat(R_out)
            legacy = Pose()
            legacy.position.x = float(p_out[0])
            legacy.position.y = float(p_out[1])
            legacy.position.z = float(p_out[2])
            legacy.orientation.x = float(qxl)
            legacy.orientation.y = float(qyl)
            legacy.orientation.z = float(qzl)
            legacy.orientation.w = float(qwl)
            self._pub_legacy.publish(legacy)

        self.get_logger().info(
            f"Grasp publicado: pick=({geom.pick_p[0]:.3f}, {geom.pick_p[1]:.3f}, "
            f"{geom.pick_p[2]:.3f})  eje_tubo=({geom.tube_axis_world[0]:.2f}, "
            f"{geom.tube_axis_world[1]:.2f}, {geom.tube_axis_world[2]:.2f})",
            throttle_duration_sec=1.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Parámetros dinámicos
    # ─────────────────────────────────────────────────────────────────────────

    def _on_param_change(self, params) -> SetParametersResult:
        """Acepta cambios en caliente de varios parámetros del nodo.

        Útil para tests que quieren alternar entre "solo calcular y publicar el
        grasp" (publish_legacy_pose=False) y "mover el brazo" (=True), sin
        necesidad de reiniciar el nodo. Los topics y frames se fijan en
        `__init__` y no se aceptan aquí (requieren reinicio).
        """
        for p in params:
            if p.name == 'publish_legacy_pose':
                new_val = bool(p.value)
                if new_val != self._publish_legacy:
                    self.get_logger().info(
                        f"publish_legacy_pose: {self._publish_legacy} → {new_val}")
                    self._publish_legacy = new_val
            elif p.name == 'legacy_tip_link':
                if p.value not in ('gripper_base_link', 'L6_link'):
                    return SetParametersResult(
                        successful=False,
                        reason=f"legacy_tip_link='{p.value}' no soportado")
                self._legacy_tip = p.value
            elif p.name == 'finger_offset_z':
                self._finger_off_z = float(p.value)
            elif p.name == 'approach_height':
                self._approach_h = float(p.value)
            elif p.name == 'tilt_threshold_deg':
                self._tilt_max_deg = float(p.value)
        return SetParametersResult(successful=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TF helper (replicado del icp_pose_estimator_node para no acoplar
    # parol6_apps con parol6_vision; las firmas de tf2_geometry_msgs varían
    # entre distros de ROS 2).
    # ─────────────────────────────────────────────────────────────────────────

    def _transform_pose(self, pose: PoseStamped, target_frame: str
                       ) -> Optional[PoseStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame, pose.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f"TF {pose.header.frame_id} → {target_frame} no disponible "
                f"({exc.__class__.__name__}); descarto el mensaje.",
                throttle_duration_sec=2.0)
            return None

        t = tf.transform.translation
        r = tf.transform.rotation
        T_tf = make_T(quat_to_mat(r.x, r.y, r.z, r.w),
                      np.array([t.x, t.y, t.z]))

        p = pose.pose.position
        o = pose.pose.orientation
        T_pose = make_T(quat_to_mat(o.x, o.y, o.z, o.w),
                        np.array([p.x, p.y, p.z]))

        T_out = T_tf @ T_pose
        out = PoseStamped()
        out.header.stamp = pose.header.stamp
        out.header.frame_id = target_frame
        out.pose.position.x = float(T_out[0, 3])
        out.pose.position.y = float(T_out[1, 3])
        out.pose.position.z = float(T_out[2, 3])
        qx, qy, qz, qw = mat_to_quat(T_out[:3, :3])
        out.pose.orientation.x = float(qx)
        out.pose.orientation.y = float(qy)
        out.pose.orientation.z = float(qz)
        out.pose.orientation.w = float(qw)
        return out


# ════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TubeGraspPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
