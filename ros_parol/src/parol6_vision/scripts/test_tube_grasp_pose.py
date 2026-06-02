#!/usr/bin/env python3
"""
test_tube_grasp_pose.py — Test end-to-end del nodo `tube_grasp_pose_node`.

Para cada pose de prueba (tubo en world):
  1. Borra el tubo previo de Gazebo y lo respawnea estático en esa pose.
  2. Publica esa MISMA pose en `/vision/tube_pose` (bypass del ICP).
  3. Espera la pose de agarre publicada por el nodo en `/grasp/pick_pose` y
     `/grasp/approach_pose`.
  4. Espera `/robot_motion_status` para saber si MoveIt resolvió IK y el
     brazo terminó el movimiento.
  5. Lee TF `base_link -> gripper_base_link` para conocer la pose REAL del
     gripper tras el movimiento.
  6. Compara:
       • pick_pose publicado  vs  pick_pose esperado (cálculo redundante)
       • TF final del gripper vs  pick_pose publicado
  7. Devuelve PASS/FAIL por pose.

╔══════════════════════════════════════════════════════════════════════════╗
║  PRE-REQUISITOS                                                            ║
║    • Gazebo + MoveIt arrancados:                                           ║
║         ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py             ║
║                                       camera:=navcam                       ║
║    • Pipeline de grasp (SIN visión, para evitar competencia con ICP):      ║
║         ros2 launch parol6_apps tube_grasp_pipeline.launch.py              ║
║                                  enable_vision:=false                      ║
║                                  enable_ik:=true                           ║
║                                  publish_legacy_pose:=true                 ║
╚══════════════════════════════════════════════════════════════════════════╝

¿POR QUÉ PUBLICAMOS LA POSE DEL SPAWN (Y NO LEEMOS /model_states)?
  La pose del spawn es la GROUND TRUTH exacta y conocida — no hay ruido de
  estimación. Eso aísla el test del comportamiento del ICP y prueba solo el
  nodo de agarre + IK + MoveIt. `/model_states` añadiría dependencia del
  callback de Gazebo y latencia adicional sin aportar nada.

CRITERIOS DE ÉXITO (por pose)
  • grasp_received     ← `/grasp/pick_pose` se publicó en < grasp_timeout
  • pick_err_calc_mm   ← cálculo del nodo coincide con el local (< 0.5 mm)
  • orient_err_calc_°  ← orientación del nodo coincide con la local (< 0.5°)
  • Si --check-arm:
      motion_status == 'done' (en < motion_timeout)
      tf_err_mm < tf_pos_tol_mm (default 15 mm)
      tf_err_° < tf_orient_tol_deg (default 5°)

USO
  ros2 run parol6_vision test_tube_grasp_pose.py
  ros2 run parol6_vision test_tube_grasp_pose.py --check-arm
  ros2 run parol6_vision test_tube_grasp_pose.py --csv /tmp/grasp.csv
  ros2 run parol6_vision test_tube_grasp_pose.py --no-spawn   # no spawnea (test solo cálculo)
  ros2 run parol6_vision test_tube_grasp_pose.py --poses-file mis_poses.yaml
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

import tf2_ros
from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String


# ════════════════════════════════════════════════════════════════════════════
#  Posición HOME del brazo (joint values medidos en simulación).
#  El test mueve el brazo a esta configuración ANTES de cada spawn de tubo
#  para que el brazo nunca esté en la zona donde va a aparecer el tubo, lo
#  que provocaba colisiones en Gazebo al respawnear.
# ════════════════════════════════════════════════════════════════════════════
ARM_JOINT_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
ARM_HOME_POSITIONS = [
    -0.0001287080347163183,
    -0.9599771561327288,
     1.28999018740207,
    -0.005668835824959473,
     0.2581663987092836,
     0.017957190239044962,
]
ARM_GROUP_NAME = 'arm_group'


# ════════════════════════════════════════════════════════════════════════════
#  Poses por defecto (frame world == base_link en este repo).
#  Convención de pose del cuerpo del tubo: eje LARGO sobre +Z del frame.
#    pitch=π/2 → eje del cilindro a lo largo de +X mundo (tubo tumbado).
#    yaw=θ    → rotación adicional alrededor de +Z mundo.
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_TEST_POSES: list[tuple[str, float, float, float, float, float, float]] = [
    # (name,                    x,    y,    z,    R,            P,                              Y)

    # ── Frontal a lo largo de +X (yaw=0) ─────────────────────────────────
    ("frontal_close",           0.18, 0.00, 0.04, 0.0,          math.pi / 2,                    0.0),
    ("frontal_mid",             0.22, 0.00, 0.04, 0.0,          math.pi / 2,                    0.0),
    ("frontal_far",             0.26, 0.00, 0.04, 0.0,          math.pi / 2,                    0.0),

    # ── Lateral a lo largo de Y (yaw=±90°) ───────────────────────────────
    ("lateral_+Y",              0.20, 0.08, 0.04, 0.0,          math.pi / 2,                    math.pi / 2),
    ("lateral_-Y",              0.20, -0.08, 0.04, 0.0,         math.pi / 2,                   -math.pi / 2),
    ("lateral_+Y_close",        0.16, 0.10, 0.04, 0.0,          math.pi / 2,                    math.pi / 2),
    ("lateral_-Y_close",        0.16, -0.10, 0.04, 0.0,         math.pi / 2,                   -math.pi / 2),

    # ── Diagonales (yaw=±30°/45°/60°) ────────────────────────────────────
    ("diag_yaw_+30",            0.20, 0.05, 0.04, 0.0,          math.pi / 2,                    math.pi / 6),
    ("diag_yaw_-30",            0.20, -0.05, 0.04, 0.0,         math.pi / 2,                   -math.pi / 6),
    ("diag_yaw_+45",            0.20, 0.07, 0.04, 0.0,          math.pi / 2,                    math.pi / 4),
    ("diag_yaw_-45",            0.20, -0.07, 0.04, 0.0,         math.pi / 2,                   -math.pi / 4),
    ("diag_yaw_+60",            0.18, 0.08, 0.04, 0.0,          math.pi / 2,                    math.pi / 3),
    ("diag_yaw_-60",            0.18, -0.08, 0.04, 0.0,         math.pi / 2,                   -math.pi / 3),

    # ── Diferentes alturas del tubo (Z) ──────────────────────────────────
    ("height_low",              0.20, 0.00, 0.02, 0.0,          math.pi / 2,                    0.0),
    ("height_high",             0.20, 0.00, 0.10, 0.0,          math.pi / 2,                    0.0),
    ("height_very_high",        0.20, 0.00, 0.15, 0.0,          math.pi / 2,                    0.0),

    # ── Tilts (eje del tubo NO perfectamente horizontal) ─────────────────
    ("tilt_+10_deg",            0.20, 0.00, 0.05, 0.0,          math.pi / 2 + math.radians(10), 0.0),
    ("tilt_-10_deg",            0.20, 0.00, 0.05, 0.0,          math.pi / 2 - math.radians(10), 0.0),
    ("tilt_+20_deg",            0.20, 0.00, 0.06, 0.0,          math.pi / 2 + math.radians(20), 0.0),
    ("tilt_-20_deg",            0.20, 0.00, 0.06, 0.0,          math.pi / 2 - math.radians(20), 0.0),

    # ── Mezclas (yaw + tilt) ─────────────────────────────────────────────
    ("yaw+45_tilt+10",          0.20, 0.06, 0.05, 0.0,          math.pi / 2 + math.radians(10), math.pi / 4),
    ("yaw-45_tilt+10",          0.20, -0.06, 0.05, 0.0,         math.pi / 2 + math.radians(10), -math.pi / 4),
    ("yaw+90_tilt-15",          0.20, 0.07, 0.05, 0.0,          math.pi / 2 - math.radians(15), math.pi / 2),

    # ── Borderline aceptable (eje 65° respecto a vertical, justo bajo el
    #                          umbral por defecto de 70°) ───────────────
    ("borderline_25deg_PASS",   0.20, 0.00, 0.10, 0.0,          math.radians(25),               0.0),

    # ── Casos degenerados (DEBEN rechazarse por el nodo) ─────────────────
    # Cilindro vertical (eje +Z): tilt 90°, sin agarre top-down posible.
    ("vertical_REJECT",         0.18, 0.00, 0.10, 0.0,          0.0,                            0.0),
    # Eje a 15° de la vertical → tilt 75° > umbral 70° → REJECT.
    ("almost_vertical_REJECT",  0.18, 0.00, 0.10, 0.0,          math.radians(15),               0.0),
]


# ════════════════════════════════════════════════════════════════════════════
#  Utilidades de álgebra (mismas que el nodo bajo test; se duplican aquí
#  intencionalmente para no acoplar el test a parol6_apps).
# ════════════════════════════════════════════════════════════════════════════

def rpy_to_quat(roll: float, pitch: float, yaw: float
               ) -> Tuple[float, float, float, float]:
    """RPY (XYZ-fixed, convención URDF) → cuaternión (x, y, z, w)."""
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def mat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Matriz 3x3 → cuaternión (x, y, z, w). Algoritmo de Shepperd."""
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


def gripper_base_to_L6(R_grasp: np.ndarray, p_grasp: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte la pose de `gripper_base_link` a la de `L6_link` aplicando la
    inversa del joint fijo `gripper_base_joint`. Réplica de la función del
    nodo bajo test (se duplica para no acoplar el test a parol6_apps)."""
    R_L6_to_gripper = rpy_to_mat(0.0, -math.pi / 2.0, 0.0)
    p_L6_to_gripper = np.array([0.025, 0.0, -0.065], dtype=np.float64)
    T_L6_to_gripper = np.eye(4)
    T_L6_to_gripper[:3, :3] = R_L6_to_gripper
    T_L6_to_gripper[:3, 3] = p_L6_to_gripper
    T_gripper_to_L6 = np.linalg.inv(T_L6_to_gripper)
    T_base_grasp = np.eye(4)
    T_base_grasp[:3, :3] = R_grasp
    T_base_grasp[:3, 3] = p_grasp
    T_base_L6 = T_base_grasp @ T_gripper_to_L6
    return T_base_L6[:3, :3], T_base_L6[:3, 3]


def expected_grasp(R_tube: np.ndarray, p_tube: np.ndarray,
                   *, finger_offset_z: float, approach_height: float,
                   tilt_threshold_deg: float
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Réplica del cálculo del nodo tube_grasp_pose_node. Devuelve
    (R_grasp, p_pick, p_approach) o None si el caso es degenerado."""
    axis = R_tube @ np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None
    axis = axis / n
    if axis[0] < -1e-6 or (abs(axis[0]) < 1e-6 and axis[1] < 0.0):
        axis = -axis
    x_g = axis
    z_world = np.array([0.0, 0.0, 1.0])
    z_g = z_world - float(np.dot(z_world, x_g)) * x_g
    nz = float(np.linalg.norm(z_g))
    if nz < 1e-6:
        return None
    z_g = z_g / nz
    cos_tilt = float(np.clip(np.dot(z_g, z_world), -1.0, 1.0))
    if math.degrees(math.acos(cos_tilt)) > tilt_threshold_deg:
        return None
    y_g = np.cross(z_g, x_g)
    y_g = y_g / (np.linalg.norm(y_g) + 1e-12)
    z_g = np.cross(x_g, y_g)
    z_g = z_g / (np.linalg.norm(z_g) + 1e-12)
    R_grasp = np.column_stack([x_g, y_g, z_g])
    p_pick = p_tube + finger_offset_z * z_g
    p_approach = p_pick + approach_height * z_world
    return R_grasp, p_pick, p_approach


def angle_between_R(R1: np.ndarray, R2: np.ndarray) -> float:
    """Ángulo (grados) entre dos rotaciones (acos de la traza de R1ᵀR2)."""
    M = R1.T @ R2
    cos_t = float(np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos_t))


# ════════════════════════════════════════════════════════════════════════════
#  Resultado por pose
# ════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TestResult:
    name: str
    spawn_xyz: tuple
    spawn_rpy: tuple
    degenerate: bool                  # caso que DEBE rechazarse (vertical, etc.)
    grasp_received: bool
    approach_received: bool
    pick_err_calc_mm: float           # error entre pick publicado y pick calculado
    orient_err_calc_deg: float
    motion_status: str                # 'done' | 'error' | 'timeout' | 'skipped'
    variant_used: str                 # 'primary' | 'alt' | 'none' | 'skipped'
    tf_err_pos_mm: float              # error TF gripper_base_link vs pick publicado
    tf_err_orient_deg: float
    passed: bool
    fail_reason: str

    @property
    def csv_row(self) -> list:
        return [
            self.name,
            f"{self.spawn_xyz[0]:.4f}", f"{self.spawn_xyz[1]:.4f}", f"{self.spawn_xyz[2]:.4f}",
            f"{math.degrees(self.spawn_rpy[0]):.1f}",
            f"{math.degrees(self.spawn_rpy[1]):.1f}",
            f"{math.degrees(self.spawn_rpy[2]):.1f}",
            "yes" if self.degenerate else "no",
            "yes" if self.grasp_received else "no",
            "yes" if self.approach_received else "no",
            f"{self.pick_err_calc_mm:.2f}" if not math.isnan(self.pick_err_calc_mm) else "n/a",
            f"{self.orient_err_calc_deg:.2f}" if not math.isnan(self.orient_err_calc_deg) else "n/a",
            self.motion_status,
            self.variant_used,
            f"{self.tf_err_pos_mm:.2f}" if not math.isnan(self.tf_err_pos_mm) else "n/a",
            f"{self.tf_err_orient_deg:.2f}" if not math.isnan(self.tf_err_orient_deg) else "n/a",
            "PASS" if self.passed else f"FAIL ({self.fail_reason})",
        ]


# ════════════════════════════════════════════════════════════════════════════
#  Nodo de test
# ════════════════════════════════════════════════════════════════════════════

class TubeGraspTester(Node):

    TUBE_ENTITY = "test_tube_grasp"

    def __init__(self, mesh_path: str, finger_offset_z: float,
                 approach_height: float, tilt_threshold_deg: float) -> None:
        super().__init__("tube_grasp_tester")
        self.mesh_path = mesh_path
        self.finger_offset_z = finger_offset_z
        self.approach_height = approach_height
        self.tilt_threshold_deg = tilt_threshold_deg

        # ── Estado por test ──────────────────────────────────────────────────
        self._last_pick: Optional[PoseStamped] = None
        self._last_approach: Optional[PoseStamped] = None
        self._last_pick_alt: Optional[PoseStamped] = None
        self._last_approach_alt: Optional[PoseStamped] = None
        self._last_motion_status: Optional[str] = None
        # Contador de transiciones de estado, para distinguir el evento de este
        # ciclo de un evento previo persistente.
        self._motion_status_seq = 0

        # ── QoS coherente con el nodo bajo test ──────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Subs ────────────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, "/grasp/pick_pose", self._pick_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, "/grasp/approach_pose", self._approach_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, "/grasp/pick_pose_alt",
            self._pick_alt_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, "/grasp/approach_pose_alt",
            self._approach_alt_cb, sensor_qos)
        self.create_subscription(
            String, "/robot_motion_status", self._status_cb, 10)

        # ── Pub ─────────────────────────────────────────────────────────────
        self._pub_tube = self.create_publisher(
            PoseStamped, "/vision/tube_pose", sensor_qos)
        # El test toma el control del topic legacy: secuencia approach→pick.
        # El nodo bajo test debe tener publish_legacy_pose=False para no
        # competir con nosotros (lo forzamos en main()).
        self._pub_legacy = self.create_publisher(
            Pose, "/pose_target_cuaterniones", 10)

        # ── Gazebo services ─────────────────────────────────────────────────
        self._cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self._cli_delete = self.create_client(DeleteEntity, "/delete_entity")

        # ── Cliente del servicio para cambiar params del nodo bajo test ─────
        self._cli_set_params = self.create_client(
            SetParameters, "/tube_grasp_pose_node/set_parameters")

        # ── Cliente del action de MoveIt (para llevar el brazo a HOME) ──────
        self._move_action_client = ActionClient(self, MoveGroup, "/move_action")

        # ── TF (para leer la pose final del gripper tras moverse) ───────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _pick_cb(self, msg: PoseStamped) -> None:
        self._last_pick = msg

    def _approach_cb(self, msg: PoseStamped) -> None:
        self._last_approach = msg

    def _pick_alt_cb(self, msg: PoseStamped) -> None:
        self._last_pick_alt = msg

    def _approach_alt_cb(self, msg: PoseStamped) -> None:
        self._last_approach_alt = msg

    def _status_cb(self, msg: String) -> None:
        self._last_motion_status = msg.data
        self._motion_status_seq += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Configuración dinámica del nodo bajo test
    # ─────────────────────────────────────────────────────────────────────────

    def set_grasp_params(self, params: dict, timeout_s: float = 3.0) -> bool:
        """Cambia uno o más parámetros del nodo tube_grasp_pose_node por
        servicio `/tube_grasp_pose_node/set_parameters`.

        El tipo del parámetro se infiere del tipo Python del valor (bool,
        int, float, str). Devuelve True solo si TODOS los cambios se
        aplicaron con éxito.
        """
        if not self._cli_set_params.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().warn(
                "Servicio /tube_grasp_pose_node/set_parameters no disponible.")
            return False
        req = SetParameters.Request()
        for name, value in params.items():
            p = Parameter()
            p.name = name
            pv = ParameterValue()
            if isinstance(value, bool):
                pv.type = ParameterType.PARAMETER_BOOL
                pv.bool_value = value
            elif isinstance(value, int):
                pv.type = ParameterType.PARAMETER_INTEGER
                pv.integer_value = value
            elif isinstance(value, float):
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = float(value)
            elif isinstance(value, str):
                pv.type = ParameterType.PARAMETER_STRING
                pv.string_value = value
            else:
                self.get_logger().error(
                    f"set_grasp_params: tipo no soportado para '{name}': "
                    f"{type(value).__name__}")
                return False
            p.value = pv
            req.parameters.append(p)
        fut = self._cli_set_params.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_s)
        if not fut.done() or fut.result() is None:
            return False
        return all(r.successful for r in fut.result().results)

    def set_publish_legacy(self, value: bool, timeout_s: float = 3.0) -> bool:
        """Wrapper de conveniencia para set_grasp_params({publish_legacy_pose})."""
        return self.set_grasp_params(
            {"publish_legacy_pose": bool(value)}, timeout_s=timeout_s)

    # ─────────────────────────────────────────────────────────────────────────
    # Spawn / Delete (SDF estático con la misma escala que mars2020tube)
    # ─────────────────────────────────────────────────────────────────────────

    def _wait_for_service(self, cli, timeout_s: float = 10.0) -> bool:
        return cli.wait_for_service(timeout_sec=timeout_s)

    def delete_tube(self) -> bool:
        if not self._wait_for_service(self._cli_delete, 5.0):
            return False
        req = DeleteEntity.Request()
        req.name = self.TUBE_ENTITY
        fut = self._cli_delete.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.done() and fut.result() is not None:
            return bool(fut.result().success)
        return False

    def spawn_tube(self, x: float, y: float, z: float,
                   roll: float, pitch: float, yaw: float) -> bool:
        if not self._wait_for_service(self._cli_spawn, 5.0):
            self.get_logger().error("/spawn_entity no disponible")
            return False
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        sdf = _SDF_TEMPLATE.format(mesh_path=self.mesh_path)
        req = SpawnEntity.Request()
        req.name = self.TUBE_ENTITY
        req.xml = sdf
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)
        req.initial_pose.orientation.x = qx
        req.initial_pose.orientation.y = qy
        req.initial_pose.orientation.z = qz
        req.initial_pose.orientation.w = qw
        fut = self._cli_spawn.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if fut.done() and fut.result() is not None and fut.result().success:
            return True
        msg = fut.result().status_message if fut.result() else "<no result>"
        self.get_logger().error(f"spawn failed: {msg}")
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Publicación de la pose en /vision/tube_pose
    # ─────────────────────────────────────────────────────────────────────────

    def publish_tube_pose(self, x: float, y: float, z: float,
                          roll: float, pitch: float, yaw: float,
                          frame_id: str = "base_link") -> None:
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._pub_tube.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Mover el brazo a HOME (joint goal directo vía /move_action de MoveIt).
    # Bloqueante. Se usa entre tests para evitar que el brazo esté en la zona
    # del próximo spawn.
    # ─────────────────────────────────────────────────────────────────────────

    def move_arm_to_home(self, timeout_s: float = 15.0,
                         velocity_scaling: float = 0.5) -> str:
        """Devuelve 'done', 'no_server', 'rejected', 'send_timeout',
        'exec_timeout' o 'error_<code>'."""
        if not self._move_action_client.wait_for_server(timeout_sec=5.0):
            return "no_server"

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name                      = ARM_GROUP_NAME
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 5.0
        req.max_velocity_scaling_factor     = velocity_scaling
        req.max_acceleration_scaling_factor = velocity_scaling

        constraints = Constraints()
        for jname, pos in zip(ARM_JOINT_NAMES, ARM_HOME_POSITIONS):
            jc = JointConstraint()
            jc.joint_name      = jname
            jc.position        = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request                          = req
        goal.planning_options                 = PlanningOptions()
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        send_future = self._move_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        if not send_future.done():
            return "send_timeout"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return "rejected"

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout_s)
        if not result_future.done():
            return "exec_timeout"
        result = result_future.result()
        if result is None:
            return "no_result"
        code = result.result.error_code.val
        return "done" if code == 1 else f"error_{code}"

    # ─────────────────────────────────────────────────────────────────────────
    # Envío al brazo: convierte la pose (gripper_base_link) a la de L6_link y
    # la publica en /pose_target_cuaterniones. Espera /robot_motion_status.
    # ─────────────────────────────────────────────────────────────────────────

    def send_to_arm(self, R_grasp: np.ndarray, p_grasp: np.ndarray,
                    motion_timeout_s: float, label: str = "") -> str:
        """Devuelve 'done', 'error', 'timeout' o 'no_subscriber'."""
        R_L6, p_L6 = gripper_base_to_L6(R_grasp, p_grasp)
        qx, qy, qz, qw = mat_to_quat(R_L6)
        msg = Pose()
        msg.position.x = float(p_L6[0])
        msg.position.y = float(p_L6[1])
        msg.position.z = float(p_L6[2])
        msg.orientation.x = float(qx)
        msg.orientation.y = float(qy)
        msg.orientation.z = float(qz)
        msg.orientation.w = float(qw)

        # Asegurar que pose_ik_subscriber está escuchando antes de publicar.
        sub_deadline = time.monotonic() + 2.0
        while (self._pub_legacy.get_subscription_count() < 1
               and time.monotonic() < sub_deadline):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._pub_legacy.get_subscription_count() < 1:
            return "no_subscriber"

        # Reset y publicación ÚNICA — el QoS RELIABLE garantiza la entrega.
        seq_before = self._motion_status_seq
        self._last_motion_status = None
        self._pub_legacy.publish(msg)
        if label:
            self.get_logger().info(
                f"  → enviado al brazo [{label}] "
                f"L6=({p_L6[0]:+.3f},{p_L6[1]:+.3f},{p_L6[2]:+.3f})")

        # Esperar al status de fin de movimiento (sin republicar).
        deadline = time.monotonic() + motion_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self._motion_status_seq > seq_before
                    and self._last_motion_status in ("done", "error")):
                return self._last_motion_status
        return "timeout"

    # ─────────────────────────────────────────────────────────────────────────
    # Lookup TF de la pose final del gripper
    # ─────────────────────────────────────────────────────────────────────────

    def lookup_gripper_pose(self, timeout_s: float = 1.0
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                "base_link", "gripper_base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_s))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        p = np.array([t.x, t.y, t.z], dtype=np.float64)
        R = quat_to_mat(r.x, r.y, r.z, r.w)
        return R, p

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de test para una pose
    # ─────────────────────────────────────────────────────────────────────────

    def run_pose_case(self, name: str,
                      x: float, y: float, z: float,
                      roll: float, pitch: float, yaw: float,
                      do_spawn: bool, check_arm: bool,
                      grasp_timeout_s: float, motion_timeout_s: float,
                      pos_tol_mm: float, orient_tol_deg: float,
                      ) -> TestResult:
        # ── Cálculo local de la pose esperada (para comparación) ─────────────
        R_tube_local = rpy_to_mat(roll, pitch, yaw)
        p_tube_local = np.array([x, y, z], dtype=np.float64)
        expected = expected_grasp(
            R_tube_local, p_tube_local,
            finger_offset_z=self.finger_offset_z,
            approach_height=self.approach_height,
            tilt_threshold_deg=self.tilt_threshold_deg,
        )
        degenerate = expected is None

        # ── Limpieza, retorno a HOME y spawn ─────────────────────────────────
        # Orden importante: primero borramos el tubo previo (libera el espacio
        # para que el path planning a HOME no encuentre obstáculos), luego
        # llevamos el brazo a HOME (estado seguro fuera del workspace donde
        # va a aparecer el siguiente tubo), y por último spawnamos el nuevo
        # tubo. Solo movemos a HOME en modo --check-arm: sin check_arm el
        # brazo no se mueve y mover a home sería trabajo innecesario.
        if do_spawn:
            self.delete_tube()
            time.sleep(0.3)
            if check_arm:
                home_status = self.move_arm_to_home()
                if home_status != "done":
                    self.get_logger().warn(
                        f"No se pudo llevar el brazo a HOME ({home_status}); "
                        f"continúo, pero el siguiente spawn podría colisionar.")
                else:
                    self.get_logger().info("Brazo en HOME, listo para spawn.")
            if not self.spawn_tube(x, y, z, roll, pitch, yaw):
                return _failure(name, (x, y, z), (roll, pitch, yaw),
                                degenerate, "spawn_failed")
            time.sleep(0.3)

        # ── Reset estado ─────────────────────────────────────────────────────
        self._last_pick = None
        self._last_approach = None
        self._last_pick_alt = None
        self._last_approach_alt = None
        self._last_motion_status = None

        # ── Esperar a que el nodo grasp esté escuchando /vision/tube_pose ───
        # Con QoS RELIABLE y un subscriber confirmado, basta UNA publicación.
        sub_deadline = time.monotonic() + 2.0
        while (self._pub_tube.get_subscription_count() < 1
               and time.monotonic() < sub_deadline):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Publicar UNA SOLA VEZ. Si no hay subscriber, no tiene sentido seguir.
        if self._pub_tube.get_subscription_count() < 1:
            return _failure(name, (x, y, z), (roll, pitch, yaw), False,
                            "no_subscriber para /vision/tube_pose")
        self.publish_tube_pose(x, y, z, roll, pitch, yaw)

        # ── Esperar las 4 poses (principal + alt) ───────────────────────────
        grasp_deadline = time.monotonic() + grasp_timeout_s
        while rclpy.ok() and time.monotonic() < grasp_deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self._last_pick is not None
                    and self._last_approach is not None
                    and self._last_pick_alt is not None
                    and self._last_approach_alt is not None):
                break

        # Caso degenerado: esperamos que NO se publique nada → PASS si así fue.
        if degenerate:
            return TestResult(
                name=name,
                spawn_xyz=(x, y, z),
                spawn_rpy=(roll, pitch, yaw),
                degenerate=True,
                grasp_received=self._last_pick is not None,
                approach_received=self._last_approach is not None,
                pick_err_calc_mm=float("nan"),
                orient_err_calc_deg=float("nan"),
                motion_status="skipped",
                variant_used="skipped",
                tf_err_pos_mm=float("nan"),
                tf_err_orient_deg=float("nan"),
                passed=self._last_pick is None,
                fail_reason="" if self._last_pick is None
                            else "pose degenerada admitida por el nodo",
            )

        if self._last_pick is None:
            return _failure(name, (x, y, z), (roll, pitch, yaw), False,
                            "timeout esperando /grasp/pick_pose")
        if self._last_approach is None:
            return _failure(name, (x, y, z), (roll, pitch, yaw), False,
                            "timeout esperando /grasp/approach_pose")

        # ── Comparación cálculo nodo vs cálculo local ───────────────────────
        assert expected is not None
        R_exp, p_pick_exp, _ = expected
        ps = self._last_pick.pose
        p_pub = np.array([ps.position.x, ps.position.y, ps.position.z])
        R_pub = quat_to_mat(ps.orientation.x, ps.orientation.y,
                            ps.orientation.z, ps.orientation.w)
        pick_err_mm = float(np.linalg.norm(p_pub - p_pick_exp) * 1000.0)
        orient_err_deg = angle_between_R(R_pub, R_exp)

        # ── Movimiento (solo si --check-arm) ─────────────────────────────────
        # Secuencia approach → pick para evitar trayectorias rectas que
        # atraviesen el tubo. Cada pose se publica UNA sola vez.
        # Fallback automático: si la pose PRINCIPAL falla (IK no resoluble,
        # configuración inalcanzable, etc.), reintenta con la ALT (rotada
        # 180° en z_g del gripper). El cilindro es simétrico, así que ambas
        # son geométricamente equivalentes; sólo cambia la configuración de
        # joints requerida.
        motion_status = "skipped"
        variant_used = "skipped"
        tf_err_pos_mm = float("nan")
        tf_err_orient_deg = float("nan")
        # `R_final` y `p_final` apuntan a la pose realmente comandada al
        # brazo (primary o alt) — sobre ella se mide el error TF posterior.
        R_final = R_pub
        p_final = p_pub
        if check_arm:
            # Posición del pick es la misma en primary y alt (sólo cambia
            # la orientación), así que extraemos también la pose alt para
            # poder reintentar si la primary falla.
            ps_alt = self._last_pick_alt.pose
            R_pub_alt = quat_to_mat(
                ps_alt.orientation.x, ps_alt.orientation.y,
                ps_alt.orientation.z, ps_alt.orientation.w)
            p_pub_alt = np.array(
                [ps_alt.position.x, ps_alt.position.y, ps_alt.position.z])

            ap = self._last_approach.pose
            R_appr = quat_to_mat(ap.orientation.x, ap.orientation.y,
                                  ap.orientation.z, ap.orientation.w)
            p_appr = np.array([ap.position.x, ap.position.y, ap.position.z])

            ap_alt = self._last_approach_alt.pose
            R_appr_alt = quat_to_mat(
                ap_alt.orientation.x, ap_alt.orientation.y,
                ap_alt.orientation.z, ap_alt.orientation.w)
            p_appr_alt = np.array(
                [ap_alt.position.x, ap_alt.position.y, ap_alt.position.z])

            # Lista de variantes a probar en orden: primary primero, alt
            # como fallback. Cada entry es (label, R_approach, p_approach,
            # R_pick, p_pick).
            variants = [
                ("primary", R_appr,     p_appr,     R_pub,     p_pub),
                ("alt",     R_appr_alt, p_appr_alt, R_pub_alt, p_pub_alt),
            ]

            for label, R_a, p_a, R_p, p_p in variants:
                status_appr = self.send_to_arm(
                    R_a, p_a, motion_timeout_s, label=f"APPROACH-{label}")
                if status_appr != "done":
                    self.get_logger().info(
                        f"  variante '{label}' descartada en approach: "
                        f"{status_appr}")
                    continue
                status_pick = self.send_to_arm(
                    R_p, p_p, motion_timeout_s, label=f"PICK-{label}")
                if status_pick == "done":
                    motion_status = "done"
                    variant_used = label
                    R_final, p_final = R_p, p_p
                    break
                self.get_logger().info(
                    f"  variante '{label}' descartada en pick: {status_pick}")
                motion_status = f"{label}_pick:{status_pick}"
            else:
                # Ninguna variante completó el movimiento (ningún 'break').
                variant_used = "none"
                if motion_status == "skipped":
                    # Se llegó aquí sin haber asignado motion_status — lo que
                    # significa que el último intento falló en approach.
                    motion_status = f"{variants[-1][0]}_approach_failed"

            if motion_status == "done":
                time.sleep(0.4)  # esperar a que TF refleje la pose final
                gripper = self.lookup_gripper_pose(timeout_s=1.0)
                if gripper is not None:
                    R_arm, p_arm = gripper
                    tf_err_pos_mm = float(
                        np.linalg.norm(p_arm - p_final) * 1000.0)
                    tf_err_orient_deg = angle_between_R(R_arm, R_final)

        # ── Veredicto ────────────────────────────────────────────────────────
        passed = True
        fail_reason = ""
        # Cálculo
        if pick_err_mm > 0.5:
            passed = False
            fail_reason = f"pick_err_calc={pick_err_mm:.2f}mm > 0.5"
        elif orient_err_deg > 0.5:
            passed = False
            fail_reason = f"orient_err_calc={orient_err_deg:.2f}° > 0.5"
        # Movimiento
        if check_arm and passed:
            if motion_status != "done":
                passed = False
                fail_reason = f"motion_status={motion_status}"
            elif (not math.isnan(tf_err_pos_mm)
                  and tf_err_pos_mm > pos_tol_mm):
                passed = False
                fail_reason = f"tf_pos_err={tf_err_pos_mm:.1f}mm > {pos_tol_mm}"
            elif (not math.isnan(tf_err_orient_deg)
                  and tf_err_orient_deg > orient_tol_deg):
                passed = False
                fail_reason = f"tf_orient_err={tf_err_orient_deg:.2f}° > {orient_tol_deg}"

        return TestResult(
            name=name,
            spawn_xyz=(x, y, z),
            spawn_rpy=(roll, pitch, yaw),
            degenerate=False,
            grasp_received=True,
            approach_received=self._last_approach is not None,
            pick_err_calc_mm=pick_err_mm,
            orient_err_calc_deg=orient_err_deg,
            motion_status=motion_status,
            variant_used=variant_used,
            tf_err_pos_mm=tf_err_pos_mm,
            tf_err_orient_deg=tf_err_orient_deg,
            passed=passed,
            fail_reason=fail_reason,
        )


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _failure(name, xyz, rpy, degenerate, reason) -> TestResult:
    return TestResult(
        name=name,
        spawn_xyz=tuple(xyz),
        spawn_rpy=tuple(rpy),
        degenerate=degenerate,
        grasp_received=False,
        approach_received=False,
        pick_err_calc_mm=float("nan"),
        orient_err_calc_deg=float("nan"),
        motion_status="error",
        variant_used="none",
        tf_err_pos_mm=float("nan"),
        tf_err_orient_deg=float("nan"),
        passed=False,
        fail_reason=reason,
    )


_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="test_tube_grasp">
    <static>true</static>
    <link name="tube_link">
      <visual name="visual">
        <pose>-0.010497 -0.010489 -0.093000 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>file://{mesh_path}</uri>
            <scale>0.022939 0.022939 0.033212</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.75 0.75 0.75 1</ambient>
          <diffuse>0.85 0.85 0.85 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.01016</radius>
            <length>0.186</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def _resolve_mesh_path(user_path: Optional[str]) -> str:
    if user_path:
        if not os.path.isfile(user_path):
            raise FileNotFoundError(user_path)
        return os.path.abspath(user_path)
    share = get_package_share_directory("parol6_gazebo")
    candidate = os.path.join(share, "meshes", "mars2020tube.stl")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"No se encuentra {candidate}. Pasa --mesh-path con la ruta.")
    return candidate


def _load_poses_yaml(path: str) -> list[tuple]:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    return [(e["name"], float(e["x"]), float(e["y"]), float(e["z"]),
             float(e.get("roll", 0.0)), float(e.get("pitch", 0.0)),
             float(e.get("yaw", 0.0))) for e in d["poses"]]


def _print_table(results: list[TestResult], check_arm: bool) -> None:
    cols = [
        ("pose",         18),
        ("tubo(x,y,z)",  22),
        ("RPY(°)",       18),
        ("gr/ap",        7),
        ("pick_err mm",  13),
        ("ori_err°",     10),
    ]
    if check_arm:
        cols += [
            ("motion",   10),
            ("variant",   9),
            ("tf_pos mm", 11),
            ("tf_ori°",   10),
        ]
    cols.append(("estado", 22))
    width = sum(w for _, w in cols)
    print("─" * width)
    for h, w in cols:
        print(f"{h:<{w}}", end="")
    print()
    print("─" * width)
    for r in results:
        xyz = f"({r.spawn_xyz[0]:+.2f},{r.spawn_xyz[1]:+.2f},{r.spawn_xyz[2]:+.2f})"
        rpy_deg = (math.degrees(r.spawn_rpy[0]),
                   math.degrees(r.spawn_rpy[1]),
                   math.degrees(r.spawn_rpy[2]))
        rpy = f"({rpy_deg[0]:+.0f},{rpy_deg[1]:+.0f},{rpy_deg[2]:+.0f})"
        gp = ("Y" if r.grasp_received else "N") + "/" + (
             "Y" if r.approach_received else "N")
        pick = f"{r.pick_err_calc_mm:.2f}" if not math.isnan(r.pick_err_calc_mm) else "—"
        ori = f"{r.orient_err_calc_deg:.2f}" if not math.isnan(r.orient_err_calc_deg) else "—"
        row = [r.name, xyz, rpy, gp, pick, ori]
        if check_arm:
            tfp = f"{r.tf_err_pos_mm:.1f}" if not math.isnan(r.tf_err_pos_mm) else "—"
            tfo = f"{r.tf_err_orient_deg:.2f}" if not math.isnan(r.tf_err_orient_deg) else "—"
            row += [r.motion_status, r.variant_used, tfp, tfo]
        status = "PASS" if r.passed else f"FAIL: {r.fail_reason[:18]}"
        row.append(status)
        for v, (_, w) in zip(row, cols):
            print(f"{str(v):<{w}}", end="")
        print()
    print("─" * width)


def _print_summary(results: list[TestResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    deg = sum(1 for r in results if r.degenerate)
    primary = sum(1 for r in results if r.variant_used == "primary")
    alt = sum(1 for r in results if r.variant_used == "alt")
    none = sum(1 for r in results
               if r.variant_used == "none" and not r.degenerate)
    print()
    print(f"  TOTAL                : {total}")
    print(f"  PASS                 : {passed}  ({100 * passed / total:.0f}%)")
    print(f"  FAIL                 : {total - passed}")
    print(f"  Casos degenerados    : {deg}  (deben rechazarse por el nodo)")
    if primary or alt or none:
        print(f"  Brazo con variante   : primary={primary}  alt={alt}  "
              f"ambas_fallaron={none}")
    if passed != total:
        print()
        print("  Fallos:")
        for r in results:
            if not r.passed:
                print(f"    • {r.name:<20} → {r.fail_reason}")
    print()


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-arm", action="store_true",
                        help="Esperar a /robot_motion_status y validar la pose "
                             "final del gripper con TF (requiere pose_ik_subscriber).")
    parser.add_argument("--no-spawn", action="store_true",
                        help="No spawnea el tubo en Gazebo (test solo del cálculo).")
    parser.add_argument("--grasp-timeout", type=float, default=4.0,
                        help="Segundos máx. esperando /grasp/pick_pose (default 4).")
    parser.add_argument("--motion-timeout", type=float, default=20.0,
                        help="Segundos máx. esperando el fin del movimiento (default 20).")
    parser.add_argument("--inter-test-delay", type=float, default=5.0,
                        help="Segundos de espera entre tests. Útil cuando "
                             "--check-arm para dejar al brazo estabilizarse "
                             "antes del siguiente spawn (default 5.0).")
    parser.add_argument("--pos-tol-mm", type=float, default=15.0,
                        help="Tolerancia (mm) de pose final del gripper vs pick (default 15).")
    parser.add_argument("--orient-tol-deg", type=float, default=5.0,
                        help="Tolerancia (°) de orientación final vs pick (default 5).")
    parser.add_argument("--finger-offset-z", type=float, default=0.035,
                        help="Offset gripper_base_link→centro entre dedos (m). "
                             "Se propaga al nodo via set_parameters (default 0.035).")
    parser.add_argument("--approach-height", type=float, default=0.04,
                        help="Altura de la pose de approach sobre el pick (m). "
                             "Se propaga al nodo. Default bajado a 0.04 m porque "
                             "el PAROL6 tiene workspace limitado y un approach "
                             "alto a menudo es inalcanzable aunque el pick sí lo sea.")
    parser.add_argument("--tilt-threshold-deg", type=float, default=70.0,
                        help="Umbral de inclinación máx. para aceptar grasp. "
                             "Se propaga al nodo (default 70°).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Escribir resultados en CSV en esta ruta.")
    parser.add_argument("--mesh-path", type=str, default=None,
                        help="Ruta absoluta al .stl (default: parol6_gazebo).")
    parser.add_argument("--poses-file", type=str, default=None,
                        help="YAML con poses personalizadas {poses: [...]}.")
    args = parser.parse_args(argv)

    try:
        mesh_path = _resolve_mesh_path(args.mesh_path) if not args.no_spawn else ""
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    poses = (_load_poses_yaml(args.poses_file) if args.poses_file
             else DEFAULT_TEST_POSES)

    rclpy.init()
    tester = TubeGraspTester(
        mesh_path,
        finger_offset_z=args.finger_offset_z,
        approach_height=args.approach_height,
        tilt_threshold_deg=args.tilt_threshold_deg,
    )

    try:
        # Esperar a que el nodo bajo test esté online (un publisher subscrito)
        deadline = time.monotonic() + 5.0
        while (tester._pub_tube.get_subscription_count() < 1
               and time.monotonic() < deadline):
            rclpy.spin_once(tester, timeout_sec=0.1)
        if tester._pub_tube.get_subscription_count() < 1:
            tester.get_logger().warn(
                "Nadie escucha /vision/tube_pose. ¿Está corriendo "
                "`tube_grasp_pose_node`? Continúo de todos modos.")

        # SIEMPRE silenciamos el topic legacy del nodo bajo test: este test
        # asume el control de /pose_target_cuaterniones para enviar la
        # secuencia approach → pick a su propio ritmo (una sola publicación
        # por movimiento, evitando el "temblor" causado por el flujo continuo
        # del nodo bajo test publicando con cada mensaje recibido).
        # Propagamos también los parámetros geométricos (approach_height,
        # finger_offset_z, tilt_threshold_deg) para que la pose publicada
        # por el nodo refleje los valores que el test usa en su cálculo
        # local de comparación.
        ok_set = tester.set_grasp_params({
            "publish_legacy_pose": False,
            "approach_height":     float(args.approach_height),
            "finger_offset_z":     float(args.finger_offset_z),
            "tilt_threshold_deg":  float(args.tilt_threshold_deg),
        })
        tester.get_logger().info(
            f"Nodo bajo test configurado: publish_legacy_pose=False, "
            f"approach_height={args.approach_height:.3f} m, "
            f"finger_offset_z={args.finger_offset_z:.3f} m, "
            f"tilt_threshold_deg={args.tilt_threshold_deg:.1f}° "
            f"({'OK' if ok_set else 'FALLO al setear parámetros'})")

        results: list[TestResult] = []
        for i, (name, x, y, z, R, P, Y) in enumerate(poses):
            if i > 0 and args.inter_test_delay > 0:
                time.sleep(args.inter_test_delay)
            tester.get_logger().info(
                f"--- Test '{name}'  tubo=({x:.3f},{y:.3f},{z:.3f}) "
                f"RPY=({math.degrees(R):.0f}°,{math.degrees(P):.0f}°,"
                f"{math.degrees(Y):.0f}°)")
            r = tester.run_pose_case(
                name, x, y, z, R, P, Y,
                do_spawn=not args.no_spawn,
                check_arm=args.check_arm,
                grasp_timeout_s=args.grasp_timeout,
                motion_timeout_s=args.motion_timeout,
                pos_tol_mm=args.pos_tol_mm,
                orient_tol_deg=args.orient_tol_deg,
            )
            results.append(r)
            tag = "PASS" if r.passed else f"FAIL ({r.fail_reason})"
            tester.get_logger().info(
                f"    pick_err={r.pick_err_calc_mm:.2f}mm  "
                f"ori_err={r.orient_err_calc_deg:.2f}°  "
                f"motion={r.motion_status}  {tag}")

        if not args.no_spawn:
            tester.delete_tube()

        print()
        print("═" * 70)
        print(" RESULTADOS test_tube_grasp_pose")
        print("═" * 70)
        _print_table(results, args.check_arm)
        _print_summary(results)

        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "x", "y", "z", "roll_deg", "pitch_deg",
                            "yaw_deg", "degenerate", "grasp_received",
                            "approach_received", "pick_err_calc_mm",
                            "orient_err_calc_deg", "motion_status",
                            "variant_used",
                            "tf_err_pos_mm", "tf_err_orient_deg", "status"])
                for r in results:
                    w.writerow(r.csv_row)
            print(f"CSV escrito: {args.csv}")

        return 0 if all(r.passed for r in results) else 1
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
