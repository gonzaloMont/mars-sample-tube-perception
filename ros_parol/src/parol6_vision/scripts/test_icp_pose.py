#!/usr/bin/env python3
"""
test_icp_pose.py — Test de rendimiento end-to-end del pipeline YOLO + ICP.

Spawnea el tubo `mars2020tube` en una serie de poses conocidas, mide las
estimaciones publicadas por `/vision/tube_pose`, y calcula métricas
de calidad y rendimiento.

╔══════════════════════════════════════════════════════════════════════════╗
║  PRE-REQUISITOS                                                            ║
║    • Gazebo arrancado con la NavCam estéreo:                               ║
║         ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py             ║
║                                       camera:=navcam                       ║
║    • Nube estéreo + pipeline ICP corriendo:                                ║
║         ros2 run parol6_vision stereo_pointcloud_node.py                   ║
║         ros2 launch parol6_vision icp_pose.launch.py                       ║
║    • YOLO activado (el test lo activa por su cuenta, ver --auto-activate). ║
╚══════════════════════════════════════════════════════════════════════════╝

MÉTRICAS POR POSE
  • error_trans (mm)   ── ||EST_pos − GT_pos|| en el frame `world`
  • error_axis (deg)   ── ángulo entre el eje del cilindro GT y el estimado
                          (con ambigüedad de 180° del cilindro resuelta por abs)
  • fitness, rmse_mm   ── métricas del ICP de Open3D, último valor visto
  • latency_ms         ── time-to-first-pose desde el spawn
  • repetibilidad      ── stddev (mm, deg) sobre las N muestras de la pose

CRITERIO DE ÉXITO (por pose)
  • error_trans < trans_thresh_mm  (default 5.0)
  • error_axis  < axis_thresh_deg  (default 5.0)
  • pose efectivamente publicada (no rechazada por fitness)

USO
  ros2 run parol6_vision test_icp_pose.py                # poses por defecto
  ros2 run parol6_vision test_icp_pose.py --csv /tmp/r.csv
  ros2 run parol6_vision test_icp_pose.py --samples 5 --timeout 15
  ros2 run parol6_vision test_icp_pose.py --poses-file mis_poses.yaml
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
import rclpy.time
from rclpy.node import Node
from rclpy.task import Future
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
import tf2_ros

from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

try:
    import cv2
    from cv_bridge import CvBridge
    _DEBUG_DEPS_OK = True
except ImportError:
    _DEBUG_DEPS_OK = False

try:
    import open3d as o3d
    _O3D_OK = True
except ImportError:
    _O3D_OK = False


# ════════════════════════════════════════════════════════════════════════════
#  POSES DE TEST (por defecto). Cada entrada: (x, y, z, roll, pitch, yaw).
#  Todas en el frame `world`. Convenios:
#    • pitch=π/2  → tubo horizontal: el eje largo (Z_link) cae en plano XY.
#    • pitch=0    → tubo vertical:   el eje largo apunta a world +Z.
#    • yaw rota el eje horizontal del cilindro alrededor de world Z.
#
#  HUELLA DE LA NAVCAM EN MÁSTIL (default `navcam_pose:=mast`)
#  El rig está en base_link (-0.3,-0.3,1.11) con RPY(0, 0.91, 0.3): mira al
#  suelo inclinado 52° y girado 17°. Su eje óptico corta el plano del suelo
#  en ≈ (0.50, 0.00) a ~1.39 m, y la zona visible útil sobre el suelo es
#  aprox.  x ∈ [-0.25, 0.80],  y ∈ [-0.75, 0.60]  (con ~1.2–1.5 m de
#  profundidad, dentro del rango estéreo [0.30, 2.0] m).
#
#  ZONA MUERTA DEL ESTÉREO
#  El SGBM no calcula disparidad en las primeras `num_disparities` columnas de
#  la imagen rectificada (escalada por scale_factor). Con num_disparities=96 y
#  scale_factor=0.5 eso son ~192 columnas en la imagen original 1024² (u_min).
#  Las poses de abajo se eligieron para que el tubo (centro Y AMBAS PUNTAS)
#  proyecte con u ≥ u_min+margen y caiga en el FOV del mástil. Se verificaron
#  con la cadena real de TF (ver /tmp scratch o re-derivar de navcam_stereo.urdf
#  + el static_transform del launch). Si cambias `navcam_pose`/`mast_*` o la
#  pose del robot, re-verifica proyectando con las extrínsecas de TF.
#
#  GRUPOS DE POSES — diseñados para AISLAR la causa de fallo:
#    A) Posición XY (yaw=0, pitch=π/2 fijo): ¿depende del lugar?
#    B) Yaw (XY fijo, pitch=π/2): ¿depende de la rotación del eje horizontal?
#    C) Altura Z (XY y orientación fijos): ¿depende de la altura sobre el suelo?
#    D) Pitch (XY fijo, yaw=0): ¿depende de la inclinación (horiz vs vert)?
#    E) Edge cases conocidos del set original (regresión).
# ════════════════════════════════════════════════════════════════════════════
#  Centradas en la huella del MÁSTIL (eje óptico → suelo ≈ (0.50, 0.00)).
#  Verificadas: centro + ambas puntas del tubo dentro de FOV + zona útil.
DEFAULT_TEST_POSES: list[tuple[str, float, float, float, float, float, float]] = [
    # ── Grupo A: barrido de posición XY (tubo horizontal yaw=0) ──────────────
    # (name,                  x,     y,     z,    R, P,           Y)
    ("A1_center",             0.50,  0.00, 0.01, 0, math.pi / 2, 0.0),
    ("A2_y_neg",              0.50, -0.20, 0.01, 0, math.pi / 2, 0.0),
    ("A3_y_pos",              0.45,  0.20, 0.01, 0, math.pi / 2, 0.0),
    ("A4_far_x",              0.65,  0.00, 0.01, 0, math.pi / 2, 0.0),
    ("A5_near_x",             0.35,  0.00, 0.01, 0, math.pi / 2, 0.0),

    # ── Grupo B: barrido de yaw (XY fijo en el centro, pitch=π/2) ────────────
    ("B1_yaw_0",              0.50,  0.00, 0.01, 0, math.pi / 2,  0.0),
    ("B2_yaw_30",             0.50,  0.00, 0.01, 0, math.pi / 2,  math.pi / 6),
    ("B3_yaw_45",             0.50,  0.00, 0.01, 0, math.pi / 2,  math.pi / 4),
    ("B4_yaw_60",             0.50,  0.00, 0.01, 0, math.pi / 2,  math.pi / 3),
    ("B5_yaw_90",             0.50,  0.00, 0.01, 0, math.pi / 2,  math.pi / 2),
    ("B6_yaw_135",            0.50,  0.00, 0.01, 0, math.pi / 2,  3 * math.pi / 4),
    ("B7_yaw_neg_45",         0.50,  0.00, 0.01, 0, math.pi / 2, -math.pi / 4),
    ("B8_yaw_neg_90",         0.50,  0.00, 0.01, 0, math.pi / 2, -math.pi / 2),

    # ── Grupo C: barrido de altura Z (XY fijo, horizontal yaw=0) ─────────────
    ("C1_z_0.01",             0.50,  0.00, 0.01, 0, math.pi / 2, 0.0),
    ("C2_z_0.05",             0.50,  0.00, 0.05, 0, math.pi / 2, 0.0),
    ("C3_z_0.10",             0.50,  0.00, 0.10, 0, math.pi / 2, 0.0),

    # ── Grupo D: barrido de pitch (XY fijo, yaw=0) ───────────────────────────
    # pitch=0 => tubo vertical (eje en world +Z); π/4 => 45°; π/2 => horizontal
    ("D1_vertical_pitch_0",   0.50,  0.00, 0.10, 0, 0.0,         0.0),
    ("D2_pitch_45",           0.50,  0.00, 0.05, 0, math.pi / 4, 0.0),
    ("D3_pitch_90",           0.50,  0.00, 0.01, 0, math.pi / 2, 0.0),

    # ── Grupo E: edge cases (bordes de la huella, lejos, laterales) ──────────
    ("E1_left_edge",          0.40,  0.30, 0.01, 0, math.pi / 2,  0.0),
    ("E2_right",              0.55, -0.30, 0.01, 0, math.pi / 2,  0.0),
    ("E3_far_yaw_45",         0.62, -0.10, 0.01, 0, math.pi / 2,  math.pi / 4),
    ("E4_near_yaw_90",        0.38,  0.10, 0.01, 0, math.pi / 2,  math.pi / 2),
]


# ════════════════════════════════════════════════════════════════════════════
#  Geometría auxiliar
# ════════════════════════════════════════════════════════════════════════════

def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """RPY (XYZ-fixed) → cuaternión (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


# ─────────────────────────────────────────────────────────────────────────────
#  Extrínsecas de la cámara (su pose en `world`).
#
#  ⚠️  NO se hardcodean. Se obtienen en runtime de TF2, igual que hace el nodo
#      ICP para publicar en `world`. Así el test es AGNÓSTICO a la cámara y a su
#      pose: funciona con la NavCam en mástil (`navcam_pose:=mast`, el default),
#      cenital (`navcam_pose:=original`) o la cámara overhead, sin tocar código.
#
#      Antes estaban FIJADAS a la pose cenital (rig en (0.2,0,0.75), 90° en Y).
#      Al pasar a la NavCam en mástil eso rompía las proyecciones de debug y la
#      conversión óptico→world. Ahora se leen de TF (ver IcpPoseTester._world_from).
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WORLD_FRAME = "world"
DEFAULT_CAMERA_FRAME = "navcam_left_optical_frame"


# ════════════════════════════════════════════════════════════════════════════
#  Resultado por pose
# ════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class PoseResult:
    name: str
    gt_xyz: tuple[float, float, float]
    gt_rpy: tuple[float, float, float]
    n_samples: int
    n_received: int
    error_trans_mm: float
    error_axis_deg: float
    repeat_trans_mm: float            # stddev de la traslación entre muestras
    repeat_axis_deg: float            # stddev del eje entre muestras
    latency_ms: float                 # tiempo desde spawn hasta 1ª pose
    fitness_mean: float
    rmse_mm_mean: float
    passed_trans: bool
    passed_axis: bool
    passed: bool                      # both criteria

    @property
    def csv_row(self) -> list:
        return [
            self.name,
            f"{self.gt_xyz[0]:.4f}", f"{self.gt_xyz[1]:.4f}", f"{self.gt_xyz[2]:.4f}",
            f"{math.degrees(self.gt_rpy[0]):.1f}",
            f"{math.degrees(self.gt_rpy[1]):.1f}",
            f"{math.degrees(self.gt_rpy[2]):.1f}",
            f"{self.n_received}/{self.n_samples}",
            f"{self.error_trans_mm:.2f}",
            f"{self.error_axis_deg:.2f}",
            f"{self.repeat_trans_mm:.2f}",
            f"{self.repeat_axis_deg:.2f}",
            f"{self.latency_ms:.0f}",
            f"{self.fitness_mean:.3f}",
            f"{self.rmse_mm_mean:.2f}",
            "PASS" if self.passed else ("FAIL_AXIS" if not self.passed_axis else "FAIL_TRANS"),
        ]


# ════════════════════════════════════════════════════════════════════════════
#  Nodo de orquestación
# ════════════════════════════════════════════════════════════════════════════

class IcpPoseTester(Node):

    TUBE_ENTITY = "test_icp_tube"
    AXIS_OBJ = np.array([0.0, 0.0, 1.0])  # eje largo del cilindro en object frame

    def __init__(self, mesh_path: str, debug_dir: Optional[str] = None,
                 world_frame: str = DEFAULT_WORLD_FRAME,
                 camera_frame: str = DEFAULT_CAMERA_FRAME) -> None:
        super().__init__("icp_pose_tester")
        self.mesh_path = mesh_path
        self.debug_dir = debug_dir

        # ── Frames (extrínsecas vía TF, NO hardcodeadas) ─────────────────────
        # world_frame:  donde vive el ground-truth (/model_states de Gazebo) y
        #               donde el nodo ICP publica la pose (output_frame: world).
        # camera_frame: frame óptico de la cámara; se autocorrige con el
        #               header.frame_id del camera_info cuando llega.
        self.world_frame = world_frame
        self.camera_frame = camera_frame
        # Cache de extrínsecas world<-frame (la TF es estática).
        self._tf_world_from: dict = {}

        # ── Estado ────────────────────────────────────────────────────────────
        self._gt_pos: Optional[np.ndarray] = None
        self._gt_R: Optional[np.ndarray] = None
        self._estimates: list[tuple[float, np.ndarray, np.ndarray]] = []
        #     cada item = (t_recepcion, pos_world, R_world)

        # ── Estado de debug ──────────────────────────────────────────────────
        self._last_image: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None
        self._last_bbox: Optional[Tuple[float, float, float, float]] = None
        self._bridge = CvBridge() if _DEBUG_DEPS_OK else None
        # Cache de proyecciones del mesh en pose ICP para overlay
        self._mesh_points_obj: Optional[np.ndarray] = None
        if debug_dir and _DEBUG_DEPS_OK and _O3D_OK:
            try:
                # Cargar mesh con misma escala que el nodo ICP (no recentrar:
                # debe quedar tal cual el visual del SDF, alineado al link)
                mesh = o3d.io.read_triangle_mesh(self.mesh_path)
                scale = np.array([0.022939, 0.022939, 0.033212])
                verts = np.asarray(mesh.vertices) * scale
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                bbox = mesh.get_axis_aligned_bounding_box()
                mesh.translate(-bbox.get_center())
                pcd = mesh.sample_points_uniformly(number_of_points=500)
                self._mesh_points_obj = np.asarray(pcd.points)
            except Exception as exc:
                self.get_logger().warn(f"No se pudo cargar mesh para debug: {exc}")

        # ── Sub/Pub ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self.create_subscription(ModelStates, "/model_states", self._gt_cb, 10)
        self.create_subscription(
            PoseStamped, "/vision/tube_pose", self._pose_cb, sensor_qos)
        if debug_dir and _DEBUG_DEPS_OK:
            self.create_subscription(
                Image, "/navcam/left/navcam_left/image_raw",
                self._image_cb, best_effort_qos)
            self.create_subscription(
                CameraInfo, "/navcam/left/navcam_left/camera_info",
                self._ci_cb, best_effort_qos)
            self.create_subscription(
                Detection2DArray, "/vision/detections", self._det_cb, 10)
            os.makedirs(debug_dir, exist_ok=True)
            self.get_logger().info(f"DEBUG ON. Imágenes en: {debug_dir}")
        self._pub_activate = self.create_publisher(
            String, "/orchestrator/vision_activate", 10)

        # ── Clientes de servicio Gazebo ──────────────────────────────────────
        self._cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self._cli_delete = self.create_client(DeleteEntity, "/delete_entity")

        # ── TF2: extrínsecas de la cámara leídas en runtime ──────────────────
        # Misma fuente que usa el nodo ICP para transformar a `world`. Agnóstico
        # a la pose de la cámara (mástil/cenital/overhead).
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _gt_cb(self, msg: ModelStates) -> None:
        for i, n in enumerate(msg.name):
            if n == self.TUBE_ENTITY:
                p = msg.pose[i].position
                o = msg.pose[i].orientation
                self._gt_pos = np.array([p.x, p.y, p.z])
                self._gt_R = quat_to_R(o.x, o.y, o.z, o.w)
                return

    def _world_from(self, frame: str, spin: bool = False,
                    timeout_s: float = 5.0
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(R, t) tal que  p_world = R @ p_frame + t, leído de TF2 y cacheado.

        Es la pose del `frame` (p.ej. el óptico de la cámara) en `world`, de la
        MISMA fuente que usa el nodo ICP. Por eso vale para cualquier cámara/pose
        sin hardcodear nada. `spin=True` rellena el buffer TF desde el hilo
        principal (NO usar con spin=True dentro de un callback de suscripción).
        """
        if frame in self._tf_world_from:
            return self._tf_world_from[frame]
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self.world_frame, frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                if spin and rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    continue
                return None
            tr = tf.transform.translation
            q = tf.transform.rotation
            R = quat_to_R(q.x, q.y, q.z, q.w)
            t = np.array([tr.x, tr.y, tr.z])
            self._tf_world_from[frame] = (R, t)
            self.get_logger().info(
                f"Extrínsecas TF {self.world_frame}<-{frame}: "
                f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
            return self._tf_world_from[frame]

    def _pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        t_w = np.array([p.x, p.y, p.z])
        R_w = quat_to_R(o.x, o.y, o.z, o.w)
        frame = msg.header.frame_id
        # La pose suele llegar ya en `world` (output_frame: world); en ese caso
        # se usa tal cual. Si llega en el frame óptico (output_frame vacío) la
        # pasamos a world con la extrínseca de TF (sin spin: dentro de callback).
        if frame and frame != self.world_frame:
            ext = self._world_from(frame, spin=False)
            if ext is not None:
                R_wo, t_wo = ext
                t_w = R_wo @ t_w + t_wo
                R_w = R_wo @ R_w
            else:
                self.get_logger().warn(
                    f"Pose en '{frame}' sin TF a '{self.world_frame}'; se usa "
                    f"sin transformar (la métrica puede salir mal).")
        self._estimates.append((time.monotonic(), t_w, R_w))

    def _image_cb(self, msg: Image) -> None:
        if self._bridge is None:
            return
        try:
            self._last_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _ci_cb(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        # El frame óptico real lo dicta el propio camera_info (robusto a renombrados).
        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def _det_cb(self, msg: Detection2DArray) -> None:
        if not msg.detections:
            return
        best = max(msg.detections,
                   key=lambda d: d.results[0].hypothesis.score if d.results else 0)
        cx, cy = best.bbox.center.position.x, best.bbox.center.position.y
        sx, sy = best.bbox.size_x, best.bbox.size_y
        self._last_bbox = (cx - sx/2, cy - sy/2, cx + sx/2, cy + sy/2)

    # ── Debug: generación de imágenes anotadas ─────────────────────────────

    def _world_to_optical(self, p_world: np.ndarray) -> Optional[np.ndarray]:
        """Convierte un punto de `world` al frame óptico de la cámara (vía TF)."""
        ext = self._world_from(self.camera_frame)
        if ext is None:
            return None
        R_wo, t_wo = ext
        return R_wo.T @ (p_world - t_wo)

    def _project_world_point(self, p_world: np.ndarray
                             ) -> Optional[Tuple[float, float, float]]:
        """Proyecta (X,Y,Z) en world a (u, v, z_optical). None si detrás de cámara."""
        if self._K is None:
            return None
        p_opt = self._world_to_optical(p_world)
        if p_opt is None or p_opt[2] <= 1e-6:
            return None
        u = self._K[0, 0] * p_opt[0] / p_opt[2] + self._K[0, 2]
        v = self._K[1, 1] * p_opt[1] / p_opt[2] + self._K[1, 2]
        return (float(u), float(v), float(p_opt[2]))

    def save_debug_image(self, name: str, gt_pos: np.ndarray, gt_R: np.ndarray,
                         estimates: list, result: PoseResult) -> None:
        """Guarda una imagen anotada con bbox, GT, estimaciones y mesh proyectado.

        Layout:
          - rojo:   bbox de YOLO
          - verde:  GT (centro + eje del cilindro de ±100mm)
          - azul:   ICP estimado (cada muestra como punto pequeño; eje en azul oscuro)
          - amarillo: nube del mesh proyectada usando la pose ICP MEDIA
          - texto:  error_trans, error_axis, n_samples, status
        """
        if self.debug_dir is None or not _DEBUG_DEPS_OK:
            return
        if self._last_image is None or self._K is None:
            self.get_logger().warn(
                f"[debug {name}] sin imagen o sin K; no se genera imagen.")
            return

        # Extrínsecas de la cámara desde TF (spin=True: estamos en el hilo
        # principal). Sin ellas no se puede proyectar nada al plano imagen.
        ext = self._world_from(self.camera_frame, spin=True)
        if ext is None:
            self.get_logger().warn(
                f"[debug {name}] sin TF {self.world_frame}<-{self.camera_frame}; "
                f"no se genera imagen.")
            return
        R_wo, t_wo = ext

        img = self._last_image.copy()
        h, w = img.shape[:2]

        # ── GT: centro + eje del cilindro ──────────────────────────────────
        gt_axis_world = gt_R @ self.AXIS_OBJ
        gt_axis_world /= np.linalg.norm(gt_axis_world) + 1e-12
        gt_tip_a = gt_pos + 0.10 * gt_axis_world
        gt_tip_b = gt_pos - 0.10 * gt_axis_world
        gt_pix_c = self._project_world_point(gt_pos)
        gt_pix_a = self._project_world_point(gt_tip_a)
        gt_pix_b = self._project_world_point(gt_tip_b)
        if gt_pix_c:
            uc, vc, _ = gt_pix_c
            cv2.circle(img, (int(uc), int(vc)), 6, (0, 255, 0), -1)
            cv2.putText(img, "GT", (int(uc) + 10, int(vc) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if gt_pix_a and gt_pix_b:
            cv2.line(img,
                     (int(gt_pix_a[0]), int(gt_pix_a[1])),
                     (int(gt_pix_b[0]), int(gt_pix_b[1])),
                     (0, 255, 0), 2)

        # ── ICP: cada muestra como punto pequeño + media + eje ─────────────
        if estimates:
            est_positions = np.stack([e[1] for e in estimates])
            for p_world in est_positions:
                px = self._project_world_point(p_world)
                if px:
                    cv2.circle(img, (int(px[0]), int(px[1])), 3,
                               (255, 200, 0), -1)
            mean_pos = est_positions.mean(axis=0)
            # Promediar eje (con resolución de simetría ±)
            axes = np.stack([e[2] @ self.AXIS_OBJ for e in estimates])
            axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
            signs = np.sign(axes @ axes[0])
            signs[signs == 0] = 1.0
            mean_axis = (axes * signs[:, None]).mean(axis=0)
            mean_axis /= np.linalg.norm(mean_axis) + 1e-12
            icp_tip_a = mean_pos + 0.10 * mean_axis
            icp_tip_b = mean_pos - 0.10 * mean_axis
            icp_pix_c = self._project_world_point(mean_pos)
            icp_pix_a = self._project_world_point(icp_tip_a)
            icp_pix_b = self._project_world_point(icp_tip_b)
            if icp_pix_c:
                uc, vc, _ = icp_pix_c
                cv2.circle(img, (int(uc), int(vc)), 6, (255, 0, 0), -1)
                cv2.putText(img, "ICP", (int(uc) + 10, int(vc) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if icp_pix_a and icp_pix_b:
                cv2.line(img,
                         (int(icp_pix_a[0]), int(icp_pix_a[1])),
                         (int(icp_pix_b[0]), int(icp_pix_b[1])),
                         (200, 0, 0), 2)

            # Línea de error GT->ICP
            if gt_pix_c and icp_pix_c:
                cv2.line(img,
                         (int(gt_pix_c[0]), int(gt_pix_c[1])),
                         (int(icp_pix_c[0]), int(icp_pix_c[1])),
                         (0, 165, 255), 1)

            # ── Mesh proyectado en pose ICP media ──────────────────────────
            if self._mesh_points_obj is not None:
                # Transformar mesh: world = R_icp @ pts_obj + t_icp
                mean_R = self._mean_rotation(estimates)
                pts_world = (mean_R @ self._mesh_points_obj.T).T + mean_pos
                # Proyectar cada punto (extrínseca world->óptico desde TF)
                pts_opt = (R_wo.T @ (pts_world - t_wo).T).T
                in_front = pts_opt[:, 2] > 1e-6
                pts_fwd = pts_opt[in_front]
                if pts_fwd.shape[0] > 0:
                    us = self._K[0, 0] * pts_fwd[:, 0] / pts_fwd[:, 2] + self._K[0, 2]
                    vs = self._K[1, 1] * pts_fwd[:, 1] / pts_fwd[:, 2] + self._K[1, 2]
                    valid = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
                    for u_, v_ in zip(us[valid].astype(int), vs[valid].astype(int)):
                        cv2.circle(img, (int(u_), int(v_)), 1, (0, 255, 255), -1)

        # ── Bbox YOLO ──────────────────────────────────────────────────────
        if self._last_bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in self._last_bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ── Texto con métricas ─────────────────────────────────────────────
        status = ("PASS" if result.passed else
                  ("FAIL_AX" if not result.passed_axis else "FAIL_T"))
        lines = [
            f"{name}  [{status}]",
            f"GT world: ({gt_pos[0]:+.3f}, {gt_pos[1]:+.3f}, {gt_pos[2]:+.3f})",
            f"e_trans:  {result.error_trans_mm:.2f} mm  (criterio <5)",
            f"e_axis:   {result.error_axis_deg:.2f} deg  (criterio <5)",
            f"n_recv:   {result.n_received}/{result.n_samples}",
            f"sigma_t:  {result.repeat_trans_mm:.2f} mm",
            f"sigma_ax: {result.repeat_axis_deg:.2f} deg",
        ]
        # Fondo semi-transparente para legibilidad
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (430, 25 + 20 * len(lines)),
                      (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        for i, ln in enumerate(lines):
            color = (0, 255, 0) if (i == 0 and result.passed) else (
                (0, 0, 255) if i == 0 else (255, 255, 255))
            cv2.putText(img, ln, (15, 25 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Leyenda
        legend_y = h - 90
        for txt, color in [("GT (verde) + eje", (0, 255, 0)),
                           ("ICP media + muestras", (255, 0, 0)),
                           ("Mesh proyectado", (0, 255, 255)),
                           ("Bbox YOLO", (0, 0, 255))]:
            cv2.putText(img, txt, (15, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            legend_y += 18

        path = os.path.join(self.debug_dir, f"{name}.png")
        cv2.imwrite(path, img)
        self.get_logger().info(f"[debug] guardado {path}")

    @staticmethod
    def _mean_rotation(estimates: list) -> np.ndarray:
        """Aproxima la rotación media de las muestras ICP (cuat → matriz)."""
        Rs = np.stack([e[2] for e in estimates])
        # Promediar matrices y reortogonalizar con SVD (proyección a SO(3))
        Rbar = Rs.mean(axis=0)
        U, _, Vt = np.linalg.svd(Rbar)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        return Rm

    # ── Activación de YOLO ───────────────────────────────────────────────────

    def activate_yolo(self, timeout_s: float = 10.0) -> bool:
        """Espera al subscriber de YOLO en /orchestrator/vision_activate y
        publica `activate`. Devuelve True si publicó al menos una vez con
        un subscriber conocido por DDS discovery.

        Sin este wait, las primeras llamadas a publish() se descartan en
        silencio porque el subscriber aún no ha sido descubierto por DDS.
        """
        msg = String()
        msg.data = "activate"
        deadline = time.monotonic() + timeout_s
        while self._pub_activate.get_subscription_count() < 1:
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    "Sin subscriber en /orchestrator/vision_activate; "
                    "publico de todas formas (puede no llegar a YOLO).")
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        # Publicar varias veces por robustez (latched no aplica aquí)
        for _ in range(3):
            self._pub_activate.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.15)
        self.get_logger().info(
            f"YOLO activado (subscribers detectados: "
            f"{self._pub_activate.get_subscription_count()}).")
        return self._pub_activate.get_subscription_count() > 0

    # ── Spawn / Delete ───────────────────────────────────────────────────────

    def _wait_for_service(self, cli, timeout_s: float = 10.0) -> bool:
        return cli.wait_for_service(timeout_sec=timeout_s)

    def delete_tube(self) -> bool:
        if not self._wait_for_service(self._cli_delete, 5.0):
            self.get_logger().warn("/delete_entity no disponible")
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

    # ── Ejecutar un caso de test ─────────────────────────────────────────────

    def run_pose_case(self, name: str,
                      x: float, y: float, z: float,
                      roll: float, pitch: float, yaw: float,
                      n_samples: int,
                      timeout_s: float,
                      trans_thresh_mm: float,
                      axis_thresh_deg: float,
                      ) -> PoseResult:
        # 1) limpiar entidad previa y resetear estado
        self.delete_tube()
        self._gt_pos = None
        self._gt_R = None
        self._estimates.clear()

        # 2) spawn
        ok = self.spawn_tube(x, y, z, roll, pitch, yaw)
        if not ok:
            self.get_logger().error(f"Spawn failed for '{name}'")
            return self._make_failure_result(name, (x, y, z), (roll, pitch, yaw),
                                             n_samples)

        spawn_time = time.monotonic()
        self.get_logger().info(f"Tube spawned for '{name}'")

        # 3) esperar GT (a veces tarda un par de mensajes en aparecer)
        t_start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t_start) < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._gt_pos is not None:
                self.get_logger().info(f"GT received: {self._gt_pos}")
                break
        if self._gt_pos is None:
            self.get_logger().warn(f"No GT received for '{name}' after 3s")

        # 3b) periodo de gracia: descartar estimaciones que el ICP haya producido
        #     con la detección YOLO ANTIGUA (del tubo anterior). Sin esto, las
        #     primeras N muestras pueden corresponder a la pose del tubo del
        #     test previo, ya que YOLO/ICP funcionan en pipeline a 2 Hz y no
        #     se invalidan al borrar+spawnear.
        grace_period_s = 2.5
        t_grace_end = time.monotonic() + grace_period_s
        while rclpy.ok() and time.monotonic() < t_grace_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        # Descartar todas las estimaciones recibidas durante el periodo de gracia
        n_discarded = len(self._estimates)
        self._estimates.clear()
        if n_discarded > 0:
            self.get_logger().info(
                f"Descartadas {n_discarded} estimaciones del periodo de gracia "
                f"(probablemente del tubo anterior).")

        # 4) esperar las N estimaciones NUEVAS (o timeout)
        t_deadline = spawn_time + timeout_s
        while rclpy.ok() and time.monotonic() < t_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self._estimates) >= n_samples:
                break

        result = self._compute_metrics(
            name, (x, y, z), (roll, pitch, yaw), n_samples,
            trans_thresh_mm, axis_thresh_deg, spawn_time)

        # 5) Guardar imagen de debug si está habilitado
        if self.debug_dir and self._gt_pos is not None and self._gt_R is not None:
            try:
                self.save_debug_image(name, self._gt_pos, self._gt_R,
                                      list(self._estimates), result)
            except Exception as exc:
                self.get_logger().warn(f"[debug {name}] fallo guardando: {exc}")

        return result

    # ── Cómputo de métricas ──────────────────────────────────────────────────

    def _compute_metrics(self, name: str,
                         xyz: tuple, rpy: tuple,
                         n_samples: int,
                         trans_thresh_mm: float,
                         axis_thresh_deg: float,
                         spawn_time: float) -> PoseResult:
        est = self._estimates
        if not est or self._gt_pos is None or self._gt_R is None:
            return self._make_failure_result(name, xyz, rpy, n_samples)

        # Latencia: tiempo del 1er mensaje desde el spawn
        latency_ms = (est[0][0] - spawn_time) * 1000.0

        # Usar todas las muestras recibidas
        positions = np.stack([e[1] for e in est], axis=0)
        axes = np.stack([e[2] @ self.AXIS_OBJ for e in est], axis=0)
        # Normalizar ejes
        axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)

        # Promedio de pose para el error final (más robusto que la última)
        pos_mean = positions.mean(axis=0)
        # Promediar dirección de eje: alineamos signos primero (el cilindro
        # tiene simetría 180°, así que ±axis son equivalentes).
        ref = axes[0]
        signs = np.sign(axes @ ref)
        signs[signs == 0] = 1.0
        axes_aligned = axes * signs[:, None]
        ax_mean = axes_aligned.mean(axis=0)
        ax_mean /= (np.linalg.norm(ax_mean) + 1e-12)

        # GT
        gt_pos = self._gt_pos
        gt_ax = self._gt_R @ self.AXIS_OBJ
        gt_ax /= (np.linalg.norm(gt_ax) + 1e-12)

        err_trans_mm = float(np.linalg.norm(pos_mean - gt_pos) * 1000.0)
        cos = float(np.clip(abs(np.dot(ax_mean, gt_ax)), -1.0, 1.0))
        err_axis_deg = float(math.degrees(math.acos(cos)))

        # Repetibilidad
        repeat_trans_mm = float(positions.std(axis=0).mean() * 1000.0)
        # Para el eje, usamos la dispersión angular entre cada muestra y la media
        cosines = np.clip(np.abs(axes_aligned @ ax_mean), -1.0, 1.0)
        ang_devs = np.degrees(np.arccos(cosines))
        repeat_axis_deg = float(ang_devs.std())

        # ICP metrics (fitness / rmse) no llegan en la PoseStamped: las
        # tomamos del log si las exportamos, o las dejamos en NaN. Aquí las
        # rellenamos con NaN; un usuario avanzado puede subscribir a un
        # tópico de diagnóstico si lo añade al nodo ICP.
        fitness_mean = float("nan")
        rmse_mm_mean = float("nan")

        passed_trans = err_trans_mm < trans_thresh_mm
        passed_axis = err_axis_deg < axis_thresh_deg

        return PoseResult(
            name=name,
            gt_xyz=xyz,
            gt_rpy=rpy,
            n_samples=n_samples,
            n_received=len(est),
            error_trans_mm=err_trans_mm,
            error_axis_deg=err_axis_deg,
            repeat_trans_mm=repeat_trans_mm,
            repeat_axis_deg=repeat_axis_deg,
            latency_ms=latency_ms,
            fitness_mean=fitness_mean,
            rmse_mm_mean=rmse_mm_mean,
            passed_trans=passed_trans,
            passed_axis=passed_axis,
            passed=(passed_trans and passed_axis),
        )

    def _make_failure_result(self, name: str, xyz, rpy, n_samples) -> PoseResult:
        return PoseResult(
            name=name,
            gt_xyz=tuple(xyz),
            gt_rpy=tuple(rpy),
            n_samples=n_samples,
            n_received=0,
            error_trans_mm=float("nan"),
            error_axis_deg=float("nan"),
            repeat_trans_mm=float("nan"),
            repeat_axis_deg=float("nan"),
            latency_ms=float("nan"),
            fitness_mean=float("nan"),
            rmse_mm_mean=float("nan"),
            passed_trans=False,
            passed_axis=False,
            passed=False,
        )


# ════════════════════════════════════════════════════════════════════════════
#  SDF del tubo estático embebido (gravedad desactivada). El test fija la
#  pose, sin física, para que GT no se desplace por colisiones.
# ════════════════════════════════════════════════════════════════════════════
_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="test_icp_tube">
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


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def _print_table(results: list[PoseResult]) -> None:
    cols = [
        ("pose",      18),
        ("gt(x,y,z)", 22),
        ("gt(R,P,Y°)", 18),
        ("n",          7),
        ("e_t mm",     9),
        ("e_ax°",      8),
        ("σ_t mm",     8),
        ("σ_ax°",      8),
        ("lat ms",     8),
        ("estado",     10),
    ]
    print("─" * sum(w for _, w in cols))
    for h, w in cols:
        print(f"{h:<{w}}", end="")
    print()
    print("─" * sum(w for _, w in cols))
    for r in results:
        gt_xyz = f"({r.gt_xyz[0]:+.2f},{r.gt_xyz[1]:+.2f},{r.gt_xyz[2]:+.2f})"
        gt_rpy_deg = (math.degrees(r.gt_rpy[0]),
                      math.degrees(r.gt_rpy[1]),
                      math.degrees(r.gt_rpy[2]))
        gt_rpy = f"({gt_rpy_deg[0]:+.0f},{gt_rpy_deg[1]:+.0f},{gt_rpy_deg[2]:+.0f})"
        n = f"{r.n_received}/{r.n_samples}"
        status = "PASS" if r.passed else ("FAIL_T" if not r.passed_trans else "FAIL_AX")
        row = [
            r.name, gt_xyz, gt_rpy, n,
            f"{r.error_trans_mm:.2f}" if not math.isnan(r.error_trans_mm) else "n/a",
            f"{r.error_axis_deg:.2f}" if not math.isnan(r.error_axis_deg) else "n/a",
            f"{r.repeat_trans_mm:.2f}" if not math.isnan(r.repeat_trans_mm) else "n/a",
            f"{r.repeat_axis_deg:.2f}" if not math.isnan(r.repeat_axis_deg) else "n/a",
            f"{r.latency_ms:.0f}" if not math.isnan(r.latency_ms) else "n/a",
            status,
        ]
        for v, (_, w) in zip(row, cols):
            print(f"{str(v):<{w}}", end="")
        print()
    print("─" * sum(w for _, w in cols))


def _print_summary(results: list[PoseResult],
                   trans_thresh: float, axis_thresh: float) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed_axis = sum(1 for r in results if not r.passed_axis)
    failed_trans = sum(1 for r in results
                       if r.passed_axis and not r.passed_trans)
    received = [r for r in results if r.n_received > 0]
    if received:
        mean_lat = np.mean([r.latency_ms for r in received])
        mean_etrans = np.mean([r.error_trans_mm for r in received])
        mean_eaxis = np.mean([r.error_axis_deg for r in received])
    else:
        mean_lat = mean_etrans = mean_eaxis = float("nan")

    print()
    print(f"  TOTAL                          : {total}")
    print(f"  PASS  (e_t<{trans_thresh:.1f}mm AND e_ax<{axis_thresh:.1f}°): {passed} ({100*passed/total:.0f}%)")
    print(f"  FAIL_AXIS                      : {failed_axis}")
    print(f"  FAIL_TRANS (sólo)              : {failed_trans}")
    print(f"  Latencia media (1ª pose)       : {mean_lat:.0f} ms")
    print(f"  Error medio traslación         : {mean_etrans:.2f} mm")
    print(f"  Error medio eje                : {mean_eaxis:.2f}°")
    print()


def _resolve_mesh_path(user_path: Optional[str]) -> str:
    if user_path:
        if not os.path.isfile(user_path):
            raise FileNotFoundError(user_path)
        return os.path.abspath(user_path)
    # Default: parol6_gazebo/meshes/mars2020tube.stl
    share = get_package_share_directory("parol6_gazebo")
    candidate = os.path.join(share, "meshes", "mars2020tube.stl")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"No se encuentra {candidate}. Pasa --mesh-path con la ruta absoluta.")
    return candidate


def _load_poses_yaml(path: str) -> list[tuple]:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    out = []
    for entry in d["poses"]:
        out.append((
            entry["name"],
            float(entry["x"]), float(entry["y"]), float(entry["z"]),
            float(entry.get("roll", 0.0)),
            float(entry.get("pitch", 0.0)),
            float(entry.get("yaw", 0.0)),
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=5,
                        help="Estimaciones por pose para promediar y medir σ (default 5)")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Segundos máx. por pose esperando estimaciones (default 20)")
    parser.add_argument("--trans-thresh-mm", type=float, default=5.0,
                        help="Umbral de error de traslación (mm) (default 5.0)")
    parser.add_argument("--axis-thresh-deg", type=float, default=5.0,
                        help="Umbral de error de eje (grados) (default 5.0)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Escribir resultados en CSV en esta ruta")
    parser.add_argument("--mesh-path", type=str, default=None,
                        help="Ruta absoluta al .stl del tubo (default: parol6_gazebo)")
    parser.add_argument("--poses-file", type=str, default=None,
                        help="YAML con poses personalizadas (formato {poses: [...]})")
    parser.add_argument("--no-auto-activate", action="store_true",
                        help="No publicar /orchestrator/vision_activate al arrancar")
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="Directorio donde guardar imágenes anotadas de debug "
                             "(bbox YOLO, GT, ICP, mesh proyectado, métricas)")
    parser.add_argument("--world-frame", type=str, default=DEFAULT_WORLD_FRAME,
                        help=f"Frame del ground-truth y de la pose publicada "
                             f"(default '{DEFAULT_WORLD_FRAME}').")
    parser.add_argument("--camera-frame", type=str, default=DEFAULT_CAMERA_FRAME,
                        help=f"Frame óptico de la cámara para proyectar el debug; "
                             f"se autocorrige con camera_info "
                             f"(default '{DEFAULT_CAMERA_FRAME}').")
    args = parser.parse_args(argv)

    try:
        mesh_path = _resolve_mesh_path(args.mesh_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.poses_file:
        poses = _load_poses_yaml(args.poses_file)
    else:
        poses = DEFAULT_TEST_POSES

    rclpy.init()
    tester = IcpPoseTester(mesh_path, debug_dir=args.debug_dir,
                           world_frame=args.world_frame,
                           camera_frame=args.camera_frame)

    try:
        if not args.no_auto_activate:
            if not tester.activate_yolo():
                tester.get_logger().warn(
                    "Activación de YOLO sin confirmación; el test puede fallar "
                    "si el detector no procesa imágenes.")

        results: list[PoseResult] = []
        for name, x, y, z, R, P, Y in poses:
            tester.get_logger().info(
                f"--- Test '{name}' GT=({x:.3f},{y:.3f},{z:.3f})  "
                f"RPY=({math.degrees(R):.0f}°,{math.degrees(P):.0f}°,{math.degrees(Y):.0f}°)")
            r = tester.run_pose_case(
                name, x, y, z, R, P, Y,
                n_samples=args.samples,
                timeout_s=args.timeout,
                trans_thresh_mm=args.trans_thresh_mm,
                axis_thresh_deg=args.axis_thresh_deg,
            )
            results.append(r)
            tester.get_logger().info(
                f"    {r.n_received}/{r.n_samples} muestras  "
                f"e_t={r.error_trans_mm:.2f}mm  e_ax={r.error_axis_deg:.2f}°  "
                f"{'PASS' if r.passed else 'FAIL'}")

        # Limpieza final
        tester.delete_tube()

        print()
        print("═" * 70)
        print(" RESULTADOS  (criterio: e_t < {:.1f} mm AND e_ax < {:.1f}°)".format(
            args.trans_thresh_mm, args.axis_thresh_deg))
        print("═" * 70)
        _print_table(results)
        _print_summary(results, args.trans_thresh_mm, args.axis_thresh_deg)

        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "gt_x", "gt_y", "gt_z",
                            "gt_roll_deg", "gt_pitch_deg", "gt_yaw_deg",
                            "n_received_over_samples",
                            "error_trans_mm", "error_axis_deg",
                            "repeat_trans_mm", "repeat_axis_deg",
                            "latency_ms",
                            "fitness_mean", "rmse_mm_mean",
                            "status"])
                for r in results:
                    w.writerow(r.csv_row)
            print(f"CSV escrito: {args.csv}")

        all_passed = all(r.passed for r in results)
        return 0 if all_passed else 1
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
