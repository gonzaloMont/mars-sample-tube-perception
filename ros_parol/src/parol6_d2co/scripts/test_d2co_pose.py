#!/usr/bin/env python3
"""
test_d2co_pose.py — Test end-to-end del estimador de pose 6D D2CO.

Equivalente a parol6_vision/test_icp_pose.py pero para el nodo
`d2co_pose_estimator_node` (edge-based, monocular). Spawnea el tubo
`mars2020tube` en una serie de poses conocidas, mide las estimaciones
publicadas en `/vision/tube_pose`, calcula métricas vs verdad-terreno y
GUARDA DOS IMÁGENES DE DEBUG por pose para evaluación visual.

╔══════════════════════════════════════════════════════════════════════════╗
║  PRE-REQUISITOS                                                            ║
║    • Gazebo con la NavCam (usa el mundo OPTIMIZED, va mucho más fluido):   ║
║        ros2 launch parol6_gazebo gazebo_moveit_mars_optimized.launch.py \  ║
║                                  camera:=navcam                            ║
║    • Nodo D2CO corriendo SIN YOLO (el test provee las bbox):              ║
║        ros2 launch parol6_d2co d2co_pose.launch.py \                       ║
║                                enable_yolo:=false output_frame:=world      ║
║                                                                            ║
║  POR QUÉ SIN YOLO                                                          ║
║    YOLO congela el equipo si corre en continuo. Este test NO lo necesita: ║
║    como conoce la pose verdad-terreno del tubo que spawnea, CALCULA la     ║
║    bbox proyectando el modelo 3D en la imagen y la publica él mismo en     ║
║    `/vision/detections`. Eso aísla el algoritmo D2CO del ruido de YOLO y   ║
║    no consume recursos. Con --bbox-margin-px y --bbox-jitter-px se puede   ║
║    emular lo holgada/descentrada que llega una bbox real de YOLO.          ║
╚══════════════════════════════════════════════════════════════════════════╝

IMÁGENES DE DEBUG (--debug-dir)
  Por cada pose se guardan dos PNG:
    • {name}_overlay.png  ── el overlay PROPIO del nodo (/vision/d2co_overlay):
                             raster real del STL proyectado en la pose que
                             estimó D2CO (puntos amarillos) + bbox (rojo).
                             Es lo que de verdad ajustó el optimizador.
    • {name}_annot.png    ── imagen anotada por el test: eje GT (verde),
                             estimación D2CO media + cilindro proyectado (azul/
                             amarillo), bbox usada (rojo), línea de error y
                             panel con métricas. Compara GT vs estimación.

MÉTRICAS POR POSE
  • error_trans (mm)   ── ||EST_pos − GT_pos|| en `world`
  • error_axis (deg)   ── ángulo entre eje GT y estimado (simetría 180° resuelta)
  • repetibilidad      ── stddev (mm, deg) entre las N muestras
  • latency_ms         ── time-to-first-pose desde el spawn
  • avg_dist           ── distancia media del chamfer (del log del nodo; n/a aquí)

CRITERIO DE ÉXITO (por pose) — umbrales realistas para monocular edge-based:
  • error_trans < trans_thresh_mm  (default 80.0)
  • error_axis  < axis_thresh_deg  (default 30.0)

USO
  ros2 run parol6_d2co test_d2co_pose.py --debug-dir /tmp/d2co_test
  ros2 run parol6_d2co test_d2co_pose.py --debug-dir /tmp/d2co_test --samples 3
  ros2 run parol6_d2co test_d2co_pose.py --poses-file mis_poses.yaml --csv /tmp/r.csv
  ros2 run parol6_d2co test_d2co_pose.py --bbox-margin-px 24 --bbox-jitter-px 12
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
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)

import tf2_ros

try:
    import cv2
    from cv_bridge import CvBridge
    _DEBUG_DEPS_OK = True
except ImportError:
    _DEBUG_DEPS_OK = False


# ════════════════════════════════════════════════════════════════════════════
#  POSES DE TEST (por defecto). Cada entrada: (name, x, y, z, roll, pitch, yaw).
#  Convenios (idénticos a test_icp_pose.py):
#    • pitch=π/2 → tubo horizontal (eje largo Z_link cae en plano XY).
#    • pitch=0   → tubo vertical (eje largo apunta a world +Z).
#    • yaw rota el eje horizontal alrededor de world Z.
#
#  GRUPOS (para aislar la causa de un fallo):
#    A) Posición XY    B) Yaw    C) Altura Z    D) Pitch (inclinación)
#  Se ha recortado respecto al test ICP porque D2CO es más lento (~1.5–3.5 s/
#  ciclo) y la inclinación fuera de plano es poco observable en monocular.
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_TEST_POSES: list[tuple[str, float, float, float, float, float, float]] = [
    # ── Grupo A: posición XY (tubo horizontal, yaw=0) ────────────────────────
    ("A1_center",        0.50,  0.00, 0.01, 0, math.pi / 2, 0.0),
    ("A2_right_y_neg",   0.30, -0.20, 0.01, 0, math.pi / 2, 0.0),
    ("A3_left_y_pos",    0.30,  0.20, 0.01, 0, math.pi / 2, 0.0),
    ("A4_near_x",        0.25,  0.10, 0.01, 0, math.pi / 2, 0.0),

    # ── Grupo B: yaw (XY fijo, pitch=π/2) ────────────────────────────────────
    ("B1_yaw_0",         0.32, -0.18, 0.01, 0, math.pi / 2,  0.0),
    ("B2_yaw_45",        0.32, -0.18, 0.01, 0, math.pi / 2,  math.pi / 4),
    ("B3_yaw_90",        0.32, -0.18, 0.01, 0, math.pi / 2,  math.pi / 2),
    ("B4_yaw_neg_45",    0.32, -0.18, 0.01, 0, math.pi / 2, -math.pi / 4),

    # ── Grupo C: altura Z (XY fijo, horizontal yaw=0) ────────────────────────
    ("C1_z_0.01",        0.30, -0.15, 0.01, 0, math.pi / 2, 0.0),
    ("C2_z_0.10",        0.30, -0.15, 0.10, 0, math.pi / 2, 0.0),

    # ── Grupo D: pitch (XY fijo, yaw=0) ──────────────────────────────────────
    ("D1_pitch_45",      0.30, -0.15, 0.05, 0, math.pi / 4, 0.0),
    ("D2_vertical",      0.30, -0.15, 0.10, 0, 0.0,         0.0),
]


# ════════════════════════════════════════════════════════════════════════════
#  Geometría auxiliar (idéntica a test_icp_pose.py: mismo rig NavCam)
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


# La pose de la cámara (frame `navcam_left_optical_frame` en `world`) NO se
# hardcodea: depende de la configuración del mástil del mundo cargado. Se lee
# de TF en tiempo de ejecución (ver D2coPoseTester._ensure_camera_pose), que es
# además la MISMA transformada que usa el nodo D2CO para pasar la pose a world.
OPTICAL_FRAME = "navcam_left_optical_frame"
WORLD_FRAME = "world"


def make_cylinder_points(length: float, radius: float,
                         n_circ: int = 48, n_len: int = 14,
                         n_cap_rings: int = 5) -> np.ndarray:
    """Nube de puntos de la superficie de un cilindro centrado en el origen,
    eje largo en +Z (object frame). Sirve para proyectar bbox y overlay sin
    depender de open3d (el tubo es esencialmente un cilindro)."""
    pts = []
    thetas = np.linspace(0.0, 2 * math.pi, n_circ, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for z in np.linspace(-length / 2.0, length / 2.0, n_len):
        for ct, st in zip(cos_t, sin_t):
            pts.append((radius * ct, radius * st, z))
    for z in (-length / 2.0, length / 2.0):          # tapas
        for rr in np.linspace(radius / n_cap_rings, radius, n_cap_rings):
            for ct, st in zip(cos_t, sin_t):
                pts.append((rr * ct, rr * st, z))
    return np.asarray(pts, dtype=np.float64)


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
    repeat_trans_mm: float
    repeat_axis_deg: float
    latency_ms: float
    est_z_m: float                    # profundidad óptica estimada (diagnóstico)
    passed_trans: bool
    passed_axis: bool
    passed: bool

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
            f"{self.est_z_m:.3f}",
            "PASS" if self.passed else ("FAIL_AXIS" if not self.passed_axis else "FAIL_TRANS"),
        ]


# ════════════════════════════════════════════════════════════════════════════
#  Nodo de orquestación
# ════════════════════════════════════════════════════════════════════════════

class D2coPoseTester(Node):

    TUBE_ENTITY = "test_d2co_tube"
    AXIS_OBJ = np.array([0.0, 0.0, 1.0])  # eje largo del cilindro en object frame

    def __init__(self, mesh_path: str,
                 tube_length: float, tube_diameter: float,
                 bbox_margin_px: float, bbox_jitter_px: float,
                 debug_dir: Optional[str] = None) -> None:
        super().__init__("d2co_pose_tester")
        self.mesh_path = mesh_path
        self.debug_dir = debug_dir
        self.bbox_margin_px = bbox_margin_px
        self.bbox_jitter_px = bbox_jitter_px
        self._rng = np.random.default_rng(0)

        # Modelo de cilindro (object frame) para proyectar bbox y overlay.
        self._cyl_obj = make_cylinder_points(tube_length, tube_diameter / 2.0)

        # ── Estado ────────────────────────────────────────────────────────────
        self._gt_pos: Optional[np.ndarray] = None
        self._gt_R: Optional[np.ndarray] = None
        self._estimates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
        #     cada item = (t_recepcion, pos_world, R_world, z_optical)

        # ── Estado de debug ──────────────────────────────────────────────────
        self._last_image: Optional[np.ndarray] = None
        self._last_overlay: Optional[np.ndarray] = None   # /vision/d2co_overlay
        self._K: Optional[np.ndarray] = None
        self._cur_bbox: Optional[Tuple[float, float, float, float]] = None  # x1y1x2y2
        self._bridge = CvBridge() if _DEBUG_DEPS_OK else None

        # ── Pose de cámara (optical→world), leída de TF ──────────────────────
        self._R_world_optical: Optional[np.ndarray] = None   # optical → world (3x3)
        self._t_world_optical: Optional[np.ndarray] = None   # origen óptico en world
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS ───────────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE, depth=10)
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE, depth=5)

        # ── Sub/Pub ───────────────────────────────────────────────────────────
        self.create_subscription(ModelStates, "/model_states", self._gt_cb, 10)
        self.create_subscription(
            PoseStamped, "/vision/tube_pose", self._pose_cb, reliable_qos)
        self.create_subscription(
            Image, "/navcam/left/navcam_left/image_raw",
            self._image_cb, best_effort_qos)
        self.create_subscription(
            CameraInfo, "/navcam/left/navcam_left/camera_info",
            self._ci_cb, best_effort_qos)
        self.create_subscription(
            Image, "/vision/d2co_overlay", self._overlay_cb, best_effort_qos)

        # Publicamos la bbox proyectada (sustituye a YOLO) a 5 Hz: el nodo D2CO
        # solo la cachea y procesa en su propio timer, así que es coste ~cero.
        self._pub_det = self.create_publisher(
            Detection2DArray, "/vision/detections", 10)
        self._bbox_timer = self.create_timer(0.2, self._publish_bbox)

        # ── Clientes de servicio Gazebo ──────────────────────────────────────
        self._cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self._cli_delete = self.create_client(DeleteEntity, "/delete_entity")

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            self.get_logger().info(f"DEBUG ON. Imágenes en: {debug_dir}")

    # ── Callbacks de sensores/estimación ──────────────────────────────────────

    def _gt_cb(self, msg: ModelStates) -> None:
        for i, n in enumerate(msg.name):
            if n == self.TUBE_ENTITY:
                p = msg.pose[i].position
                o = msg.pose[i].orientation
                self._gt_pos = np.array([p.x, p.y, p.z])
                self._gt_R = quat_to_R(o.x, o.y, o.z, o.w)
                return

    def _pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        t_opt = np.array([p.x, p.y, p.z])
        R_opt = quat_to_R(o.x, o.y, o.z, o.w)
        if msg.header.frame_id == OPTICAL_FRAME and self._ensure_camera_pose():
            t_w = self._R_world_optical @ t_opt + self._t_world_optical
            R_w = self._R_world_optical @ R_opt
            z_opt = float(t_opt[2])
        else:
            # El nodo ya publica en `world` (output_frame:=world).
            t_w = t_opt
            R_w = R_opt
            p_o = self._world_to_optical(t_w)
            z_opt = float(p_o[2]) if p_o is not None else float("nan")
        self._estimates.append((time.monotonic(), t_w, R_w, z_opt))

    def _image_cb(self, msg: Image) -> None:
        if self._bridge is None:
            return
        try:
            self._last_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _overlay_cb(self, msg: Image) -> None:
        if self._bridge is None:
            return
        try:
            self._last_overlay = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _ci_cb(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    # ── Proyección y bbox ─────────────────────────────────────────────────────

    def _ensure_camera_pose(self) -> bool:
        """Lee de TF la pose `optical→world` (una vez; la cámara es estática)."""
        if self._R_world_optical is not None:
            return True
        try:
            tf = self._tf_buffer.lookup_transform(
                WORLD_FRAME, OPTICAL_FRAME, rclpy.time.Time())
        except Exception:
            return False
        q = tf.transform.rotation
        t = tf.transform.translation
        self._R_world_optical = quat_to_R(q.x, q.y, q.z, q.w)
        self._t_world_optical = np.array([t.x, t.y, t.z])
        self.get_logger().info(
            f"Pose de cámara (TF): origen óptico en world="
            f"({t.x:+.3f},{t.y:+.3f},{t.z:+.3f})")
        return True

    def _world_to_optical(self, p_world: np.ndarray) -> Optional[np.ndarray]:
        if not self._ensure_camera_pose():
            return None
        return self._R_world_optical.T @ (p_world - self._t_world_optical)

    def _project_world_points(self, pts_world: np.ndarray) -> Optional[np.ndarray]:
        """(N,3) world → (M,2) pixeles (solo los que quedan delante de cámara)."""
        if self._K is None or not self._ensure_camera_pose():
            return None
        pts_opt = (self._R_world_optical.T
                   @ (pts_world - self._t_world_optical).T).T
        fwd = pts_opt[:, 2] > 1e-6
        pts_opt = pts_opt[fwd]
        if pts_opt.shape[0] == 0:
            return None
        us = self._K[0, 0] * pts_opt[:, 0] / pts_opt[:, 2] + self._K[0, 2]
        vs = self._K[1, 1] * pts_opt[:, 1] / pts_opt[:, 2] + self._K[1, 2]
        return np.stack([us, vs], axis=1)

    def _project_world_point(self, p_world: np.ndarray
                             ) -> Optional[Tuple[float, float]]:
        out = self._project_world_points(p_world.reshape(1, 3))
        if out is None or out.shape[0] == 0:
            return None
        return float(out[0, 0]), float(out[0, 1])

    def _compute_bbox_from_gt(self) -> Optional[Tuple[float, float, float, float]]:
        """Proyecta el cilindro en la pose GT y devuelve (cx, cy, sx, sy) con
        margen + jitter, emulando una detección YOLO."""
        if self._gt_pos is None or self._gt_R is None or self._K is None:
            return None
        pts_world = (self._gt_R @ self._cyl_obj.T).T + self._gt_pos
        px = self._project_world_points(pts_world)
        if px is None or px.shape[0] < 4:
            return None
        x1, y1 = px.min(axis=0)
        x2, y2 = px.max(axis=0)
        m = self.bbox_margin_px
        x1 -= m; y1 -= m; x2 += m; y2 += m
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        sx, sy = (x2 - x1), (y2 - y1)
        if self.bbox_jitter_px > 0.0:
            j = self.bbox_jitter_px
            cx += float(self._rng.uniform(-j, j))
            cy += float(self._rng.uniform(-j, j))
            sx += float(self._rng.uniform(-j, j))
            sy += float(self._rng.uniform(-j, j))
        return (cx, cy, max(4.0, sx), max(4.0, sy))

    def _publish_bbox(self) -> None:
        bb = self._compute_bbox_from_gt()
        if bb is None:
            return
        cx, cy, sx, sy = bb
        self._cur_bbox = (cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)
        arr = Detection2DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = "navcam_left_optical_frame"
        det = Detection2D()
        det.header = arr.header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "tube"
        hyp.hypothesis.score = 0.95
        det.results.append(hyp)
        det.bbox.center.position.x = float(cx)
        det.bbox.center.position.y = float(cy)
        det.bbox.size_x = float(sx)
        det.bbox.size_y = float(sy)
        arr.detections.append(det)
        self._pub_det.publish(arr)

    # ── Debug: imágenes anotadas ──────────────────────────────────────────────

    def save_debug_images(self, name: str, gt_pos: np.ndarray, gt_R: np.ndarray,
                          estimates: list, result: PoseResult) -> None:
        if self.debug_dir is None or not _DEBUG_DEPS_OK:
            return

        # 1) Overlay propio del nodo (lo más fiel a lo que ajustó D2CO).
        if self._last_overlay is not None:
            path = os.path.join(self.debug_dir, f"{name}_overlay.png")
            cv2.imwrite(path, self._last_overlay)
            self.get_logger().info(f"[debug] guardado {path}")

        # 2) Imagen anotada del test (GT vs estimación).
        if self._last_image is None or self._K is None:
            self.get_logger().warn(
                f"[debug {name}] sin imagen o sin K; no se genera anotada.")
            return
        img = self._last_image.copy()
        h, w = img.shape[:2]

        # GT: centro + eje
        gt_axis = gt_R @ self.AXIS_OBJ
        gt_axis /= np.linalg.norm(gt_axis) + 1e-12
        gt_c = self._project_world_point(gt_pos)
        gt_a = self._project_world_point(gt_pos + 0.10 * gt_axis)
        gt_b = self._project_world_point(gt_pos - 0.10 * gt_axis)
        if gt_a and gt_b:
            cv2.line(img, (int(gt_a[0]), int(gt_a[1])),
                     (int(gt_b[0]), int(gt_b[1])), (0, 255, 0), 2)
        if gt_c:
            cv2.circle(img, (int(gt_c[0]), int(gt_c[1])), 6, (0, 255, 0), -1)
            cv2.putText(img, "GT", (int(gt_c[0]) + 10, int(gt_c[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # D2CO: muestras + media + eje + cilindro proyectado
        d2co_c = None
        if estimates:
            est_pos = np.stack([e[1] for e in estimates])
            for p_w in est_pos:
                px = self._project_world_point(p_w)
                if px:
                    cv2.circle(img, (int(px[0]), int(px[1])), 3, (255, 200, 0), -1)
            mean_pos = est_pos.mean(axis=0)
            mean_R = self._mean_rotation(estimates)
            mean_axis = mean_R @ self.AXIS_OBJ
            mean_axis /= np.linalg.norm(mean_axis) + 1e-12

            # cilindro proyectado en la pose estimada media (amarillo)
            cyl_world = (mean_R @ self._cyl_obj.T).T + mean_pos
            cyl_px = self._project_world_points(cyl_world)
            if cyl_px is not None:
                for u_, v_ in cyl_px.astype(int):
                    if 0 <= u_ < w and 0 <= v_ < h:
                        cv2.circle(img, (int(u_), int(v_)), 1, (0, 255, 255), -1)

            d2co_c = self._project_world_point(mean_pos)
            d2co_a = self._project_world_point(mean_pos + 0.10 * mean_axis)
            d2co_b = self._project_world_point(mean_pos - 0.10 * mean_axis)
            if d2co_a and d2co_b:
                cv2.line(img, (int(d2co_a[0]), int(d2co_a[1])),
                         (int(d2co_b[0]), int(d2co_b[1])), (200, 0, 0), 2)
            if d2co_c:
                cv2.circle(img, (int(d2co_c[0]), int(d2co_c[1])), 6, (255, 0, 0), -1)
                cv2.putText(img, "D2CO", (int(d2co_c[0]) + 10, int(d2co_c[1]) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if gt_c and d2co_c:
                cv2.line(img, (int(gt_c[0]), int(gt_c[1])),
                         (int(d2co_c[0]), int(d2co_c[1])), (0, 165, 255), 1)

        # bbox usada (la proyectada de GT)
        if self._cur_bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in self._cur_bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # panel de métricas
        status = ("PASS" if result.passed else
                  ("FAIL_AX" if not result.passed_axis else "FAIL_T"))
        lines = [
            f"{name}  [{status}]",
            f"GT world: ({gt_pos[0]:+.3f}, {gt_pos[1]:+.3f}, {gt_pos[2]:+.3f})",
            f"e_trans:  {result.error_trans_mm:.1f} mm  (crit <{result.thr_t:.0f})",
            f"e_axis:   {result.error_axis_deg:.1f} deg  (crit <{result.thr_a:.0f})",
            f"n_recv:   {result.n_received}/{result.n_samples}",
            f"z_est:    {result.est_z_m:.3f} m",
            f"sigma_t:  {result.repeat_trans_mm:.1f} mm",
        ]
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (440, 25 + 20 * len(lines)), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        for i, ln in enumerate(lines):
            color = (0, 255, 0) if (i == 0 and result.passed) else (
                (0, 0, 255) if i == 0 else (255, 255, 255))
            cv2.putText(img, ln, (15, 25 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        legend_y = h - 90
        for txt, color in [("GT (verde) + eje", (0, 255, 0)),
                           ("D2CO media + muestras", (255, 0, 0)),
                           ("Cilindro proyectado", (0, 255, 255)),
                           ("Bbox usada", (0, 0, 255))]:
            cv2.putText(img, txt, (15, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            legend_y += 18

        path = os.path.join(self.debug_dir, f"{name}_annot.png")
        cv2.imwrite(path, img)
        self.get_logger().info(f"[debug] guardado {path}")

    @staticmethod
    def _mean_rotation(estimates: list) -> np.ndarray:
        Rs = np.stack([e[2] for e in estimates])
        Rbar = Rs.mean(axis=0)
        U, _, Vt = np.linalg.svd(Rbar)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        return Rm

    # ── Spawn / Delete ────────────────────────────────────────────────────────

    def _wait_for_service(self, cli, timeout_s: float = 10.0) -> bool:
        return cli.wait_for_service(timeout_sec=timeout_s)

    def delete_tube(self) -> bool:
        if not self._wait_for_service(self._cli_delete, 5.0):
            return False
        req = DeleteEntity.Request()
        req.name = self.TUBE_ENTITY
        fut = self._cli_delete.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return bool(fut.done() and fut.result() and fut.result().success)

    def spawn_tube(self, x, y, z, roll, pitch, yaw) -> bool:
        if not self._wait_for_service(self._cli_spawn, 5.0):
            self.get_logger().error("/spawn_entity no disponible")
            return False
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        req = SpawnEntity.Request()
        req.name = self.TUBE_ENTITY
        req.xml = _SDF_TEMPLATE.format(mesh_path=self.mesh_path)
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)
        req.initial_pose.orientation.x = qx
        req.initial_pose.orientation.y = qy
        req.initial_pose.orientation.z = qz
        req.initial_pose.orientation.w = qw
        fut = self._cli_spawn.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if fut.done() and fut.result() and fut.result().success:
            return True
        msg = fut.result().status_message if fut.result() else "<no result>"
        self.get_logger().error(f"spawn failed: {msg}")
        return False

    # ── Ejecutar un caso ──────────────────────────────────────────────────────

    def run_pose_case(self, name, x, y, z, roll, pitch, yaw,
                      n_samples, timeout_s, grace_s,
                      trans_thresh_mm, axis_thresh_deg) -> PoseResult:
        self.delete_tube()
        self._gt_pos = None
        self._gt_R = None
        self._estimates.clear()

        if not self.spawn_tube(x, y, z, roll, pitch, yaw):
            return self._make_failure_result(name, (x, y, z), (roll, pitch, yaw),
                                             n_samples, trans_thresh_mm, axis_thresh_deg)
        spawn_time = time.monotonic()
        self.get_logger().info(f"Tube spawned for '{name}'")

        # esperar GT
        t0 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t0) < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._gt_pos is not None:
                break
        if self._gt_pos is None:
            self.get_logger().warn(f"No GT received for '{name}'")

        # periodo de gracia: descartar estimaciones del tubo anterior (D2CO
        # mantiene una pose "tracked"; al saltar el tubo la nueva bbox la
        # rompe, pero las primeras muestras pueden ser de la pose previa).
        t_grace = time.monotonic() + grace_s
        while rclpy.ok() and time.monotonic() < t_grace:
            rclpy.spin_once(self, timeout_sec=0.1)
        n_disc = len(self._estimates)
        self._estimates.clear()
        if n_disc:
            self.get_logger().info(f"Descartadas {n_disc} muestras del periodo de gracia.")

        # recoger N estimaciones nuevas (o timeout)
        t_deadline = spawn_time + timeout_s
        while rclpy.ok() and time.monotonic() < t_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self._estimates) >= n_samples:
                break

        result = self._compute_metrics(name, (x, y, z), (roll, pitch, yaw),
                                       n_samples, trans_thresh_mm, axis_thresh_deg,
                                       spawn_time)
        if self.debug_dir and self._gt_pos is not None and self._gt_R is not None:
            try:
                self.save_debug_images(name, self._gt_pos, self._gt_R,
                                       list(self._estimates), result)
            except Exception as exc:
                self.get_logger().warn(f"[debug {name}] fallo guardando: {exc}")
        return result

    # ── Métricas ──────────────────────────────────────────────────────────────

    def _compute_metrics(self, name, xyz, rpy, n_samples,
                         trans_thresh_mm, axis_thresh_deg, spawn_time) -> PoseResult:
        est = self._estimates
        if not est or self._gt_pos is None or self._gt_R is None:
            return self._make_failure_result(name, xyz, rpy, n_samples,
                                             trans_thresh_mm, axis_thresh_deg)

        latency_ms = (est[0][0] - spawn_time) * 1000.0
        positions = np.stack([e[1] for e in est], axis=0)
        axes = np.stack([e[2] @ self.AXIS_OBJ for e in est], axis=0)
        axes /= (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)
        z_est = float(np.mean([e[3] for e in est]))

        pos_mean = positions.mean(axis=0)
        ref = axes[0]
        signs = np.sign(axes @ ref)
        signs[signs == 0] = 1.0
        axes_aligned = axes * signs[:, None]
        ax_mean = axes_aligned.mean(axis=0)
        ax_mean /= (np.linalg.norm(ax_mean) + 1e-12)

        gt_pos = self._gt_pos
        gt_ax = self._gt_R @ self.AXIS_OBJ
        gt_ax /= (np.linalg.norm(gt_ax) + 1e-12)

        err_trans_mm = float(np.linalg.norm(pos_mean - gt_pos) * 1000.0)
        cos = float(np.clip(abs(np.dot(ax_mean, gt_ax)), -1.0, 1.0))
        err_axis_deg = float(math.degrees(math.acos(cos)))

        repeat_trans_mm = float(positions.std(axis=0).mean() * 1000.0)
        cosines = np.clip(np.abs(axes_aligned @ ax_mean), -1.0, 1.0)
        repeat_axis_deg = float(np.degrees(np.arccos(cosines)).std())

        passed_trans = err_trans_mm < trans_thresh_mm
        passed_axis = err_axis_deg < axis_thresh_deg
        r = PoseResult(
            name=name, gt_xyz=xyz, gt_rpy=rpy, n_samples=n_samples,
            n_received=len(est), error_trans_mm=err_trans_mm,
            error_axis_deg=err_axis_deg, repeat_trans_mm=repeat_trans_mm,
            repeat_axis_deg=repeat_axis_deg, latency_ms=latency_ms,
            est_z_m=z_est, passed_trans=passed_trans, passed_axis=passed_axis,
            passed=(passed_trans and passed_axis))
        r.thr_t = trans_thresh_mm
        r.thr_a = axis_thresh_deg
        return r

    def _make_failure_result(self, name, xyz, rpy, n_samples,
                             trans_thresh_mm, axis_thresh_deg) -> PoseResult:
        r = PoseResult(
            name=name, gt_xyz=tuple(xyz), gt_rpy=tuple(rpy), n_samples=n_samples,
            n_received=0, error_trans_mm=float("nan"), error_axis_deg=float("nan"),
            repeat_trans_mm=float("nan"), repeat_axis_deg=float("nan"),
            latency_ms=float("nan"), est_z_m=float("nan"),
            passed_trans=False, passed_axis=False, passed=False)
        r.thr_t = trans_thresh_mm
        r.thr_a = axis_thresh_deg
        return r


# ════════════════════════════════════════════════════════════════════════════
#  SDF del tubo estático (sin física: GT no se mueve). Igual que test_icp_pose.
# ════════════════════════════════════════════════════════════════════════════
_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="test_d2co_tube">
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
#  CLI / impresión
# ════════════════════════════════════════════════════════════════════════════

def _print_table(results: list[PoseResult]) -> None:
    cols = [("pose", 18), ("gt(x,y,z)", 22), ("gt(R,P,Y°)", 18), ("n", 7),
            ("e_t mm", 9), ("e_ax°", 8), ("σ_t mm", 8), ("z_est", 8),
            ("lat ms", 8), ("estado", 10)]
    print("─" * sum(w for _, w in cols))
    for h, w in cols:
        print(f"{h:<{w}}", end="")
    print()
    print("─" * sum(w for _, w in cols))
    for r in results:
        gt_xyz = f"({r.gt_xyz[0]:+.2f},{r.gt_xyz[1]:+.2f},{r.gt_xyz[2]:+.2f})"
        rpy = (math.degrees(r.gt_rpy[0]), math.degrees(r.gt_rpy[1]),
               math.degrees(r.gt_rpy[2]))
        gt_rpy = f"({rpy[0]:+.0f},{rpy[1]:+.0f},{rpy[2]:+.0f})"
        n = f"{r.n_received}/{r.n_samples}"
        status = "PASS" if r.passed else ("FAIL_T" if not r.passed_trans else "FAIL_AX")
        row = [r.name, gt_xyz, gt_rpy, n,
               f"{r.error_trans_mm:.1f}" if not math.isnan(r.error_trans_mm) else "n/a",
               f"{r.error_axis_deg:.1f}" if not math.isnan(r.error_axis_deg) else "n/a",
               f"{r.repeat_trans_mm:.1f}" if not math.isnan(r.repeat_trans_mm) else "n/a",
               f"{r.est_z_m:.3f}" if not math.isnan(r.est_z_m) else "n/a",
               f"{r.latency_ms:.0f}" if not math.isnan(r.latency_ms) else "n/a",
               status]
        for v, (_, w) in zip(row, cols):
            print(f"{str(v):<{w}}", end="")
        print()
    print("─" * sum(w for _, w in cols))


def _print_summary(results: list[PoseResult], thr_t: float, thr_a: float) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    received = [r for r in results if r.n_received > 0]
    if received:
        mean_lat = np.mean([r.latency_ms for r in received])
        mean_et = np.mean([r.error_trans_mm for r in received])
        mean_ea = np.mean([r.error_axis_deg for r in received])
    else:
        mean_lat = mean_et = mean_ea = float("nan")
    print()
    print(f"  TOTAL                          : {total}")
    print(f"  PASS (e_t<{thr_t:.0f}mm AND e_ax<{thr_a:.0f}°): {passed} ({100*passed/total:.0f}%)")
    print(f"  Sin estimación                 : {total - len(received)}")
    print(f"  Latencia media (1ª pose)       : {mean_lat:.0f} ms")
    print(f"  Error medio traslación         : {mean_et:.1f} mm")
    print(f"  Error medio eje                : {mean_ea:.1f}°")
    print()


def _resolve_mesh_path(user_path: Optional[str]) -> str:
    if user_path:
        if not os.path.isfile(user_path):
            raise FileNotFoundError(user_path)
        return os.path.abspath(user_path)
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
    for e in d["poses"]:
        out.append((e["name"], float(e["x"]), float(e["y"]), float(e["z"]),
                    float(e.get("roll", 0.0)), float(e.get("pitch", 0.0)),
                    float(e.get("yaw", 0.0))))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=3,
                        help="Estimaciones por pose para promediar y medir σ (default 3)")
    parser.add_argument("--timeout", type=float, default=50.0,
                        help="Segundos máx. por pose (default 50; D2CO ~2–9 s/ciclo "
                             "según carga de la máquina)")
    parser.add_argument("--grace", type=float, default=10.0,
                        help="Periodo de gracia tras spawn para descartar las primeras "
                             "estimaciones (bbox aún obsoleta / pose previa). Debe "
                             "cubrir ≥1 ciclo completo del nodo (default 10 s)")
    parser.add_argument("--trans-thresh-mm", type=float, default=80.0,
                        help="Umbral de error de traslación (mm) (default 80)")
    parser.add_argument("--axis-thresh-deg", type=float, default=30.0,
                        help="Umbral de error de eje (grados) (default 30)")
    parser.add_argument("--bbox-margin-px", type=float, default=20.0,
                        help="Margen añadido a la bbox proyectada (px) (default 20)")
    parser.add_argument("--bbox-jitter-px", type=float, default=0.0,
                        help="Ruido uniforme ± en centro/tamaño de la bbox (px) "
                             "para emular YOLO (default 0 = bbox limpia)")
    parser.add_argument("--tube-length", type=float, default=0.186,
                        help="Longitud del tubo (m) para la bbox/overlay (default 0.186)")
    parser.add_argument("--tube-diameter", type=float, default=0.0208,
                        help="Diámetro del tubo (m) para la bbox/overlay (default 0.0208)")
    parser.add_argument("--csv", type=str, default=None, help="Escribir CSV aquí")
    parser.add_argument("--mesh-path", type=str, default=None,
                        help="Ruta absoluta al .stl del tubo (default: parol6_gazebo)")
    parser.add_argument("--poses-file", type=str, default=None,
                        help="YAML con poses {poses: [...]}")
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="Directorio para las imágenes de debug (2 por pose)")
    args = parser.parse_args(argv)

    try:
        mesh_path = _resolve_mesh_path(args.mesh_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    poses = _load_poses_yaml(args.poses_file) if args.poses_file else DEFAULT_TEST_POSES

    rclpy.init()
    tester = D2coPoseTester(
        mesh_path, args.tube_length, args.tube_diameter,
        args.bbox_margin_px, args.bbox_jitter_px, debug_dir=args.debug_dir)

    try:
        results: list[PoseResult] = []
        for name, x, y, z, R, P, Y in poses:
            tester.get_logger().info(
                f"--- Test '{name}' GT=({x:.3f},{y:.3f},{z:.3f}) "
                f"RPY=({math.degrees(R):.0f}°,{math.degrees(P):.0f}°,{math.degrees(Y):.0f}°)")
            r = tester.run_pose_case(
                name, x, y, z, R, P, Y,
                n_samples=args.samples, timeout_s=args.timeout, grace_s=args.grace,
                trans_thresh_mm=args.trans_thresh_mm,
                axis_thresh_deg=args.axis_thresh_deg)
            results.append(r)
            tester.get_logger().info(
                f"    {r.n_received}/{r.n_samples} muestras  "
                f"e_t={r.error_trans_mm:.1f}mm  e_ax={r.error_axis_deg:.1f}°  "
                f"{'PASS' if r.passed else 'FAIL'}")

        tester.delete_tube()

        print()
        print("═" * 70)
        print(" RESULTADOS D2CO  (criterio: e_t < {:.0f} mm AND e_ax < {:.0f}°)".format(
            args.trans_thresh_mm, args.axis_thresh_deg))
        print("═" * 70)
        _print_table(results)
        _print_summary(results, args.trans_thresh_mm, args.axis_thresh_deg)

        if args.csv:
            with open(args.csv, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["name", "gt_x", "gt_y", "gt_z",
                             "gt_roll_deg", "gt_pitch_deg", "gt_yaw_deg",
                             "n_received_over_samples", "error_trans_mm",
                             "error_axis_deg", "repeat_trans_mm", "repeat_axis_deg",
                             "latency_ms", "est_z_m", "status"])
                for r in results:
                    wr.writerow(r.csv_row)
            print(f"CSV escrito: {args.csv}")

        return 0 if all(r.passed for r in results) else 1
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
