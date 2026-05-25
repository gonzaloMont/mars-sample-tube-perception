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
from rclpy.node import Node
from rclpy.task import Future
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# ════════════════════════════════════════════════════════════════════════════
#  POSES DE TEST (por defecto). Cada entrada: (x, y, z, roll, pitch, yaw).
#  Todas en el frame `world`. El convenio "tubo tumbado" usa pitch=π/2
#  para que el eje largo del cilindro (Z del link) caiga sobre world X+.
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_TEST_POSES: list[tuple[str, float, float, float, float, float, float]] = [
    # (name,            x,    y,    z,    R, P, Y)
    ("horiz_left_X",    0.30, 0.30, 0.01, 0, math.pi / 2, 0.0),
    ("horiz_left_X2",   0.35, 0.20, 0.01, 0, math.pi / 2, 0.0),
    ("horiz_yaw_45",    0.25, 0.35, 0.01, 0, math.pi / 2, math.pi / 4),
    ("horiz_yaw_90",    0.32, 0.28, 0.01, 0, math.pi / 2, math.pi / 2),
    ("horiz_yaw_-45",   0.28, 0.32, 0.01, 0, math.pi / 2, -math.pi / 4),
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


# Frame `navcam_left_optical_frame` en world. Constantes derivadas del URDF
# `navcam_stereo.urdf.xacro`: rig a (0.2,0,0.75) con rot 90° en Y, offset
# (0,0.075,0) al left_link, y RPY(-π/2,0,-π/2) al optical_frame.
R_WORLD_RIG = quat_to_R(0.0, 0.7071068, 0.0, 0.7071068)
T_WORLD_RIG = np.array([0.2, 0.0, 0.75])
T_RIG_LEFT = np.array([0.0, 0.075, 0.0])
R_LINK_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
T_WORLD_LEFT = R_WORLD_RIG @ T_RIG_LEFT + T_WORLD_RIG
R_WORLD_OPTICAL = R_WORLD_RIG @ R_LINK_OPTICAL


def transform_pose_to_world(t_optical: np.ndarray,
                            R_optical: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una pose expresada en `navcam_left_optical_frame` a `world`."""
    t_world = R_WORLD_OPTICAL @ t_optical + T_WORLD_LEFT
    R_world = R_WORLD_OPTICAL @ R_optical
    return t_world, R_world


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

    def __init__(self, mesh_path: str) -> None:
        super().__init__("icp_pose_tester")
        self.mesh_path = mesh_path

        # ── Estado ────────────────────────────────────────────────────────────
        self._gt_pos: Optional[np.ndarray] = None
        self._gt_R: Optional[np.ndarray] = None
        self._estimates: list[tuple[float, np.ndarray, np.ndarray]] = []
        #     cada item = (t_recepcion, pos_world, R_world)

        # ── Sub/Pub ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.create_subscription(ModelStates, "/model_states", self._gt_cb, 10)
        self.create_subscription(
            PoseStamped, "/vision/tube_pose", self._pose_cb, sensor_qos)
        self._pub_activate = self.create_publisher(
            String, "/orchestrator/vision_activate", 10)

        # ── Clientes de servicio Gazebo ──────────────────────────────────────
        self._cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self._cli_delete = self.create_client(DeleteEntity, "/delete_entity")

    # ── Callbacks ────────────────────────────────────────────────────────────

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
        if msg.header.frame_id == "navcam_left_optical_frame":
            t_w, R_w = transform_pose_to_world(
                np.array([p.x, p.y, p.z]),
                quat_to_R(o.x, o.y, o.z, o.w),
            )
        else:
            t_w = np.array([p.x, p.y, p.z])
            R_w = quat_to_R(o.x, o.y, o.z, o.w)
        self._estimates.append((time.monotonic(), t_w, R_w))

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

        # 4) esperar las N estimaciones (o timeout)
        t_deadline = spawn_time + timeout_s
        while rclpy.ok() and time.monotonic() < t_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self._estimates) >= n_samples:
                break

        return self._compute_metrics(
            name, (x, y, z), (roll, pitch, yaw), n_samples,
            trans_thresh_mm, axis_thresh_deg, spawn_time)

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
    tester = IcpPoseTester(mesh_path)

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
