#!/usr/bin/env python3
"""
eval_pose_dataset.py — Evaluación OFFLINE y COMPARABLE de los estimadores de
pose 6D del tubo (ICP estéreo y D2CO monocular) sobre el dataset de evaluación
de pose (generador v2 en modo `stereo_capture` + `placement_mode:=footprint`).

╔══════════════════════════════════════════════════════════════════════════╗
║  PRINCIPIO: MISMAS IMÁGENES, MISMAS CAJAS, MISMAS MÉTRICAS                ║
║                                                                           ║
║  Cada instancia del dataset se evalúa con la MISMA imagen izquierda y la  ║
║  MISMA bbox 2D para todos los estimadores. El monocular (D2CO) solo ve la ║
║  izquierda; el estéreo (ICP) ve además la derecha (SGBM offline con el    ║
║  MISMO código que el nodo, parol6_vision.stereo_utils) o el depth GT.     ║
║  Las salidas se comparan contra el GT BOP (cam_R_m2c/cam_t_m2c, metros,   ║
║  frame navcam_left_optical_frame) con ADD-S (simetría continua en Z).     ║
╚══════════════════════════════════════════════════════════════════════════╝

ESTIMADORES (--estimator)
  icp_lib    llama directamente a parol6_vision.icp_utils con los parámetros
             de config/icp_pose_params.yaml (lo mismo que ejecuta el nodo, sin
             ROS; determinista y rápido → barridos de parámetros).
  icp_node   reproduce las entradas hacia icp_pose_estimator_node.py por
             topics y fuerza la estimación con ~/estimate_now (confirma el
             cableado real del nodo). Requiere el nodo corriendo:
               ros2 run parol6_vision icp_pose_estimator_node.py --ros-args \\
                 --params-file $(ros2 pkg prefix parol6_vision)/share/parol6_vision/config/icp_pose_params.yaml \\
                 -p cloud_source:=depth -p depth_scale:=0.001 -p output_frame:="" \\
                 -p broadcast_tf:=false -p publish_debug_clouds:=false \\
                 -p process_rate:=0.1 -p detection_max_age_s:=30.0
             (con --cloud sgbm usa -p cloud_source:=pointcloud)
  d2co_node  igual, hacia d2co_pose_estimator_node (rama d2co / paquete
             parol6_d2co). Requiere DISPLAY (OpenGL) y:
               ros2 run parol6_d2co d2co_pose_estimator_node --ros-args \\
                 -p tracking_max_age_s:=0.0 -p output_frame:="" -p broadcast_tf:=false \\
                 -p publish_debug_overlay:=false -p process_rate:=0.1 \\
                 -p detection_max_age_s:=30.0

FUENTE DE LA NUBE (ICP, --cloud)
  depth_gt   depth GT de Gazebo (cota superior: aísla el registro ICP de la
             reconstrucción estéreo)
  sgbm       par estéreo → StereoMatcher (SGBM 3WAY + WLS + zona muerta), el
             mismo código que stereo_pointcloud_node.py, con un par único
             (en Gazebo las cámaras no tienen ruido: la fusión multi-frame
             del nodo no aporta información).

FUENTE DE LA BBOX (--bbox)
  gt_visib   bbox de la parte visible (lo que daría un detector perfecto);
             --bbox-margin-px / --bbox-jitter-px emulan holgura/descentrado
  gt_obj     bbox amodal
  yolo       detecciones reales cacheadas por cache_yolo_detections.py,
             asociadas a cada instancia por IoU con bbox_visib (--yolo-iou-min)

MÉTRICAS (por instancia, CSV) y RESUMEN
  ADD-S [mm] y recall ADD-S < 0.1·diámetro SOBRE EL TOTAL de instancias
  (una instancia sin pose es un fallo, no un dato ausente), cobertura,
  error de traslación total / radial (línea de visión) / lateral, error de
  eje (plegado a [0,90°] por la simetría), estratificado por visibilidad,
  modo (apoyado/enterrado), visibilidad de la cabeza, brazo y oclusión.

USO
  # 1) cota superior de ICP (depth GT) con bbox perfecta, sin ROS
  ros2 run parol6_vision eval_pose_dataset.py --estimator icp_lib --cloud depth_gt \\
      --split test --csv results/icp_lib_depthgt_gtbbox.csv
  # 2) ICP realista (SGBM + YOLO cacheado)
  ros2 run parol6_vision cache_yolo_detections.py --dataset ~/parol6_pose_eval_dataset_v3
  ros2 run parol6_vision eval_pose_dataset.py --estimator icp_lib --cloud sgbm \\
      --bbox yolo --split test --csv results/icp_lib_sgbm_yolo.csv
  # 3) D2CO con el nodo real
  ros2 run parol6_vision eval_pose_dataset.py --estimator d2co_node --bbox gt_visib \\
      --split test --csv results/d2co_gtbbox.csv
  # 4) comparar
  ros2 run parol6_vision compare_pose_results.py results/*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ADDS_DIAMETER_FRACTION = 0.1
DEFAULT_DATASET = os.path.expanduser('~/parol6_pose_eval_dataset_v3')
DEFAULT_FRAME = 'navcam_left_optical_frame'


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    im_id: int
    inst: int
    split: str
    path: str
    right_path: str
    depth_path: str
    cam: dict
    K: np.ndarray
    R_gt: np.ndarray
    t_gt: np.ndarray
    bbox_visib: tuple
    bbox_obj: tuple
    visib: float
    head_visib: bool
    mode: str
    arm_mode: str
    occl_plan: str
    cell: int
    n_tubes: int
    dist: float
    meta: dict = field(default_factory=dict)


def load_dataset(root: str, splits, min_visib: float, ids=None) -> list[Sample]:
    bop = os.path.join(root, 'bop')
    with open(os.path.join(bop, 'scene_camera.json')) as f:
        scene_camera = json.load(f)
    with open(os.path.join(bop, 'scene_gt.json')) as f:
        scene_gt = json.load(f)
    with open(os.path.join(bop, 'scene_gt_info.json')) as f:
        scene_gt_info = json.load(f)
    metas = {}
    with open(os.path.join(root, 'meta.jsonl')) as f:
        for line in f:
            m = json.loads(line)
            metas[int(m['im_id'])] = m
    out = []
    for k, gts in scene_gt.items():
        im_id = int(k)
        m = metas.get(im_id)
        if m is None or m.get('kind') != 'pos':
            continue
        if splits and m.get('split') not in splits:
            continue
        if ids and im_id not in ids:
            continue
        cam = scene_camera[k]
        K = np.asarray(cam['cam_K'], dtype=float).reshape(3, 3)
        infos = scene_gt_info.get(k, [])
        cells = m.get('cells', [])
        for inst, g in enumerate(gts):
            info = infos[inst] if inst < len(infos) else {}
            bv = info.get('bbox_visib') or [0, 0, 0, 0]
            if bv[2] <= 0 or bv[3] <= 0:
                continue
            visib = float(info.get('visib_fract', 1.0))
            if visib < min_visib:
                continue
            kpv = info.get('keypoints_visib', [])
            t = np.asarray(g['cam_t_m2c'], dtype=float)
            out.append(Sample(
                im_id=im_id, inst=inst, split=m.get('split', ''),
                path=os.path.join(root, m['file']),
                right_path=os.path.join(root, m['right_file']) if m.get('right_file') else '',
                depth_path=os.path.join(root, m['depth_file']) if m.get('depth_file') else '',
                cam=cam, K=K,
                R_gt=np.asarray(g['cam_R_m2c'], dtype=float).reshape(3, 3), t_gt=t,
                bbox_visib=tuple(float(v) for v in bv),
                bbox_obj=tuple(float(v) for v in (info.get('bbox_obj') or bv)),
                visib=visib, head_visib=bool(kpv[1]) if len(kpv) > 1 else True,
                mode=info.get('mode', ''), arm_mode=m.get('arm_mode', ''),
                occl_plan=m.get('occl_plan', ''),
                cell=int(cells[inst]) if inst < len(cells) else -1,
                n_tubes=len(gts), dist=float(np.linalg.norm(t)), meta=m))
    out.sort(key=lambda s: (s.im_id, s.inst))
    return out


def load_model_points(stl_path: str, n_points: int, seed: int = 0) -> np.ndarray:
    """Puntos de la superficie del tubo en el frame tube_link (el del GT):
    malla × escala no uniforme + offset del visual, como en el generador."""
    import open3d as o3d
    from parol6_vision.icp_utils import DEFAULT_MESH_SCALE
    # Offset del <visual><pose> de mars2020tube.sdf (= mesh_utils.DEFAULT_FRAME_OFFSET,
    # que no se importa para no exigir trimesh): lleva la malla escalada al
    # origen de tube_link, el frame del GT.
    frame_offset = (-0.010497, -0.010489, -0.093000)
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f'No se pudo cargar el modelo: {stl_path}')
    verts = np.asarray(mesh.vertices) * np.asarray(DEFAULT_MESH_SCALE) + np.asarray(frame_offset)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    return np.asarray(pcd.points, dtype=float)


# ════════════════════════════════════════════════════════════════════════════
# Métricas
# ════════════════════════════════════════════════════════════════════════════

def add_s(points, R_gt, t_gt, R_est, t_est) -> float:
    """ADD-S: distancia media de cada punto del modelo en la pose estimada al
    punto MÁS CERCANO del modelo en la pose GT (simetría continua en Z)."""
    p_gt = points @ R_gt.T + t_gt
    p_est = points @ R_est.T + t_est
    d2 = ((p_est[:, None, :] - p_gt[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min(axis=1)).mean())


def translation_errors(t_gt, t_est):
    err = t_est - t_gt
    total = float(np.linalg.norm(err))
    n = float(np.linalg.norm(t_gt))
    if n < 1e-9:
        return total, float('nan'), float('nan')
    u = t_gt / n
    radial = float(np.dot(err, u))
    lateral = float(np.linalg.norm(err - radial * u))
    return total, radial, lateral


def axis_error_deg(R_gt, R_est) -> float:
    c = abs(float(np.dot(R_gt[:, 2], R_est[:, 2])))
    return math.degrees(math.acos(min(1.0, max(0.0, c))))


def quat_to_rot(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bbox_iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Fuentes de bbox
# ════════════════════════════════════════════════════════════════════════════

class BboxProvider:
    def __init__(self, args):
        self.source = args.bbox
        self.margin = args.bbox_margin_px
        self.jitter = args.bbox_jitter_px
        self.rng = np.random.default_rng(args.seed)
        self.yolo = None
        self.iou_min = args.yolo_iou_min
        if self.source == 'yolo':
            path = args.yolo_json or os.path.join(args.dataset, 'yolo_detections.json')
            with open(path) as f:
                self.yolo = json.load(f)['detections']

    def get(self, s: Sample):
        """Devuelve (bbox_xywh, info) o (None, info) si no hay caja."""
        if self.source == 'yolo':
            dets = self.yolo.get(str(s.im_id), [])
            best, best_iou = None, 0.0
            for d in dets:
                x1, y1, x2, y2 = d['xyxy']
                cand = (x1, y1, x2 - x1, y2 - y1)
                iou = bbox_iou(cand, s.bbox_visib)
                if iou > best_iou:
                    best, best_iou = (cand, d), iou
            if best is None or best_iou < self.iou_min:
                return None, {'det_conf': '', 'det_iou': round(best_iou, 3), 'n_dets': len(dets)}
            cand, d = best
            return cand, {'det_conf': d['conf'], 'det_iou': round(best_iou, 3), 'n_dets': len(dets)}
        base = s.bbox_visib if self.source == 'gt_visib' else s.bbox_obj
        x, y, w, h = base
        if self.jitter > 0:
            x += float(self.rng.uniform(-self.jitter, self.jitter))
            y += float(self.rng.uniform(-self.jitter, self.jitter))
        x -= self.margin
        y -= self.margin
        w += 2 * self.margin
        h += 2 * self.margin
        return (x, y, w, h), {'det_conf': 1.0, 'det_iou': 1.0, 'n_dets': 1}


# ════════════════════════════════════════════════════════════════════════════
# Nubes de puntos (ICP)
# ════════════════════════════════════════════════════════════════════════════

class CloudProvider:
    def __init__(self, mode: str, stereo_params: Optional[dict]):
        self.mode = mode
        self.stereo_params = stereo_params or {}
        self._matchers = {}

    def _matcher(self, s: Sample):
        import cv2
        key = (tuple(s.cam['cam_K']), tuple(s.cam.get('cam_t_l2r', ())))
        m = self._matchers.get(key)
        if m is None:
            from parol6_vision.stereo_utils import StereoMatcher
            img = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
            m = StereoMatcher.from_scene_camera(s.cam, (img.shape[1], img.shape[0]),
                                                self.stereo_params)
            self._matchers[key] = m
        return m

    def depth_m(self, s: Sample) -> np.ndarray:
        import cv2
        if not s.depth_path:
            raise RuntimeError('el dataset no tiene depth GT (genera con stereo_capture:=true)')
        d = cv2.imread(s.depth_path, cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) * float(s.cam.get('depth_scale', 0.001))

    def points(self, s: Sample) -> np.ndarray:
        """Nube (N,3) en el frame óptico izquierdo. depth_gt: organizada con
        NaN en píxeles inválidos (igual que icp_utils.depth_to_xyz en el nodo)."""
        import cv2
        from parol6_vision import icp_utils
        if self.mode == 'depth_gt':
            return icp_utils.depth_to_xyz(self.depth_m(s), s.K, depth_scale=1.0)
        if self.mode == 'sgbm':
            if not s.right_path:
                raise RuntimeError('el dataset no tiene imagen derecha (stereo_capture:=true)')
            left = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(s.right_path, cv2.IMREAD_GRAYSCALE)
            return self._matcher(s).compute_points(left, right)
        raise ValueError(self.mode)


# ════════════════════════════════════════════════════════════════════════════
# Estimador ICP en librería (sin ROS)
# ════════════════════════════════════════════════════════════════════════════

class IcpLibEstimator:
    def __init__(self, params_yaml: str, stl_path: str, clouds: CloudProvider):
        import yaml
        from parol6_vision import icp_utils
        with open(params_yaml) as f:
            y = yaml.safe_load(f)
        p = y['icp_pose_estimator_node']['ros__parameters']
        self.p = p
        self.icp_utils = icp_utils
        self.clouds = clouds
        self.model_cloud = icp_utils.load_reference_cloud(
            stl_path, n_points=int(p.get('model_points', 5000)))
        self.bbox_margin = float(p.get('bbox_margin_px', 8.0))
        self.name = 'icp_lib'

    def estimate(self, s: Sample, bbox_xywh) -> dict:
        p, iu = self.p, self.icp_utils
        t0 = time.time()
        xyz = self.clouds.points(s)
        x, y, w, h = bbox_xywh
        cropped = iu.crop_cloud_by_bbox(xyz, s.K, (x, y, x + w, y + h), margin_px=self.bbox_margin)
        extras = {'n_crop': int(cropped.shape[0]), 'n_scene': 0, 'fitness': '', 'rmse_mm': ''}
        min_pts = int(p.get('min_cloud_points', 60))
        if cropped.shape[0] < min_pts:
            return dict(ok=False, T=None, msg=f'pocos puntos tras recorte ({cropped.shape[0]} < {min_pts})',
                        latency_ms=(time.time() - t0) * 1e3, **extras)
        scene = iu.preprocess_scene_cloud(
            iu.make_o3d_cloud(cropped),
            voxel_size=float(p.get('voxel_size', 0.003)),
            sor_neighbors=int(p.get('sor_neighbors', 20)),
            sor_std_ratio=float(p.get('sor_std_ratio', 2.0)),
            remove_plane=bool(p.get('remove_plane', True)),
            plane_dist_thresh=float(p.get('plane_dist_thresh', 0.004)),
            cluster_eps=float(p.get('cluster_eps', 0.012)),
            cluster_min_points=int(p.get('cluster_min_points', 30)))
        n_scene = len(scene.points)
        extras['n_scene'] = n_scene
        if n_scene < min_pts:
            return dict(ok=False, T=None, msg=f'pocos puntos tras limpieza ({n_scene} < {min_pts})',
                        latency_ms=(time.time() - t0) * 1e3, **extras)
        corr = float(p.get('icp_max_corr_dist', 0.0))
        res = iu.register_tube_icp(
            scene, self.model_cloud,
            voxel_size=float(p.get('voxel_size', 0.003)),
            max_corr_dist=(corr if corr > 0 else None),
            use_point_to_plane=bool(p.get('icp_point_to_plane', True)),
            max_iter=int(p.get('icp_max_iter', 80)),
            use_global_registration=bool(p.get('use_global_registration', False)))
        T, fit, rmse = res['T_cam_obj'], res['fitness'], res['rmse']
        extras['fitness'] = round(float(fit), 4)
        extras['rmse_mm'] = round(float(rmse) * 1e3, 3) if np.isfinite(rmse) else ''
        lat = (time.time() - t0) * 1e3
        if T is None:
            return dict(ok=False, T=None, msg='ICP no convergio', latency_ms=lat, **extras)
        if fit < float(p.get('min_fitness', 0.45)) or rmse > float(p.get('max_rmse', 0.006)):
            return dict(ok=False, T=T, msg=f'pose rechazada (fitness={fit:.3f} rmse={rmse*1e3:.2f}mm)',
                        latency_ms=lat, **extras)
        return dict(ok=True, T=T, msg=f'pose aceptada (fitness={fit:.3f} rmse={rmse*1e3:.2f}mm)',
                    latency_ms=lat, **extras)


# ════════════════════════════════════════════════════════════════════════════
# Estimadores por nodo ROS (icp_node / d2co_node)
# ════════════════════════════════════════════════════════════════════════════

class NodeEstimator:
    """Reproduce las entradas de una instancia hacia el nodo, fuerza la
    estimación con ~/estimate_now y correlaciona la pose por stamp."""

    def __init__(self, kind: str, args, clouds: Optional[CloudProvider]):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from std_srvs.srv import Trigger
        from vision_msgs.msg import Detection2DArray

        self.kind = kind
        self.name = kind
        self.args = args
        self.clouds = clouds
        self.rclpy = rclpy
        rclpy.init()
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=5)

        class Feeder(Node):
            def __init__(inner):
                super().__init__('pose_eval_feeder')
                inner.bridge = CvBridge()
                inner.pub_info = inner.create_publisher(CameraInfo, args.camera_info_topic, qos)
                inner.pub_img = inner.create_publisher(Image, args.image_topic, qos)
                inner.pub_depth = inner.create_publisher(Image, args.depth_topic, qos)
                inner.pub_cloud = inner.create_publisher(PointCloud2, args.pointcloud_topic, qos)
                inner.pub_det = inner.create_publisher(Detection2DArray, args.detections_topic, 10)
                inner.create_subscription(PoseStamped, args.pose_topic, inner._pose_cb, 10)
                service = args.service or {
                    'icp_node': '/icp_pose_estimator_node/estimate_now',
                    'd2co_node': '/d2co_pose_estimator_node/estimate_now'}[kind]
                inner.cli = inner.create_client(Trigger, service)
                inner.poses = {}

            def _pose_cb(inner, msg):
                inner.poses[int(msg.header.stamp.sec)] = msg

        self.node = Feeder()
        if not self.node.cli.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(f'servicio {self.node.cli.srv_name} no disponible: ¿está el nodo corriendo?')

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()

    def _spin(self, seconds):
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)

    def _make_cloud_msg(self, xyz, stamp, frame):
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        pts = np.ascontiguousarray(xyz[np.isfinite(xyz).all(axis=1)], dtype=np.float32)
        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=frame)
        msg.height, msg.width = 1, int(pts.shape[0])
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate('xyz')]
        msg.is_bigendian, msg.point_step, msg.row_step = False, 12, 12 * int(pts.shape[0])
        msg.data = pts.tobytes()
        msg.is_dense = True
        return msg

    def estimate(self, s: Sample, bbox_xywh, idx: int) -> dict:
        import cv2
        from builtin_interfaces.msg import Time as TimeMsg
        from sensor_msgs.msg import CameraInfo
        from std_srvs.srv import Trigger
        from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
        a = self.args
        stamp_sec = idx + 1
        stamp = TimeMsg(sec=stamp_sec, nanosec=0)
        img = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, a.frame_id
        info.height, info.width = img.shape[:2]
        info.k = [float(v) for v in s.K.flatten()]
        info.d = [0.0] * 5
        info.distortion_model = 'plumb_bob'
        img_msg = self.node.bridge.cv2_to_imgmsg(img, encoding='mono8')
        img_msg.header.stamp, img_msg.header.frame_id = stamp, a.frame_id
        cloud_msg = depth_msg = None
        if self.kind == 'icp_node':
            if self.clouds.mode == 'depth_gt':
                d_mm = np.clip(np.round(self.clouds.depth_m(s) * 1000.0), 0, 65535).astype(np.uint16)
                depth_msg = self.node.bridge.cv2_to_imgmsg(d_mm, encoding='16UC1')
                depth_msg.header.stamp, depth_msg.header.frame_id = stamp, a.frame_id
            else:
                cloud_msg = self._make_cloud_msg(self.clouds.points(s), stamp, a.frame_id)
        x, y, w, h = bbox_xywh
        det = Detection2D()
        det.header.stamp, det.header.frame_id = stamp, a.frame_id
        det.bbox.center.position.x, det.bbox.center.position.y = float(x + w / 2), float(y + h / 2)
        det.bbox.size_x, det.bbox.size_y = float(w), float(h)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id, hyp.hypothesis.score = a.class_id, 1.0
        det.results.append(hyp)
        det_arr = Detection2DArray()
        det_arr.header.stamp, det_arr.header.frame_id = stamp, a.frame_id
        det_arr.detections.append(det)

        self.node.poses.clear()
        pose, msg_txt, latency_ms, attempts, stale = None, '', 0.0, 0, 0
        for attempt in range(1, a.max_attempts + 1):
            attempts = attempt
            for _ in range(a.repeat_publish):
                self.node.pub_info.publish(info)
                self.node.pub_img.publish(img_msg)
                if depth_msg is not None:
                    self.node.pub_depth.publish(depth_msg)
                if cloud_msg is not None:
                    self.node.pub_cloud.publish(cloud_msg)
                self.node.pub_det.publish(det_arr)
                self._spin(a.settle_s / a.repeat_publish)
            t0 = time.time()
            fut = self.node.cli.call_async(Trigger.Request())
            self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=a.timeout_s)
            latency_ms = (time.time() - t0) * 1e3
            msg_txt = fut.result().message if (fut.done() and fut.result() is not None) else 'timeout del servicio'
            self._spin(0.05)
            pose = self.node.poses.pop(stamp_sec, None)
            if pose is not None:
                break
            if self.node.poses:      # pose de OTRA muestra: entradas no asentadas
                stale += 1
                self.node.poses.clear()
                continue
            if any(k in msg_txt for k in ('rechazada', 'obsoleta', 'pocos puntos', 'no convergio', 'sin ')):
                break
        extras = {'attempts': attempts, 'stale': stale, 'n_crop': '', 'n_scene': '', 'fitness': '', 'rmse_mm': ''}
        if pose is None:
            return dict(ok=False, T=None, msg=msg_txt, latency_ms=latency_ms, **extras)
        T = np.eye(4)
        o, p = pose.pose.orientation, pose.pose.position
        T[:3, :3] = quat_to_rot(o.x, o.y, o.z, o.w)
        T[:3, 3] = [p.x, p.y, p.z]
        return dict(ok=True, T=T, msg=msg_txt, latency_ms=latency_ms, **extras)


# ════════════════════════════════════════════════════════════════════════════
# Overlays de depuración
# ════════════════════════════════════════════════════════════════════════════

def save_overlay(path, s: Sample, bbox_xywh, points, T_est, row):
    import cv2
    img = cv2.imread(s.path)
    K = s.K

    def draw(R, t, color):
        p = points @ R.T + t
        ok = p[:, 2] > 1e-6
        u = (K[0, 0] * p[ok, 0] / p[ok, 2] + K[0, 2]).astype(int)
        v = (K[1, 1] * p[ok, 1] / p[ok, 2] + K[1, 2]).astype(int)
        m = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        img[v[m], u[m]] = color
    draw(s.R_gt, s.t_gt, (0, 255, 0))
    if T_est is not None:
        draw(T_est[:3, :3], T_est[:3, 3], (0, 0, 255))
    if bbox_xywh is not None:
        x, y, w, h = [int(round(v)) for v in bbox_xywh]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 1)
    x0 = max(0, int(s.bbox_obj[0]) - 60)
    y0 = max(0, int(s.bbox_obj[1]) - 60)
    x1 = min(img.shape[1], int(s.bbox_obj[0] + s.bbox_obj[2]) + 60)
    y1 = min(img.shape[0], int(s.bbox_obj[1] + s.bbox_obj[3]) + 60)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    txt = (f"im{s.im_id}#{s.inst} {row.get('msg', '')[:40]} ADD-S={row.get('add_s_mm', '')} "
           f"rad={row.get('err_radial_mm', '')} ax={row.get('err_axis_deg', '')}")
    cv2.putText(crop, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, crop)


# ════════════════════════════════════════════════════════════════════════════
# Resumen
# ════════════════════════════════════════════════════════════════════════════

CSV_COLS = ['im_id', 'inst', 'split', 'cell', 'mode', 'visib', 'head_visib', 'arm_mode', 'occl_plan',
            'n_tubes', 'dist_gt_m', 'z_gt_m', 'bbox_w', 'bbox_h', 'det_conf', 'det_iou', 'n_dets',
            'pose_ok', 'z_est_m', 'add_s_mm', 'add_s_pass', 'err_trans_mm', 'err_radial_mm',
            'err_lateral_mm', 'err_axis_deg', 'latency_ms', 'fitness', 'rmse_mm', 'n_crop', 'n_scene',
            'attempts', 'stale', 'msg']


def _stats(vals):
    if not vals:
        return '      n/a'
    a = np.asarray(vals, dtype=float)
    return f'med {np.median(a):8.2f} | media {a.mean():8.2f} | p95 {np.percentile(a, 95):8.2f}'


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def summarize(rows, adds_thresh_m, elapsed, label='') -> dict:
    n = len(rows)
    if n == 0:
        print('\nSin resultados.')
        return {}
    ok = [r for r in rows if r['pose_ok']]
    n_pass = sum(int(r['add_s_pass']) for r in rows)
    out = {'label': label, 'n': n, 'coverage': len(ok) / n, 'recall_total': n_pass / n,
           'recall_accepted': (n_pass / len(ok)) if ok else float('nan')}
    print('\n' + '=' * 76)
    print(f'RESUMEN {label}')
    print('=' * 76)
    print(f'Instancias evaluadas : {n}')
    print(f'Cobertura            : {100 * out["coverage"]:.1f}%  ({len(ok)}/{n} con pose)')
    print(f'Tiempo total         : {elapsed:.1f} s ({elapsed / n:.2f} s/instancia)')
    n_to = sum(1 for r in rows if 'timeout' in str(r.get('msg', '')))
    if n_to:
        print(f'  *** AVISO: {n_to} instancias cortadas por TIMEOUT del harness (cobertura infravalorada)')
    print(f'\nRecall ADD-S < {adds_thresh_m * 1e3:.1f} mm')
    print(f'  sobre el total     : {100 * out["recall_total"]:.1f}%  ({n_pass}/{n})')
    if ok:
        print(f'  sobre las aceptadas: {100 * out["recall_accepted"]:.1f}%  ({n_pass}/{len(ok)})')
        for key, name in [('add_s_mm', 'ADD-S      '), ('err_trans_mm', 'trans total'),
                          ('err_radial_mm', '|radial|   '), ('err_lateral_mm', 'lateral    '),
                          ('err_axis_deg', 'eje (deg)  ')]:
            vals = [abs(_f(r[key])) if key == 'err_radial_mm' else _f(r[key]) for r in ok]
            out['med_' + key] = float(np.median(vals))
        print('\nErrores (solo instancias con pose publicada) [mm / grados]')
        print(f"  ADD-S       : {_stats([_f(r['add_s_mm']) for r in ok])}")
        print(f"  trans total : {_stats([_f(r['err_trans_mm']) for r in ok])}")
        print(f"  |radial|    : {_stats([abs(_f(r['err_radial_mm'])) for r in ok])}")
        print(f"  lateral     : {_stats([_f(r['err_lateral_mm']) for r in ok])}")
        print(f"  eje (deg)   : {_stats([_f(r['err_axis_deg']) for r in ok])}")
        bias = float(np.mean([_f(r['err_radial_mm']) for r in ok]))
        out['radial_bias_mm'] = bias
        print(f'\n  Sesgo radial medio: {bias:+.1f} mm ({"sobreestima" if bias > 0 else "subestima"} la profundidad)')
        lat = [_f(r['latency_ms']) for r in rows if _f(r['latency_ms']) == _f(r['latency_ms'])]
        if lat:
            print(f'  Latencia por instancia: med {np.median(lat):.0f} ms')

    def strat(title, keyfn, bins_or_cats):
        print(f'\nEstratificado por {title}')
        print(f"  {'grupo':>16} | {'n':>4} | {'cobert.':>7} | {'ADD-S rec':>9} | {'ADD-S med':>9} | {'|rad| med':>9} | {'eje med':>7}")
        print('  ' + '-' * 78)
        for name, sel in bins_or_cats(keyfn):
            if not sel:
                continue
            okk = [r for r in sel if r['pose_ok']]
            cov = 100 * len(okk) / len(sel)
            rec = 100 * sum(int(r['add_s_pass']) for r in sel) / len(sel)
            med = np.median([_f(r['add_s_mm']) for r in okk]) if okk else float('nan')
            rad = np.median([abs(_f(r['err_radial_mm'])) for r in okk]) if okk else float('nan')
            ax = np.median([_f(r['err_axis_deg']) for r in okk]) if okk else float('nan')
            print(f'  {name:>16} | {len(sel):4d} | {cov:6.1f}% | {rec:8.1f}% | {med:9.2f} | {rad:9.2f} | {ax:7.1f}')
            out[f'strat/{title}/{name}'] = {'n': len(sel), 'coverage': cov / 100, 'recall': rec / 100}

    def by_bins(edges):
        def f(keyfn):
            for lo, hi in zip(edges[:-1], edges[1:]):
                yield f'{lo:.2f}-{hi:.2f}', [r for r in rows if lo <= _f(r[keyfn]) < hi]
        return f

    def by_cat(keyfn):
        cats = sorted({str(r[keyfn]) for r in rows})
        for c in cats:
            yield c, [r for r in rows if str(r[keyfn]) == c]

    strat('visibilidad', 'visib', by_bins([0.0, 0.25, 0.5, 0.75, 0.9, 1.01]))
    strat('modo', 'mode', by_cat)
    strat('cabeza visible', 'head_visib', by_cat)
    strat('brazo', 'arm_mode', by_cat)
    strat('oclusión planificada', 'occl_plan', by_cat)
    strat('nº tubos en imagen', 'n_tubes', by_cat)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Evaluación offline comparable de estimadores de pose (ICP/D2CO).',
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--dataset', default=DEFAULT_DATASET)
    p.add_argument('--split', nargs='*', default=[], help='train/val/test (vacío = todos)')
    p.add_argument('--ids', type=int, nargs='*', default=None, help='im_ids concretos')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--min-visib', type=float, default=0.0)
    p.add_argument('--estimator', choices=['icp_lib', 'icp_node', 'd2co_node'], required=True)
    p.add_argument('--cloud', choices=['depth_gt', 'sgbm'], default='sgbm', help='fuente de la nube (ICP)')
    p.add_argument('--stereo-params', default='', help='YAML del nodo estéreo para --cloud sgbm (default: DEFAULT_PARAMS)')
    p.add_argument('--icp-params', default='', help='YAML de icp_pose_estimator_node (default: config del paquete)')
    p.add_argument('--stl', default='', help='mars2020tube.stl ORIGINAL (default: parol6_gazebo/meshes)')
    p.add_argument('--model-points', type=int, default=800, help='puntos del modelo para ADD-S')
    p.add_argument('--bbox', choices=['gt_visib', 'gt_obj', 'yolo'], default='gt_visib')
    p.add_argument('--bbox-margin-px', type=float, default=0.0)
    p.add_argument('--bbox-jitter-px', type=float, default=0.0)
    p.add_argument('--yolo-json', default='', help='JSON de cache_yolo_detections.py (default: <dataset>/yolo_detections.json)')
    p.add_argument('--yolo-iou-min', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--csv', default='', help='CSV de resultados por instancia (+ .summary.json)')
    p.add_argument('--overlay-dir', default='', help='guardar overlays de depuración')
    p.add_argument('--overlay-n', type=int, default=20)
    p.add_argument('--progress-every', type=int, default=25)
    p.add_argument('--verbose', action='store_true')
    # Nodo ROS
    p.add_argument('--image-topic', default='/navcam/left/navcam_left/image_raw')
    p.add_argument('--camera-info-topic', default='/navcam/left/navcam_left/camera_info')
    p.add_argument('--depth-topic', default='/navcam/left/navcam_left/depth/image_raw')
    p.add_argument('--pointcloud-topic', default='/navcam/pointcloud')
    p.add_argument('--detections-topic', default='/vision/detections')
    p.add_argument('--pose-topic', default='/vision/tube_pose')
    p.add_argument('--service', default='', help='servicio estimate_now (default según estimador)')
    p.add_argument('--frame-id', default=DEFAULT_FRAME)
    p.add_argument('--class-id', default='0')
    p.add_argument('--settle-s', type=float, default=0.4)
    p.add_argument('--repeat-publish', type=int, default=2)
    p.add_argument('--timeout-s', type=float, default=180.0)
    p.add_argument('--max-attempts', type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset = os.path.expanduser(args.dataset)
    if not os.path.isdir(args.dataset):
        print(f'ERROR: no existe el dataset {args.dataset}', file=sys.stderr)
        return 1
    from ament_index_python.packages import get_package_share_directory
    if not args.stl:
        args.stl = os.path.join(get_package_share_directory('parol6_gazebo'), 'meshes', 'mars2020tube.stl')
    if not args.icp_params:
        args.icp_params = os.path.join(get_package_share_directory('parol6_vision'), 'config', 'icp_pose_params.yaml')

    samples = load_dataset(args.dataset, args.split, args.min_visib, set(args.ids) if args.ids else None)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        print('ERROR: no hay instancias tras los filtros.', file=sys.stderr)
        return 1
    with open(os.path.join(args.dataset, 'bop', 'models_info.json')) as f:
        diam = float(json.load(f)['1']['diameter'])
    adds_thresh = ADDS_DIAMETER_FRACTION * diam
    points = load_model_points(args.stl, args.model_points, args.seed)

    stereo_params = None
    if args.stereo_params:
        import yaml
        with open(args.stereo_params) as f:
            y = yaml.safe_load(f)
        stereo_params = next(iter(y.values()))['ros__parameters'] if 'ros__parameters' not in y else y['ros__parameters']
    clouds = CloudProvider(args.cloud, stereo_params) if args.estimator.startswith('icp') else None
    bboxes = BboxProvider(args)

    if args.estimator == 'icp_lib':
        est = IcpLibEstimator(args.icp_params, args.stl, clouds)
    else:
        est = NodeEstimator(args.estimator, args, clouds)

    label = f'{args.estimator}' + (f' cloud={args.cloud}' if clouds else '') + f' bbox={args.bbox}'
    if args.bbox_margin_px or args.bbox_jitter_px:
        label += f' (margen {args.bbox_margin_px}px, jitter {args.bbox_jitter_px}px)'
    print(f'Instancias a evaluar : {len(samples)}   splits {args.split or "todos"}')
    print(f'Estimador            : {label}')
    print(f'Umbral ADD-S         : {adds_thresh * 1e3:.2f} mm (0.1 x diámetro {diam:.4f} m)')
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)

    rows = []
    t_start = time.time()
    try:
        for idx, s in enumerate(samples):
            bbox, binfo = bboxes.get(s)
            row = {'im_id': s.im_id, 'inst': s.inst, 'split': s.split, 'cell': s.cell, 'mode': s.mode,
                   'visib': round(s.visib, 4), 'head_visib': int(s.head_visib), 'arm_mode': s.arm_mode,
                   'occl_plan': s.occl_plan, 'n_tubes': s.n_tubes, 'dist_gt_m': round(s.dist, 4),
                   'z_gt_m': round(float(s.t_gt[2]), 4),
                   'bbox_w': round(bbox[2], 1) if bbox else '', 'bbox_h': round(bbox[3], 1) if bbox else '',
                   'det_conf': binfo.get('det_conf', ''), 'det_iou': binfo.get('det_iou', ''),
                   'n_dets': binfo.get('n_dets', ''), 'pose_ok': 0, 'z_est_m': '', 'add_s_mm': '',
                   'add_s_pass': 0, 'err_trans_mm': '', 'err_radial_mm': '', 'err_lateral_mm': '',
                   'err_axis_deg': '', 'latency_ms': '', 'fitness': '', 'rmse_mm': '', 'n_crop': '',
                   'n_scene': '', 'attempts': '', 'stale': '', 'msg': ''}
            res = None
            if bbox is None:
                row['msg'] = 'sin deteccion (YOLO no encontro el tubo)'
            else:
                try:
                    res = est.estimate(s, bbox) if args.estimator == 'icp_lib' else est.estimate(s, bbox, idx)
                except Exception as e:  # noqa: BLE001
                    res = dict(ok=False, T=None, msg=f'excepcion: {e}', latency_ms='')
                for k in ('latency_ms', 'fitness', 'rmse_mm', 'n_crop', 'n_scene', 'attempts', 'stale'):
                    if k in res:
                        row[k] = (round(res[k], 1) if isinstance(res[k], float) else res[k])
                row['msg'] = str(res.get('msg', '')).replace(',', ';')[:120]
                if res.get('ok') and res.get('T') is not None:
                    T = res['T']
                    R_est, t_est = T[:3, :3], T[:3, 3]
                    e_tot, e_rad, e_lat = translation_errors(s.t_gt, t_est)
                    a = add_s(points, s.R_gt, s.t_gt, R_est, t_est)
                    row.update({'pose_ok': 1, 'z_est_m': round(float(t_est[2]), 4),
                                'add_s_mm': round(a * 1e3, 3), 'add_s_pass': int(a < adds_thresh),
                                'err_trans_mm': round(e_tot * 1e3, 3), 'err_radial_mm': round(e_rad * 1e3, 3),
                                'err_lateral_mm': round(e_lat * 1e3, 3),
                                'err_axis_deg': round(axis_error_deg(s.R_gt, R_est), 3)})
            rows.append(row)
            if args.overlay_dir and idx < args.overlay_n:
                save_overlay(os.path.join(args.overlay_dir, f'ov_{s.im_id:06d}_{s.inst}.png'),
                             s, bbox, points, (res or {}).get('T'), row)
            if args.verbose or (idx + 1) % args.progress_every == 0:
                if row['pose_ok']:
                    print(f"  [{idx + 1}/{len(samples)}] im{s.im_id}#{s.inst} z={s.t_gt[2]:.2f}m "
                          f"ADD-S={row['add_s_mm']:.1f}mm rad={row['err_radial_mm']:+.1f} "
                          f"lat={row['err_lateral_mm']:.1f} axis={row['err_axis_deg']:.1f}deg")
                else:
                    print(f"  [{idx + 1}/{len(samples)}] im{s.im_id}#{s.inst} SIN POSE ({row['msg'][:70]})")
    finally:
        if hasattr(est, 'close'):
            est.close()
    elapsed = time.time() - t_start

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'\nCSV escrito en {args.csv}')
    summary = summarize(rows, adds_thresh, elapsed, label)
    if args.csv:
        summary['args'] = {k: v for k, v in vars(args).items()}
        with open(os.path.splitext(args.csv)[0] + '.summary.json', 'w') as f:
            json.dump(summary, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
