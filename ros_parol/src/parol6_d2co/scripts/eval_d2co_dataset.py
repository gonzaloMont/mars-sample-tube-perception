#!/usr/bin/env python3
"""
eval_d2co_dataset.py — Evaluación OFFLINE del estimador de pose 6D D2CO sobre
el dataset sintético con verdad-terreno estilo BOP.

╔══════════════════════════════════════════════════════════════════════════╗
║  QUÉ HACE                                                                 ║
║                                                                           ║
║  Reproduce las imágenes del dataset hacia el nodo `d2co_pose_estimator_   ║
║  node` (CameraInfo + Image + Detection2DArray con la bbox de la VERDAD-   ║
║  TERRENO), fuerza una estimación por muestra con el servicio              ║
║  `~/estimate_now` y compara la pose publicada contra `cam_R_m2c` /        ║
║  `cam_t_m2c` del GT BOP.                                                  ║
║                                                                           ║
║  Aísla el algoritmo de YOLO: la bbox se toma de `scene_gt_info.json`,     ║
║  con margen y jitter opcionales para emular una detección real.           ║
╚══════════════════════════════════════════════════════════════════════════╝

DOS MODOS
─────────
1) `--check-gt`  (NO necesita ROS ni el nodo D2CO)
   Valida la cadena dataset + modelo + intrínsecas ANTES de medir nada:
   proyecta el modelo en la pose GT y comprueba que la bbox y los keypoints
   resultantes coinciden con los del dataset. Si esto falla, ningún número
   posterior es fiable. EJECÚTALO SIEMPRE PRIMERO.

2) modo evaluación (por defecto)
   Requiere el nodo D2CO corriendo. Ver "PRE-REQUISITOS".

PRE-REQUISITOS (modo evaluación)
────────────────────────────────
  D2CO necesita un contexto OpenGL (GLFW), así que hace falta DISPLAY.
  Lánzalo SIN YOLO y con estos overrides, que son OBLIGATORIOS para que la
  evaluación sea válida:

    ros2 run parol6_d2co d2co_pose_estimator_node --ros-args \
      -p tracking_max_age_s:=0.0 \
      -p output_frame:="" \
      -p broadcast_tf:=false \
      -p publish_debug_overlay:=false \
      -p process_rate:=0.1 \
      -p z_max:=3.6

  · tracking_max_age_s:=0.0  ── IMPRESCINDIBLE. Las muestras del dataset son
      escenas INDEPENDIENTES; con el tracking activo la pose de la muestra N
      se usaría como semilla de la N+1 y la evaluación no mediría el
      algoritmo sino un tracking sobre datos inconexos.
  · output_frame:=""         ── deja la pose en el frame óptico de la cámara,
      que es exactamente el frame del GT BOP (`cam_R_m2c`/`cam_t_m2c`).
      Así no hace falta ninguna TF.
  · process_rate:=0.1        ── el pipeline lo dispara este script vía
      servicio; el timer solo estorbaría.
  · z_max                    ── el default (2.0 m) DESCARTA el 55% del
      dataset por configuración. Súbelo o restringe con --max-dist, pero
      decídelo ANTES de medir.

MÉTRICAS
────────
  · ADD-S            ── distancia media al punto más cercano del modelo en la
                        pose GT. El tubo es un sólido de revolución
                        (`symmetries_continuous` en Z), así que ADD-S es la
                        métrica correcta, no ADD.
                        Recall estándar BOP: ADD-S < 0.1 x diámetro.
  · err_trans        ── ||t_est - t_gt||, descompuesto en:
      · err_radial   ── componente a lo largo de la línea de visión (la
                        profundidad: donde un método monocular debe sufrir)
      · err_lateral  ── componente perpendicular
  · err_axis         ── ángulo entre el eje del cilindro GT y el estimado,
                        plegado a [0, 90] por la simetría de 180 grados.
  · cobertura        ── % de muestras con pose publicada. SE REPORTA SIEMPRE
                        junto al error: el nodo rechaza poses por
                        `max_avg_dist`, así que los errores están
                        condicionados a aceptación y sin la cobertura no
                        significan nada.

USO
───
  # 1. Validar el dataset (sin ROS)
  ros2 run parol6_d2co eval_d2co_dataset.py --check-gt

  # 2. Smoke test del harness sobre unas pocas muestras de TRAIN
  ros2 run parol6_d2co eval_d2co_dataset.py --split train --limit 20

  # 3. Desarrollo / tuning (train+val); test se reserva para el número final
  ros2 run parol6_d2co eval_d2co_dataset.py --split train val --csv /tmp/dev.csv

  # 4. Cifra final, UNA sola vez, cuando ya no vayas a tocar parámetros
  ros2 run parol6_d2co eval_d2co_dataset.py --split test --csv /tmp/test.csv
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


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_DATASET = os.path.expanduser("~/parol6_synth_dataset")
DEFAULT_STL = (
    "/home/gonzalo/parol6_ws/src/PAROL6/ros_parol/src/parol6_d2co/models/"
    "mars2020tube_scaled.stl"
)
# Fracción del diámetro para el umbral de recall ADD-S (convenio BOP).
ADDS_DIAMETER_FRACTION = 0.1


# ════════════════════════════════════════════════════════════════════════════
# CARGA DEL DATASET
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    """Una INSTANCIA de tubo en una imagen (una imagen puede tener varias)."""
    im_id: int
    inst: int                 # índice de la instancia dentro de la imagen
    path: str                 # ruta absoluta al PNG
    split: str
    K: np.ndarray             # (3,3)
    R_gt: np.ndarray          # (3,3) modelo -> cámara
    t_gt: np.ndarray          # (3,)   metros
    bbox: tuple               # (x, y, w, h) de la verdad-terreno
    visib: float
    dist: Optional[float]     # distancia cámara->tubo [m] de meta.jsonl
    world: str = ""
    keypoints_2d: list = field(default_factory=list)


def load_dataset(root: str, splits: list[str], bbox_source: str) -> list[Sample]:
    """Carga las instancias positivas del dataset como lista de `Sample`."""
    bop = os.path.join(root, "bop")
    with open(os.path.join(bop, "scene_camera.json"), encoding="utf-8") as fp:
        scene_camera = json.load(fp)
    with open(os.path.join(bop, "scene_gt.json"), encoding="utf-8") as fp:
        scene_gt = json.load(fp)
    with open(os.path.join(bop, "scene_gt_info.json"), encoding="utf-8") as fp:
        scene_gt_info = json.load(fp)

    meta_by_id = {}
    with open(os.path.join(root, "meta.jsonl"), encoding="utf-8") as fp:
        for line in fp:
            m = json.loads(line)
            meta_by_id[int(m["im_id"])] = m

    samples: list[Sample] = []
    for im_id_str, gts in scene_gt.items():
        im_id = int(im_id_str)
        meta = meta_by_id.get(im_id)
        if meta is None or meta.get("kind") != "pos":
            continue
        if splits and meta.get("split") not in splits:
            continue

        cam = scene_camera[im_id_str]
        K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        infos = scene_gt_info.get(im_id_str, [])
        path = os.path.join(root, meta["file"])

        for inst, gt in enumerate(gts):
            info = infos[inst] if inst < len(infos) else {}
            bbox = info.get(bbox_source) or info.get("bbox_obj")
            if not bbox:
                continue
            samples.append(Sample(
                im_id=im_id,
                inst=inst,
                path=path,
                split=meta.get("split", ""),
                K=K,
                R_gt=np.asarray(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                t_gt=np.asarray(gt["cam_t_m2c"], dtype=np.float64),
                bbox=tuple(float(v) for v in bbox),
                visib=float(info.get("visib_fract", 1.0)),
                # Distancia POR INSTANCIA, calculada del propio GT.
                # OJO: meta.jsonl trae un campo `dist` que es POR IMAGEN: en
                # imagenes con varios tubos se refiere solo a uno de ellos
                # (discrepancias medidas de hasta 2.15 m). Usarlo para filtrar
                # por distancia mete instancias fuera del rango pedido y deja
                # fuera otras que si estaban dentro.
                dist=float(np.linalg.norm(
                    np.asarray(gt["cam_t_m2c"], dtype=np.float64))),
                world=meta.get("world", ""),
                keypoints_2d=info.get("keypoints_2d", []),
            ))
    samples.sort(key=lambda s: (s.im_id, s.inst))
    return samples


def load_model_points(stl_path: str, n_points: int) -> np.ndarray:
    """Muestrea la superficie del STL PRE-ESCALADO (ya en metros y centrado)."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"No se pudo cargar el modelo: {stl_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    return np.asarray(pcd.points, dtype=np.float64)


def model_diameter(points: np.ndarray) -> float:
    """Diámetro del modelo (mayor distancia entre dos puntos), vía extensión."""
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


# ════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ════════════════════════════════════════════════════════════════════════════

def add_s(points: np.ndarray, R_gt, t_gt, R_est, t_est) -> float:
    """ADD-S: distancia media al punto MÁS CERCANO del modelo en la pose GT.

    Es la variante simétrica de ADD, obligatoria aquí: el tubo declara
    `symmetries_continuous` en el eje Z, así que ADD penalizaría giros
    alrededor del eje que son físicamente indistinguibles.
    """
    from scipy.spatial import cKDTree
    p_gt = points @ R_gt.T + t_gt
    p_est = points @ R_est.T + t_est
    dists, _ = cKDTree(p_gt).query(p_est, k=1)
    return float(np.mean(dists))


def translation_errors(t_gt: np.ndarray, t_est: np.ndarray):
    """Error de traslación total, radial (línea de visión) y lateral.

    La descomposición es la métrica más informativa de la comparación
    monocular vs estéreo: el error radial ES la profundidad.
    """
    err = t_est - t_gt
    total = float(np.linalg.norm(err))
    n = float(np.linalg.norm(t_gt))
    if n < 1e-9:
        return total, float("nan"), float("nan")
    u = t_gt / n                       # línea de visión al centro del objeto
    radial = float(np.dot(err, u))
    lateral = float(np.linalg.norm(err - radial * u))
    return total, radial, lateral


def axis_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """Ángulo entre los ejes del cilindro (eje Z del modelo), plegado a [0,90]."""
    a_gt = R_gt[:, 2]
    a_est = R_est[:, 2]
    c = abs(float(np.dot(a_gt, a_est)))          # |.| resuelve la simetría 180
    return math.degrees(math.acos(min(1.0, max(0.0, c))))


def quat_to_rot(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════
# MODO 1 — VALIDACIÓN DEL GT (sin ROS)
# ════════════════════════════════════════════════════════════════════════════

def project(points: np.ndarray, K, R, t) -> np.ndarray:
    """Proyecta puntos del modelo a píxeles con el modelo pinhole."""
    p_cam = points @ R.T + t
    p_cam = p_cam[p_cam[:, 2] > 1e-6]
    if p_cam.shape[0] == 0:
        return np.empty((0, 2))
    uv = (K @ p_cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def bbox_iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def run_check_gt(args) -> int:
    """Valida que dataset + modelo + intrínsecas son mutuamente coherentes.

    Proyecta el modelo en la pose GT y compara con `bbox_obj` y con los
    keypoints del dataset (centro y las dos puntas del eje Z). Es la prueba
    que detecta un origen de modelo distinto, un eje permutado o una escala
    mal aplicada ANTES de que contaminen cualquier medición.
    """
    print(f"Dataset : {args.dataset}")
    print(f"Modelo  : {args.stl}")
    points = load_model_points(args.stl, args.model_points)
    diam = model_diameter(points)
    print(f"Modelo cargado: {len(points)} puntos, extensión "
          f"{(points.max(axis=0) - points.min(axis=0)).round(4)} m, "
          f"diámetro {diam:.4f} m")

    samples = load_dataset(args.dataset, args.split, "bbox_obj")
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        print("ERROR: no hay muestras que validar.", file=sys.stderr)
        return 1
    print(f"Validando {len(samples)} instancias...\n")

    # Tamaño de imagen, para recortar la proyección igual que hace el dataset
    # con las bbox de objetos que se salen del encuadre.
    import cv2
    probe = cv2.imread(samples[0].path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = (probe.shape[:2] if probe is not None else (1024, 1024))

    ious, kp_center, kp_tip = [], [], []
    # Puntas del eje del modelo: el tubo mide 0.186 m sobre Z, centrado.
    half = float(points[:, 2].max())
    tips = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, half], [0.0, 0.0, -half]])

    for s in samples:
        uv = project(points, s.K, s.R_gt, s.t_gt)
        if uv.shape[0] < 10:
            continue
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        x0, y0 = max(x0, 0.0), max(y0, 0.0)
        x1, y1 = min(x1, float(img_w)), min(y1, float(img_h))
        ious.append(bbox_iou((x0, y0, x1 - x0, y1 - y0), s.bbox))

        if len(s.keypoints_2d) >= 3:
            kp = np.asarray(s.keypoints_2d[:3], dtype=np.float64)
            puv = project(tips, s.K, s.R_gt, s.t_gt)
            if puv.shape[0] == 3:
                kp_center.append(float(np.linalg.norm(puv[0] - kp[0])))
                # Las puntas pueden venir en cualquier orden: nos quedamos con
                # el emparejamiento que minimiza el error (el eje es simétrico).
                d_direct = (np.linalg.norm(puv[1] - kp[1])
                            + np.linalg.norm(puv[2] - kp[2])) / 2
                d_swap = (np.linalg.norm(puv[1] - kp[2])
                          + np.linalg.norm(puv[2] - kp[1])) / 2
                kp_tip.append(float(min(d_direct, d_swap)))

    ious = np.asarray(ious)
    print("── Reproyección de la bbox (modelo en pose GT vs bbox_obj) ──")
    print(f"  IoU  media {ious.mean():.4f} | mediana {np.median(ious):.4f} | "
          f"min {ious.min():.4f}")
    print(f"  IoU > 0.90 : {100.0 * np.mean(ious > 0.90):.1f}% de instancias")
    print("  (Comprobación GRUESA: la bbox proyectada queda algo por dentro de")
    print("   la real porque un muestreo finito de la superficie no alcanza la")
    print("   silueta exacta. Un IoU ~0.94 es lo esperado y no indica error.)")

    ok = ious.mean() > 0.85
    if kp_center:
        kc, kt = np.asarray(kp_center), np.asarray(kp_tip)
        print("\n── Reproyección de keypoints (px) — LA COMPROBACIÓN DECISIVA ──")
        print("  Detecta un origen de modelo distinto, un eje permutado o una")
        print("  escala mal aplicada. Debe salir por debajo de ~1 px.")
        print(f"  centro    : mediana {np.median(kc):6.2f} | p95 "
              f"{np.percentile(kc, 95):6.2f}")
        print(f"  puntas eje: mediana {np.median(kt):6.2f} | p95 "
              f"{np.percentile(kt, 95):6.2f}")
        ok = ok and float(np.median(kc)) < 5.0 and float(np.median(kt)) < 5.0

    print()
    if ok:
        print("RESULTADO: OK — el modelo, las intrínsecas y el GT son coherentes.")
        print("           Las convenciones de origen y de eje coinciden con las")
        print("           que usa D2CO (setBBCenterOrigOffset + eje largo en Z).")
        return 0
    print("RESULTADO: FALLO — hay una incoherencia entre modelo, K y GT.")
    print("           NO evalúes hasta resolverlo: cualquier error de pose que")
    print("           midas estaría contaminado por este desajuste.")
    return 1


# ════════════════════════════════════════════════════════════════════════════
# MODO 2 — EVALUACIÓN CONTRA EL NODO D2CO
# ════════════════════════════════════════════════════════════════════════════

def run_eval(args) -> int:
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
    from builtin_interfaces.msg import Time as TimeMsg
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_srvs.srv import Trigger
    from vision_msgs.msg import (Detection2D, Detection2DArray,
                                 ObjectHypothesisWithPose)

    sensor_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )

    class Feeder(Node):
        def __init__(self):
            super().__init__("d2co_dataset_feeder")
            self.bridge = CvBridge()
            self.pub_info = self.create_publisher(
                CameraInfo, args.camera_info_topic, sensor_qos)
            self.pub_img = self.create_publisher(
                Image, args.image_topic, sensor_qos)
            self.pub_det = self.create_publisher(
                Detection2DArray, args.detections_topic, 10)
            self.create_subscription(
                PoseStamped, args.pose_topic, self._pose_cb, 10)
            self.cli = self.create_client(Trigger, args.service)
            self.poses: dict[int, PoseStamped] = {}

        def _pose_cb(self, msg: PoseStamped) -> None:
            # Correlación por stamp: cada muestra se publica con sec = idx+1,
            # así que la pose recibida se atribuye sin ambigüedad ni carreras.
            self.poses[int(msg.header.stamp.sec)] = msg

    rclpy.init()
    node = Feeder()

    if not node.cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            f"El servicio {args.service} no responde. ¿Está corriendo "
            f"d2co_pose_estimator_node? (necesita DISPLAY para OpenGL)")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    samples = load_dataset(args.dataset, args.split, args.bbox_source)
    if args.min_dist is not None:
        samples = [s for s in samples
                   if s.dist is None or s.dist >= args.min_dist]
    if args.max_dist is not None:
        samples = [s for s in samples
                   if s.dist is None or s.dist <= args.max_dist]
    if args.min_visib is not None:
        samples = [s for s in samples if s.visib >= args.min_visib]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        print("ERROR: no hay muestras tras aplicar los filtros.", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    points = load_model_points(args.stl, args.model_points)
    diam = model_diameter(points)
    adds_thresh = ADDS_DIAMETER_FRACTION * diam
    rng = np.random.default_rng(args.seed)

    print(f"Muestras a evaluar : {len(samples)}")
    print(f"Splits             : {args.split or 'todos'}")
    print(f"Umbral ADD-S       : {adds_thresh * 1000:.2f} mm "
          f"(0.1 x diámetro {diam:.4f} m)")
    print(f"Fuente de bbox     : {args.bbox_source} "
          f"(margen {args.bbox_margin_px} px, jitter {args.bbox_jitter_px} px)")
    print()

    rows = []
    t_start = time.time()

    for idx, s in enumerate(samples):
        stamp_sec = idx + 1
        stamp = TimeMsg(sec=stamp_sec, nanosec=0)

        img = cv2.imread(s.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [{idx}] AVISO: no se pudo leer {s.path}")
            continue

        # ── CameraInfo ──────────────────────────────────────────────────────
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = args.frame_id
        info.height, info.width = img.shape[:2]
        info.k = [float(v) for v in s.K.flatten()]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.distortion_model = "plumb_bob"

        # ── Imagen ──────────────────────────────────────────────────────────
        img_msg = node.bridge.cv2_to_imgmsg(img, encoding="mono8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = args.frame_id

        # ── Detección con la bbox de la verdad-terreno ──────────────────────
        bx, by, bw, bh = s.bbox
        if args.bbox_jitter_px > 0:
            bx += rng.uniform(-args.bbox_jitter_px, args.bbox_jitter_px)
            by += rng.uniform(-args.bbox_jitter_px, args.bbox_jitter_px)
        bw += 2.0 * args.bbox_margin_px
        bh += 2.0 * args.bbox_margin_px

        det = Detection2D()
        det.bbox.center.position.x = bx + (s.bbox[2]) / 2.0
        det.bbox.center.position.y = by + (s.bbox[3]) / 2.0
        det.bbox.size_x = float(bw)
        det.bbox.size_y = float(bh)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = args.class_id
        hyp.hypothesis.score = 1.0
        det.results.append(hyp)
        det_arr = Detection2DArray()
        det_arr.header.stamp = stamp
        det_arr.header.frame_id = args.frame_id
        det_arr.detections.append(det)

        # ── Publicar, estimar y CORRELACIONAR POR STAMP ─────────────────────
        # Los tópicos de imagen van en BEST_EFFORT, así que un mensaje puede
        # perderse (sobre todo en el primer envío, con el descubrimiento aún
        # en curso). Si el nodo no ha recibido todavía la imagen de ESTA
        # muestra, estimaría sobre la ANTERIOR y publicaría con su stamp: esa
        # pose NO puede atribuirse a esta muestra. Por eso se reintenta hasta
        # obtener una pose cuyo stamp coincida exactamente.
        node.poses.clear()
        pose, msg_txt, latency_ms, attempts, stale = None, "", 0.0, 0, 0
        for attempt in range(1, args.max_attempts + 1):
            attempts = attempt
            for _ in range(args.repeat_publish):
                node.pub_info.publish(info)
                node.pub_img.publish(img_msg)
                node.pub_det.publish(det_arr)
                _spin(rclpy, node, args.settle_s / args.repeat_publish)

            t0 = time.time()
            future = node.cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(
                node, future, timeout_sec=args.timeout_s)
            latency_ms = (time.time() - t0) * 1000.0

            if future.done() and future.result() is not None:
                msg_txt = future.result().message
            else:
                msg_txt = "timeout del servicio"

            _spin(rclpy, node, 0.05)
            pose = node.poses.pop(stamp_sec, None)
            if pose is not None:
                break
            if node.poses:
                # Se publicó una pose, pero de otra muestra: imagen aún no
                # asentada. Reintentamos en vez de contaminar la medida.
                stale += 1
                node.poses.clear()
                continue
            if "rechazada" in msg_txt or "obsoleta" in msg_txt:
                break      # rechazo legítimo del nodo: no insistir

        row = {
            "im_id": s.im_id, "inst": s.inst, "split": s.split,
            "world": s.world,
            "dist_gt_m": round(s.dist, 4) if s.dist else "",
            "visib": round(s.visib, 4),
            "bbox_w": round(s.bbox[2], 1), "bbox_h": round(s.bbox[3], 1),
            "z_gt_m": round(float(s.t_gt[2]), 4),
            "pose_ok": int(pose is not None),
            "latency_ms": round(latency_ms, 1),
            "attempts": attempts,
            "stale": stale,
            "msg": msg_txt.replace(",", ";")[:120],
        }

        if pose is not None:
            t_est = np.array([pose.pose.position.x, pose.pose.position.y,
                              pose.pose.position.z])
            R_est = quat_to_rot(pose.pose.orientation.x, pose.pose.orientation.y,
                                pose.pose.orientation.z, pose.pose.orientation.w)
            e_tot, e_rad, e_lat = translation_errors(s.t_gt, t_est)
            a = add_s(points, s.R_gt, s.t_gt, R_est, t_est)
            row.update({
                "z_est_m": round(float(t_est[2]), 4),
                "add_s_mm": round(a * 1000.0, 3),
                "add_s_pass": int(a < adds_thresh),
                "err_trans_mm": round(e_tot * 1000.0, 3),
                "err_radial_mm": round(e_rad * 1000.0, 3),
                "err_lateral_mm": round(e_lat * 1000.0, 3),
                "err_axis_deg": round(axis_error_deg(s.R_gt, R_est), 3),
            })
        else:
            row.update({"z_est_m": "", "add_s_mm": "", "add_s_pass": 0,
                        "err_trans_mm": "", "err_radial_mm": "",
                        "err_lateral_mm": "", "err_axis_deg": ""})

        rows.append(row)

        if args.verbose or (idx + 1) % args.progress_every == 0:
            if pose is not None:
                print(f"  [{idx + 1}/{len(samples)}] im{s.im_id}#{s.inst} "
                      f"d={s.dist:.2f}m ADD-S={row['add_s_mm']:.1f}mm "
                      f"rad={row['err_radial_mm']:+.1f} "
                      f"lat={row['err_lateral_mm']:.1f} "
                      f"axis={row['err_axis_deg']:.1f}deg")
            else:
                print(f"  [{idx + 1}/{len(samples)}] im{s.im_id}#{s.inst} "
                      f"SIN POSE ({msg_txt[:70]})")

    elapsed = time.time() - t_start
    node.destroy_node()
    rclpy.shutdown()

    if args.csv:
        _write_csv(args.csv, rows)
        print(f"\nCSV escrito en {args.csv}")

    _summary(rows, adds_thresh, elapsed)
    return 0


def _spin(rclpy, node, seconds: float) -> None:
    """Gira el executor durante `seconds` para procesar callbacks pendientes."""
    end = time.time() + max(0.0, seconds)
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["im_id", "inst", "split", "world", "dist_gt_m", "visib",
            "bbox_w", "bbox_h", "z_gt_m", "z_est_m", "pose_ok",
            "add_s_mm", "add_s_pass", "err_trans_mm", "err_radial_mm",
            "err_lateral_mm", "err_axis_deg", "latency_ms", "attempts",
            "stale", "msg"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ════════════════════════════════════════════════════════════════════════════

def _stats(vals: list[float]) -> str:
    if not vals:
        return "      n/a"
    a = np.asarray(vals, dtype=np.float64)
    return (f"med {np.median(a):8.2f} | media {a.mean():8.2f} | "
            f"p95 {np.percentile(a, 95):8.2f}")


def _summary(rows: list[dict], adds_thresh: float, elapsed: float) -> None:
    n = len(rows)
    if n == 0:
        print("\nSin resultados.")
        return
    ok = [r for r in rows if r["pose_ok"]]
    cov = 100.0 * len(ok) / n

    print("\n" + "=" * 74)
    print("RESUMEN")
    print("=" * 74)
    print(f"Instancias evaluadas : {n}")
    print(f"Cobertura            : {cov:.1f}%  ({len(ok)}/{n} con pose)")
    print(f"Tiempo total         : {elapsed:.1f} s "
          f"({elapsed / max(n, 1):.2f} s/instancia)")
    # Salud del harness: si esto no es ~0, el asentamiento va justo y
    # conviene subir --settle-s (no afecta a la validez, solo al coste).
    n_retry = sum(1 for r in rows if r.get("attempts", 1) > 1)
    n_stale = sum(r.get("stale", 0) for r in rows)
    print(f"Reintentos           : {n_retry} muestras "
          f"({n_stale} poses descartadas por stamp ajeno)")
    # Un timeout NO es un rechazo del algoritmo: es el harness cortando antes
    # de tiempo. Si aparece, la cobertura esta INFRAVALORADA y la comparacion
    # entre configuraciones no es valida hasta repetir con --timeout-s mayor.
    n_to = sum(1 for r in rows if "timeout" in str(r.get("msg", "")))
    if n_to:
        print(f"\n  *** AVISO: {n_to} instancias ({100.0 * n_to / n:.1f}%) "
              f"cortadas por TIMEOUT del harness.")
        print("  *** No son rechazos del algoritmo: la cobertura de abajo esta")
        print("  *** INFRAVALORADA. Repite con --timeout-s mayor antes de")
        print("  *** comparar configuraciones o de citar estas cifras.")

    if not ok:
        print("\nNinguna pose publicada: revisa el nodo, el rango z_max y los")
        print("umbrales de aceptación (max_avg_dist).")
        return

    # ── ADD-S sobre el TOTAL, no solo sobre las aceptadas ───────────────────
    # Una muestra sin pose es un fallo, no un dato ausente: contarla como
    # ausente inflaría el recall al premiar el rechazo de casos difíciles.
    n_pass = sum(r["add_s_pass"] for r in rows)
    print(f"\nRecall ADD-S < {adds_thresh * 1000:.1f} mm")
    print(f"  sobre el total     : {100.0 * n_pass / n:.1f}%  ({n_pass}/{n})")
    print(f"  sobre las aceptadas: {100.0 * n_pass / len(ok):.1f}%  "
          f"({n_pass}/{len(ok)})")

    print("\nErrores (solo instancias con pose publicada) [mm / grados]")
    print(f"  ADD-S       : {_stats([r['add_s_mm'] for r in ok])}")
    print(f"  trans total : {_stats([r['err_trans_mm'] for r in ok])}")
    print(f"  |radial|    : {_stats([abs(r['err_radial_mm']) for r in ok])}")
    print(f"  lateral     : {_stats([r['err_lateral_mm'] for r in ok])}")
    print(f"  eje (deg)   : {_stats([r['err_axis_deg'] for r in ok])}")

    sesgo = float(np.mean([r["err_radial_mm"] for r in ok]))
    print(f"\n  Sesgo radial medio: {sesgo:+.1f} mm "
          f"({'sobreestima' if sesgo > 0 else 'subestima'} la profundidad)")

    _strat(rows, "distancia [m]", "dist_gt_m",
           [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 99)])
    _strat(rows, "visibilidad", "visib",
           [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)])


def _strat(rows: list[dict], title: str, key: str, bins) -> None:
    print(f"\nEstratificado por {title}")
    print(f"  {'rango':>12} | {'n':>4} | {'cobert.':>7} | {'ADD-S rec':>9} | "
          f"{'ADD-S med':>9} | {'|rad| med':>9}")
    print("  " + "-" * 66)
    for lo, hi in bins:
        sel = [r for r in rows
               if r[key] != "" and lo <= float(r[key]) < hi]
        if not sel:
            continue
        ok = [r for r in sel if r["pose_ok"]]
        cov = 100.0 * len(ok) / len(sel)
        rec = 100.0 * sum(r["add_s_pass"] for r in sel) / len(sel)
        med = np.median([r["add_s_mm"] for r in ok]) if ok else float("nan")
        rad = np.median([abs(r["err_radial_mm"]) for r in ok]) if ok else float("nan")
        print(f"  {lo:5.2f}-{hi:5.2f} | {len(sel):4d} | {cov:6.1f}% | "
              f"{rec:8.1f}% | {med:9.2f} | {rad:9.2f}")


# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluación offline de D2CO sobre el dataset BOP.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Raíz del dataset sintético.")
    p.add_argument("--stl", default=DEFAULT_STL,
                   help="STL PRE-ESCALADO del tubo (el mismo que usa D2CO).")
    p.add_argument("--split", nargs="*", default=[],
                   help="Splits a evaluar (train/val/test). Vacío = todos. "
                        "Usa train+val para tuning y reserva test para la "
                        "cifra final.")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluar como mucho N instancias (smoke test).")
    p.add_argument("--min-dist", type=float, default=None,
                   help="Descartar muestras con distancia GT menor que esta. "
                        "Con --max-dist acota la evaluacion al SOBRE OPERATIVO "
                        "(para el PAROL6 con NavCam en mastil: 1.10-1.41 m, "
                        "derivado del alcance del brazo y la pose del rig).")
    p.add_argument("--max-dist", type=float, default=None,
                   help="Descartar muestras con distancia GT mayor que esta. "
                        "Útil si mantienes z_max=2.0 en el nodo.")
    p.add_argument("--min-visib", type=float, default=None,
                   help="Descartar instancias con visib_fract menor que esta.")
    p.add_argument("--model-points", type=int, default=800,
                   help="Puntos muestreados del modelo para ADD-S.")

    p.add_argument("--check-gt", action="store_true",
                   help="Solo validar la coherencia dataset/modelo/K. Sin ROS.")

    p.add_argument("--bbox-source", default="bbox_visib",
                   choices=["bbox_visib", "bbox_obj"],
                   help="bbox_visib (parte visible, como daría YOLO) o "
                        "bbox_obj (objeto completo).")
    p.add_argument("--bbox-margin-px", type=float, default=0.0,
                   help="Margen añadido a la bbox, para emular una detección "
                        "holgada.")
    p.add_argument("--bbox-jitter-px", type=float, default=0.0,
                   help="Desplazamiento aleatorio de la bbox, para emular una "
                        "detección descentrada.")
    p.add_argument("--seed", type=int, default=0, help="Semilla del jitter.")

    p.add_argument("--image-topic", default="/navcam/left/navcam_left/image_raw")
    p.add_argument("--camera-info-topic",
                   default="/navcam/left/navcam_left/camera_info")
    p.add_argument("--detections-topic", default="/vision/detections")
    p.add_argument("--pose-topic", default="/vision/tube_pose")
    p.add_argument("--service", default="/d2co_pose_estimator_node/estimate_now")
    p.add_argument("--frame-id", default="navcam_left_optical_frame")
    p.add_argument("--class-id", default="0")

    p.add_argument("--settle-s", type=float, default=0.4,
                   help="Tiempo de asentamiento tras publicar cada muestra.")
    p.add_argument("--repeat-publish", type=int, default=2,
                   help="Veces que se republica cada muestra (QoS best-effort).")
    p.add_argument("--timeout-s", type=float, default=180.0,
                   help="Timeout de la llamada a ~/estimate_now. OJO: con un "
                        "prior de profundidad estrecho el refinado LM puede "
                        "tardar ~30 s/instancia; un timeout corto se "
                        "contabilizaria como falta de cobertura y FALSEARIA "
                        "la comparacion entre configuraciones.")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Reintentos por muestra si el nodo aún no había "
                        "recibido la imagen (correlación por stamp).")

    p.add_argument("--csv", default="", help="Ruta del CSV de resultados.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.dataset):
        print(f"ERROR: no existe el dataset {args.dataset}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.stl):
        print(f"ERROR: no existe el modelo {args.stl}", file=sys.stderr)
        return 1
    if args.check_gt:
        return run_check_gt(args)
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
