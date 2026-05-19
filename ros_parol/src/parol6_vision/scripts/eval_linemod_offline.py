#!/usr/bin/env python3
"""
eval_linemod_offline.py — Verificación OFFLINE del detector LINEMOD.

Herramienta de diagnóstico (no es un nodo ROS). Carga la base de datos
generada por train_linemod_templates.py y ejecuta el matching + extrae pose 6D.

⚠️  DIFERENCIA CON NODO REAL:
  • Nodo real (linemod_pose_estimator_node.py):
    - Recibe imagen de cámara + detecciones YOLO
    - Recorta ROI en el bbox del tubo
    - Ejecuta LINEMOD solo en el ROI
    - Extrae pose 6D en el ROI

  • Este script:
    - Ejecuta LINEMOD sobre la imagen COMPLETA (sin ROI YOLO)
    - Mejor para imágenes sintéticas o cuando hay un solo tubo
    - Menos fiable en imágenes reales con clutter

Uso:
  python3 eval_linemod_offline.py \\
      --db    src/parol6_vision/models/linemod \\
      --image captura.png \\
      --threshold 80 \\
      --out    match_debug.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import cv2
    from parol6_vision import linemod_db, pose_solver
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import cv2
    from parol6_vision import linemod_db, pose_solver


def main() -> int:
    p = argparse.ArgumentParser(
        description="Verifica el matching LINEMOD offline y extrae pose 6D."
    )
    p.add_argument("--db", required=True, help="Directorio de la base de datos LINEMOD.")
    p.add_argument("--image", required=True, help="Imagen sobre la que hacer matching.")
    p.add_argument("--threshold", type=float, default=80.0,
                   help="Umbral de similitud LINEMOD [0..100].")
    p.add_argument("--out", default="match_debug.png", help="Imagen de salida anotada.")
    p.add_argument("--camera-z", type=float, default=0.75,
                   help="Altura de la cámara sobre el suelo [m].")
    p.add_argument("--ground-z", type=float, default=0.03,
                   help="Altura del tubo sobre el suelo [m].")
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] No se pudo leer la imagen: {args.image}", file=sys.stderr)
        return 1

    print(f"Cargando base de datos LINEMOD desde {args.db} ...")
    db = linemod_db.LinemodDatabase.load(args.db)
    print(f"  {len(db.entries)} templates cargados.")
    print()

    matches = db.match(img, threshold=args.threshold)
    if not matches:
        print("Sin matches por encima del umbral. Prueba a bajar --threshold.")
        return 0

    print(f"{len(matches)} matches encontrados. Mejor:")
    best = matches[0]
    entry = db.entries.get(int(best.template_id))
    print()
    print("▸ MATCHING (cv2.linemod - qué template coincide):")
    print(f"  template_id   = {best.template_id}")
    print(f"  similitud     = {best.similarity:.1f}/100")
    print(f"  esquina (x,y) = ({best.x}, {best.y}) píxeles")
    
    cx, cy = float(best.x), float(best.y)
    if entry is not None:
        cx = best.x + float(entry.centroid_offset[0])
        cy = best.y + float(entry.centroid_offset[1])
        print(f"  centro obj    = ({cx:.1f}, {cy:.1f}) píxeles")
        print(f"  plantilla     = yaw={entry.yaw_deg:.1f}° tilt={entry.tilt_deg:.1f}°")

    # Diagnóstico adicional: tamaño del template (bbox de matching).
    tpl_w, tpl_h = None, None
    try:
        tpls = db.ensure_detector().getTemplates(db.class_id, int(best.template_id))
        if tpls:
            tpl = tpls[0]
            tpl_w = int(tpl.width)
            tpl_h = int(tpl.height)
            print(f"  bbox template = {tpl_w}×{tpl_h} píxeles")
    except Exception as exc:
        pass

    # Diagnóstico adicional: error respecto al centroide observado en la imagen.
    # Solo es válido para renders sintéticos con fondo casi negro.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_frac = float(np.mean(gray < 20))
    obs_centroid = None
    if dark_frac > 0.60:
        _, fg = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        ys, xs = np.nonzero(fg)
        if len(xs) > 0:
            obs_centroid = (float(xs.mean()), float(ys.mean()))
            err_px = float(np.hypot(cx - obs_centroid[0], cy - obs_centroid[1]))
            print(f"  centro obs    = ({obs_centroid[0]:.1f}, {obs_centroid[1]:.1f}) píxeles")
            print(f"  error visual  = {err_px:.2f} píxeles")

    # ── Extracción de pose 6D ───────────────────────────────────────────────
    print()
    print("▸ POSE 6D (solve_pose_from_match - rotación + traslación):")
    print("  frame: óptico de cámara (Z delante, X derecha, Y abajo)")

    roi_offset = (0, 0)  # No hay ROI: matching es en imagen completa
    # Fallback: cámara cenital perfecta mirando hacia abajo
    R_cam_plane = np.array([[0.0, -1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0]], dtype=np.float64)
    t_cam_plane = np.array([0.0, 0.0, args.camera_z], dtype=np.float64)
    plane_z = args.ground_z

    try:
        pose_cam = pose_solver.solve_pose_from_match(
            match=best,
            db=db,
            roi_offset=roi_offset,
            R_cam_plane=R_cam_plane,
            t_cam_plane=t_cam_plane,
            plane_z=plane_z,
        )
        if pose_cam is not None:
            print(f"  rotación (R):   template_id={pose_cam.template_id}")
            if entry:
                print(f"                  yaw={entry.yaw_deg:.1f}° tilt={entry.tilt_deg:.1f}°")
            print(f"  traslación (t): ({pose_cam.t_cam_obj[0]:7.4f}, "
                  f"{pose_cam.t_cam_obj[1]:7.4f}, {pose_cam.t_cam_obj[2]:7.4f}) m")
            qx, qy, qz, qw = pose_solver.rotation_to_quaternion(pose_cam.R_cam_obj)
            print(f"  quaternión:     ({qx:7.3f}, {qy:7.3f}, {qz:7.3f}, {qw:7.3f})")
            print()
            print("  ⓘ Nota: Esta pose está en el frame óptico de la cámara.")
            print("    El nodo real transforma esto a base_link con tf2.")
        else:
            print("  None (solve_pose_from_match falló)")
    except Exception as e:
        print(f"  Error: {e}")

    # Anotar el match en la imagen.
    vis = img.copy()
    cv2.drawMarker(vis, (int(round(cx)), int(round(cy))), (0, 255, 0),
                   cv2.MARKER_CROSS, 24, 2)
    cv2.circle(vis, (int(best.x), int(best.y)), 4, (255, 255, 0), -1)
    if tpl_w is not None and tpl_h is not None:
        cv2.rectangle(vis, (int(best.x), int(best.y)),
                      (int(best.x + tpl_w), int(best.y + tpl_h)),
                      (0, 165, 255), 1)
    if obs_centroid is not None:
        cv2.drawMarker(vis, (int(round(obs_centroid[0])), int(round(obs_centroid[1]))),
                       (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 1)
    cv2.putText(vis, f"sim={best.similarity:.0f} id={best.template_id}",
                (int(round(cx)) + 8, int(round(cy)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(args.out, vis)
    print()
    print(f"Imagen anotada guardada en: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
