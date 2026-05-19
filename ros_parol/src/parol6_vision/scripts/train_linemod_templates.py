#!/usr/bin/env python3
"""
train_linemod_templates.py — Generación OFFLINE de la base de datos LINEMOD.

╔══════════════════════════════════════════════════════════════════════════╗
║  Este script NO es un nodo ROS. Es una herramienta offline que se ejecuta ║
║  UNA sola vez (o cada vez que cambie la geometría del tubo o la cámara).  ║
║                                                                            ║
║  Flujo:                                                                    ║
║    1. Carga mars2020tube.stl y le aplica el escalado OBLIGATORIO           ║
║       [0.022939, 0.022939, 0.033212]  (delegado en mesh_utils).            ║
║    2. Muestrea una rejilla de orientaciones (yaw × tilt).                  ║
║    3. Renderiza cada orientación con pyrender usando las intrínsecas       ║
║       reales de la cámara cenital.                                         ║
║    4. Registra cada render como template en cv2.linemod, guardando su      ║
║       pose 6D asociada.                                                    ║
║    5. Vuelca la base de datos a disco (templates + poses + metadatos).     ║
╚══════════════════════════════════════════════════════════════════════════╝

Uso típico
──────────
  python3 train_linemod_templates.py \\
      --stl   src/parol6_gazebo/meshes/mars2020tube.stl \\
      --out   src/parol6_vision/models/linemod \\
      --distance 0.72 \\
      --yaw-steps 36 --tilt-steps 3

El artefacto resultante en --out lo carga directamente el nodo de inferencia
linemod_pose_estimator_node.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Permite ejecutar el script tanto instalado como desde el árbol de fuentes.
try:
    from parol6_vision import mesh_utils, rendering, linemod_db
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parol6_vision import mesh_utils, rendering, linemod_db


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera la base de datos de templates LINEMOD del tubo.")
    p.add_argument("--stl", required=True,
                   help="Ruta al fichero mars2020tube.stl original.")
    p.add_argument("--out", required=True,
                   help="Directorio de salida para la base de datos LINEMOD.")
    p.add_argument("--distance", type=float, default=0.72,
                   help="Distancia cámara->tubo [m] (altura cámara sobre suelo). "
                        "Cámara a z=0.75, tubo a z≈0.03 -> 0.72 por defecto.")
    p.add_argument("--yaw-steps", type=int, default=36,
                   help="Nº de orientaciones de yaw (densidad angular).")
    p.add_argument("--tilt-steps", type=int, default=3,
                   help="Nº de inclinaciones (1 = solo tubo perfectamente plano).")
    p.add_argument("--tilt-max-deg", type=float, default=18.0,
                   help="Inclinación máxima muestreada [grados].")
    p.add_argument("--width", type=int, default=640, help="Ancho de imagen.")
    p.add_argument("--height", type=int, default=480, help="Alto de imagen.")
    p.add_argument("--hfov", type=float, default=1.3962634,
                   help="FOV horizontal [rad] (de overhead_camera.urdf.xacro).")
    p.add_argument("--camera-info", default=None,
                   help="YAML de camera_info opcional; si se da, sobreescribe "
                        "--width/--height/--hfov con las intrínsecas reales.")
    p.add_argument("--debug-dir", default=None,
                   help="Si se indica, guarda los renders PNG para inspección.")
    return p.parse_args()


def load_intrinsics(args) -> tuple[np.ndarray, int, int]:
    """Obtiene K, width, height desde un camera_info YAML o desde el FOV."""
    if args.camera_info:
        import yaml
        with open(args.camera_info, "r", encoding="utf-8") as fp:
            ci = yaml.safe_load(fp)
        width = int(ci["image_width"])
        height = int(ci["image_height"])
        k = ci["camera_matrix"]["data"]
        K = np.array(k, dtype=np.float64).reshape(3, 3)
        return K, width, height
    K = mesh_utils.intrinsics_from_hfov(args.width, args.height, args.hfov)
    return K, args.width, args.height


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.stl):
        print(f"[ERROR] No existe el .stl: {args.stl}", file=sys.stderr)
        return 1

    # ── 1. Malla escalada (REQUISITO GEOMÉTRICO) ────────────────────────────
    print("[1/5] Cargando y escalando la malla del tubo...")
    print(f"      escala no uniforme = {mesh_utils.DEFAULT_MESH_SCALE}")
    model = mesh_utils.load_tube_mesh(args.stl)
    print(f"      tubo escalado: longitud={model.length:.4f} m, "
          f"radio={model.radius:.4f} m")

    # ── 2. Intrínsecas + viewpoints ─────────────────────────────────────────
    K, width, height = load_intrinsics(args)
    print(f"[2/5] Intrínsecas: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}  ({width}x{height})")

    viewpoints = rendering.sample_tube_viewpoints(
        yaw_steps=args.yaw_steps,
        tilt_steps=args.tilt_steps,
        tilt_max_deg=args.tilt_max_deg,
    )
    print(f"      {len(viewpoints)} viewpoints muestreados "
          f"({args.yaw_steps} yaw × {args.tilt_steps} tilt)")

    # ── 3. Renderer + base de datos ─────────────────────────────────────────
    print("[3/5] Inicializando renderer y detector LINEMOD...")
    renderer = rendering.TemplateRenderer(
        tube_mesh=model.mesh, K=K, width=width, height=height,
        render_distance=args.distance,
    )
    db = linemod_db.LinemodDatabase(
        class_id="tube",
        render_distance=args.distance,
        image_size=(width, height),
        K=K,
    )

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    # ── 4. Render + registro de templates ───────────────────────────────────
    print("[4/5] Renderizando y registrando templates...")
    import cv2  # solo para volcar PNGs de debug
    accepted, rejected = 0, 0
    for idx, vp in enumerate(viewpoints):
        result = renderer.render(vp.rotation)
        if result is None:
            rejected += 1
            continue

        tid = db.add_template(
            color_bgr=result.color,
            mask=result.mask,
            rotation=vp.rotation,
            centroid_uv=result.centroid_uv,
            yaw_deg=vp.yaw_deg,
            tilt_deg=vp.tilt_deg,
        )
        if tid is None:
            rejected += 1
            continue
        accepted += 1

        if args.debug_dir:
            cv2.imwrite(
                os.path.join(args.debug_dir, f"tpl_{tid:04d}.png"),
                result.color,
            )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(viewpoints):
            print(f"      progreso: {idx + 1}/{len(viewpoints)}  "
                  f"(aceptados={accepted}, rechazados={rejected})")

    renderer.close()

    if accepted == 0:
        print("[ERROR] LINEMOD rechazó todos los templates. Revisa la "
              "iluminación del render o la malla.", file=sys.stderr)
        return 2

    # Validación: si >10% de templates se rechazan, advertir
    rejection_rate = rejected / len(viewpoints)
    if rejection_rate > 0.10:
        print(f"[WARNING] Alta tasa de rechazo: {rejected}/{len(viewpoints)} templates "
            f"rechazados ({rejection_rate*100:.1f}%).", file=sys.stderr)
        print(f"          Esto sugiere que los gradientes detectados son insuficientes.",
            file=sys.stderr)
        print(f"          Posibles causas:", file=sys.stderr)
        print(f"            • Iluminación del render demasiado plana (gradientes débiles)",
            file=sys.stderr)
        print(f"            • num_features demasiado alto (reduce en linemod_db.py)",
            file=sys.stderr)
        print(f"            • Malla con geometría problematica", file=sys.stderr)
        print(f"          Ver LINEMOD_DIAGNOSTICO.md para más detalles.", file=sys.stderr)

    # ── 5. Guardado ──────────────────────────────────────────────────────────
    print(f"[5/5] Guardando base de datos en: {args.out}")
    db.save(args.out)
    print(f"      OK — {accepted} templates guardados.")
    print()
    print("Artefactos generados:")
    print(f"  • {os.path.join(args.out, 'templates_tube.yml')}")
    print(f"  • {os.path.join(args.out, 'linemod_meta.json')}")
    print(f"  • {os.path.join(args.out, 'linemod_poses.npz')}")
    print()
    print("Ahora puedes lanzar la inferencia:")
    print("  ros2 run parol6_vision linemod_pose_estimator_node.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
