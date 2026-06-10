#!/usr/bin/env python3
"""
Convierte el terreno marciano de Space ROS (Gazebo Ignition) al formato que usa
Gazebo Classic 11 en este proyecto.

Entrada:
    spaceros_gz_demos/models/martian_surface/resources/martian_surface.glb
    (malla .glb, no soportada por Gazebo Classic).

Salida (en esta misma carpeta):
    martian_surface.obj            visual completo texturizado (~520k tri)
    material.mtl + Material.png     material/atlas de textura (se reescribe el .mtl
                                    a mano tras generar, ver README/world)
    martian_surface_collision.obj  colisión decimada (~40k tri) para no penalizar ODE

Transformaciones HORNEADAS en los vértices (para no depender de <scale>/<pose> en
el .world, que ODE maneja peor en colisiones de malla):
    * escala x100  (igual que el <scale>100 100 100</scale> del model.sdf original)
    * rotación +90° sobre X  (Y-up de la malla -> Z-up de Gazebo)
    * recentrado en XY al origen y suelo del origen llevado a z≈0
    * RECORTE a un cuadrado de +-CROP_RADIUS m alrededor del robot (igual criterio
      que worlds/mars_optimized): el .glb completo son 200x200 m / ~520k tri / 50 MB,
      demasiado para gzclient. Recortar por máscara de caras PRESERVA la textura
      (las UV se mantienen) y baja a ~50k tri. Pon CROP_RADIUS=None para no recortar.

Requiere: trimesh, open3d, pillow.
Uso:
    python3 convert_from_spaceros.py [ruta/al/martian_surface.glb]
"""
import os
import sys
import numpy as np
import trimesh
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.expanduser(
    "~/Downloads/spaceros_gz_demos-main/spaceros_gz_demos/"
    "models/martian_surface/resources/martian_surface.glb"
)
# ±15 m (~12k tri) iguala el peso de worlds/mars_bumpy (~12.8k), que renderiza bien
# con las cámaras estéreo en la iGPU Intel. Subirlo penaliza el RTF (la iGPU renderiza
# la malla 2 veces por frame estéreo). None = terreno completo de 200 m.
CROP_RADIUS = 15.0
COLLISION_TARGET_TRIS = 6000
# Features para el estéreo: la textura del .glb es lisa y de bajo contraste, y el
# matcher no encuentra correspondencias en el terreno (sí en el brazo, por sus bordes).
# Re-normalizamos las UV del parche y teselamos (UV_TILING) + realzamos el contraste y
# añadimos detalle multi-octava de alta frecuencia (rocoso). RTF-neutral (solo muestreo
# de textura). El período de repetición (~30 m / UV_TILING) es >> al rango de disparidad,
# así que no introduce ambigüedad estéreo.
UV_TILING = 4.0       # repeticiones de la textura sobre el parche (~30 m)
TEX_UPSCALE = 2048    # resolución de la textura realzada
TEX_CONTRAST = 1.6    # factor de contraste
TEX_DETAIL = 42.0     # amplitud del detalle inyectado (sobre 0-255)


def enhance_texture(path, upscale=TEX_UPSCALE, contrast=TEX_CONTRAST,
                    detail_amp=TEX_DETAIL, seed=0):
    """Realza la textura para el estéreo: upscale + contraste + detalle multi-octava."""
    from PIL import Image
    base = np.asarray(Image.open(path).convert("RGB").resize((upscale, upscale),
                      Image.BILINEAR)).astype(np.float32)
    H, W = base.shape[:2]
    rng = np.random.default_rng(seed)
    detail = np.zeros((H, W), np.float32)
    for cells, amp in [(16, 1.0), (32, 0.6), (64, 0.35), (128, 0.2), (256, 0.12)]:
        n = rng.normal(0, 1, (cells, cells)).astype(np.float32)
        n = (n - n.min()) / max(n.max() - n.min(), 1e-6)
        n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((W, H),
                       Image.BICUBIC), dtype=np.float32) / 255.0 - 0.5
        detail += amp * n
    detail /= max(np.abs(detail).max(), 1e-6)         # [-1, 1]
    mean = base.mean(axis=(0, 1), keepdims=True)
    base = mean + (base - mean) * contrast            # contraste
    base = base + detail[..., None] * detail_amp       # detalle de alta frecuencia
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(path)
    print(f"Textura realzada -> {upscale}² (contraste x{contrast}, detalle ±{detail_amp:.0f})")


def main(src):
    scene = trimesh.load(src)
    mesh = scene.to_geometry()  # funde las 5 sub-mallas en una con atlas de textura

    # escala x100 + rotación +90° en X (Y-up -> Z-up)
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = 100.0
    Rx = trimesh.transformations.rotation_matrix(np.radians(90.0), [1, 0, 0])
    mesh.apply_transform(Rx @ S)

    # centrar en XY
    cx = (mesh.bounds[0, 0] + mesh.bounds[1, 0]) / 2.0
    cy = (mesh.bounds[0, 1] + mesh.bounds[1, 1]) / 2.0
    mesh.apply_transform(trimesh.transformations.translation_matrix([-cx, -cy, 0]))

    # suelo del origen -> z = 0
    v = mesh.vertices
    r = np.hypot(v[:, 0], v[:, 1])
    ground = float(np.median(v[r < 1.0][:, 2]))
    mesh.apply_transform(trimesh.transformations.translation_matrix([0, 0, -ground]))

    # RECORTE a +-CROP_RADIUS m por máscara de caras (preserva UV y textura).
    if CROP_RADIUS is not None:
        fv = mesh.vertices[mesh.faces]  # (F,3,3)
        inside = ((np.abs(fv[:, :, 0]) <= CROP_RADIUS).all(axis=1) &
                  (np.abs(fv[:, :, 1]) <= CROP_RADIUS).all(axis=1))
        mesh = mesh.submesh([np.where(inside)[0]], append=True)
        print(f"Recortado a +-{CROP_RADIUS} m: {len(mesh.faces)} tri (desde {len(inside)})")

    mesh.vertex_normals  # fuerza el cálculo de normales suaves
    print("Bounds finales:", np.round(mesh.bounds[0], 2), np.round(mesh.bounds[1], 2))

    # Re-normalizar las UV del parche a [0,1] y teselar (UV_TILING) para dar features
    # finos al estéreo. Gazebo/OGRE usa wrap, así que vt > 1 repite la textura.
    if getattr(mesh.visual, "uv", None) is not None:
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        span = uv.max(axis=0) - uv.min(axis=0)
        span[span == 0] = 1.0
        mesh.visual.uv = (uv - uv.min(axis=0)) / span * UV_TILING

    # VISUAL: OBJ completo con normales + textura (precisión 4 decimales)
    mesh.export(os.path.join(HERE, "martian_surface.obj"),
                include_normals=True, include_texture=True,
                mtl_name="material.mtl", digits=4)

    # Realce de la textura para el estéreo (sobre la Material.png recién escrita)
    enhance_texture(os.path.join(HERE, "Material.png"))

    # COLISIÓN: decimada con open3d, solo geometría
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    om.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    om = om.simplify_quadric_decimation(target_number_of_triangles=COLLISION_TARGET_TRIS)
    om.remove_degenerate_triangles()
    om.remove_duplicated_vertices()
    cm = trimesh.Trimesh(vertices=np.asarray(om.vertices),
                         faces=np.asarray(om.triangles), process=True)
    cm.export(os.path.join(HERE, "martian_surface_collision.obj"), include_texture=False)
    print(f"Colisión: {len(cm.faces)} tri (desde {len(mesh.faces)})")
    print("OJO: revisa material.mtl (trimesh pone Kd=0.4; este proyecto usa Kd 1 1 1, illum 1).")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.isfile(src):
        sys.exit(f"No existe el .glb de entrada: {src}")
    main(src)
