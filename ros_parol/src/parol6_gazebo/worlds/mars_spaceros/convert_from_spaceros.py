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
COLLISION_TARGET_TRIS = 40000


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
    mesh.vertex_normals  # fuerza el cálculo de normales suaves

    print("Bounds finales:", np.round(mesh.bounds[0], 2), np.round(mesh.bounds[1], 2))

    # VISUAL: OBJ completo con normales + textura (precisión 4 decimales)
    mesh.export(os.path.join(HERE, "martian_surface.obj"),
                include_normals=True, include_texture=True,
                mtl_name="material.mtl", digits=4)

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
