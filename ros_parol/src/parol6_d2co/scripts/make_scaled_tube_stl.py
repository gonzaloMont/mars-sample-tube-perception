#!/usr/bin/env python3
"""
make_scaled_tube_stl.py — Genera el STL del tubo PRE-ESCALADO para D2CO.

¿Por qué? El mars2020tube.stl original tiene proporciones incorrectas y el
resto del proyecto (Gazebo, ICP, LINEMOD) le aplica la escala NO UNIFORME:

    scale = (0.022939, 0.022939, 0.033212)   # (X, Y, Z)

D2CO carga el modelo con OpenMesh y solo admite escala UNIFORME (unidad de
medida: m, mm, …), así que necesita un STL ya escalado y en METROS. Este
script replica exactamente la transformación de
`parol6_vision.icp_utils.load_reference_cloud` (escala anisótropa + recentrado
en el centro del bounding box) y guarda el resultado.

Tras el escalado el tubo mide ~0.186 m de largo (eje Z) y ~0.021 m de
diámetro, igual que el cilindro de colisión del SDF.

Uso:
    python3 make_scaled_tube_stl.py \
        [--stl  ruta/al/mars2020tube.stl] \
        [--out  ruta/de/salida/mars2020tube_scaled.stl]

Sin argumentos usa el STL de parol6_gazebo y escribe en parol6_d2co/models/.
"""

import argparse
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("Se requiere open3d:  pip3 install open3d")

# Misma constante que parol6_vision/parol6_vision/icp_utils.py
DEFAULT_MESH_SCALE = (0.022939, 0.022939, 0.033212)


def find_default_paths():
    """Resuelve las rutas por defecto relativas a este script (en el repo)."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(here, "..", ".."))
    stl = os.path.join(src_root, "parol6_gazebo", "meshes", "mars2020tube.stl")
    out = os.path.join(src_root, "parol6_d2co", "models",
                       "mars2020tube_scaled.stl")
    return stl, out


def main() -> None:
    default_stl, default_out = find_default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stl", default=default_stl,
                    help="STL original (default: %(default)s)")
    ap.add_argument("--out", default=default_out,
                    help="STL de salida (default: %(default)s)")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.stl)
    if mesh.is_empty():
        sys.exit(f"No se pudo cargar la malla: {args.stl}")

    # Escala NO uniforme sobre los vértices (open3d.scale solo es uniforme).
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts *= np.asarray(DEFAULT_MESH_SCALE, dtype=np.float64)
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # Recentrado en el centro del bounding box: la pose estimada por D2CO
    # será la del CENTRO geométrico del tubo, igual que en el pipeline ICP.
    bbox = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bbox.get_center())

    # Normales coherentes (el STL las almacena por triángulo).
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not o3d.io.write_triangle_mesh(args.out, mesh):
        sys.exit(f"No se pudo escribir: {args.out}")

    ext = mesh.get_axis_aligned_bounding_box().get_extent()
    print(f"STL escalado escrito en: {args.out}")
    print(f"  extents (X,Y,Z) = ({ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}) m")
    print(f"  longitud ~{ext[2]:.3f} m, diametro ~{ext[0]:.4f} m")


if __name__ == "__main__":
    main()
