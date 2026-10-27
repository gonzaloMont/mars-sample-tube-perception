#!/usr/bin/env python3
"""crear_cilindro.py — cilindro_simple.stl: Ø 20.8 mm, 186 mm, 64 facetas, centrado y con el eje en Z
(mismo marco que mars2020tube_scaled.stl), para comparar con la malla detallada del paquete."""
import os
import open3d as o3d
c = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0104, height=0.186, resolution=64, split=1)
c.compute_triangle_normals()
c.compute_vertex_normals()
o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cilindro_simple.stl'), c)
