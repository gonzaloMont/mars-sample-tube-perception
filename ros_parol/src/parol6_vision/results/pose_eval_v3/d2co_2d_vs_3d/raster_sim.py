#!/usr/bin/env python3
"""
raster_sim.py — De dónde salen los puntos del modelo que D2CO compara con la
imagen. Reproduce en Python el criterio de RasterObjectModel3D::updateRaster
(arista viva si las normales de las dos caras difieren más de normal_epsilon
= 0.4, o arista de silueta) y el muestreo de addVisibleLine (un punto cada
2 mm; al menos uno por arista visible), con la oclusión por raycasting, para
las poses GT de las 12 primeras imágenes de test. Escribe raster_composicion.md.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
STL = os.path.join(SRC, 'parol6_d2co', 'models', 'mars2020tube_scaled.stl')
D = os.path.expanduser('~/parol6_pose_eval_dataset_v3')
STEP, NORMAL_EPS = 0.002, 0.4


def main():
    m = o3d.io.read_triangle_mesh(STL)
    P, inv = np.unique(np.round(np.asarray(m.vertices), 7), axis=0, return_inverse=True)   # soldar como OpenMesh
    F = inv[np.asarray(m.triangles)]
    P = P - (P.min(0) + P.max(0)) / 2                                                         # setBBCenterOrigOffset
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    E = defaultdict(list)
    for fi, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            E[(min(a, b), max(a, b))].append((fi, a, b))
    half = np.array([(fi, a, b, l[1 - i][0]) for l in E.values() for i, (fi, a, b) in enumerate(l)])
    crease = np.linalg.norm(n[half[:, 0]] - n[half[:, 3]], axis=1) > NORMAL_EPS
    gt = json.load(open(D + '/bop/scene_gt.json'))
    cam = json.load(open(D + '/bop/scene_camera.json'))
    meta = {json.loads(l)['im_id']: json.loads(l) for l in open(D + '/meta.jsonl')}
    ids = [k for k in sorted(gt, key=int) if meta[int(k)]['split'] == 'test'][:12]
    out = ['# Composición del raster del modelo de D2CO (poses GT de test)\n',
           f'Malla `parol6_d2co/models/mars2020tube_scaled.stl`: {len(F)} triángulos, {int(crease.sum() / 2)} aristas vivas '
           f'(|Δn| > {NORMAL_EPS}). Cabeza = z > 60 mm, tapa inferior = z < −80 mm (frame del tubo). Fuente: `raster_sim.py`.\n',
           '| imagen | puntos visibles | cabeza | % cabeza | tamaño cabeza (px a 512) | cuerpo | de ellos silueta | tapa inferior |',
           '|---|---|---|---|---|---|---|---|']
    for k in ids:
        R = np.array(gt[k][0]['cam_R_m2c']).reshape(3, 3)
        t = np.array(gt[k][0]['cam_t_m2c'])
        Pc, nc = P @ R.T + t, n @ R.T
        mid = (Pc[half[:, 1]] + Pc[half[:, 2]]) / 2
        sil = (np.einsum('ij,ij->i', mid, nc[half[:, 0]]) * np.einsum('ij,ij->i', mid, nc[half[:, 3]])) < 0
        sel = crease | sil
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(Pc.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        pts, tag = [], []
        for (fi, a, b, fo), isc in zip(half[sel], crease[sel]):
            p0, p1 = Pc[a], Pc[b]
            ns = int(np.linalg.norm(p1 - p0) / STEP)
            s = [p0] if ns == 0 else [p0 + (i / ns) * (p1 - p0) for i in range(ns + 1)]
            pts += s
            tag += ['viva' if isc else 'silueta'] * len(s)
        pts = np.array(pts, np.float32)
        d = np.linalg.norm(pts, axis=1)
        rays = np.hstack([np.zeros_like(pts), pts / d[:, None]]).astype(np.float32)
        vis = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy() >= d - 0.0005
        z = ((pts[vis] - t) @ R)[:, 2]
        reg = np.where(z > 0.06, 'cabeza', np.where(z < -0.08, 'tapa', 'cuerpo'))
        tg = np.array(tag)[vis]
        K = np.array(cam[k]['cam_K']).reshape(3, 3) / 2
        K[2, 2] = 1
        uv = pts[vis] @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        c = Counter(reg)
        hp = np.ptp(uv[reg == 'cabeza'], axis=0)
        out.append(f"| {k} | {vis.sum()} | {c['cabeza']} | {100 * c['cabeza'] / vis.sum():.0f} | {hp[0]:.1f} × {hp[1]:.1f} "
                   f"| {c['cuerpo']} | {int(((reg == 'cuerpo') & (tg == 'silueta')).sum())} | {c['tapa']} |")
    txt = '\n'.join(out) + '\n'
    open(os.path.join(HERE, 'raster_composicion.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
