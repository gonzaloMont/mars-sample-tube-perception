#!/usr/bin/env python3
"""
analiza.py — Registro 2D frente a error 3D de D2CO (sin ROS). Escribe
resultados.md y metrica_2d_malla.csv.

1. Sobre d2co_malla_test60.csv (con T_est), por pose publicada:
   - error 2D unidireccional: media, para cada punto del modelo proyectado con
     la pose de D2CO, de la distancia al punto más cercano del modelo
     proyectado con la pose real (indulgente: un tubo proyectado más corto o
     que cruza al real puntúa bien);
   - error 2D bidireccional: el máximo de los dos sentidos (exige además que
     la proyección de D2CO cubra todo el tubo real);
   - longitud proyectada D2CO / real y ángulo entre ejes en la imagen.
2. Malla del paquete frente a cilindro simple (mismas 60 instancias).
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))      # ros_parol/src
STL = os.path.join(SRC, 'parol6_d2co', 'models', 'mars2020tube_scaled.stl')
D = os.path.expanduser('~/parol6_pose_eval_dataset_v3')
THR = 18.69   # ADD-S, 0.1 × diámetro (models_info.json)


def rows(name):
    return list(csv.DictReader(open(os.path.join(HERE, name))))


def f(v):
    try:
        return float(v)
    except ValueError:
        return float('nan')


def pct(x):
    return f'{100 * x:.1f}'


def metrics_2d():
    gt = json.load(open(D + '/bop/scene_gt.json'))
    cam = json.load(open(D + '/bop/scene_camera.json'))
    mesh = o3d.io.read_triangle_mesh(STL)
    o3d.utility.random.seed(0)
    pts = np.asarray(mesh.sample_points_uniformly(800).points)
    pts -= (pts.min(0) + pts.max(0)) / 2
    ends = np.array([[0, 0, -0.093], [0, 0, 0.093]])
    out = []
    for r in rows('d2co_malla_test60.csv'):
        if r['pose_ok'] != '1':
            continue
        g = gt[r['im_id']][int(r['inst'])]
        R = np.array(g['cam_R_m2c']).reshape(3, 3)
        t = np.array(g['cam_t_m2c'])
        K = np.array(cam[r['im_id']]['cam_K']).reshape(3, 3)
        T = np.array([float(v) for v in r['T_est'].split()]).reshape(3, 4)

        def pr(P, R_, t_):
            q = (P @ R_.T + t_) @ K.T
            return q[:, :2] / q[:, 2:]
        a, b = pr(pts, R, t), pr(pts, T[:, :3], T[:, 3])
        d = np.sqrt(((b[:, None] - a[None]) ** 2).sum(-1))
        uni = d.min(1).mean()
        bi = max(uni, d.min(0).mean())
        ea, eb = pr(ends, R, t), pr(ends, T[:, :3], T[:, 3])
        va, vb = ea[1] - ea[0], eb[1] - eb[0]
        ang = np.degrees(np.arccos(min(1.0, abs(va @ vb) / np.linalg.norm(va) / np.linalg.norm(vb))))
        out.append({'im_id': r['im_id'], 'inst': r['inst'], 'proj2d_uni_px': round(uni, 2),
                    'proj2d_bi_px': round(bi, 2), 'ratio_longitud': round(np.linalg.norm(vb) / np.linalg.norm(va), 3),
                    'angulo_eje_img_deg': round(ang, 2), 'add_s_mm': f(r['add_s_mm'])})
    with open(os.path.join(HERE, 'metrica_2d_malla.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out, len(rows('d2co_malla_test60.csv'))


def summary_3d(name):
    rr = rows(name)
    ok = [r for r in rr if r['pose_ok'] == '1']
    rad = np.array([f(r['err_radial_mm']) for r in ok])
    lat = np.array([f(r['err_lateral_mm']) for r in ok])
    ax = np.array([f(r['err_axis_deg']) for r in ok])
    k = sum(int(r['add_s_pass']) for r in rr)
    return (f"| {len(rr)} | {pct(len(ok) / len(rr))} | {pct(k / len(rr))} ({k}) | {np.median(np.abs(rad)):.0f} "
            f"| {np.median(rad):+.0f} | {np.median(lat):.1f} | {pct((lat < THR).mean())} | {np.median(ax):.1f} "
            f"| {pct((ax < 10).mean())} | {pct((np.abs(rad) < THR).mean())} |")


def main():
    m, n_tot = metrics_2d()
    o = np.array([[x['proj2d_uni_px'], x['proj2d_bi_px'], x['ratio_longitud'], x['angulo_eje_img_deg'], x['add_s_mm']]
                  for x in m])
    good = o[:, 1] < 5
    L = ['# D2CO: registro 2D frente a error 3D\n',
         f'Malla del paquete, 60 primeras instancias de test, caja GT, sin umbral: {len(m)} poses publicadas de {n_tot}. '
         'Fuente: `d2co_malla_test60.csv` → `metrica_2d_malla.csv` (`analiza.py`).\n',
         '| Métrica 2D (px a 1024) | Mediana | < 5 px | < 10 px |', '|---|---|---|---|',
         f'| Unidireccional (D2CO → real; indulgente) | {np.median(o[:, 0]):.1f} | {pct((o[:, 0] < 5).mean())} % '
         f'| {pct((o[:, 0] < 10).mean())} % |',
         f'| **Bidireccional** (cubre también el tubo real) | **{np.median(o[:, 1]):.1f}** | **{pct((o[:, 1] < 5).mean())} %** '
         f'| {pct((o[:, 1] < 10).mean())} % |',
         f'\n- Ángulo entre los ejes del tubo en la imagen: mediana {np.median(o[:, 3]):.1f}°; < 5° en el '
         f'{pct((o[:, 3] < 5).mean())} %.',
         f'- Longitud proyectada D2CO / real: mediana {np.median(o[:, 2]):.2f} (p25 {np.percentile(o[:, 2], 25):.2f}, '
         f'p75 {np.percentile(o[:, 2], 75):.2f}).',
         f'- Poses con registro 2D completo (bidireccional < 5 px): {good.sum()}; de ellas cumplen ADD-S '
         f'(< {THR} mm): {(o[good, 4] < THR).sum()}; ADD-S mediano {np.median(o[good, 4]):.0f} mm.\n',
         '## Malla del paquete frente a cilindro simple (mismas 60 instancias)\n',
         '| modelo | n | cobertura % | recall ADD-S % | \\|radial\\| med mm | radial med mm | lateral med mm '
         '| lateral < umbral % | eje med ° | eje < 10° % | \\|radial\\| < umbral % |',
         '|---|---|---|---|---|---|---|---|---|---|---|',
         '| malla (`d2co_malla_test60.csv`) ' + summary_3d('d2co_malla_test60.csv'),
         '| cilindro (`d2co_cilindro_test60.csv`) ' + summary_3d('d2co_cilindro_test60.csv')]
    txt = '\n'.join(L) + '\n'
    open(os.path.join(HERE, 'resultados.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
