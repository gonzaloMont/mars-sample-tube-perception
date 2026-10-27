#!/usr/bin/env python3
"""
eval_sgbm.py — Evaluación de la reconstrucción estéreo (SGBM + WLS, el mismo
código que el nodo: parol6_vision.stereo_utils.StereoMatcher) frente a la
profundidad real de la cámara izquierda, sin ICP.

Para cada par de la partición elegida de ~/parol6_pose_eval_dataset_v3 y cada
configuración estéreo, cada punto de SGBM se proyecta con la K de la cámara
izquierda y se compara su profundidad con la real en ese píxel
(error = z_SGBM − z_real). Regiones, en la rejilla de trabajo de SGBM (celdas
de 2×2 píxeles):
  imagen          profundidad real dentro de [min_depth, max_depth] = [0.3, 2] m
  zona de trabajo la región de la imagen donde caen los tubos agarrables
                  (u 173–688, v 567–923; docs/pose_evaluation.md)
  tubo            píxeles visibles de los tubos (modelo en su pose real,
                  visible si z_modelo ≤ z_real + 5 mm)
Métricas: cobertura (fracción de celdas de la región con algún punto SGBM) y,
sobre los puntos de la región, error mediano (sesgo), |error| mediano y
porcentaje con |error| < 10 mm, < 30 mm y > 100 mm.

Uso (entorno ROS cargado):
  eval_sgbm.py --split test --configs ../pose_eval_v3/configs/stereo_wls8000.yaml ../pose_eval_v3/configs/stereo_wls500.yaml
Escribe sgbm_<config>_<split>.csv (por imagen y región) y resumen_<split>.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser('~/parol6_pose_eval_dataset_v3')
ZONA = (173, 688, 567, 923)          # u_min, u_max, v_min, v_max (px a 1024)
REGIONES = ('imagen', 'zona de trabajo', 'tubo')


def load_params(path):
    return next(iter(yaml.safe_load(open(path)).values()))['ros__parameters']


def model_points(n=20000):
    import open3d as o3d
    stl = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'parol6_d2co', 'models', 'mars2020tube_scaled.stl'))
    m = o3d.io.read_triangle_mesh(stl)
    o3d.utility.random.seed(0)
    p = np.asarray(m.sample_points_uniformly(n).points)
    return p - (p.min(0) + p.max(0)) / 2


def evaluate(config, split, pts_model, gt, cam, metas):
    from parol6_vision.stereo_utils import StereoMatcher
    params = load_params(config)
    zmin, zmax = float(params['min_depth']), float(params['max_depth'])
    rows, errs = [], {r: [] for r in REGIONES}
    matcher = None
    t_total = 0.0
    for m in metas:
        k = str(m['im_id'])
        left = cv2.imread(os.path.join(D, m['file']), 0)
        right = cv2.imread(os.path.join(D, m['right_file']), 0)
        depth = cv2.imread(os.path.join(D, m['depth_file']), cv2.IMREAD_UNCHANGED).astype(np.float32) \
            * float(cam[k].get('depth_scale', 0.001))
        if matcher is None:
            matcher = StereoMatcher.from_scene_camera(cam[k], (left.shape[1], left.shape[0]), params)
        t0 = time.time()
        xyz = matcher.compute_points(left, right)
        t_total += time.time() - t0
        K = np.array(cam[k]['cam_K']).reshape(3, 3)
        H, W = depth.shape
        h2, w2 = H // 2, W // 2
        # máscaras de región en la rejilla de trabajo (celda = píxel superior izquierdo)
        dc = depth[0::2, 0::2]
        valid = (dc > zmin) & (dc < zmax)
        zona = np.zeros_like(valid)
        zona[ZONA[2] // 2:ZONA[3] // 2 + 1, ZONA[0] // 2:ZONA[1] // 2 + 1] = True
        tubo = np.zeros_like(valid)
        for g in gt.get(k, []):
            R = np.array(g['cam_R_m2c']).reshape(3, 3)
            t = np.array(g['cam_t_m2c'])
            p = pts_model @ R.T + t
            u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z = u[ok], v[ok], p[ok, 2]
            zg = depth[v, u]
            vis = (zg > 0) & (z <= zg + 0.005)
            tubo[v[vis] // 2, u[vis] // 2] = True
        masks = {'imagen': valid, 'zona de trabajo': valid & zona, 'tubo': tubo}
        # puntos SGBM → píxel de la imagen izquierda → profundidad real
        q = xyz[xyz[:, 2] > 1e-6]
        u = np.round(K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        q, u, v = q[ok], u[ok], v[ok]
        zg = depth[v, u]
        cu, cv_ = np.minimum(u // 2, w2 - 1), np.minimum(v // 2, h2 - 1)
        hit = np.zeros_like(valid)
        hit[cv_, cu] = True
        row = {'im_id': m['im_id'], 'luz_g': (m.get('light_diffuse') or [float('nan')])[0], 'n_puntos': len(q)}
        for r in REGIONES:
            mk = masks[r]
            n_cells = int(mk.sum())
            sel = mk[cv_, cu] & (zg > 0)
            e = (q[sel, 2] - zg[sel]) * 1e3
            errs[r].append(e)
            row[f'{r}|celdas'] = n_cells
            row[f'{r}|cobertura'] = round(float((hit & mk).sum() / n_cells), 4) if n_cells else ''
            row[f'{r}|err_med_mm'] = round(float(np.median(e)), 2) if len(e) else ''
            row[f'{r}|abs_med_mm'] = round(float(np.median(np.abs(e))), 2) if len(e) else ''
        rows.append(row)
    name = os.path.splitext(os.path.basename(config))[0]
    with open(os.path.join(HERE, f'sgbm_{name}_{split}.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return name, rows, {r: np.concatenate(v) for r, v in errs.items()}, t_total / len(metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--configs', nargs='+', required=True)
    a = ap.parse_args()
    gt = json.load(open(D + '/bop/scene_gt.json'))
    cam = json.load(open(D + '/bop/scene_camera.json'))
    metas = [json.loads(l) for l in open(D + '/meta.jsonl')]
    metas = [m for m in metas if m['split'] == a.split]
    pts = model_points()
    L = [f'# SGBM frente a la profundidad real — partición {a.split} ({len(metas)} pares)\n',
         'Puntos de SGBM proyectados en la imagen izquierda; error = z_SGBM − z_real. Cobertura = media por imagen de '
         'la fracción de celdas (2×2 px) de la región con algún punto. Errores sobre todos los puntos de la región. '
         'Fuente: `eval_sgbm.py` → `sgbm_<config>_' + a.split + '.csv`.\n',
         '| configuración | región | cobertura % | error mediano (sesgo) mm | \\|error\\| mediano mm | < 10 mm % | < 30 mm % '
         '| > 100 mm % | puntos |', '|---|---|---|---|---|---|---|---|---|']
    luz = []
    for c in a.configs:
        name, rows, errs, t_img = evaluate(c, a.split, pts, gt, cam, metas)
        for r in REGIONES:
            e = errs[r]
            cov = np.mean([x[f'{r}|cobertura'] for x in rows if x[f'{r}|cobertura'] != ''])
            ae = np.abs(e)
            L.append(f'| {name} | {r} | {100 * cov:.1f} | {np.median(e):+.1f} | {np.median(ae):.1f} | {100 * (ae < 10).mean():.1f} '
                     f'| {100 * (ae < 30).mean():.1f} | {100 * (ae > 100).mean():.1f} | {len(e)} |')
        luz.append((name, rows))
        L.append(f'| {name} | tiempo por par (SGBM + WLS, 512²) | {1e3 * t_img:.0f} ms | | | | | | |')
    g = np.array([x['luz_g'] for x in luz[0][1]])
    cuts = np.percentile(g, [100 / 3, 200 / 3])
    L += ['\n## Por iluminación (mediana por imagen del \\|error\\| en la región; terciles de la difusa g)\n',
          f'| configuración | región | g < {cuts[0]:.3f} | {cuts[0]:.3f}–{cuts[1]:.3f} | > {cuts[1]:.3f} |', '|---|---|---|---|---|']
    for name, rows in luz:
        for r in ('zona de trabajo', 'tubo'):
            cells = []
            for lo, hi in ((-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 9)):
                v = [x[f'{r}|abs_med_mm'] for x in rows if lo <= x['luz_g'] < hi and x[f'{r}|abs_med_mm'] != '']
                cv = [x[f'{r}|cobertura'] for x in rows if lo <= x['luz_g'] < hi and x[f'{r}|cobertura'] != '']
                cells.append(f'{np.median(v):.1f} mm (cob. {100 * np.mean(cv):.0f} %, n={len(v)})')
            L.append(f'| {name} | {r} | ' + ' | '.join(cells) + ' |')
    # Sensibilidad teórica: z = f·B/d  →  |dz/dd| = z²/(f·B), a la resolución de trabajo de SGBM
    k0 = str(metas[0]['im_id'])
    f_work = np.array(cam[k0]['cam_K']).reshape(3, 3)[0, 0] * float(load_params(a.configs[0])['scale_factor'])
    base = float(np.linalg.norm(cam[k0]['cam_t_l2r']))
    z_med = float(np.median([np.linalg.norm(gg['cam_t_m2c']) for m in metas for gg in gt.get(str(m['im_id']), [])]))
    L.append(f'\nSensibilidad: f = {f_work:.2f} px (resolución de trabajo), B = {base:.3f} m; a la distancia mediana '
             f'cámara–tubo de esta partición ({z_med:.3f} m) la disparidad es {f_work * base / z_med:.1f} px y 1 px de '
             f'disparidad equivale a {1e3 * z_med ** 2 / (f_work * base):.1f} mm de profundidad.')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(HERE, f'resumen_{a.split}.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    sys.exit(main())
