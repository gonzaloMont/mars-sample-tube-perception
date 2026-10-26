#!/usr/bin/env python3
"""
calidad_sgbm.py — Calidad de la reconstrucción SGBM en los píxeles del TUBO,
sin ICP: mide directamente si el estéreo empeora entre bases y con la luz.

Por instancia: se proyecta la superficie del modelo (pose GT) y se queda con
los píxeles visibles (z_modelo ≤ z_depthGT + 5 mm, como el generador), en la
rejilla de trabajo de SGBM (media resolución). Sobre esas celdas:
  cobertura  = fracción de celdas del tubo con algún punto SGBM
  error      = z_SGBM − z_tubo (z_tubo = mínimo z del modelo en la celda)
Se resume por base de datos y, en v3, por terciles de la difusa g.

Uso: calidad_sgbm.py [--stereo-params YAML] [--md analysis/calidad_sgbm.md]
Requiere el entorno de ROS cargado (importa parol6_vision).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import wilson, pct  # noqa: E402,F401

DATASETS = {'v1': '~/parol6_pose_eval_dataset', 'v3': '~/parol6_pose_eval_dataset_v3'}


def correlations(v3, v1, n_boot=4000, seed=0):
    """Spearman g ↔ calidad en v3 (sin NaN) y bootstrap de la diferencia de
    medias v3-oscuro (g < 0.69) − v1 en la fracción de celdas buenas."""
    def rank(x):
        return np.argsort(np.argsort(x)).astype(float)
    sel = [r for r in v3 if r['frac_10mm'] == r['frac_10mm']]
    g = np.array([r['g'] for r in sel])
    lines = []
    for k, lab in (('frac_10mm', 'fracción de celdas con |error| < 10 mm'), ('abs_med_mm', '|error z| mediano')):
        y = np.array([r[k] for r in sel])
        rho = np.corrcoef(rank(g), rank(y))[0, 1]
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(g), len(g))
            bs.append(np.corrcoef(rank(g[i]), rank(y[i]))[0, 1])
        lines.append(f'Spearman (v3) entre g y {lab}: ρ = {rho:+.2f} (IC 95 % {np.percentile(bs, 2.5):+.2f} … '
                     f'{np.percentile(bs, 97.5):+.2f}, n = {len(g)}).')
    rng = np.random.default_rng(seed)
    a = np.array([r['frac_10mm'] for r in v1 if r['frac_10mm'] == r['frac_10mm']])
    for lo, hi, lab in ((-1, 0.69, 'g < 0.69'), (0.9, 2, 'g ≥ 0.9')):
        b = np.array([r['frac_10mm'] for r in sel if lo <= r['g'] < hi])
        d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n_boot)]
        lines.append(f'Fracción media de celdas buenas, v3 {lab} − v1: {100 * (b.mean() - a.mean()):+.1f} pp '
                     f'(IC 95 % bootstrap {100 * np.percentile(d, 2.5):+.1f} … {100 * np.percentile(d, 97.5):+.1f}).')
    return '\n' + '\n'.join(lines)


def from_csv(path):
    import csv
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({k: (v if k == 'base' else float(v)) for k, v in r.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stereo-params', default='')
    ap.add_argument('--md', default=os.path.join(HERE, 'calidad_sgbm.md'))
    ap.add_argument('--csv', default=os.path.join(HERE, 'calidad_sgbm.csv'))
    a = ap.parse_args()
    from ament_index_python.packages import get_package_share_directory
    sys.path.insert(0, os.path.join(get_package_share_directory('parol6_vision'), '..', '..', 'lib', 'parol6_vision'))
    from eval_pose_dataset import load_dataset, load_model_points
    from parol6_vision.stereo_utils import StereoMatcher
    params = None
    if a.stereo_params:
        import yaml
        params = next(iter(yaml.safe_load(open(a.stereo_params)).values()))['ros__parameters']
    stl = os.path.join(get_package_share_directory('parol6_gazebo'), 'meshes', 'mars2020tube.stl')
    pts = load_model_points(stl, 20000, 0)

    rows = []
    for tag, d in DATASETS.items():
        root = os.path.expanduser(d)
        samples = load_dataset(root, [], 0.0)
        metas = {}
        with open(os.path.join(root, 'meta.jsonl')) as fh:
            for line in fh:
                m = json.loads(line)
                metas[int(m['im_id'])] = m
        matcher, cache = None, {}
        for s in samples:
            if s.im_id not in cache:
                left = cv2.imread(s.path, 0)
                right = cv2.imread(s.right_path, 0)
                if matcher is None:
                    matcher = StereoMatcher.from_scene_camera(s.cam, (left.shape[1], left.shape[0]), params)
                xyz = matcher.compute_points(left, right)
                depth = cv2.imread(s.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * float(s.cam.get('depth_scale', 0.001))
                cache = {s.im_id: (xyz, depth, float(left.mean()))}
            xyz, depth, bright = cache[s.im_id]
            K = s.K
            # superficie del tubo visible, en píxeles de 1024 y celdas de 512
            p = pts @ s.R_gt.T + s.t_gt
            u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
            ok = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
            u, v, z = u[ok], v[ok], p[ok, 2]
            zg = depth[v, u]
            vis = (zg > 0) & (z <= zg + 0.005)
            cells = {}
            for cu, cv_, zz in zip(u[vis] // 2, v[vis] // 2, z[vis]):
                c = (int(cu), int(cv_))
                cells[c] = min(cells.get(c, 9.0), float(zz))
            if not cells:
                continue
            q = xyz[xyz[:, 2] > 1e-6]
            su = np.floor((K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]) / 2).astype(int)
            sv = np.floor((K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]) / 2).astype(int)
            hit = {}
            for cu, cv_, zz in zip(su, sv, q[:, 2]):
                c = (int(cu), int(cv_))
                if c in cells:
                    hit.setdefault(c, []).append(float(zz))
            err = np.array([np.median(hit[c]) - cells[c] for c in hit]) * 1e3
            m = metas[s.im_id]
            rows.append({'base': tag, 'im_id': s.im_id, 'inst': s.inst, 'visib': s.visib,
                         'g': (m.get('light_diffuse') or [0.9])[0], 'brillo': bright, 'celdas': len(cells),
                         'cobertura': len(hit) / len(cells),
                         'err_med_mm': float(np.median(err)) if len(err) else float('nan'),
                         'abs_med_mm': float(np.median(np.abs(err))) if len(err) else float('nan'),
                         'frac_10mm': float((np.abs(err) < 10).mean()) if len(err) else float('nan')})
    import csv
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def block(sel):
        f = lambda k: np.nanmedian([r[k] for r in sel])
        return (f"{len(sel)} | {f('brillo'):.1f} | {100 * f('cobertura'):.1f} | {f('err_med_mm'):+.1f} | {f('abs_med_mm'):.1f} "
                f"| {100 * f('frac_10mm'):.1f}")

    out = ['# Calidad de SGBM en los píxeles del tubo (sin ICP)\n',
           f"Estéreo: {a.stereo_params or 'stereo_utils.DEFAULT_PARAMS'}. Medianas por instancia; solo instancias "
           f"con visibilidad ≥ 0.5 para no mezclar oclusión. Fuente: `analysis/calidad_sgbm.csv`.\n",
           '| grupo | n | brillo medio imagen | cobertura del tubo % | error z mediano mm | \\|error z\\| mediano mm | celdas con \\|error\\| < 10 mm % |',
           '|---|---|---|---|---|---|---|']
    good = [r for r in rows if r['visib'] >= 0.5]
    for tag in ('v1', 'v3'):
        out.append(f'| {tag} (todas) | ' + block([r for r in good if r['base'] == tag]) + ' |')
    v3 = [r for r in good if r['base'] == 'v3']
    cuts = np.percentile([r['g'] for r in rows if r['base'] == 'v3'], [100 / 3, 200 / 3])
    for lo, hi, lab in [(-1, cuts[0], f'g < {cuts[0]:.3f}'), (cuts[0], cuts[1], f'{cuts[0]:.3f}–{cuts[1]:.3f}'),
                        (cuts[1], 0.9, f'{cuts[1]:.3f}–0.9'), (0.9, 2, 'g ≥ 0.9 (más luz que v1)')]:
        sel = [r for r in v3 if lo <= r['g'] < hi]
        if sel:
            out.append(f'| v3, {lab} | ' + block(sel) + ' |')
    out.append(correlations(v3, [r for r in good if r['base'] == 'v1']))
    txt = '\n'.join(out) + '\n'
    open(a.md, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
