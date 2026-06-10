#!/usr/bin/env python3
"""
dataset_stats.py — Histogramas de cobertura del dataset sintético v2.

Comprueba que la estratificación funcionó: distribución de tamaño de bbox,
posición del centro en la imagen, distancia cámara↔tubo y visib_fract, más
recuentos por split / cámara / modo de colocación.

Salida: PNGs en <dataset>/qa_stats/ (si hay matplotlib) y resumen por texto.

Uso:
    ros2 run parol6_vision dataset_stats.py --dataset_dir ~/parol6_synth_dataset
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def text_hist(values, bins, title, unit=''):
    if not len(values):
        return
    hist, edges = np.histogram(values, bins=bins)
    peak = max(1, hist.max())
    print(f'\n{title}')
    for c, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        bar = '█' * int(round(30 * c / peak))
        print(f'  [{e0:8.3f},{e1:8.3f}){unit} {c:5d} {bar}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--bins', type=int, default=12)
    args = ap.parse_args()
    root = os.path.expanduser(args.dataset_dir)

    with open(os.path.join(root, 'bop', 'scene_gt_info.json')) as f:
        gt_info = json.load(f)
    with open(os.path.join(root, 'bop', 'scene_camera.json')) as f:
        cams = json.load(f)
    metas = {}
    with open(os.path.join(root, 'meta.jsonl')) as f:
        for line in f:
            m = json.loads(line)
            metas[m['im_id']] = m

    # Resolución (de la K guardada se obtiene cx,cy; mejor leer una imagen)
    any_meta = next(iter(metas.values()))
    import cv2
    img = cv2.imread(os.path.join(root, any_meta['file']))
    h, w = img.shape[:2]

    sizes, cxs, cys, visibs, dists = [], [], [], [], []
    for k, infos in gt_info.items():
        for info in infos:
            bv = info['bbox_visib']
            if bv[2] <= 0 or bv[3] <= 0:
                continue
            sizes.append(float(np.sqrt(bv[2] * bv[3])))
            cxs.append((bv[0] + bv[2] / 2) / w)
            cys.append((bv[1] + bv[3] / 2) / h)
            visibs.append(info['visib_fract'])
        m = metas.get(int(k))
        if m and m.get('dist') is not None:
            dists.append(m['dist'])

    n_pos = sum(1 for m in metas.values() if m['kind'] == 'pos')
    n_neg = sum(1 for m in metas.values() if m['kind'] == 'neg')
    splits = Counter(m['split'] for m in metas.values())
    cams_c = Counter(m['cam'] for m in metas.values())
    modes = Counter(md for m in metas.values() for md in m.get('modes', []))
    hards = sum(1 for m in metas.values() if m.get('hard'))
    occl = Counter(m.get('occl_plan', 'none') for m in metas.values()
                   if m['kind'] == 'pos')
    multi = sum(1 for m in metas.values() if m.get('n_tubes', 0) > 1)

    print(f'Dataset: {root}')
    print(f'Imágenes: {len(metas)}  (positivas={n_pos}, negativas={n_neg}, '
          f'hard={hards}, multi-tubo={multi})')
    print(f'Splits:   {dict(splits)}')
    print(f'Cámara:   {dict(cams_c)}')
    print(f'Modos:    {dict(modes)}')
    print(f'Oclusión planificada (positivas): {dict(occl)}')
    if visibs:
        v = np.array(visibs)
        print(f'visib_fract: media={v.mean():.3f}  '
              f'<0.5: {(v < 0.5).mean() * 100:.1f}%  '
              f'<{0.1}: {(v < 0.1).mean() * 100:.1f}%')

    text_hist(sizes, args.bins, 'Tamaño de bbox visible (sqrt(área), px)')
    text_hist(visibs, args.bins, 'visib_fract')
    text_hist(dists, args.bins, 'Distancia cámara↔tubo (m)', ' m')
    text_hist(cxs, args.bins, 'Centro del bbox: u normalizado')
    text_hist(cys, args.bins, 'Centro del bbox: v normalizado')

    if not HAS_MPL:
        print('\n(matplotlib no disponible: solo histogramas de texto)')
        return
    out = os.path.join(root, 'qa_stats')
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    plots = [
        (sizes, 'Tamaño bbox visible (px)', None),
        (visibs, 'visib_fract', (0, 1)),
        (dists, 'Distancia cámara↔tubo (m)', None),
        (cxs, 'Centro bbox u (norm.)', (0, 1)),
        (cys, 'Centro bbox v (norm.)', (0, 1)),
    ]
    for ax, (vals, title, rng) in zip(axes, plots):
        ax.hist(vals, bins=args.bins, range=rng, color='#4477aa')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[5].hist2d(cxs, cys, bins=10, range=[[0, 1], [0, 1]], cmap='viridis')
    axes[5].invert_yaxis()
    axes[5].set_title('Mapa de posición de centros (u,v)')
    fig.tight_layout()
    path = os.path.join(out, 'dataset_stats.png')
    fig.savefig(path, dpi=110)
    print(f'\nGráficas guardadas en {path}')


if __name__ == '__main__':
    main()
