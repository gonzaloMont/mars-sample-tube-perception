#!/usr/bin/env python3
"""
yolo_metrics.py — Métricas de DETECCIÓN de las cachés de cache_yolo_detections.py
sobre un dataset de evaluación de pose (sin ROS).

Por instancia (bbox_visib > 0): mejor IoU con alguna detección → recall a
IoU ≥ 0.3 (la asociación que usa eval_pose_dataset.py --bbox yolo) y ≥ 0.5.
Por detección: falso positivo si su IoU con todas las bbox_visib GT de la
imagen es < 0.3. "top-1 correcto": la detección de mayor confianza (la que
eligen los nodos ICP y D2CO) cae sobre algún tubo (IoU ≥ 0.3).

Uso: yolo_metrics.py --dataset DIR --cache JSON [--cache JSON ...] [--labels ...] [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import wilson, pct, VIS_EDGES  # noqa: E402


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate(dataset, cache_path):
    root = os.path.expanduser(dataset)
    metas = {}
    with open(os.path.join(root, 'meta.jsonl')) as fh:
        for line in fh:
            m = json.loads(line)
            metas[int(m['im_id'])] = m
    info = json.load(open(os.path.join(root, 'bop', 'scene_gt_info.json')))
    dets = json.load(open(cache_path))['detections']
    inst, images = [], []
    for k, infos in info.items():
        im = int(k)
        m = metas[im]
        gts = [i['bbox_visib'] for i in infos if i.get('bbox_visib') and i['bbox_visib'][2] > 0 and i['bbox_visib'][3] > 0]
        vis = [i['visib_fract'] for i in infos if i.get('bbox_visib') and i['bbox_visib'][2] > 0 and i['bbox_visib'][3] > 0]
        boxes = [(d['xyxy'][0], d['xyxy'][1], d['xyxy'][2] - d['xyxy'][0], d['xyxy'][3] - d['xyxy'][1], d['conf'])
                 for d in dets.get(k, [])]
        for g, v in zip(gts, vis):
            best = max((iou(b[:4], g) for b in boxes), default=0.0)
            inst.append({'split': m['split'], 'visib': v, 'hard': m.get('hard', False), 'iou': best})
        n_fp = sum(1 for b in boxes if max((iou(b[:4], g) for g in gts), default=0.0) < 0.3)
        top1 = max(boxes, key=lambda b: b[4]) if boxes else None
        top1_ok = top1 is not None and max((iou(top1[:4], g) for g in gts), default=0.0) >= 0.3
        images.append({'split': m['split'], 'hard': m.get('hard', False), 'n_det': len(boxes), 'n_fp': n_fp,
                       'top1_ok': top1_ok, 'has_det': bool(boxes)})
    return inst, images


def table(dataset, caches, labels, splits=('train', 'val', 'test', None)):
    lines = ['| detector | partición | instancias | recall IoU≥0.3 % (IC) | recall IoU≥0.5 % | imágenes '
             '| detecciones | FP (IoU<0.3) | FP/imagen | top-1 sobre un tubo % |',
             '|---|---|---|---|---|---|---|---|---|---|']
    vis_lines = ['| detector | ' + ' | '.join(f'visib {lo:.2f}–{min(hi, 1):.2f}' for lo, hi in zip(VIS_EDGES[:-1], VIS_EDGES[1:])) + ' |',
                 '|---|' + '---|' * (len(VIS_EDGES) - 1)]
    for cache, lab in zip(caches, labels):
        inst, images = evaluate(dataset, cache)
        for sp in splits:
            I = [r for r in inst if sp is None or r['split'] == sp]
            M = [r for r in images if sp is None or r['split'] == sp]
            k3 = sum(r['iou'] >= 0.3 for r in I)
            k5 = sum(r['iou'] >= 0.5 for r in I)
            lo, hi = wilson(k3, len(I))
            nd = sum(r['n_det'] for r in M)
            nfp = sum(r['n_fp'] for r in M)
            t1 = sum(r['top1_ok'] for r in M)
            lines.append(f"| {lab} | {sp or 'todas'} | {len(I)} | {pct(k3 / len(I))} ({pct(lo)}–{pct(hi)}) "
                         f"| {pct(k5 / len(I))} | {len(M)} | {nd} | {nfp} | {nfp / len(M):.2f} | {pct(t1 / len(M))} |")
        cells = []
        for lo_, hi_ in zip(VIS_EDGES[:-1], VIS_EDGES[1:]):
            sel = [r for r in inst if lo_ <= r['visib'] < hi_]
            k = sum(r['iou'] >= 0.3 for r in sel)
            cells.append(f'{pct(k / len(sel)) if sel else "—"} (n={len(sel)})')
        vis_lines.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n\nRecall IoU≥0.3 por visibilidad (todas las particiones):\n\n' + '\n'.join(vis_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--cache', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*')
    ap.add_argument('--md', default='')
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(c) for c in a.cache]
    txt = table(a.dataset, a.cache, labels)
    print(txt)
    if a.md:
        with open(a.md, 'w') as fh:
            fh.write(f'Dataset: {a.dataset}\n\n' + txt + '\n')


if __name__ == '__main__':
    main()
