#!/usr/bin/env python3
"""
eval_detector.py — Evaluación de YOLO26n (models/best.pt) en su propio conjunto
de datos (~/parol6_synth_dataset_v3: 700/200/100 imágenes).

1. Métricas estándar de Ultralytics (model.val, conf 0.001) en val y en test:
   precisión y recall (en el punto de máximo F1), mAP50 y mAP50-95.
2. Punto de operación del nodo (conf 0.3, la de yolo26n_camera_subscriber.py) en
   test: emparejamiento voraz por IoU ≥ 0.5 → TP, FP, FN, precisión, recall y F1,
   por tanda (image_bins = cualquier parte de la imagen, footprint = zona de
   trabajo) y FP en las imágenes sin tubo.
Escribe deteccion_conjunto_detector.md y deteccion_conjunto_detector.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.normpath(os.path.join(HERE, '..', '..', 'models', 'best.pt'))
D = os.path.expanduser('~/parol6_synth_dataset_v3')
CONF_NODE, IOU_MATCH = 0.3, 0.5


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    from ultralytics import YOLO
    import cv2
    model = YOLO(MODEL)
    out = {'model': MODEL, 'dataset': D}
    for split in ('val', 'test'):
        m = model.val(data=os.path.join(D, 'data.yaml'), split=split, imgsz=640, batch=16, conf=0.001,
                      plots=False, verbose=False, project=os.path.join(HERE, 'runs'), name=split, exist_ok=True)
        out[split] = {'precision': float(m.box.mp), 'recall': float(m.box.mr),
                      'mAP50': float(m.box.map50), 'mAP50_95': float(m.box.map)}
    metas = [json.loads(l) for l in open(os.path.join(D, 'meta.jsonl'))]
    metas = [x for x in metas if x['split'] == 'test']
    cnt = {}
    for x in metas:
        img = cv2.imread(os.path.join(D, x['file']))
        h, w = img.shape[:2]
        lab = os.path.join(D, x['file'].replace('images/', 'labels/').rsplit('.', 1)[0] + '.txt')
        gts = []
        if os.path.isfile(lab):
            for line in open(lab):
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:])
                    gts.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        r = model.predict(source=img, conf=CONF_NODE, verbose=False)[0]
        dets = sorted([(float(b.conf.item()), b.xyxy[0].tolist()) for b in r.boxes], key=lambda d: -d[0])
        used, tp = set(), 0
        for c, d in dets:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j not in used and iou(d, g) > best:
                    best, bj = iou(d, g), j
            if best >= IOU_MATCH:
                used.add(bj)
                tp += 1
        key = x.get('placement', '') + (' (sin tubo)' if x['kind'] == 'neg' else '')
        c = cnt.setdefault(key, {'imagenes': 0, 'cajas': 0, 'tp': 0, 'fp': 0, 'fn': 0})
        c['imagenes'] += 1
        c['cajas'] += len(gts)
        c['tp'] += tp
        c['fp'] += len(dets) - tp
        c['fn'] += len(gts) - tp
    tot = {k: sum(c[k] for c in cnt.values()) for k in ('imagenes', 'cajas', 'tp', 'fp', 'fn')}
    cnt['total'] = tot
    out['test_conf0.3_iou0.5'] = cnt

    def prf(c):
        p = c['tp'] / (c['tp'] + c['fp']) if c['tp'] + c['fp'] else float('nan')
        r = c['tp'] / c['cajas'] if c['cajas'] else float('nan')
        f = 2 * p * r / (p + r) if p == p and r == r and p + r else float('nan')
        return p, r, f
    L = ['# YOLO26n en su conjunto de datos (`~/parol6_synth_dataset_v3`)\n',
         f'Modelo `models/best.pt`. Fuente: `eval_detector.py` → `deteccion_conjunto_detector.json`.\n',
         '## Métricas de Ultralytics (`model.val`, imgsz 640, conf 0.001)\n',
         '| partición | imágenes | precisión | recall | mAP50 | mAP50-95 |', '|---|---|---|---|---|---|']
    n_img = {'val': 200, 'test': 100}
    for s in ('val', 'test'):
        v = out[s]
        L.append(f"| {s} | {n_img[s]} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |")
    L += ['\nPrecisión y recall en el punto de máximo F1 (criterio de Ultralytics).\n',
          f'## Punto de operación del nodo en test (conf {CONF_NODE}, IoU ≥ {IOU_MATCH})\n',
          '| subconjunto | imágenes | tubos | TP | FP | FN | precisión | recall | F1 |', '|---|---|---|---|---|---|---|---|---|']
    names = {'image_bins': 'tubo en cualquier parte', 'footprint': 'zona de trabajo',
             'image_bins (sin tubo)': 'sin tubo (tanda general)', 'footprint (sin tubo)': 'sin tubo (zona de trabajo)',
             'total': '**total**'}
    for k in ('image_bins', 'footprint', 'image_bins (sin tubo)', 'footprint (sin tubo)', 'total'):
        if k not in cnt:
            continue
        c = cnt[k]
        p, r, f = prf(c)
        fmt = lambda v: '—' if v != v else f'{v:.3f}'
        L.append(f"| {names[k]} | {c['imagenes']} | {c['cajas']} | {c['tp']} | {c['fp']} | {c['fn']} | {fmt(p)} | {fmt(r)} | {fmt(f)} |")
    txt = '\n'.join(L) + '\n'
    open(os.path.join(HERE, 'deteccion_conjunto_detector.md'), 'w').write(txt)
    json.dump(out, open(os.path.join(HERE, 'deteccion_conjunto_detector.json'), 'w'), indent=1)
    print(txt)


if __name__ == '__main__':
    main()
