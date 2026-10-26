#!/usr/bin/env python3
"""
cache_yolo_detections.py — Ejecuta el detector YOLO (best.pt) UNA vez sobre
las imágenes izquierdas de un dataset y guarda las detecciones en JSON.

Para qué: la evaluación de pose (eval_pose_dataset.py --bbox yolo) necesita
las cajas REALES del detector para medir el pipeline completo, pero correr
YOLO en vivo bloquea la máquina. Con el caché, cada instancia del dataset se
evalúa con la misma detección en todas las corridas y estimadores.

Replica la llamada del nodo yolo26n_camera_subscriber.py (model.predict con
conf_threshold, sin imgsz explícito) para que las cajas sean las mismas que
vería el pipeline.

Formato de salida (JSON):
    {"model": "...", "conf": 0.3,
     "detections": {"<im_id>": [{"xyxy": [x1,y1,x2,y2], "conf": c, "cls": "name"}, ...]}}

Uso:
    ros2 run parol6_vision cache_yolo_detections.py --dataset ~/parol6_pose_eval_dataset
    ros2 run parol6_vision cache_yolo_detections.py --dataset DIR --split test --conf 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=os.path.expanduser('~/parol6_pose_eval_dataset'))
    ap.add_argument('--split', nargs='*', default=[], help='splits a procesar (vacío = todos)')
    ap.add_argument('--model', default='best.pt',
                    help='ruta o nombre en parol6_vision/models (default best.pt)')
    ap.add_argument('--conf', type=float, default=0.3, help='conf_threshold del nodo (default 0.3)')
    ap.add_argument('--out', default='', help='JSON de salida (default <dataset>/yolo_detections.json)')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    root = os.path.expanduser(args.dataset)
    metas = [json.loads(l) for l in open(os.path.join(root, 'meta.jsonl'))]
    if args.split:
        metas = [m for m in metas if m['split'] in args.split]
    if args.limit:
        metas = metas[:args.limit]
    if not metas:
        print('sin imágenes', file=sys.stderr)
        return 1

    # Ruta del modelo igual que el nodo (chdir a models/ para que ultralytics
    # resuelva pesos relativos y no descargue nada al directorio actual)
    from ament_index_python.packages import get_package_share_directory
    from ultralytics import YOLO
    models_dir = os.path.join(get_package_share_directory('parol6_vision'), 'models')
    model_path = args.model if os.path.isfile(args.model) else os.path.join(models_dir, args.model)
    if not os.path.isfile(model_path):
        print(f'modelo no encontrado: {model_path}', file=sys.stderr)
        return 1
    prev = os.getcwd()
    os.chdir(models_dir)
    try:
        model = YOLO(model_path)
    finally:
        os.chdir(prev)

    import cv2
    out_path = args.out or os.path.join(root, 'yolo_detections.json')
    dets = {}
    t0 = time.time()
    n_with = 0
    for i, m in enumerate(metas):
        frame = cv2.imread(os.path.join(root, m['file']))  # BGR 3 canales, como el nodo
        res = model.predict(source=frame, conf=args.conf, verbose=False)[0]
        rows = []
        if res.boxes is not None and len(res.boxes):
            names = res.names
            for box in res.boxes:
                cls_idx = int(box.cls.item())
                name = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                rows.append({'xyxy': [round(float(v), 2) for v in box.xyxy[0].tolist()],
                             'conf': round(float(box.conf.item()), 4), 'cls': name})
        dets[str(m['im_id'])] = rows
        n_with += bool(rows)
        if (i + 1) % 50 == 0:
            print(f'  [{i + 1}/{len(metas)}] {(time.time() - t0) / (i + 1):.2f} s/img')
    with open(out_path, 'w') as f:
        json.dump({'model': model_path, 'conf': args.conf, 'n_images': len(metas),
                   'detections': dets}, f)
    print(f'{len(metas)} imágenes, {n_with} con alguna detección '
          f'({100 * n_with / len(metas):.1f}%), {(time.time() - t0):.0f} s → {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
