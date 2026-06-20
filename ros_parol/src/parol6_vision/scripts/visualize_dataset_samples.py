#!/usr/bin/env python3
"""
visualize_dataset_samples.py — Verificación VISUAL del dataset sintético v2.

Dibuja sobre una muestra de imágenes:
  * bbox_visib (verde) y bbox_obj amodal (naranja) de bop/scene_gt_info.json
  * el bbox de la etiqueta YOLO re-leída de labels/ (amarillo fino: debe
    solaparse exactamente con bbox_visib)
  * los keypoints proyectados (relleno = visible, anillo = ocluido;
    cabeza +Z en rojo, cola −Z en azul, centro en blanco)
  * el eje del tubo REPROYECTADO desde la pose 6D del BOP (cian): si la pose
    R|t y la K son correctas, la línea debe caer sobre el tubo en la imagen.

Úsalo SIEMPRE antes de lanzar una generación larga: detecta errores de
proyección/etiquetado con 10-20 muestras.

Uso:
    ros2 run parol6_vision visualize_dataset_samples.py \
        --dataset_dir ~/parol6_synth_dataset --n 20
    # añade --check para un PASS/FAIL numérico de la reproyección

No usa ROS: lee solo los ficheros del dataset.
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

TUBE_HALF = 0.093  # m: semieje del tubo (cilindro de colisión)


def load_dataset(root):
    bop = {}
    for name in ('scene_camera', 'scene_gt', 'scene_gt_info'):
        with open(os.path.join(root, 'bop', f'{name}.json')) as f:
            bop[name] = json.load(f)
    metas = {}
    with open(os.path.join(root, 'meta.jsonl')) as f:
        for line in f:
            m = json.loads(line)
            metas[m['im_id']] = m
    return bop, metas


def yolo_boxes(label_path, w, h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts)
            boxes.append([int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                          int(bw * w), int(bh * h)])
    return boxes


def project(K, R, t, pts):
    """Proyecta puntos (frame del modelo) con la pose BOP R|t (en METROS)."""
    p = np.asarray(pts) @ np.asarray(R).reshape(3, 3).T + np.asarray(t)
    z = np.clip(p[:, 2], 1e-6, None)
    u = K[0, 0] * p[:, 0] / z + K[0, 2]
    v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), p[:, 2]


def draw_sample(root, im_id, bop, meta, out_dir, check_stats):
    img_path = os.path.join(root, meta['file'])
    img = cv2.imread(img_path)
    if img is None:
        print(f'  [WARN] no pude leer {img_path}')
        return
    h, w = img.shape[:2]
    k = str(im_id)
    cam = bop['scene_camera'][k]
    K = np.array(cam['cam_K']).reshape(3, 3)
    gts = bop['scene_gt'].get(k, [])
    infos = bop['scene_gt_info'].get(k, [])

    label_path = os.path.join(
        root, 'labels', meta['split'],
        os.path.splitext(os.path.basename(meta['file']))[0] + '.txt')
    for x, y, bw, bh in yolo_boxes(label_path, w, h):
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 255), 1)

    for gt, info in zip(gts, infos):
        bx = info['bbox_obj']
        cv2.rectangle(img, (bx[0], bx[1]), (bx[0] + bx[2], bx[1] + bx[3]),
                      (0, 128, 255), 1)
        bv = info['bbox_visib']
        if bv[2] > 0:
            cv2.rectangle(img, (bv[0], bv[1]), (bv[0] + bv[2], bv[1] + bv[3]),
                          (0, 220, 0), 2)
        # Eje del tubo reproyectado desde la pose 6D (validación de R|t)
        ax2d, _ = project(K, gt['cam_R_m2c'], gt['cam_t_m2c'],
                          [[0, 0, -TUBE_HALF], [0, 0, 0], [0, 0, TUBE_HALF]])
        p0, pc, p1 = (tuple(np.round(p).astype(int)) for p in ax2d)
        cv2.line(img, p0, p1, (255, 255, 0), 2)
        cv2.circle(img, pc, 3, (255, 255, 0), -1)

        # Keypoints guardados (0 centro, 1 cabeza+Z, 2 cola−Z, 3-5 borde cabeza)
        colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0),
                  (0, 0, 200), (0, 0, 200), (0, 0, 200)]
        for kp, vis, col in zip(info['keypoints_2d'],
                                info['keypoints_visib'], colors):
            if kp[0] < 0:
                continue
            c = (int(round(kp[0])), int(round(kp[1])))
            if vis:
                cv2.circle(img, c, 4, col, -1)
            else:
                cv2.circle(img, c, 4, col, 1)

        # Chequeo numérico: el centro reproyectado debe caer en bbox_obj inflado
        m = 6
        ok = (bx[0] - m <= pc[0] <= bx[0] + bx[2] + m and
              bx[1] - m <= pc[1] <= bx[1] + bx[3] + m)
        check_stats.append((im_id, ok, info['visib_fract']))
        cv2.putText(img, f"vis={info['visib_fract']:.2f} {info['mode']}",
                    (bx[0], max(12, bx[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

    tag = ' HARD' if meta.get('hard') else ''
    cv2.putText(img, f"id={im_id} {meta['split']} cam={meta['cam']} "
                     f"{meta['kind']}{tag}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    out = os.path.join(out_dir, f'vis_{im_id:06d}.png')
    cv2.imwrite(out, img)
    print(f'  {out}  ({meta["kind"]}, instancias={len(gts)})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--n', type=int, default=12,
                    help='nº de muestras aleatorias a dibujar (0 = todas)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ids', type=int, nargs='*', default=None,
                    help='ids concretos en vez de muestreo aleatorio')
    ap.add_argument('--check', action='store_true',
                    help='resumen PASS/FAIL de la reproyección del centro')
    args = ap.parse_args()

    root = os.path.expanduser(args.dataset_dir)
    bop, metas = load_dataset(root)
    out_dir = os.path.join(root, 'qa_vis')
    os.makedirs(out_dir, exist_ok=True)

    ids = args.ids if args.ids else sorted(metas.keys())
    if not args.ids and args.n > 0 and len(ids) > args.n:
        rng = random.Random(args.seed)
        # mezclar positivos con/sin oclusión y algún negativo
        ids = sorted(rng.sample(ids, args.n))

    check_stats = []
    for im_id in ids:
        draw_sample(root, im_id, bop, metas[im_id], out_dir, check_stats)

    if check_stats:
        n_ok = sum(1 for _, ok, _ in check_stats if ok)
        print(f'\nReproyección del centro dentro de bbox_obj: '
              f'{n_ok}/{len(check_stats)}')
        if args.check:
            bad = [i for i, ok, _ in check_stats if not ok]
            if bad:
                print(f'FAIL en ids: {sorted(set(bad))}')
                sys.exit(1)
            print('PASS')


if __name__ == '__main__':
    main()
