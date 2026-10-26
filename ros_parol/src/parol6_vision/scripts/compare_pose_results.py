#!/usr/bin/env python3
"""
compare_pose_results.py — Tabla comparativa de varias corridas de
eval_pose_dataset.py (CSV por instancia) sobre las MISMAS instancias.

Para cada CSV: cobertura, recall ADD-S sobre el total, medianas de ADD-S,
error radial, lateral y de eje (solo poses publicadas) y latencia. Después
restringe a las instancias COMUNES a todas las corridas (comparación
emparejada) y cuenta en cuántas acierta cada una.

Uso:
    ros2 run parol6_vision compare_pose_results.py results/*.csv
    ros2 run parol6_vision compare_pose_results.py a.csv b.csv --labels "ICP depthGT" "ICP SGBM"
    ros2 run parol6_vision compare_pose_results.py results/*.csv --md tabla.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['pose_ok'] = int(_f(r.get('pose_ok', 0)) or 0)
        r['add_s_pass'] = int(_f(r.get('add_s_pass', 0)) or 0)
    return rows


def key(r):
    return (int(r['im_id']), int(r['inst']))


def stats(rows):
    n = len(rows)
    ok = [r for r in rows if r['pose_ok']]
    med = lambda k, f=lambda x: x: (float(np.median([f(_f(r[k])) for r in ok])) if ok else float('nan'))
    return {
        'n': n,
        'cobertura': 100 * len(ok) / n if n else float('nan'),
        'recall_total': 100 * sum(r['add_s_pass'] for r in rows) / n if n else float('nan'),
        'recall_acept': 100 * sum(r['add_s_pass'] for r in ok) / len(ok) if ok else float('nan'),
        'adds_med': med('add_s_mm'),
        'rad_med': med('err_radial_mm', abs),
        'rad_bias': float(np.mean([_f(r['err_radial_mm']) for r in ok])) if ok else float('nan'),
        'lat_med': med('err_lateral_mm'),
        'axis_med': med('err_axis_deg'),
        'lat_ms': med('latency_ms'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--md', default='', help='escribir la tabla en Markdown a este fichero')
    args = ap.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.csv]
    if len(labels) != len(args.csv):
        print('--labels debe tener tantos elementos como CSV', file=sys.stderr)
        return 1
    runs = [load(p) for p in args.csv]

    hdr = ['corrida', 'n', 'cobert.%', 'recall ADD-S % (total)', 'recall % (acept.)', 'ADD-S med mm',
           '|radial| med mm', 'sesgo radial mm', 'lateral med mm', 'eje med °', 'latencia med ms']
    lines = []

    def emit(title, table):
        lines.append(f'\n{title}')
        lines.append('| ' + ' | '.join(hdr) + ' |')
        lines.append('|' + '---|' * len(hdr))
        for lab, st in table:
            lines.append('| ' + ' | '.join([lab, str(st['n']), f"{st['cobertura']:.1f}", f"{st['recall_total']:.1f}",
                                            f"{st['recall_acept']:.1f}", f"{st['adds_med']:.1f}", f"{st['rad_med']:.1f}",
                                            f"{st['rad_bias']:+.1f}", f"{st['lat_med']:.1f}", f"{st['axis_med']:.1f}",
                                            f"{st['lat_ms']:.0f}"]) + ' |')

    emit('Todas las instancias de cada corrida', [(lab, stats(rows)) for lab, rows in zip(labels, runs)])

    common = set.intersection(*[set(map(key, rows)) for rows in runs])
    if len(runs) > 1 and common:
        sub = [[r for r in rows if key(r) in common] for rows in runs]
        emit(f'Instancias comunes a todas las corridas (n={len(common)})',
             [(lab, stats(rows)) for lab, rows in zip(labels, sub)])
        # Emparejado: en cuántas instancias acierta cada corrida y cuántas todas/ninguna
        passes = []
        for rows in sub:
            d = {key(r): r['add_s_pass'] for r in rows}
            passes.append(np.array([d[k] for k in sorted(common)], dtype=bool))
        P = np.stack(passes)
        lines.append('')
        lines.append(f'ADD-S superado en las {len(common)} instancias comunes: '
                     + ', '.join(f'{lab} {int(p.sum())}' for lab, p in zip(labels, P))
                     + f' | todas {int(P.all(0).sum())} | ninguna {int((~P).all(0).sum())}')
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                only_i = int((P[i] & ~P[j]).sum())
                only_j = int((P[j] & ~P[i]).sum())
                lines.append(f'  solo {labels[i]}: {only_i}   solo {labels[j]}: {only_j}')
    text = '\n'.join(lines)
    print(text)
    if args.md:
        with open(args.md, 'w') as f:
            f.write(text + '\n')
        print(f'\n(tabla escrita en {args.md})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
