#!/usr/bin/env python3
"""
fase3_test.py — Cifra final en test (results/pose_eval_v3/test/*.csv).
Resumen con IC, estratos (visibilidad, modo, brazo) y comparaciones
emparejadas dentro de test. Escribe analysis/test.md.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import load, summary, summary_row, SUMMARY_HDR, paired, pct, strata_table, key  # noqa: E402

D = os.path.join(HERE, '..', 'test')
RUNS = [('ICP, profundidad real, caja GT', 'icp_depthgt_gt_B_defaults_nodo'),
        ('ICP, SGBM, caja GT', 'icp_sgbmwls500_gt_B_defaults_nodo'),
        ('ICP, profundidad real, caja YOLO', 'icp_depthgt_yolo_B_defaults_nodo'),
        ('ICP, SGBM, caja YOLO (cadena completa)', 'icp_sgbmwls500_yolo_B_defaults_nodo'),
        ('D2CO con umbral, caja GT', 'd2co_gt_umbral'),
        ('D2CO sin umbral, caja GT', 'd2co_gt_sinumbral'),
        ('D2CO con umbral, caja YOLO', 'd2co_yolo_umbral')]


def avg_dist(msg):
    m = re.search(r'avg_dist=([0-9.]+)', msg)
    if not m:
        return float('nan')
    return float('inf') if ('179769313486' in msg or '340282346638' in msg) else float(m.group(1))


def main():
    R = {lab: load(os.path.join(D, f + '.csv')) for lab, f in RUNS if os.path.isfile(os.path.join(D, f + '.csv'))}
    out = ['# Fase 3 — cifra final en test (222 instancias, una sola corrida por configuración)\n',
           'ICP: config B (defaults del nodo) + estéreo `wls_lambda` 500 (elegidas en val). D2CO: nodo de la rama '
           '`d2co` (45743e7). Detector: best.pt actual. Umbral ADD-S 18.69 mm.\n', SUMMARY_HDR]
    for lab, rows in R.items():
        out.append(summary_row(lab, summary(rows)))
    labs = list(R)
    out.append('\n## Estratos (recall ADD-S sobre el total, IC 95 % Wilson)\n')
    icp = [(l.replace('ICP, ', ''), R[l]) for l in labs if l.startswith('ICP')]
    d2 = [(l.replace('D2CO ', ''), R[l]) for l in labs if l.startswith('D2CO')]
    if icp:
        out.append('ICP:\n\n' + strata_table(icp))
    if d2:
        out.append('\nD2CO:\n\n' + strata_table(d2))
    out.append('\n## Comparaciones emparejadas en test\n')
    out.append('| comparación | n | recall 1.ª % | recall 2.ª % | diferencia (pp) | IC 95 % | solo 1.ª | solo 2.ª | p McNemar |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    pairs = [(labs[i], labs[j]) for i, j in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (1, 5), (3, 6), (4, 5)]
             if i < len(labs) and j < len(labs)]
    for a, b in pairs:
        c = paired(R[a], R[b])
        out.append(f"| {a} → {b} | {c['n']} | {pct(c['rec_a'])} | {pct(c['rec_b'])} | {100 * c['diff']:+.1f} "
                   f"| {100 * c['lo']:+.1f} … {100 * c['hi']:+.1f} | {c['only_a']} | {c['only_b']} | {c['p_mcnemar']:.2g} |")
    u, s = R.get('D2CO con umbral, caja GT'), R.get('D2CO sin umbral, caja GT')
    if u and s:
        du = {key(r): r for r in u}
        acc_post = sum(1 for r in s if r['pose_ok'] and avg_dist(r['msg']) <= 4.5)
        both = sum(1 for r in s if r['pose_ok'] and avg_dist(r['msg']) <= 4.5 and du[key(r)]['pose_ok'])
        dz = [abs(r['z_est_m'] - du[key(r)]['z_est_m']) * 1e3 for r in s
              if r['pose_ok'] and du[key(r)]['pose_ok']]
        ad = np.array([avg_dist(r['msg']) for r in s if r['pose_ok']])
        n_diff = sum(1 for r in s if (r['pose_ok'] and avg_dist(r['msg']) <= 4.5) != bool(du[key(r)]['pose_ok']))
        out.append(f"\n## Coherencia D2CO con/sin umbral\n\nLa corrida sin umbral publica {sum(r['pose_ok'] for r in s)} poses; "
                   f"{acc_post} tienen avg_dist ≤ 4.5 (lo que el nodo con umbral aceptaría). La corrida con umbral aceptó "
                   f"{sum(r['pose_ok'] for r in u)}; coinciden en {both}. En las {len(dz)} instancias con pose en ambas, |Δz| mediana "
                   f"{np.median(dz):.1f} mm, máximo {np.max(dz):.0f} mm: cuando las dos corridas publican, la pose es idéntica. "
                   f"Pero {n_diff} instancias cambian de resultado entre corridas (no por el umbral: el avg_dist final es otro), "
                   f"compatible con el atajo de tracking del nodo (`tracking_fast_dist` / `full_search_every`), que depende "
                   f"del contador de ciclos y este avanza también con el timer. avg_dist de las poses sin umbral: mediana "
                   f"{np.median(ad):.2f}, p90 {np.percentile(ad, 90):.2f}.")
    txt = '\n'.join(out) + '\n'
    with open(os.path.join(HERE, 'test.md'), 'w') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
