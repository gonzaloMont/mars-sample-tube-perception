#!/usr/bin/env python3
"""
fase1_train.py — Análisis del ajuste en train (results/pose_eval_v3/train/*.csv).

Las corridas de train se hicieron SIN rechazo; aquí se reconstruyen las
configuraciones completas aplicando el rechazo del nodo a posteriori:
  A      yaml anterior            remove_plane F                 min_fitness 0.20  max_rmse 5 mm
  A+rp   yaml + remove_plane    remove_plane T, plane 6 mm     min_fitness 0.20  max_rmse 5 mm
  B0.20  defaults, fitness 0.2  remove_plane T, plane 4 mm     min_fitness 0.20  max_rmse 6 mm
  B      defaults del nodo      remove_plane T, plane 4 mm     min_fitness 0.45  max_rmse 6 mm
Escribe analysis/train.md.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import (load, apply_reject, summary, summary_row, SUMMARY_HDR, paired, pct, num,  # noqa: E402
                   strata_table)

T = os.path.join(HERE, '..', 'train')

CONFIGS = [  # etiqueta, sufijo de la corrida, min_fitness, max_rmse_mm
    ('A (yaml anterior)', 'rpF', 0.20, 5.0),
    ('A+rp (yaml + remove_plane)', 'rpT006', 0.20, 5.0),
    ('B con min_fitness 0.20', 'rpT004', 0.20, 6.0),
    ('B (defaults del nodo)', 'rpT004', 0.45, 6.0),
]


def run(name):
    p = os.path.join(T, name + '.csv')
    return load(p) if os.path.isfile(p) else None


def main():
    out = ['# Fase 1 — ajuste en train (365 instancias, bbox GT visible salvo indicación)\n',
           'Corridas sin rechazo; configuraciones reconstruidas aplicando el rechazo del nodo '
           'a posteriori (`analysis/fase1_train.py`). Umbral ADD-S 18.69 mm.\n']
    for cloud, bbox, title in [('depthgt', 'gt', 'Profundidad real, bbox GT'), ('sgbm', 'gt', 'SGBM, bbox GT'),
                               ('depthgt', 'yolo', 'Profundidad real, bbox YOLO'), ('sgbm', 'yolo', 'SGBM, bbox YOLO')]:
        out.append(f'\n## {title}\n\n' + SUMMARY_HDR)
        base = {}
        for label, suf, mf, rm in CONFIGS:
            rows = run(f'icp_{cloud}_{bbox}_{suf}')
            if rows is None:
                continue
            rr = apply_reject(rows, mf, rm)
            base[label] = rr
            out.append(summary_row(label, summary(rr)))
        norej = run(f'icp_{cloud}_{bbox}_rpT004')
        if norej is not None:
            out.append(summary_row('remove_plane T, plane 4 mm, SIN rechazo', summary(norej)))
        if cloud == 'sgbm' and bbox == 'gt':
            w = run('icp_sgbmwls500_gt_rpT004')
            if w is not None:
                for mf in (0.20, 0.45):
                    out.append(summary_row(f'B min_fitness {mf:.2f} + SGBM wls_lambda 500', summary(apply_reject(w, mf, 6.0))))
        labels = list(base)
        if len(labels) >= 2:
            out.append('\nComparaciones emparejadas (mismas instancias; IC 95 % bootstrap por imagen, McNemar exacto):\n')
            out.append('| A → B | n | recall A % | recall B % | B − A (pp) | IC 95 % | solo A | solo B | p McNemar |')
            out.append('|---|---|---|---|---|---|---|---|---|')
            for i, j in [(0, 1), (1, 2), (2, 3), (0, 3)]:
                if i < len(labels) and j < len(labels):
                    c = paired(base[labels[i]], base[labels[j]])
                    out.append(f"| {labels[i]} → {labels[j]} | {c['n']} | {pct(c['rec_a'])} | {pct(c['rec_b'])} "
                               f"| {100 * c['diff']:+.1f} | {100 * c['lo']:+.1f} … {100 * c['hi']:+.1f} "
                               f"| {c['only_a']} | {c['only_b']} | {c['p_mcnemar']:.2g} |")
        if norej is not None:
            out.append('\nCurva de rechazo (remove_plane T, plane 4 mm, max_rmse 6 mm):\n')
            out.append('| min_fitness | cobertura % | recall total % | recall aceptadas % |')
            out.append('|---|---|---|---|')
            for mf in (0.0, 0.2, 0.3, 0.45, 0.6, 0.75):
                s = summary(apply_reject(norej, mf, 6.0), boot=False)
                out.append(f"| {mf:.2f} | {pct(s['cov'])} | {pct(s['rec'])} | {pct(s['rec_acc'])} |")
        if len(labels) == 4:
            out.append('\nEstratos (A frente a B):\n')
            out.append(strata_table([('A', base[labels[0]]), ('B', base[labels[3]])]))

    a, b = run('icp_depthgt_gt_rpT004'), run('icp_depthgt_gt_rpT004_rep')
    if a is not None and b is not None:
        da = {(r['im_id'], r['inst']): r for r in a}
        nd = sum(1 for r in b if abs((r['add_s_mm'] if r['add_s_mm'] == r['add_s_mm'] else -1)
                                     - (da[(r['im_id'], r['inst'])]['add_s_mm']
                                        if da[(r['im_id'], r['inst'])]['add_s_mm'] == da[(r['im_id'], r['inst'])]['add_s_mm'] else -1)) > 0.01)
        c = paired(a, b)
        out.append(f"\n## Repetibilidad\n\nMisma corrida dos veces (profundidad real, bbox GT, remove_plane T, plane 4 mm, "
                   f"sin rechazo): {nd} de {len(b)} instancias con ADD-S distinto (> 0.01 mm); recall "
                   f"{pct(c['rec_a'])} % frente a {pct(c['rec_b'])} % (solo en la 1.ª: {c['only_a']}, solo en la 2.ª: "
                   f"{c['only_b']}).\n")
    a2, b2 = run('icp_depthgt_gt_rpF'), run('repcheck_icp_depthgt_gt_rpF_60')
    if a2 is not None and b2 is not None:
        import numpy as np
        da = {(r['im_id'], r['inst']): r for r in a2}
        d = [abs(r['add_s_mm'] - da[(r['im_id'], r['inst'])]['add_s_mm']) for r in b2 if r['add_s_mm'] == r['add_s_mm']]
        flips = sum(r['add_s_pass'] != da[(r['im_id'], r['inst'])]['add_s_pass'] for r in b2)
        out.append(f"Sin `remove_plane` (sin RANSAC; primeras {len(b2)} instancias de train, repetidas): {sum(x > 0.01 for x in d)} "
                   f"con ADD-S distinto; |ΔADD-S| mediana {np.median(d):.2f} mm, p90 {np.percentile(d, 90):.2f} mm, máximo "
                   f"{max(d):.1f} mm; {flips} cambio(s) de acierto. El harness no es determinista bit a bit ni sin RANSAC "
                   f"(causa no aislada; previsiblemente el paralelismo de Open3D); el efecto sobre el recall es de décimas "
                   f"de punto, muy por debajo del IC muestral.\n")
    txt = '\n'.join(out) + '\n'
    with open(os.path.join(HERE, 'train.md'), 'w') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
