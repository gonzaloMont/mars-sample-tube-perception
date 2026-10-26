#!/usr/bin/env python3
"""
fase2_val.py — Confirmación en val (results/pose_eval_v3/val/*.csv) y decisión.

Criterio fijado ANTES de ver val (escrito mientras corría la fase 2):
  * B sustituye a A si la diferencia emparejada B − A del recall en val es ≥ 0
    en los cuatro escenarios y su IC 95 % (bootstrap por imagen) excluye el 0
    en profundidad real + caja GT (el escenario que aísla la segmentación).
  * wls_lambda 500 se adopta solo si mejora a B en los dos escenarios SGBM y
    el IC excluye el 0 en SGBM + caja GT.
Escribe analysis/val.md.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import load, summary, summary_row, SUMMARY_HDR, paired, pct, strata_table  # noqa: E402

V = os.path.join(HERE, '..', 'val')
SCEN = [('depthgt_gt', 'Profundidad real, caja GT'), ('sgbm_gt', 'SGBM, caja GT'),
        ('depthgt_yolo', 'Profundidad real, caja YOLO'), ('sgbm_yolo', 'SGBM, caja YOLO (cadena completa)')]


def run(name):
    return load(os.path.join(V, name + '.csv'))


def cmp_row(label, c):
    return (f"| {label} | {c['n']} | {pct(c['rec_a'])} | {pct(c['rec_b'])} | {100 * c['diff']:+.1f} "
            f"| {100 * c['lo']:+.1f} … {100 * c['hi']:+.1f} | {c['only_a']} | {c['only_b']} | {c['p_mcnemar']:.2g} |")


def main():
    out = ['# Fase 2 — confirmación en val (142 instancias)\n', 'Configuraciones con su rechazo real. '
           'Umbral ADD-S 18.69 mm.\n', SUMMARY_HDR]
    runs = {}
    for s, title in SCEN:
        for cfg, lab in (('A_yaml_actual', 'A'), ('B_defaults_nodo', 'B')):
            runs[(s, lab)] = run(f'icp_{s}_{cfg}')
            out.append(summary_row(f'{title} — {lab}', summary(runs[(s, lab)])))
    for s in ('gt', 'yolo'):
        p = os.path.join(V, f'icp_sgbmwls500_{s}_B_defaults_nodo.csv')
        if os.path.isfile(p):
            runs[(f'sgbm_{s}', 'B+wls500')] = load(p)
            out.append(summary_row(f'SGBM wls_lambda 500, caja {s.upper()} — B', summary(runs[(f'sgbm_{s}', 'B+wls500')])))
    hdr = ('| comparación | n | recall 1.ª % | recall 2.ª % | diferencia (pp) | IC 95 % | solo 1.ª | solo 2.ª | p McNemar |\n'
           '|---|---|---|---|---|---|---|---|---|')
    out.append('\n## A → B (emparejado)\n\n' + hdr)
    diffs = {}
    for s, title in SCEN:
        c = paired(runs[(s, 'A')], runs[(s, 'B')])
        diffs[s] = c
        out.append(cmp_row(title, c))
    ok_b = all(c['diff'] >= 0 for c in diffs.values()) and diffs['depthgt_gt']['lo'] > 0
    out.append(f"\n**Decisión A/B según el criterio fijado:** {'B' if ok_b else 'A'} "
               f"(diferencias ≥ 0 en los cuatro: {all(c['diff'] >= 0 for c in diffs.values())}; "
               f"IC en profundidad real + caja GT excluye 0: {diffs['depthgt_gt']['lo'] > 0}).")
    if ('sgbm_gt', 'B+wls500') in runs:
        out.append('\n## B → B + wls_lambda 500 (emparejado)\n\n' + hdr)
        cw = {s: paired(runs[(s, 'B')], runs[(s, 'B+wls500')]) for s in ('sgbm_gt', 'sgbm_yolo')}
        for s, c in cw.items():
            out.append(cmp_row(dict(SCEN)[s], c))
        ok_w = all(c['diff'] > 0 for c in cw.values()) and cw['sgbm_gt']['lo'] > 0
        out.append(f"\n**Decisión wls_lambda según el criterio fijado:** {'adoptar 500' if ok_w else 'mantener 8000'} "
                   f"(mejora en ambos: {all(c['diff'] > 0 for c in cw.values())}; IC en SGBM + caja GT excluye 0: "
                   f"{cw['sgbm_gt']['lo'] > 0}).")
    out.append('\n## Estratos, profundidad real + caja GT\n')
    out.append(strata_table([('A', runs[('depthgt_gt', 'A')]), ('B', runs[('depthgt_gt', 'B')])]))
    out.append('\n## Estratos, cadena completa (SGBM + YOLO)\n')
    out.append(strata_table([('A', runs[('sgbm_yolo', 'A')]), ('B', runs[('sgbm_yolo', 'B')])]))
    txt = '\n'.join(out) + '\n'
    with open(os.path.join(HERE, 'val.md'), 'w') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
