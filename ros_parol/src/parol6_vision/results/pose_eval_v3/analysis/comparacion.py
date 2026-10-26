#!/usr/bin/env python3
"""
comparacion.py — Comparación AGREGADA con la evaluación anterior
(anterior/, base ~/parol6_pose_eval_dataset = "v1") y separación de
efectos. Las escenas son distintas: no hay comparación instancia a instancia
entre bases; entre bases solo diferencias de proporciones independientes (IC
de Newcombe). Dentro de una base, los cambios de configuración o de detector
son emparejados (bootstrap por imagen + McNemar).

Secciones (escribe analysis/comparacion.md):
  1. Cifra final (test v3, config final) frente a test v1 (con y sin 'hard').
  2. Descomposición en train+val (sin muestras 'hard' en ninguna base):
     datos (v1→v3, config y detector fijos), detector (antiguo→nuevo),
     configuración (A→B, y estéreo 8000→500).
  3. Iluminación dentro de v3 (intensidad de la difusa por terciles).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import (load, apply_reject, summary, newcombe, paired, wilson, pct, mark_hard,  # noqa: E402
                   SUMMARY_HDR, summary_row)

R = os.path.join(HERE, '..')
OLD = os.path.join(R, 'anterior')
V1 = '~/parol6_pose_eval_dataset'
V3 = '~/parol6_pose_eval_dataset_v3'


def p(*a):
    return os.path.join(R, *a)


def ld(path, splits=None):
    if not os.path.isfile(path):
        return None
    rows = load(path)
    return [r for r in rows if r['split'] in splits] if splits else rows


def trval_v3(train_norej, mf, rm, val_csv):
    """train (corrida sin rechazo + rechazo a posteriori) + val (corrida real)."""
    t = ld(p('train', train_norej + '.csv'))
    v = ld(p('val', val_csv + '.csv'))
    if t is None or v is None:
        return None
    return apply_reject(t, mf, rm) + v


def rec(rows):
    k = sum(r['add_s_pass'] for r in rows)
    return k, len(rows)


def fmt_rec(rows):
    k, n = rec(rows)
    lo, hi = wilson(k, n)
    return f'{pct(k / n)} ({pct(lo)}–{pct(hi)}), n={n}'


def indep(label, a, b):
    """b − a, bases distintas."""
    ka, na = rec(a)
    kb, nb = rec(b)
    d, lo, hi = newcombe(kb, nb, ka, na)
    return (f'| {label} | {pct(ka / na)} (n={na}) | {pct(kb / nb)} (n={nb}) | {100 * d:+.1f} '
            f'| {100 * lo:+.1f} … {100 * hi:+.1f} | independiente (Newcombe) |')


def pair(label, a, b):
    c = paired(a, b)
    return (f"| {label} | {pct(c['rec_a'])} (n={c['n']}) | {pct(c['rec_b'])} | {100 * c['diff']:+.1f} "
            f"| {100 * c['lo']:+.1f} … {100 * c['hi']:+.1f} | emparejado, McNemar p={c['p_mcnemar']:.2g} |")


CMP_HDR = ('| efecto | recall antes % | recall después % | diferencia (pp) | IC 95 % | tipo |\n'
           '|---|---|---|---|---|---|')


def main():
    out = ['# Comparación con la evaluación anterior\n',
           'v1 = `~/parol6_pose_eval_dataset` (resultados en `anterior/`), v3 = '
           '`~/parol6_pose_eval_dataset_v3`. Config A = yaml anterior (`remove_plane: false`), B = defaults '
           'del nodo. Detector antiguo = best.pt de 15ac096, nuevo = best.pt actual. Umbral ADD-S 18.69 mm '
           'en todas las corridas.\n']

    # ── 1. Cifra final frente a test v1 ─────────────────────────────────────
    old = {s: mark_hard(ld(os.path.join(OLD, f + '.csv')), V1) for s, f in [
        ('dg', 'icp_depthgt_gtbbox'), ('sg', 'icp_sgbm_gtbbox'), ('dy', 'icp_depthgt_yolo'),
        ('sy', 'icp_sgbm_yolo'), ('d2u', 'd2co_asis_gtbbox'), ('d2s', 'd2co_relaxed_gtbbox')]}
    new = {s: ld(p('test', f + '.csv')) for s, f in [
        ('dg', 'icp_depthgt_gt_B_defaults_nodo'), ('sg', 'icp_sgbmwls500_gt_B_defaults_nodo'),
        ('dy', 'icp_depthgt_yolo_B_defaults_nodo'), ('sy', 'icp_sgbmwls500_yolo_B_defaults_nodo'),
        ('d2u', 'd2co_gt_umbral'), ('d2s', 'd2co_gt_sinumbral'), ('d2y', 'd2co_yolo_umbral')]}
    names = {'dg': 'ICP, profundidad real, caja GT', 'sg': 'ICP, SGBM, caja GT', 'dy': 'ICP, profundidad real, caja YOLO',
             'sy': 'ICP, SGBM, caja YOLO (cadena completa)', 'd2u': 'D2CO con umbral, caja GT',
             'd2s': 'D2CO sin umbral, caja GT', 'd2y': 'D2CO con umbral, caja YOLO'}
    out.append('## 1. Test: cifra final v3 frente a test v1\n')
    out.append('v1: config A, detector antiguo, estéreo `wls_lambda` 8000 (corridas de `anterior/` '
               'filtradas a `split == test`). v3: config B, detector nuevo, estéreo `wls_lambda` 500. '
               'Cambian a la vez datos, detector y configuración: la sección 2 los separa.\n')
    out.append('| escenario | v1 test (todas) | v1 test sin hard | v3 test | v3 − v1 sin hard (pp) | IC 95 % |')
    out.append('|---|---|---|---|---|---|')
    for s in ('dg', 'sg', 'dy', 'sy', 'd2u', 'd2s', 'd2y'):
        n_ = new.get(s)
        o = old.get(s)
        if n_ is None:
            continue
        if o is None:
            out.append(f'| {names[s]} | — | — | {fmt_rec(n_)} | — | — |')
            continue
        ot = [r for r in o if r['split'] == 'test']
        oth = [r for r in ot if not r['hard']]
        kb, nb = rec(n_)
        ka, na = rec(oth)
        d, lo, hi = newcombe(kb, nb, ka, na)
        out.append(f'| {names[s]} | {fmt_rec(ot)} | {fmt_rec(oth)} | {fmt_rec(n_)} | {100 * d:+.1f} | {100 * lo:+.1f} … {100 * hi:+.1f} |')
    out.append('\nCobertura (fracción con pose publicada), v1 test sin hard frente a v3 test:\n')
    out.append('| escenario | v1 test sin hard % | v3 test % | diferencia (pp) | IC 95 % |\n|---|---|---|---|---|')
    for s in ('dg', 'sg', 'dy', 'sy', 'd2u', 'd2s'):
        oth = [r for r in old[s] if r['split'] == 'test' and not r['hard']]
        ka, na = sum(r['pose_ok'] for r in oth), len(oth)
        kb, nb = sum(r['pose_ok'] for r in new[s]), len(new[s])
        d, lo, hi = newcombe(kb, nb, ka, na)
        out.append(f'| {names[s]} | {pct(ka / na)} | {pct(kb / nb)} | {100 * d:+.1f} | {100 * lo:+.1f} … {100 * hi:+.1f} |')
    out.append('\nResumen completo de v3 test:\n\n' + SUMMARY_HDR)
    for s in ('dg', 'sg', 'dy', 'sy', 'd2u', 'd2s', 'd2y'):
        if new.get(s) is not None:
            out.append(summary_row(names[s], summary(new[s])))
    out.append('\nResumen de v1 test sin hard (mismas métricas):\n\n' + SUMMARY_HDR)
    for s in ('dg', 'sg', 'dy', 'sy', 'd2u', 'd2s'):
        oth = [r for r in old[s] if r['split'] == 'test' and not r['hard']]
        out.append(summary_row(names[s], summary(oth)))
    hard = [r for r in old['dg'] if r['split'] == 'test' and r['hard']]
    out.append(f"\nMuestras 'hard' de v1 test: {len(hard)} instancias (de {len([r for r in old['dg'] if r['split'] == 'test'])}); "
               f"visibilidad mediana {np.median([r['visib'] for r in hard]):.3f}; aciertos ICP profundidad real + caja GT: "
               f"{sum(r['add_s_pass'] for r in hard)}.")

    # ── 2. Descomposición en train+val ──────────────────────────────────────
    TV = ('train', 'val')
    o_tv = {s: [r for r in old[s] if r['split'] in TV] for s in old}
    v3A = {'dg': trval_v3('icp_depthgt_gt_rpF', 0.20, 5.0, 'icp_depthgt_gt_A_yaml_actual'),
           'sg': trval_v3('icp_sgbm_gt_rpF', 0.20, 5.0, 'icp_sgbm_gt_A_yaml_actual'),
           'dy': trval_v3('icp_depthgt_yolo_rpF', 0.20, 5.0, 'icp_depthgt_yolo_A_yaml_actual'),
           'sy': trval_v3('icp_sgbm_yolo_rpF', 0.20, 5.0, 'icp_sgbm_yolo_A_yaml_actual')}
    v3B = {'dg': trval_v3('icp_depthgt_gt_rpT004', 0.45, 6.0, 'icp_depthgt_gt_B_defaults_nodo'),
           'sg': trval_v3('icp_sgbm_gt_rpT004', 0.45, 6.0, 'icp_sgbm_gt_B_defaults_nodo'),
           'dy': trval_v3('icp_depthgt_yolo_rpT004', 0.45, 6.0, 'icp_depthgt_yolo_B_defaults_nodo'),
           'sy': trval_v3('icp_sgbm_yolo_rpT004', 0.45, 6.0, 'icp_sgbm_yolo_B_defaults_nodo')}
    v3Bw = trval_v3('icp_sgbmwls500_gt_rpT004', 0.45, 6.0, 'icp_sgbmwls500_gt_B_defaults_nodo')
    v3A_old = {'dy': ld(p('cross', 'v3_trval_icp_depthgt_yoloant_A_yaml_actual.csv')),
               'sy': ld(p('cross', 'v3_trval_icp_sgbm_yoloant_A_yaml_actual.csv'))}
    v1B = {s: ld(p('cross', f'v1_trval_icp_{c}_B_defaults_nodo.csv')) for s, c in
           [('dg', 'depthgt_gt'), ('sg', 'sgbm_gt'), ('dy', 'depthgt_yolo'), ('sy', 'sgbm_yolo')]}
    out.append('\n## 2. Descomposición en train+val (ninguna base tiene muestras hard en train+val)\n')
    out.append('v3 train = corrida sin rechazo con el rechazo de la config aplicado a posteriori; v3 val = corrida real.\n')
    for s in ('dg', 'sg', 'dy', 'sy'):
        out.append(f'\n### {names[s]}\n\n' + CMP_HDR)
        if s in ('dg', 'sg'):
            out.append(indep('datos v1 → v3 (config A)', o_tv[s], v3A[s]))
            out.append(pair('config A → B (v3)', v3A[s], v3B[s]))
            if v1B[s] is not None:
                out.append(pair('config A → B (v1, réplica)', o_tv[s], v1B[s]))
                out.append(indep('datos v1 → v3 (config B)', v1B[s], v3B[s]))
            if s == 'sg' and v3Bw is not None:
                out.append(pair('estéreo wls_lambda 8000 → 500 (v3, config B)', v3B[s], v3Bw))
        else:
            if v3A_old[s] is not None:
                out.append(indep('datos v1 → v3 (config A, detector antiguo)', o_tv[s], v3A_old[s]))
                out.append(pair('detector antiguo → nuevo (v3, config A)', v3A_old[s], v3A[s]))
            out.append(pair('config A → B (v3, detector nuevo)', v3A[s], v3B[s]))
            if v1B[s] is not None:
                out.append(indep('datos v1 → v3 (config B, detector nuevo)', v1B[s], v3B[s]))
        line = f'\nRecalls: v1 A {fmt_rec(o_tv[s])}; v3 A {fmt_rec(v3A[s])}; v3 B {fmt_rec(v3B[s])}'
        if v1B[s] is not None:
            line += f'; v1 B {fmt_rec(v1B[s])}'
        out.append(line + '.')

    # ── 3. Iluminación dentro de v3 ─────────────────────────────────────────
    metas = {}
    with open(os.path.expanduser(V3 + '/meta.jsonl')) as fh:
        for line in fh:
            m = json.loads(line)
            metas[int(m['im_id'])] = m
    g = {i: m['light_diffuse'][0] for i, m in metas.items()}
    cuts = np.percentile(list(g.values()), [100 / 3, 200 / 3])
    out.append(f'\n## 3. Iluminación aleatoria dentro de v3\n\nv1 tenía la difusa fija del mundo, (0.9, 0.85, 0.8) '
               f'(`mars_optimized.world`), es decir g = 0.9; el {pct(np.mean([v < 0.9 for v in g.values()]), 0)} % de '
               f'las escenas de v3 están más oscuras. Difusa del sol g ~ U(0.55, 1.0) por escena; terciles '
               f'g < {cuts[0]:.3f}, {cuts[0]:.3f}–{cuts[1]:.3f}, > {cuts[1]:.3f}. Todas las particiones disponibles '
               f'(train con rechazo a posteriori).\n')
    det = json.load(open(p('yolo', 'v3_detnuevo.json')))['detections']
    info = json.load(open(os.path.expanduser(V3 + '/bop/scene_gt_info.json')))
    from yolo_metrics import iou
    det_rows = []
    for k, infos in info.items():
        boxes = [(d['xyxy'][0], d['xyxy'][1], d['xyxy'][2] - d['xyxy'][0], d['xyxy'][3] - d['xyxy'][1]) for d in det.get(k, [])]
        for i in infos:
            bv = i.get('bbox_visib')
            if bv and bv[2] > 0 and bv[3] > 0:
                det_rows.append({'im_id': int(k), 'add_s_pass': int(max((iou(b, bv) for b in boxes), default=0) >= 0.3)})
    series = [('detección YOLO nuevo (IoU ≥ 0.3)', det_rows)]
    for s, lab in [('dg', 'ICP B, profundidad real, caja GT'), ('sy', 'ICP B, SGBM 8000 + YOLO')]:
        rows = (v3B[s] or []) + ([r for r in (ld(p('test', {'dg': 'icp_depthgt_gt_B_defaults_nodo',
                                                          'sy': 'icp_sgbm_yolo_B_defaults_nodo'}[s] + '.csv')) or [])])
        series.append((lab + (' (train+val)' if s == 'sy' else ' (train+val+test)'), rows))
    series.append(('ICP A, SGBM 8000, caja GT (train+val)', v3A['sg']))
    series.append(('ICP B, SGBM 8000, caja GT (train+val)', v3B['sg']))
    if v3Bw is not None:
        series.append(('ICP B, SGBM 500, caja GT (train+val)', v3Bw))
    for nm, lab in [('d2co_gt_sinumbral', 'D2CO sin umbral, caja GT (test)')]:
        rr = ld(p('test', nm + '.csv'))
        if rr:
            series.append((lab, rr))
    out.append('| serie | g bajo | g medio | g alto |\n|---|---|---|---|')
    for lab, rows in series:
        cells = []
        for lo, hi in [(-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 2)]:
            sel = [r for r in rows if lo <= g[r['im_id']] < hi]
            k = sum(r['add_s_pass'] for r in sel)
            a, b = wilson(k, len(sel))
            cells.append(f'{pct(k / len(sel))} ({pct(a, 0)}–{pct(b, 0)}), n={len(sel)}' if sel else '—')
        out.append(f'| {lab} | ' + ' | '.join(cells) + ' |')

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(HERE, 'comparacion.md'), 'w') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
