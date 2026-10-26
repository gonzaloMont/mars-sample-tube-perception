#!/usr/bin/env python3
"""
atribucion.py — ¿Dónde falla cada solución? Atribución por instancia sobre las
corridas emparejadas de un mismo split (mismas instancias en todas).

ICP: escalera de lo ideal a lo real para la cadena completa (SGBM + YOLO):
  1. detección        YOLO no devuelve ninguna caja asociable (IoU ≥ 0.3)
  2. segmentación /   falla incluso con profundidad real y caja GT
     registro         (subtipos: sin pose, eje ≥ 10°, eje < 10° y traslación)
  3. estéreo          acierta con profundidad real + caja GT, falla con SGBM + caja GT
  4. caja YOLO        acierta con SGBM + caja GT, falla con SGBM + caja YOLO
  5. acierto
D2CO: rechazo por umbral, sin candidatos (todas las hipótesis fuera de la banda
de profundidad, avg_dist = DBL_MAX), y entre las aceptadas, qué componente
del error supera el umbral (radial = profundidad, lateral, eje).

Uso: atribucion.py --split test --icp-cfg B_defaults_nodo [--md analysis/atribucion_test.md]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import load, key, pct  # noqa: E402

THR_MM = 18.69   # 0.1 × diámetro (models_info.json)


def icp_ladder(dg, sg, dy, sy):
    """dg/sg/dy/sy: dicts key→row de depthgt+gt, sgbm+gt, depthgt+yolo, sgbm+yolo."""
    cats, sub = {}, {}
    for k in sorted(sy):
        r = sy[k]
        if 'sin deteccion' in r['msg']:
            c = '1 detección (YOLO no encuentra el tubo)'
        elif not dg[k]['add_s_pass']:
            c = '2 segmentación/registro (falla con profundidad real y caja GT)'
            g = dg[k]
            if not g['pose_ok']:
                sub[k] = 'sin pose (' + ('pocos puntos' if 'pocos puntos' in g['msg'] else 'rechazada') + ')'
            elif g['err_axis_deg'] >= 10:
                sub[k] = 'pose con eje ≥ 10°'
            else:
                sub[k] = 'eje < 10°, error de traslación'
        elif not sg[k]['add_s_pass']:
            c = '3 reconstrucción estéreo (acierta con profundidad real)'
            g = sg[k]
            if not g['pose_ok']:
                sub[k] = 'sin pose con SGBM'
            elif g['err_axis_deg'] >= 10:
                sub[k] = 'SGBM: eje ≥ 10°'
            elif abs(g['err_radial_mm']) >= THR_MM:
                sub[k] = 'SGBM: eje < 10°, |radial| ≥ umbral'
            else:
                sub[k] = 'SGBM: eje < 10°, otro'
        elif not r['add_s_pass']:
            c = '4 caja de YOLO (acierta con caja GT)'
        else:
            c = '5 acierto de la cadena completa'
        cats[k] = c
    return cats, sub


def d2co_class(rows):
    out = {}
    for r in rows:
        k = key(r)
        m = r['msg']
        if r['add_s_pass']:
            out[k] = 'acierto'
        elif 'sin deteccion' in m:
            out[k] = 'sin detección YOLO'
        elif not r['pose_ok']:
            mm = re.search(r'avg_dist=([0-9.e+]+)', m)
            v = float(mm.group(1)) if mm else float('nan')
            # DBL_MAX (1.797e308) o FLT_MAX (3.403e38) llegan truncados: el harness
            # corta msg a 120 caracteres
            out[k] = ('sin candidato (todas fuera de la banda de profundidad)'
                      if '179769313486' in m or '340282346638' in m or v > 1e30
                      else 'rechazada (avg_dist > umbral)' if v == v else 'sin pose: ' + m[:40])
        else:
            rad = abs(r['err_radial_mm']) >= THR_MM
            lat = r['err_lateral_mm'] >= THR_MM
            ax = r['err_axis_deg'] >= 10
            out[k] = 'aceptada, falla: ' + '+'.join(n for n, b in (('profundidad', rad), ('lateral', lat), ('eje', ax)) if b) \
                if (rad or lat or ax) else 'aceptada, falla: ADD-S sin componente dominante'
    return out


def counts_table(cats, title, total):
    from collections import Counter
    c = Counter(cats.values())
    lines = [f'| {title} | n | % de {total} |', '|---|---|---|']
    for name, n in sorted(c.items()):
        lines.append(f'| {name} | {n} | {pct(n / total)} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--icp-cfg', default='B_defaults_nodo')
    ap.add_argument('--md', default='')
    ap.add_argument('--files', nargs=4, metavar=('DG', 'SG', 'DY', 'SY'), default=None,
                    help='CSV explícitos (p. ej. de anterior/) en vez de <split>/icp_*_<cfg>.csv')
    ap.add_argument('--d2co', nargs='*', default=None, help='CSV de D2CO explícitos')
    ap.add_argument('--only-split', default='', help='filtrar las filas a este split (CSV con todos los splits)')
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    d = os.path.join(HERE, '..', a.split)
    out = [f'# Atribución de fallos — {a.title or f"split {a.split}, ICP config {a.icp_cfg}"}\n']
    names = {'dg': f'icp_depthgt_gt_{a.icp_cfg}', 'sg': f'icp_sgbm_gt_{a.icp_cfg}',
             'dy': f'icp_depthgt_yolo_{a.icp_cfg}', 'sy': f'icp_sgbm_yolo_{a.icp_cfg}'}
    paths = dict(zip(names, a.files)) if a.files else {s: os.path.join(d, n + '.csv') for s, n in names.items()}

    def rows_of(p):
        rows = load(p)
        return [r for r in rows if r['split'] == a.only_split] if a.only_split else rows
    if all(os.path.isfile(p) for p in paths.values()):
        R = {s: {key(r): r for r in rows_of(p)} for s, p in paths.items()}
        cats, sub = icp_ladder(R['dg'], R['sg'], R['dy'], R['sy'])
        n = len(cats)
        out.append('## ICP: cadena completa (SGBM + YOLO), escalera ideal → real\n')
        out.append(counts_table(cats, 'causa', n))
        out.append('\nSubtipos:\n')
        out.append(counts_table(sub, 'subtipo', n))
        # Instancias que la cadena completa acierta aunque un peldaño "más fácil" falle
        nm = sum(1 for k in cats if R['sy'][k]['add_s_pass'] and not (R['dg'][k]['add_s_pass'] and R['sg'][k]['add_s_pass']))
        out.append(f'\nNo monótonas (aciertan con SGBM + YOLO pero fallan en algún peldaño ideal): {nm}.')
        # Sesgo de profundidad del estéreo en las instancias con eje correcto
        ok = [R['sg'][k] for k in cats if R['sg'][k]['pose_ok'] and R['sg'][k]['err_axis_deg'] < 10]
        if ok:
            rad = np.array([r['err_radial_mm'] for r in ok])
            out.append(f'\nSGBM + caja GT, poses con eje < 10° (n={len(ok)}): error radial mediano {np.median(rad):+.1f} mm, '
                       f'media {rad.mean():+.1f} mm, |radial| < umbral en {pct((np.abs(rad) < THR_MM).mean())} %.')
        # Segmentación: tamaño del cluster en fallos frente a aciertos (profundidad real, caja GT)
        okk = [R['dg'][k] for k in cats if R['dg'][k]['pose_ok']]
        good = [r['n_scene'] for r in okk if r['add_s_pass']]
        bad = [r['n_scene'] for r in okk if not r['add_s_pass']]
        if good and bad:
            out.append(f'\nPuntos del cluster tras la limpieza (profundidad real, caja GT): mediana {np.median(good):.0f} en aciertos, '
                       f'{np.median(bad):.0f} en fallos con pose.')
        # Escalera por estrato de visibilidad
        out.append('\nCausa por visibilidad:\n')
        bins = [(0, .25), (.25, .5), (.5, .75), (.75, .9), (.9, 1.01)]
        cats_names = sorted(set(cats.values()))
        out.append('| visibilidad | n | ' + ' | '.join(cats_names) + ' |')
        out.append('|---|---|' + '---|' * len(cats_names))
        for lo, hi in bins:
            ks = [k for k in cats if lo <= R['sy'][k]['visib'] < hi]
            if ks:
                out.append(f'| {lo:.2f}–{min(hi, 1):.2f} | {len(ks)} | ' +
                           ' | '.join(str(sum(cats[k] == c for k in ks)) for c in cats_names) + ' |')
    d2co_paths = a.d2co if a.d2co is not None else [
        os.path.join(d, nm + '.csv') for nm in ('d2co_gt_umbral', 'd2co_gt_sinumbral', 'd2co_yolo_umbral')]
    for p in d2co_paths:
        if not os.path.isfile(p):
            continue
        rows = rows_of(p)
        out.append(f'\n## {os.path.splitext(os.path.basename(p))[0]}\n')
        out.append(counts_table(d2co_class(rows), 'resultado', len(rows)))
        ok = [r for r in rows if r['pose_ok']]
        if ok:
            rad = np.array([r['err_radial_mm'] for r in ok])
            lat = np.array([r['err_lateral_mm'] for r in ok])
            ax = np.array([r['err_axis_deg'] for r in ok])
            out.append(f'\nPoses publicadas (n={len(ok)}): |radial| < umbral {pct((np.abs(rad) < THR_MM).mean())} %, '
                       f'lateral < umbral {pct((lat < THR_MM).mean())} %, eje < 10° {pct((ax < 10).mean())} %; '
                       f'radial mediano {np.median(rad):+.0f} mm (sobreestima la profundidad en {pct((rad > 0).mean())} %).')
    txt = '\n'.join(out) + '\n'
    print(txt)
    if a.md:
        with open(a.md, 'w') as fh:
            fh.write(txt)


if __name__ == '__main__':
    main()
