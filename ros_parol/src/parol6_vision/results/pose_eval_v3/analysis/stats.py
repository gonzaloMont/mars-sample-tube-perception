#!/usr/bin/env python3
"""
stats.py — Utilidades de análisis para results/pose_eval_v3 (sin ROS).

Lee los CSV por instancia de eval_pose_dataset.py y calcula:
  * resumen (cobertura, recall ADD-S sobre el total, medianas de error),
  * IC 95 % del recall: Wilson (binomial) y bootstrap por imagen (las
    instancias de una misma imagen no son independientes),
  * comparación emparejada de dos corridas sobre las mismas instancias:
    McNemar exacto + IC bootstrap por imagen de la diferencia,
  * comparación de corridas independientes (bases de datos distintas):
    IC de Newcombe (método híbrido de Wilson) de la diferencia,
  * rechazo a posteriori: en las corridas "sin rechazo" (min_fitness 0,
    max_rmse 1 m) aplica cualquier (min_fitness, max_rmse) con las columnas
    fitness / rmse_mm, que es exactamente lo que hace el nodo al final.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

Z95 = 1.959963984540054
VIS_EDGES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.01]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def load(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r['im_id'] = int(r['im_id'])
        r['inst'] = int(r['inst'])
        r['pose_ok'] = int(f(r['pose_ok']) or 0)
        r['add_s_pass'] = int(f(r['add_s_pass']) or 0)
        for k in ('visib', 'add_s_mm', 'err_radial_mm', 'err_lateral_mm', 'err_axis_deg', 'err_trans_mm',
                  'latency_ms', 'fitness', 'rmse_mm', 'det_iou', 'n_scene', 'n_crop', 'z_est_m', 'z_gt_m'):
            if k in r:
                r[k] = f(r[k])
    return rows


def key(r):
    return (r['im_id'], r['inst'])


def mark_hard(rows, dataset_dir):
    """Añade r['hard'] desde meta.jsonl (muestras 'hard' del dataset antiguo)."""
    hard = {}
    with open(os.path.join(os.path.expanduser(dataset_dir), 'meta.jsonl')) as fh:
        for line in fh:
            m = json.loads(line)
            hard[int(m['im_id'])] = bool(m.get('hard', False))
    for r in rows:
        r['hard'] = hard.get(r['im_id'], False)
    return rows


def apply_reject(rows, min_fitness, max_rmse_mm):
    """Rechazo a posteriori sobre una corrida sin rechazo."""
    out = []
    for r in rows:
        r2 = dict(r)
        if r['pose_ok']:
            ok = (r['fitness'] >= min_fitness) and (r['rmse_mm'] <= max_rmse_mm)
            if not ok:
                r2['pose_ok'] = 0
                r2['add_s_pass'] = 0
                r2['msg'] = f"rechazada a posteriori (fitness={r['fitness']:.3f} rmse={r['rmse_mm']:.2f}mm)"
        out.append(r2)
    return out


# ── Intervalos ──────────────────────────────────────────────────────────────

def wilson(k, n, z=Z95):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2, z=Z95):
    """IC de p1 - p2 para muestras independientes (Newcombe 1998, método 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def mcnemar_exact(b, c):
    """p bilateral exacto de McNemar con b, c discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _by_image(rows, field='add_s_pass'):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        g[r['im_id']][0] += r[field]
        g[r['im_id']][1] += 1
    ids = sorted(g)
    return np.array([g[i][0] for i in ids], float), np.array([g[i][1] for i in ids], float)


def bootstrap_recall(rows, n_boot=4000, seed=0):
    """IC 95 % percentil del recall remuestreando IMÁGENES (cluster bootstrap)."""
    k, n = _by_image(rows)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(n_boot, len(k)))
    rec = k[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))


def paired(rows_a, rows_b, n_boot=4000, seed=0):
    """Compara A y B sobre sus instancias comunes. Devuelve dict con n, recall
    de cada una, diferencia B−A, IC bootstrap por imagen y p de McNemar."""
    da = {key(r): r for r in rows_a}
    db = {key(r): r for r in rows_b}
    common = sorted(set(da) & set(db))
    a = np.array([da[k]['add_s_pass'] for k in common], bool)
    b = np.array([db[k]['add_s_pass'] for k in common], bool)
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    ims = np.array([k[0] for k in common])
    uniq, inv = np.unique(ims, return_inverse=True)
    ka = np.bincount(inv, weights=a.astype(float))
    kb = np.bincount(inv, weights=b.astype(float))
    nn = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    diff = (kb[idx].sum(1) - ka[idx].sum(1)) / nn[idx].sum(1)
    n = len(common)
    return {'n': n, 'rec_a': a.mean() if n else float('nan'), 'rec_b': b.mean() if n else float('nan'),
            'diff': (b.mean() - a.mean()) if n else float('nan'),
            'lo': float(np.percentile(diff, 2.5)), 'hi': float(np.percentile(diff, 97.5)),
            'only_a': only_a, 'only_b': only_b, 'p_mcnemar': mcnemar_exact(only_a, only_b)}


# ── Resúmenes ───────────────────────────────────────────────────────────────

def _med(vals):
    vals = [v for v in vals if v == v]
    return float(np.median(vals)) if vals else float('nan')


def summary(rows, boot=True):
    n = len(rows)
    ok = [r for r in rows if r['pose_ok']]
    k = sum(r['add_s_pass'] for r in rows)
    lo, hi = wilson(k, n)
    s = {'n': n, 'n_ok': len(ok), 'k': k, 'cov': len(ok) / n if n else float('nan'),
         'rec': k / n if n else float('nan'), 'lo': lo, 'hi': hi,
         'rec_acc': k / len(ok) if ok else float('nan'),
         'adds_med': _med([r['add_s_mm'] for r in ok]),
         'rad_med': _med([abs(r['err_radial_mm']) for r in ok]),
         'rad_bias': float(np.mean([r['err_radial_mm'] for r in ok])) if ok else float('nan'),
         'rad_med_signed': _med([r['err_radial_mm'] for r in ok]),
         'lat_med': _med([r['err_lateral_mm'] for r in ok]),
         'axis_med': _med([r['err_axis_deg'] for r in ok]),
         'ms_med': _med([r['latency_ms'] for r in rows])}
    if boot and n:
        s['blo'], s['bhi'] = bootstrap_recall(rows)
    return s


def pct(x, d=1):
    return '—' if x != x else f'{100 * x:.{d}f}'


def num(x, d=1, sign=False):
    if x != x:
        return '—'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


SUMMARY_HDR = ('| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm '
               '| \\|radial\\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |\n'
               '|---|---|---|---|---|---|---|---|---|---|---|')


def summary_row(label, s):
    return (f"| {label} | {s['n']} | {pct(s['cov'])} | **{pct(s['rec'])}** ({pct(s['lo'])}–{pct(s['hi'])}) "
            f"| {pct(s['rec_acc'])} | {num(s['adds_med'])} | {num(s['rad_med'])} | {num(s['rad_bias'], sign=True)} "
            f"| {num(s['lat_med'])} | {num(s['axis_med'])} | {num(s['ms_med'], 0)} |")


def strata(rows):
    """Grupos para estratificar: visibilidad, modo, brazo."""
    groups = []
    for lo, hi in zip(VIS_EDGES[:-1], VIS_EDGES[1:]):
        groups.append((f'visib {lo:.2f}–{min(hi, 1.0):.2f}', [r for r in rows if lo <= r['visib'] < hi]))
    for m in sorted({r['mode'] for r in rows}):
        groups.append((f'modo {m}', [r for r in rows if r['mode'] == m]))
    for a in sorted({r['arm_mode'] for r in rows}):
        groups.append((f'brazo {a}', [r for r in rows if r['arm_mode'] == a]))
    return groups


def strata_table(runs):
    """runs: [(label, rows)] con las MISMAS instancias. Tabla recall (IC) por estrato."""
    labels = [l for l, _ in runs]
    hdr = '| estrato | n | ' + ' | '.join(f'{l}: recall % (IC)' for l in labels) + ' |'
    lines = [hdr, '|---|---|' + '---|' * len(labels)]
    ref = runs[0][1]
    for name, sel in strata(ref):
        if not sel:
            continue
        keys = {key(r) for r in sel}
        cells = []
        for _, rows in runs:
            sub = [r for r in rows if key(r) in keys]
            k = sum(r['add_s_pass'] for r in sub)
            lo, hi = wilson(k, len(sub))
            cells.append(f'{pct(k / len(sub))} ({pct(lo, 0)}–{pct(hi, 0)})')
        lines.append(f'| {name} | {len(sel)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
