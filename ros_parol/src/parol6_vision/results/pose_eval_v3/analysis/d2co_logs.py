#!/usr/bin/env python3
"""
d2co_logs.py — Mecanismo interno de D2CO a partir de los logs del nodo
(test/d2co_*.node.log, no versionados por el .gitignore) y de los tiempos de
inicio/fin de las corridas anteriores (anterior/tiempos_corridas.csv, para la
nota de latencias). Escribe analysis/d2co_mecanismo.md.

Del log del nodo solo se cuentan los ciclos disparados por el servicio
([service]: una instancia + sus reintentos), no los del timer.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
OLD = os.path.join(R, 'anterior')


def main():
    out = ['# D2CO: mecanismo interno (logs del nodo, test v3)\n',
           '| corrida | ciclos [service] | bordes ok / blob / sin bordes | corrección por anchura | k mediana | k > 1 (aleja) % '
           '| anchura medida mediana (px de trabajo) | ciclos con descartes por la banda de profundidad |',
           '|---|---|---|---|---|---|---|---|']
    for f in sorted(glob.glob(os.path.join(R, 'test', 'd2co_*.node.log'))):
        txt = open(f).read().splitlines()
        cyc = [l for l in txt if '[service] bbox' in l]
        edges = [re.search(r'bordes\((\w+)', l).group(1) for l in cyc]
        corr = [l for l in txt if '[service] correccion anchura' in l]
        k = np.array([float(re.search(r'k=([0-9.]+)', l).group(1)) for l in corr])
        w = np.array([float(re.search(r'w=([0-9.]+)px', l).group(1)) for l in corr])
        band = sum(1 for l in txt if re.search(r'\[service\] \d+ pose\(s\) refinadas descartadas', l))
        name = os.path.basename(f).replace('.node.log', '')
        out.append(f"| {name} | {len(cyc)} | {edges.count('ok')} / {edges.count('blob')} / {edges.count('NO')} "
                   f"| {len(corr)} | {np.median(k):.2f} | {100 * (k > 1).mean():.1f} | {np.median(w):.2f} | {band} |")
    out.append('\n`k` = factor por el que la corrección por anchura (z = fx·Ø/w) multiplica la traslación; se aplica '
               'solo si avg_dist > `width_correction_gate` (1.5). "blob" = los bordes no son alargados: sin eje '
               'PCA ni banda de profundidad, y barrido amplio `z_factors`.\n')
    out.append('## Corridas de la evaluación anterior: ejecución simultánea\n')
    out.append('| corrida | inicio | fin | tiempo total s |\n|---|---|---|---|')
    import csv
    for r in csv.DictReader(open(os.path.join(OLD, 'tiempos_corridas.csv'))):
        out.append(f"| {r['corrida']} | {r['inicio']} | {r['fin']} | {r['tiempo_total_s']} |")
    out.append('\n(`anterior/tiempos_corridas.csv`: inicio = fecha de modificación del CSV − "Tiempo total" del log, '
               'calculado antes de archivar los logs). Las latencias de esas corridas están infladas por la '
               'concurrencia; las comparables son las de los barridos en serie (`anterior/README.md`).')
    txt = '\n'.join(out) + '\n'
    with open(os.path.join(HERE, 'd2co_mecanismo.md'), 'w') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    sys.exit(main())
