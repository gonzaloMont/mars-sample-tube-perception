# D2CO: mecanismo interno (logs del nodo, test v3)

| corrida | ciclos [service] | bordes ok / blob / sin bordes | corrección por anchura | k mediana | k > 1 (aleja) % | anchura medida mediana (px de trabajo) | ciclos con descartes por la banda de profundidad |
|---|---|---|---|---|---|---|---|
| d2co_gt_sinumbral | 263 | 184 / 73 / 6 | 82 | 1.30 | 86.6 | 6.15 | 82 |
| d2co_gt_umbral | 256 | 167 / 78 / 11 | 77 | 1.28 | 88.3 | 6.00 | 75 |
| d2co_yolo_umbral | 224 | 165 / 55 / 4 | 83 | 1.30 | 88.0 | 6.00 | 71 |

`k` = factor por el que la corrección por anchura (z = fx·Ø/w) multiplica la traslación; se aplica solo si avg_dist > `width_correction_gate` (1.5). "blob" = los bordes no son alargados: sin eje PCA ni banda de profundidad, y barrido amplio `z_factors`.

## Corridas de la evaluación anterior: ejecución simultánea

| corrida | inicio | fin | tiempo total s |
|---|---|---|---|
| icp_depthgt_gtbbox | 2026-09-16 01:08 | 2026-09-16 02:53 | 6313 |
| icp_sgbm_gtbbox | 2026-09-16 01:08 | 2026-09-16 02:52 | 6252 |
| icp_depthgt_yolo | 2026-09-16 01:08 | 2026-09-16 02:09 | 3688 |
| icp_sgbm_yolo | 2026-09-16 01:08 | 2026-09-16 02:04 | 3407 |
| d2co_asis_gtbbox | 2026-09-16 17:43 | 2026-09-16 18:42 | 3523 |
| d2co_relaxed_gtbbox | 2026-09-16 17:43 | 2026-09-16 18:47 | 3813 |

(`anterior/tiempos_corridas.csv`: inicio = fecha de modificación del CSV − "Tiempo total" del log, calculado antes de archivar los logs). Las latencias de esas corridas están infladas por la concurrencia; las comparables son las de los barridos en serie (`anterior/README.md`).
