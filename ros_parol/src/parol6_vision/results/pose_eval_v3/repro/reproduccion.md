# Reproducción de la corrida antigua `icp_sgbm_gtbbox` (v1, config A, SGBM 8000, caja GT) con el código de hoy

| | n | cobertura % | recall % | ADD-S med mm | sesgo radial mm |
|---|---|---|---|---|---|
| septiembre (anterior/icp_sgbm_gtbbox.csv) | 729 | 92.7 | 18.5 | 71.5 | +47.4 |
| hoy (repro/v1_todas_icp_sgbm_gt_A_hoy.csv) | 729 | 92.7 | 18.1 | 71.0 | +47.5 |

Instancia a instancia: 262 de 676 poses con ADD-S idéntico (< 0.01 mm), |ΔADD-S| mediana 0.093 mm, p90 4.41 mm; 0 cambian de con/sin pose; aciertos solo en septiembre 4, solo hoy 1 (McNemar p=0.38).
