# Evaluación anterior (base v1, 16-09-2026): lo que se conserva

Corridas principales de la evaluación anterior sobre `~/parol6_pose_eval_dataset`
(config A = `config/icp_pose_params.yaml` de entonces, detector `best.pt` de
15ac096, estéreo `wls_lambda` 8000, todas las particiones). Se conservan porque
el informe de `../README.md` las usa para la comparación (`analysis/comparacion.py`,
`analysis/atribucion.py --only-split test`).

| Fichero | Corrida |
|---|---|
| `icp_depthgt_gtbbox.csv` | ICP, profundidad real, caja GT |
| `icp_sgbm_gtbbox.csv` | ICP, SGBM, caja GT |
| `icp_depthgt_yolo.csv` | ICP, profundidad real, caja YOLO |
| `icp_sgbm_yolo.csv` | ICP, SGBM, caja YOLO |
| `d2co_asis_gtbbox.csv` | D2CO con umbral, caja GT |
| `d2co_relaxed_gtbbox.csv` | D2CO sin umbral (`max_avg_dist:=100`), caja GT |
| `tiempos_corridas.csv` | inicio/fin de cada corrida (fecha del CSV − «Tiempo total» del log): las seis se ejecutaron en paralelo |

Todo lo demás (README y `comparativa_todas.md` antiguos, barridos `sweeps/`
hechos sobre test, la corrida de 40 instancias del nodo ICP, *logs* y
*overlays*) está en `~/parol6_pose_eval_v1_resultados_2026-09-16.tar.gz`,
fuera del repositorio. Cifras de esos ficheros que cita el informe:

| Dato | Valor | Fichero dentro del tar |
|---|---|---|
| Latencia mediana de las corridas principales (en paralelo) | ICP 5932 / 7222 / 9304 / 15821 ms; D2CO 3440 / 3087 ms | `pose_eval/comparativa_todas.md` |
| Latencia mediana de los barridos (en serie) | 230–839 ms | `pose_eval/sweeps/*.log` |
| Barrido estéreo con `remove_plane` en test v1 | recall 15.8 % (`wls_lambda` 8000) → 19.8 % (500) | `pose_eval/sweeps/sweep_stereo_plane.md` |
