# Resultados de evaluación de D2CO

Generados con `scripts/eval_d2co_dataset.py` sobre el dataset sintético
`~/parol6_synth_dataset` (GT 6D estilo BOP, en metros).

## Alcance de estas tiradas

Ambas evalúan las **mismas 199 instancias** de tubo, filtradas al
**sobre operativo del robot: 1.10 – 1.41 m** de distancia cámara→tubo.

Ese rango NO se eligió mirando resultados: se deriva de la geometría del
sistema, antes de medir nada.

- Alcance del PAROL6: 0.456 m desde el hombro (0.18 + 0.176 + muñeca),
  hombro a z = 0.1105 m — de `parol6_description/urdf/parol6.csv`.
- NavCam en mástil: (-0.3, -0.3, 1.11) respecto a `base_link` — de
  `gazebo_moveit_mars.launch.py` (`mast_x/y/z`).
- Barriendo la zona alcanzable sobre el suelo, la distancia de la cámara al
  tubo queda en 1.10 – 1.41 m (mediana 1.21).

Splits: `train` + `val`. El split `test` se reserva SIN USAR para la cifra
final, de modo que ningún ajuste de parámetros lo contamine.

## Ficheros

| Fichero | Configuración |
|---|---|
| `d2co_banda_operativa_baseline.*` | prior de profundidad `z ∈ [0.3, 3.6]` m |
| `d2co_banda_operativa_prior_estrecho.*` | prior de profundidad `z ∈ [1.0, 1.5]` m |

`.csv` = una fila por instancia; `.txt` = resumen impreso por el harness.

## Parámetros del nodo comunes a ambas

```
tracking_max_age_s := 0.0     # obligatorio: escenas independientes
broadcast_tf       := false
publish_debug_overlay := false
process_rate       := 0.1     # el pipeline lo dispara el harness por servicio
```

`tracking_max_age_s := 0.0` es imprescindible: las muestras son escenas sin
relación entre sí, y con el tracking activo la pose de la muestra N se usaría
como semilla de la N+1, midiendo un seguimiento sobre datos inconexos en vez
del algoritmo.

## Columnas del CSV

| Columna | Significado |
|---|---|
| `im_id`, `inst` | imagen e índice de instancia dentro de ella |
| `dist_gt_m` | distancia cámara→tubo de la verdad-terreno [m] |
| `visib` | `visib_fract` (fracción visible tras oclusión) |
| `z_gt_m`, `z_est_m` | profundidad verdadera y estimada [m] |
| `pose_ok` | 1 si el nodo publicó pose; 0 si la rechazó |
| `add_s_mm` | ADD-S [mm]. Métrica simétrica, obligatoria: el tubo declara `symmetries_continuous` en Z |
| `add_s_pass` | 1 si ADD-S < 18.83 mm (0.1 × diámetro, convenio BOP) |
| `err_radial_mm` | error de traslación a lo largo de la línea de visión (profundidad) |
| `err_lateral_mm` | error de traslación perpendicular a la línea de visión |
| `err_axis_deg` | ángulo entre el eje del cilindro real y el estimado, plegado a [0, 90] |
| `attempts`, `stale` | salud del harness: reintentos y poses descartadas por stamp ajeno |
| `msg` | mensaje devuelto por el servicio `~/estimate_now` |

## Advertencias de lectura

- **Cobertura y error van siempre juntos.** El nodo rechaza poses por
  `max_avg_dist`, así que los errores están condicionados a aceptación: un
  algoritmo puede parecer preciso simplemente rechazando lo difícil. El
  recall ADD-S se calcula sobre el TOTAL de instancias, no sobre las
  aceptadas.
- `avg_dist = 1.79e308` en `msg` es `DBL_MAX`: significa que TODAS las
  hipótesis refinadas cayeron fuera de la banda de profundidad y no hubo
  ningún candidato. Es un modo de fallo distinto de `avg_dist > 4.5`
  (ajuste pobre), aunque el nodo los reporte con el mismo texto.
- Una tirada anterior con `--timeout-s 30` produjo 55 timeouts que se
  contabilizaron como falta de cobertura. Los ficheros aquí son de la
  repetición con timeout suficiente, y no contienen ningún timeout.
