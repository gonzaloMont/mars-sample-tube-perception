# Evaluación de pose 6D sobre la base de datos v3: ICP estéreo frente a D2CO monocular

Evaluación de los dos estimadores de pose del tubo `mars2020tube` sobre
`~/parol6_pose_eval_dataset_v3` (600 escenas, 733 tubos, 729 con caja visible;
partición 300/120/180 escenas = 365/142/222 instancias; iluminación aleatoria;
sin muestras *hard*), con el banco de pruebas existente
(`scripts/eval_pose_dataset.py`) y el protocolo de `docs/pose_evaluation.md`.
Comparación agregada con la evaluación anterior (`anterior/`, base
`~/parol6_pose_eval_dataset`, que aquí se llama **v1**).

Todas las cifras salen de los ficheros indicados en cada tabla («Fuente»). Los
`.md` de `analysis/` se regeneran con los scripts de esa misma carpeta (ver
[Reproducción](#reproducción)).

## Resumen

1. **Configuración de ICP recomendada: los defaults del nodo** (config B:
   `remove_plane: true`, `plane_dist_thresh: 0.004`, `min_fitness: 0.45`,
   `max_rmse: 0.006`), es decir, quitar del YAML los cuatro valores de
   diagnóstico. En val, B sube el recall con profundidad real y caja GT del
   55.6 % al 88.0 % (+32.4 pp, IC 95 % +23.9 … +40.7) y no empeora ningún
   escenario. La mejora se replica en la base antigua (+39.2 pp).
2. **Cifras finales en test (222 instancias, una sola corrida por escenario):**
   - ICP con profundidad real y caja GT: **84.2 %** (78.9–88.4).
   - ICP con profundidad real y caja YOLO: **80.2 %** (74.4–84.9).
   - ICP con SGBM y caja GT: **23.0 %** (17.9–28.9).
   - ICP con SGBM y caja YOLO, la cadena completa: **23.0 %** (17.9–28.9).
   - D2CO: **0.5–0.9 %**, con umbral o sin él y con caja GT o de YOLO.
3. **Dónde falla la cadena completa de ICP** (escalera de lo ideal a lo real
   en test):
   - Reconstrucción estéreo: 56.8 % de las instancias. Aciertan con
     profundidad real y fallan con SGBM, sobre todo con el eje mal estimado
     (≥ 10°).
   - Detección: 15.8 %, todas con visibilidad < 0.75.
   - Segmentación/registro: 5.4 %.
   - Caja de YOLO: 4.1 %.
   Con profundidad real, ICP ya no es el problema: ADD-S mediana 2.1 mm, eje
   0.1°.
4. **D2CO no resuelve la profundidad.** De las poses que publica con umbral,
   solo el 0.9 % tiene el error radial por debajo del umbral, y el 86.1 %
   sitúa el tubo más lejos de lo real (error radial mediano +446 mm). Quitar el
   umbral sube la cobertura del 48.6 % al 82.9 % sin cambiar el recall: el
   umbral no es lo que limita a D2CO.
5. **Frente a la evaluación anterior (test, sin muestras *hard*):**
   - La cadena completa pasa del 3.3 % al 23.0 % (+19.7 pp, IC +13.6 …
     +25.9), un cambio real.
   - ICP con profundidad real y caja GT pasa del 51.2 % al 84.2 % (+33.0 pp,
     real).
   Separando efectos en train+val:
   - La mejora con profundidad real es de **configuración**; los datos no la
     cambian.
   - La mejora con caja de YOLO es del **detector nuevo** (+32.3 pp) y de la
     configuración (+34.9 pp).
   - Los **datos nuevos empeoran SGBM** (−6.1 pp con config A y −8.6 pp con
     B, los dos significativos), por la iluminación más oscura: medido
     directamente sobre la profundidad, con la misma luz que v1 la calidad
     del estéreo es igual. No es una regresión de código. Lo compensa el cambio a `wls_lambda 500`
     (+9.1 pp), por eso SGBM con caja GT no cambia de forma significativa
     (+4.5 pp, IC −3.2 … +12.1).
   - D2CO no cambia.
6. **Umbral de ADD-S: 18.69 mm** (0.1 × 0.186898 m, de `bop/models_info.json`,
   idéntico en v1 y v3), el mismo en las 36 corridas de pose (cada `.log`
   lo imprime: «Umbral ADD-S : 18.69 mm»). El 18.7 mm del README
   anterior es ese valor redondeado. El 18.83 mm viene de
   `eval_d2co_dataset.py` (rama `d2co`), que aproxima el diámetro con la
   diagonal de la caja envolvente de los puntos muestreados en lugar de la
   distancia máxima entre dos puntos (la definición BOP).

## 1. Qué se ha evaluado y con qué

| Elemento | Detalle |
|---|---|
| Banco de pruebas | `eval_pose_dataset.py`, `compare_pose_results.py` y `cache_yolo_detections.py` sin modificar. Su última modificación (16-09 00:45) es anterior a las corridas anteriores (16-09 01:08 en adelante, `anterior/tiempos_corridas.csv`): es **el mismo código** que produjo los resultados anteriores. |
| ICP | `icp_lib` (`parol6_vision.icp_utils`, el mismo código que el nodo, sin ROS). Configuraciones congeladas en `configs/` (ver abajo). |
| Estéreo | `stereo_utils.StereoMatcher` (el código del nodo), un solo par por escena. `wls_lambda` 8000 (default en el momento de la evaluación, fijado ahora con `configs/stereo_wls8000.yaml`) o 500 (`configs/stereo_wls500.yaml`, el nuevo default). |
| D2CO | Nodo real `d2co_pose_estimator_node` por servicio. Build ya instalado (`install/parol6_d2co`, 16-09), enlazado contra `~/parol6_ws/.d2co_deps/d2co_branch/parol6_d2co`; he comprobado que ese árbol es idéntico byte a byte a `d2co` HEAD (45743e7) en el nodo, el YAML, el CMake, el launch, el STL y las fuentes de la biblioteca. No se ha cambiado de rama ni recompilado. Parámetros: YAML de la rama + `tracking_max_age_s:=0.0`, `output_frame:=""`, `broadcast_tf:=false`, `publish_debug_overlay:=false`, `process_rate:=0.1`, `detection_max_age_s:=30.0`; «sin umbral» = `max_avg_dist:=100.0`. Dominio ROS aislado (`ROS_DOMAIN_ID=63`, `ROS_LOCALHOST_ONLY=1`). |
| Detector | **Nuevo**: `models/best.pt` actual (sha256 `a07ff5df…`, entrenado el 24-10-2026 sobre `parol6_synth_dataset_v3`, commit 57091b2). **Antiguo**: `best.pt` de 15ac096 (sha256 `bf9a8353…`), solo para las corridas cruzadas. Al regenerar con él la caché sobre v1 sale idéntica a la que usó la evaluación anterior (600/600 imágenes con las mismas cajas): aquella evaluación usó estos pesos. |
| Asociación YOLO → instancia | La del banco de pruebas: la detección de mayor IoU con `bbox_visib`, con IoU ≥ 0.3. |
| Métricas | Cobertura, recall ADD-S < 18.69 mm **sobre el total** (sin pose = fallo), ADD-S y errores radial, lateral y de eje (estos, solo sobre poses publicadas), latencia. IC 95 % de Wilson para el recall. Para comparar corridas sobre las mismas instancias, diferencia emparejada con IC por *bootstrap* por imagen y McNemar exacto. Para comparar bases distintas, IC de Newcombe. El IC de Wilson coincide con el de *bootstrap* por imagen con diferencias de ±0.5 pp en ICP (`analysis/ic_wilson_vs_bootstrap.md`). |

Configuraciones de ICP (`configs/`):

| Nombre | `remove_plane` | `plane_dist_thresh` | `min_fitness` | `max_rmse` | Origen |
|---|---|---|---|---|---|
| **A** | false | 0.006 | 0.20 | 0.005 | `config/icp_pose_params.yaml` en el momento de la evaluación (copia exacta; ya sustituido por B) |
| **B** | true | 0.004 | 0.45 | 0.006 | defaults declarados en `icp_pose_estimator_node.py` |
| `norej_*` | según la corrida | según la corrida | 0 | 1 m | solo en train: sin rechazo, para reconstruir cualquier umbral después |

El resto de parámetros es igual en A y B (y coincide con los defaults del
nodo). El rechazo por `min_fitness`/`max_rmse` es el último paso del pipeline
y no altera la pose. Por eso, en train, A y B (y cualquier otro umbral) se
reconstruyen exactamente aplicando el rechazo a posteriori sobre las columnas
`fitness`/`rmse_mm` de las corridas sin rechazo (`analysis/stats.py`,
`apply_reject`). En val y test se usan las configuraciones reales.

Protocolo: ajuste en **train**, confirmación en **val** con un criterio fijado
antes de ver val (cabecera de `analysis/fase2_val.py`) y **test una sola vez**
por escenario con la configuración elegida. D2CO no tiene nada que ajustar
aquí: solo se ha corrido en test.

## 2. Resultados finales en test

| Corrida | n | Cobertura % | Recall ADD-S % (IC 95 %) | Recall aceptadas % | ADD-S med mm | \|radial\| med mm | Sesgo radial medio mm | Lateral med mm | Eje med ° | Latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| ICP, profundidad real, caja GT | 222 | 98.6 | **84.2** (78.9–88.4) | 85.4 | 2.1 | 0.6 | −6.9 | 2.3 | 0.1 | 371 |
| ICP, SGBM, caja GT | 222 | 86.5 | **23.0** (17.9–28.9) | 26.6 | 43.5 | 39.0 | −1.5 | 40.0 | 22.9 | 452 |
| ICP, profundidad real, caja YOLO | 222 | 83.8 | **80.2** (74.4–84.9) | 95.7 | 2.1 | 0.5 | −0.7 | 2.2 | 0.1 | 355 |
| **ICP, SGBM, caja YOLO (cadena completa)** | 222 | 73.4 | **23.0** (17.9–28.9) | 31.3 | 37.5 | 31.7 | +8.9 | 37.1 | 15.6 | 476 |
| D2CO con umbral, caja GT | 222 | 48.6 | **0.5** (0.1–2.5) | 0.9 | 430.8 | 446.3 | +798.9 | 37.0 | 45.1 | 1994 |
| D2CO sin umbral, caja GT | 222 | 82.9 | **0.9** (0.2–3.2) | 1.1 | 311.8 | 342.9 | +542.2 | 35.0 | 41.0 | 2577 |
| D2CO con umbral, caja YOLO | 222 | 38.3 | **0.5** (0.1–2.5) | 1.2 | 254.7 | 269.3 | +563.7 | 30.0 | 45.4 | 2519 |

ICP: config B + `wls_lambda` 500 en las corridas SGBM. Fuente:
`test/*.csv` → `analysis/test.md`. Las latencias son la mediana por instancia
con las corridas en serie. En ICP incluyen la nube (SGBM o deproyección) y el
registro, pero no YOLO; en D2CO son la duración de la llamada al servicio.

Estratificado (recall % sobre el total, IC 95 % entre paréntesis):

| Estrato | n | ICP prof. real + GT | ICP SGBM + GT | ICP prof. real + YOLO | ICP SGBM + YOLO | D2CO con umbral + GT | D2CO sin umbral + GT |
|---|---|---|---|---|---|---|---|
| visib 0.00–0.25 | 19 | 26.3 (12–49) | 0.0 (0–17) | 21.1 (9–43) | 0.0 (0–17) | 0.0 | 0.0 |
| visib 0.25–0.50 | 37 | 59.5 (43–74) | 8.1 (3–21) | 43.2 (29–59) | 5.4 (1–18) | 0.0 | 0.0 |
| visib 0.50–0.75 | 20 | 80.0 (58–92) | 15.0 (5–36) | 75.0 (53–89) | 20.0 (8–42) | 0.0 | 0.0 |
| visib 0.75–0.90 | 12 | 100.0 (76–100) | 8.3 (1–35) | 100.0 (76–100) | 16.7 (5–45) | 0.0 | 0.0 |
| visib 0.90–1.00 | 134 | 98.5 (95–100) | 32.8 (25–41) | 97.8 (94–99) | 32.1 (25–40) | 0.7 | 1.5 |
| modo buried | 54 | 87.0 (76–94) | 20.4 (12–33) | 85.2 (73–92) | 22.2 (13–35) | 1.9 | 1.9 |
| modo grounded | 168 | 83.3 (77–88) | 23.8 (18–31) | 78.6 (72–84) | 23.2 (17–30) | 0.0 | 0.6 |
| brazo home | 54 | 92.6 (82–97) | 24.1 (15–37) | 92.6 (82–97) | 25.9 (16–39) | 0.0 | 0.0 |
| brazo random | 168 | 81.5 (75–87) | 22.6 (17–30) | 76.2 (69–82) | 22.0 (16–29) | 0.6 | 1.2 |

Fuente: `analysis/test.md`. Con profundidad real y el tubo visible al
≥ 75 %, ICP acierta prácticamente siempre (98–100 %). Lo que queda son
oclusiones fuertes (≤ 50 % visible) y, en menor medida, el brazo en posición
aleatoria (81.5 % frente a 92.6 % en home, con IC que se solapan). Enterrado
frente a apoyado no marca diferencia.

Comparaciones emparejadas en test (mismas 222 instancias; `analysis/test.md`):

| Paso | Diferencia (pp) | IC 95 % | Solo 1.ª / solo 2.ª | p McNemar |
|---|---|---|---|---|
| ICP: profundidad real → SGBM (caja GT) | −61.3 | −68.0 … −54.6 | 138 / 2 | 1.4e-38 |
| ICP: caja GT → YOLO (profundidad real) | −4.1 | −7.6 … −0.5 | 13 / 4 | 0.049 |
| ICP: caja GT → YOLO (SGBM) | 0.0 | −4.2 … +4.1 | 11 / 11 | 1 |
| ICP SGBM + GT → D2CO con umbral + GT | −22.5 | −28.2 … −16.9 | 51 / 1 | 2.4e-14 |
| ICP cadena completa → D2CO con umbral + YOLO | −22.5 | −28.3 … −17.0 | 51 / 1 | 2.4e-14 |

## 3. Dónde falla cada solución y por qué

### ICP: escalera de lo ideal a lo real (test)

Para cada instancia de la cadena completa (SGBM + YOLO) se busca el primer
peldaño que falla: (1) YOLO no la detecta; (2) falla incluso con profundidad
real y caja GT (segmentación o registro); (3) acierta con profundidad real
pero falla con SGBM y caja GT (reconstrucción estéreo); (4) acierta con SGBM y
caja GT pero falla con caja YOLO.

| Causa | v1 test (anterior: A, detector antiguo, `wls_lambda` 8000) | v3 test (B, detector nuevo, `wls_lambda` 500) |
|---|---|---|
| 1 Detección (YOLO no encuentra el tubo) | 197 (79.8 %) | 35 (15.8 %) |
| 2 Segmentación/registro | 26 (10.5 %) | 12 (5.4 %) |
| 3 Reconstrucción estéreo | 20 (8.1 %) | **126 (56.8 %)** |
| 4 Caja de YOLO | 1 (0.4 %) | 9 (4.1 %) |
| 5 Acierto en todos los peldaños | 3 (1.2 %) | 40 (18.0 %) |
| No monótonas (aciertan en la cadena completa pero fallan en un peldaño ideal) | 5 | 11 |
| n | 247 | 222 |

Fuente: `analysis/atribucion_test.md` y `analysis/atribucion_anterior_test.md`
(`analysis/atribucion.py`). Las no monótonas están contadas en el peldaño en
que fallan; 40 + 11 = 51 aciertos = 23.0 %.

Qué hay detrás de cada causa:

- **Reconstrucción estéreo (el cuello de botella actual).** De las 126
  instancias que pierde el estéreo, 92 terminan con el eje mal estimado
  (≥ 10°), 18 sin pose, 13 con el eje bien pero con |radial| ≥ umbral y 3 por
  otras causas.
  Es decir, la nube SGBM de un tubo de ~7–8 px de ancho a media resolución
  (13–17 px a 1024 px, `docs/pose_evaluation.md`) no conserva la forma del
  cilindro, y PCA + ICP orientan mal el modelo. El sesgo de profundidad ya no
  es el problema principal: en las poses con el eje bien (n = 58), el error
  radial mediano es −0.5 mm, pero solo el 58.6 % queda por debajo del umbral
  (`analysis/atribucion_test.md`). Con `wls_lambda` 8000 había además un sesgo
  radial medio de +22.9 mm en val y +24.9 mm en train (config B;
  `analysis/val.md`, `analysis/train.md`).
- **Detección.** Las 35 instancias no detectadas tienen todas visibilidad
  < 0.75: 13 por debajo de 0.25, 18 entre 0.25 y 0.50 y 4 entre 0.50 y 0.75
  (`analysis/atribucion_test.md`). El detector nuevo encuentra el 99.8 % de
  los tubos visibles al ≥ 90 % (`analysis/deteccion_v3.md`).
- **Segmentación/registro.** Con la config B quedan 12 instancias (5.4 %),
  concentradas en visibilidad < 0.75 (10 de 12). En los fallos con pose, el
  *cluster* tras la limpieza tiene una mediana de 175 puntos, frente a 468 en
  los aciertos: apunta a que queda poco tubo visible, más que a terreno colado
  (`analysis/atribucion_test.md`). Con la config A era distinto: 26 fallos,
  todos con el tubo visible al ≥ 90 %, 19 de ellos con el eje ≥ 10°
  (`analysis/atribucion_anterior_test.md`). Que `remove_plane` los arregle
  indica que era el suelo que quedaba en el *cluster* lo que desviaba la PCA.
- **Caja de YOLO.** Con profundidad real, la caja de YOLO cuesta 4.1 pp frente
  a la caja GT (IC −7.6 … −0.5, en el límite de la significación); con SGBM,
  0.0 pp.

### D2CO: ambigüedad de profundidad

| Resultado (test, caja GT) | Con umbral | Sin umbral |
|---|---|---|
| Acierto | 1 (0.5 %) | 2 (0.9 %) |
| Sin candidato: todas las hipótesis refinadas, fuera de la banda de profundidad | 44 (19.8 %) | 36 (16.2 %) |
| Rechazada por `avg_dist` > 4.5 | 70 (31.5 %) | 2 (0.9 %) |
| Publicada y falla en profundidad (sola o con otros errores) | 107 (48.2 %) | 182 (82.0 %) |
| Publicada y falla sin error de profundidad (lateral o eje) | 0 | 0 |

Fuente: `analysis/atribucion_test.md`. «Falla en profundidad» suma las filas
«aceptada, falla: profundidad…» (|radial| ≥ 18.69 mm). Con caja de YOLO se
añaden 35 instancias sin detección; de sus 85 poses publicadas, 81 fallan en
profundidad, 3 solo en lateral o eje y 1 acierta.

- **Profundidad.** De las 108 poses que publica con umbral, solo el 0.9 %
  tiene |radial| < 18.69 mm, y el 86.1 % sobreestima la profundidad (error
  radial mediano +446 mm). La posición lateral queda por debajo del umbral
  solo en el 24.1 %, y el eje a menos de 10° en el 5.6 %. Ese error de eje es
  3D: incluye la inclinación fuera del plano, así que no contradice que el eje
  en el plano de imagen sí se ancle (`docs/pose_evaluation.md` §4).
- **Mecanismo** (`analysis/d2co_mecanismo.md`, de los *logs* del nodo):
  - **Corrección por anchura de raíles.** Se aplicó en 82 de los 263 ciclos
    de servicio de la corrida sin umbral y multiplicó la traslación por una
    mediana de k = 1.30; en el 86.6 % de los casos aleja el tubo.
  - **Anchura medida.** Es de 6.15 px de mediana, ya restado
    `width_bias_px` = 1 px. Con fx = 401.84 px a 512 px (`cam_K` de
    `scene_camera.json`: fx = 803.68 a 1024 px), Ø = 0.0208 m (parámetro
    `tube_diameter`) y la distancia mediana del dataset (1.232 m), la anchura
    esperada sería ~6.8 px. Una anchura subestimada se traduce directamente en
    más profundidad (z = fx·Ø/w).
  - **Banda de profundidad.** Descartó poses refinadas en 82 ciclos.
  - **Bordes poco alargados.** En 73 ciclos («blob»), los bordes no eran lo
    bastante alargados para tener eje PCA ni banda de profundidad.
- **El umbral no limita la precisión.** Quitarlo publica 184 poses en lugar
  de 108 y deja el recall en 0.9 %. Solo cambia cuántas poses erróneas se
  publican.
- **El nodo no es determinista entre corridas.** Cuando las dos corridas
  publican pose para la misma instancia, la pose es idéntica (104 instancias,
  |Δz| máximo de 0 mm). Pero 15 instancias cambian de resultado entre corridas
  (`analysis/test.md`). Encaja con el atajo de seguimiento: solo se refina la
  mejor hipótesis si puntúa < `tracking_fast_dist`, salvo cada
  `full_search_every` = 5 ciclos, y ese contador avanza también con el
  temporizador, que sigue disparando el pipeline entre llamadas al servicio.
  No lo he aislado, pero es la única dependencia del estado entre llamadas que
  queda con `tracking_max_age_s = 0`.

## 4. Configuración recomendada

### ICP: config B (defaults del nodo)

Cambio propuesto en `config/icp_pose_params.yaml` (no aplicado):
`remove_plane: true`, `plane_dist_thresh: 0.004`, `min_fitness: 0.45`,
`max_rmse: 0.006`. Son exactamente los defaults que declara el nodo.

Validación (142 instancias; criterio fijado antes de ver val: B gana si la
diferencia emparejada es ≥ 0 en los cuatro escenarios y su IC excluye el 0 con
profundidad real y caja GT):

| Escenario (val) | A % | B % | B − A (pp) | IC 95 % | p McNemar |
|---|---|---|---|---|---|
| Profundidad real, caja GT | 55.6 | 88.0 | **+32.4** | +23.9 … +40.7 | 1e-11 |
| SGBM, caja GT | 16.2 | 16.2 | 0.0 | −5.1 … +5.1 | 1 |
| Profundidad real, caja YOLO | 46.5 | 83.1 | **+36.6** | +29.0 … +44.4 | 4.4e-16 |
| SGBM, caja YOLO | 16.9 | 17.6 | +0.7 | −5.3 … +6.5 | 1 |

Fuente: `val/*.csv` → `analysis/val.md`.

- **De dónde viene la mejora** (train, emparejado, `analysis/train.md`):
  `remove_plane` aporta +33.7 pp (IC +28.5 … +38.5);
  `plane_dist_thresh` 4 mm con `max_rmse` 6 mm, +3.0 pp más (+1.3 … +5.1,
  p = 0.003); y `min_fitness` 0.45 frente a 0.20, −0.3 pp (una sola
  instancia). La mejora se replica en la base antigua: +39.2 pp en v1
  train+val (`analysis/comparacion.md`).
- **Por qué `min_fitness` 0.45 y no 0.20.** El recall total es el mismo
  (curva de rechazo plana hasta 0.60 en train), pero 0.45 rechaza algunas
  poses más y sube ligeramente el recall de las aceptadas (con profundidad
  real y caja GT, 87.2 % frente a 86.0 %; `analysis/train.md`). Para agarrar,
  una pose rechazada es más barata que una errónea. A partir de 0.75 empieza a
  perder aciertos.
- **Con SGBM la config da igual** (0.0 y +0.7 pp en val), así que el cambio no
  tiene coste en la cadena real.

### Estéreo: `wls_lambda` 500 (secundaria, con menos certeza)

Fijado igual que antes: se adopta si mejora a B en los dos escenarios SGBM de
val y su IC excluye el 0 con caja GT.

| Escenario | `wls_lambda` 8000 % | 500 % | Diferencia (pp) | IC 95 % | p McNemar |
|---|---|---|---|---|---|
| val, SGBM + caja GT | 16.2 | 31.0 | **+14.8** | +6.4 … +23.1 | 0.00075 |
| val, SGBM + caja YOLO | 17.6 | 23.9 | +6.3 | −0.7 … +13.5 | 0.15 |
| train+val, SGBM + caja GT | 14.6 | 23.7 | **+9.1** | +5.1 … +13.0 | 1.6e-05 |

Fuente: `analysis/val.md`, `analysis/comparacion.md`. Aplicado en
`config/stereo_pointcloud_params.yaml` y en `stereo_utils.DEFAULT_PARAMS`
(las corridas de 8000 de este informe lo fijan con `configs/stereo_wls8000.yaml`).
La mejora es clara con caja GT; en la cadena completa la dirección es la
misma, pero no alcanza significación con 142 instancias. En test no hay
corrida con 8000 (test se usa una vez, con la configuración elegida), así que
su efecto en test no está medido directamente. La evaluación anterior ya
apuntaba en la misma dirección (15.8 % → 19.8 % en su barrido sobre test;
`anterior/README.md`).

Aun con las dos recomendaciones, la cadena completa se queda en el 23 % porque
el 56.8 % de las instancias se pierden en la reconstrucción estéreo. El
siguiente trabajo útil está ahí, no en ICP ni en el detector.

### D2CO

Ninguna configuración del umbral lo hace utilizable para agarrar (0.5–0.9 %).
Tiene sentido mantenerlo como referencia monocular, con el umbral (publica
menos poses erróneas: 108 frente a 184).

## 5. Comparación con la evaluación anterior

Las escenas son distintas: solo hay comparación agregada, con IC de Newcombe
para la diferencia de proporciones independientes. Cambian a la vez tres
cosas de los datos (escenas, iluminación aleatoria, ausencia de muestras
*hard*), el detector y la configuración. Para separarlas he usado train+val de
ambas bases, que no tienen muestras *hard* en ninguna de las dos. Fijando dos
de los tres factores se mide el tercero:

- **Datos (v1 → v3):** independiente, misma config y mismo detector.
- **Detector y configuración:** emparejados sobre las mismas instancias de v3.
- **Réplica en v1:** config B y detector nuevo sobre la base antigua
  (`cross/v1_trval_*`), y detector antiguo sobre v3 (`cross/v3_trval_*`).

### 5.1 Test: antes y ahora

| Escenario | v1 test (todas) | v1 test sin *hard* | v3 test | v3 − v1 sin *hard* (pp) | IC 95 % | ¿Cambio real? |
|---|---|---|---|---|---|---|
| ICP, profundidad real, caja GT | 44.5 (n=247) | 51.2 (n=211) | 84.2 (n=222) | +33.0 | +24.5 … +41.0 | sí |
| ICP, SGBM, caja GT | 16.6 | 18.5 | 23.0 | +4.5 | −3.2 … +12.1 | no (compatible con ruido) |
| ICP, profundidad real, caja YOLO | 8.9 | 10.4 | 80.2 | +69.8 | +62.2 … +75.6 | sí |
| ICP, SGBM, caja YOLO | 3.2 | 3.3 | 23.0 | +19.7 | +13.6 … +25.9 | sí |
| D2CO con umbral, caja GT | 0.8 | 0.9 | 0.5 | −0.5 | −3.0 … +1.7 | no |
| D2CO sin umbral, caja GT | 0.8 | 0.9 | 0.9 | −0.0 | −2.6 … +2.4 | no |

Cobertura de D2CO con umbral: 54.5 % frente a 48.6 % (−5.9 pp,
IC −15.1 … +3.5, no significativo). Cobertura de ICP SGBM + caja GT: 95.7 %
frente a 86.5 % (−9.2 pp, IC −14.7 … −3.9, real). Fuente:
`analysis/comparacion.md` §1 (v1: `anterior/*.csv` filtrado a
`split == test`; *hard* según `meta.jsonl` de v1).

**Muestras *hard*.** Son 36 instancias de v1 test (visibilidad mediana
0.032); ICP con profundidad real y caja GT acierta 2. Explican los 6.7 pp que
separan el 44.5 % del 51.2 % en v1 test. En v3 no hay muestras *hard*: en
test, ninguna instancia baja de 0.10 de visibilidad, y 19 están entre 0.10 y
0.25. En ellas, ICP con profundidad real acierta el 26.3 %.

### 5.2 Separación de efectos (train+val)

| Escenario | Datos v1 → v3 (pp, IC) | Detector antiguo → nuevo (pp, IC) | Config A → B (pp, IC) | Estéreo 8000 → 500 (pp, IC) |
|---|---|---|---|---|
| ICP prof. real + GT | +3.6 (−2.6 … +9.8) con A; −0.3 (−4.7 … +4.2) con B | — | **+35.3** (+31.2 … +39.4) en v3; **+39.2** (+34.7 … +43.8) en v1 | — |
| ICP SGBM + GT | **−6.1** (−10.7 … −1.5) con A; **−8.6** (−13.5 … −3.8) con B | — | +1.2 (−1.6 … +4.0) en v3; +3.7 (+0.8 … +6.8) en v1 | **+9.1** (+5.1 … +13.0) |
| ICP prof. real + YOLO | +1.8 (−2.2 … +5.9) con A y detector antiguo; −1.9 (−6.8 … +3.1) con B y detector nuevo | **+32.3** (+28.0 … +36.8) | **+34.9** (+30.9 … +39.1) | — |
| ICP SGBM + YOLO | −0.1 (−2.9 … +2.8) con A y detector antiguo; **−5.9** (−11.0 … −0.8) con B y detector nuevo | **+9.9** (+6.9 … +12.9) | +3.0 (+0.0 … +5.9), p = 0.07 | (val: +6.3, −0.7 … +13.5) |

Fuente: `analysis/comparacion.md` §2 (v1 A: `anterior/*.csv`
filtrado a train+val; v3 A/B: `train/` con rechazo a posteriori + `val/`;
réplicas: `cross/`). En negrita, los IC que excluyen el 0.

Lectura:

- **Configuración.** Es la causa de la mejora con profundidad real (+35 a
  +39 pp en las dos bases). Los datos no la cambian.
- **Detector.** Es la causa de la mejora con caja de YOLO.
  - A nivel de detección, sobre v3 el nuevo recupera el 85.2 % de los tubos
    (IoU ≥ 0.3) con 23 falsos positivos en 600 imágenes. El antiguo: 32.4 %
    y 416 falsos positivos.
  - Sobre v1 la mejora es la misma (83.7 % frente a 26.9 %), así que es del
    detector y no de los datos.
  - El antiguo no detectaba casi nada con visibilidad < 0.75 (0–5 %); el
    nuevo, del 28.6 % al 86.0 % según el tramo.
  Fuente: `analysis/deteccion_v3.md`, `analysis/deteccion_v1.md`.
- **Datos.** El único efecto real está en **SGBM**: v3 da entre 6 y 9 pp
  menos que v1 con la misma configuración, y menos cobertura. La candidata
  natural es la **iluminación**. v1 tenía la difusa fija del mundo,
  g = 0.9 (`mars_optimized.world`: 0.9 0.85 0.8), y en v3 el 78 % de las
  escenas está más oscuro. Pero los datos dentro de v3 solo lo apoyan a
  medias:

  | Serie (recall %, IC 95 %) | g < 0.691 | 0.691–0.853 | g > 0.853 |
  |---|---|---|---|
  | ICP B, SGBM 8000 + YOLO (train+val) | 11.4 (7–17) | 18.9 (14–26) | 23.3 (18–30) |
  | ICP A, SGBM 8000 + GT (train+val) | 9.6 (6–15) | 15.4 (11–22) | 15.1 (11–21) |
  | ICP B, SGBM 8000 + GT (train+val) | 12.0 (8–18) | 18.3 (13–25) | 13.4 (9–19) |
  | ICP B, SGBM 500 + GT (train+val) | 21.7 (16–29) | 23.1 (17–30) | 26.2 (20–33) |
  | Detección YOLO nuevo, IoU ≥ 0.3 (todas las particiones) | 83.6 (78–88) | 83.6 (78–88) | 88.4 (84–92) |
  | ICP B, profundidad real + GT (todas las particiones) | 84.0 (79–88) | 84.4 (79–88) | 86.3 (81–90) |

  A través del recall de ICP la tendencia no es concluyente: no es monótona
  en todas las series y los IC se solapan (`analysis/comparacion.md` §3).
  Por eso medí la reconstrucción directamente, sin ICP: el error de
  profundidad de SGBM frente a la profundidad real en los píxeles visibles
  del tubo (`analysis/calidad_sgbm.md`; instancias con visibilidad ≥ 0.5,
  `wls_lambda` 8000).

  | Grupo | n | Brillo medio | \|error z\| mediano mm | Celdas con \|error\| < 10 mm % |
  |---|---|---|---|---|
  | v1 | 543 | 129.7 | 10.1 | 49.5 |
  | v3, g < 0.690 | 179 | 98.7 | 14.4 | 35.8 |
  | v3, 0.690–0.852 | 191 | 114.9 | 11.5 | 44.3 |
  | v3, 0.852–0.9 | 77 | 127.1 | 11.8 | 45.4 |
  | v3, g ≥ 0.9 (tanta luz como v1 o más) | 113 | 135.1 | 10.7 | 48.5 |

  La calidad cae de forma monótona con la luz (Spearman ρ = +0.19,
  IC +0.11 … +0.27). Con g ≥ 0.9, v3 es indistinguible de v1 (+0.7 pp de
  celdas buenas, IC −3.8 … +5.5); con poca luz es 10.6 pp peor
  (IC −14.4 … −6.7). La cobertura del tubo no cambia (≈ 91 %): SGBM sigue
  devolviendo puntos, pero peores. **Conclusión: el empeoramiento de SGBM en v3
  se debe a la iluminación**, no al código. La corrida antigua repetida hoy
  sale igual: 18.1 % frente a 18.5 %, misma cobertura, McNemar p = 0.38
  (`repro/reproduccion.md`). El código de estéreo, ICP y el banco de pruebas
  y las librerías (OpenCV 4.9.0, Open3D 0.19.0) no han cambiado desde
  entonces. La iluminación no afecta de forma apreciable a la detección ni a
  ICP con profundidad real.
- **Efecto neto de SGBM + caja GT.** El empeoramiento por los datos (−8.6 pp
  con B) y la mejora de `wls_lambda` 500 (+9.1 pp) se compensan. Por eso el
  test sale «igual» que antes (+4.5 pp, no significativo): no es que no haya
  pasado nada.
- **D2CO no cambia:** sigue en torno al 1 %, con el mismo patrón de error.

### 5.3 Qué cambia también en la lectura de las cifras anteriores

- **Latencias anteriores no comparables.** Las seis corridas principales de
  la evaluación anterior se ejecutaron a la vez (las cuatro de ICP empezaron
  a las 01:08 del 16-09; las dos de D2CO a las 17:43;
  `anterior/tiempos_corridas.csv`). Por eso sus latencias (5.9–15.8 s en ICP
  y 3.1–3.4 s en D2CO) están infladas. Las comparables son las de sus
  barridos, que sí corrieron en serie (230–839 ms). Fuente de ambas:
  `anterior/README.md`. Ahora todo ha corrido en serie: 355–476 ms en ICP y
  2.0–2.6 s en D2CO.
- **Barridos anteriores sobre test.** La «mejor» configuración de los
  barridos anteriores se eligió mirando test; aquí se ha elegido en train y
  val. El resultado coincide: `remove_plane` es lo que importa.
- **Tabla principal anterior con todas las particiones.** La del README
  anterior mezclaba las tres particiones (729 instancias) y las muestras
  *hard*. Las cifras de v1 de este informe están separadas por partición.

## 6. Advertencias

- **Asociación YOLO optimista en imágenes con varios tubos.** El banco de
  pruebas asigna a cada instancia la detección con mejor IoU (un oráculo de
  asociación). Los nodos, en cambio, usan una sola detección por imagen, la
  de mayor confianza. En test, esa detección cae sobre un tubo en el 86.1 %
  de las imágenes (`analysis/deteccion_v3.md`, columna «top-1»).
- **Determinismo.**
  - ICP no es determinista bit a bit, ni siquiera sin RANSAC. Entre dos
    corridas iguales, |ΔADD-S| mediana de 0.02 mm (p90 1.8 mm) y un cambio de
    acierto en 60 instancias. Con `remove_plane` (corrida completa de train),
    84.4 % frente a 84.1 %. Son décimas de punto, frente a ±4–5 pp de IC
    muestral (`analysis/train.md`).
  - D2CO: ver §3.
  - El banco de pruebas descartó por *stamp* las poses de la muestra anterior
    (las publica el temporizador del nodo) y reintentó: ~15 % de las
    instancias de D2CO necesitó un segundo intento (columnas `attempts` y
    `stale` de `test/d2co_*.csv`). No hubo *timeouts*.
- **Tamaño de muestra.** En test, los IC de ICP rondan ±5 pp, y los estratos
  pequeños (visibilidad < 0.25: n = 19; 0.75–0.90: n = 12) solo dan órdenes de
  magnitud.
- **Detector y base de evaluación de la misma simulación.** El detector se
  entrenó en `parol6_synth_dataset_v3`, otra base generada en el mismo mundo
  y con los mismos *pools* de tubos y rocas. Sus cifras miden la
  generalización dentro de la simulación, no la transferencia a imágenes
  reales.
- **Nube estéreo de un solo par**, como en la evaluación anterior: en Gazebo
  no hay ruido, y la fusión multi-frame del nodo (`accumulate_frames`) no
  aporta información (`stereo_utils.py`).

## Reproducción

Desde `results/pose_eval_v3/`, en este orden y en serie (para que las
latencias sean válidas). Cada script deja `<corrida>.csv`,
`<corrida>.summary.json` y `<corrida>.log` (los `.log` y los *overlays* están
en el `.gitignore`).

```bash
cd ~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3

# 0. Cachés YOLO (detector nuevo y antiguo × v3 y v1). El antiguo se extrae de git:
#    git show 15ac096:ros_parol/src/parol6_vision/models/best.pt > /tmp/best_15ac096.pt
#    (ejecutado con OLD_WEIGHTS apuntando a una copia en la carpeta temporal de la sesión)
OLD_WEIGHTS=/tmp/best_15ac096.pt ./fase0_yolo.sh                 # ~2 min en GPU

# 1. Ajuste en train (sin rechazo; 12 corridas)                  # ~40 min
./fase1_train.sh
#    comprobación de repetibilidad sin RANSAC (60 instancias)
ros2 run parol6_vision eval_pose_dataset.py --dataset ~/parol6_pose_eval_dataset_v3 --split train --limit 60 \
    --estimator icp_lib --cloud depth_gt --bbox gt_visib --icp-params configs/icp_norej_rpF.yaml \
    --csv train/repcheck_icp_depthgt_gt_rpF_60.csv

# 2. Confirmación en val: A y B × 4 escenarios + B con wls_lambda 500 (añadida tras train)   # ~12 min
./fase2_val.sh

# 3. Test, una vez: ICP con B + wls500, y D2CO (nodo real, necesita DISPLAY o xvfb-run)
./fase3_icp_test.sh                                              # ~8 min
./fase3_d2co_test.sh                                             # ~45 min

# 4. Corridas cruzadas en train+val (separación de efectos)      # ~25 min
./fase4_cruzada.sh

# Análisis (sin ROS)
python3 analysis/yolo_metrics.py --dataset ~/parol6_pose_eval_dataset_v3 --cache yolo/v3_detnuevo.json yolo/v3_detantiguo.json \
    --labels "nuevo (a07ff5)" "antiguo (15ac096)" --md analysis/deteccion_v3.md
python3 analysis/yolo_metrics.py --dataset ~/parol6_pose_eval_dataset --cache yolo/v1_detnuevo.json yolo/v1_detantiguo.json \
    --labels "nuevo (a07ff5)" "antiguo (15ac096)" --md analysis/deteccion_v1.md
python3 analysis/fase1_train.py        # → analysis/train.md
python3 analysis/fase2_val.py          # → analysis/val.md (aplica el criterio de decisión)
python3 analysis/fase3_test.py         # → analysis/test.md
python3 analysis/atribucion.py --split test --title "test v3 (config B, estéreo wls_lambda 500, detector nuevo)" \
    --files test/icp_depthgt_gt_B_defaults_nodo.csv test/icp_sgbmwls500_gt_B_defaults_nodo.csv \
            test/icp_depthgt_yolo_B_defaults_nodo.csv test/icp_sgbmwls500_yolo_B_defaults_nodo.csv \
    --md analysis/atribucion_test.md
O=anterior; python3 analysis/atribucion.py --only-split test \
    --title "evaluación anterior (base antigua, config A, detector antiguo), split test" \
    --files $O/icp_depthgt_gtbbox.csv $O/icp_sgbm_gtbbox.csv $O/icp_depthgt_yolo.csv $O/icp_sgbm_yolo.csv \
    --d2co $O/d2co_asis_gtbbox.csv $O/d2co_relaxed_gtbbox.csv --md analysis/atribucion_anterior_test.md
python3 analysis/comparacion.py        # → analysis/comparacion.md
python3 analysis/d2co_logs.py          # → analysis/d2co_mecanismo.md (lee test/*.node.log)

# ¿Regresión de código en SGBM? Corrida antigua repetida hoy (~6 min) + calidad directa del estéreo (~8 min)
ros2 run parol6_vision eval_pose_dataset.py --dataset ~/parol6_pose_eval_dataset --estimator icp_lib --cloud sgbm \
    --bbox gt_visib --icp-params configs/icp_A_yaml_actual.yaml --stereo-params configs/stereo_wls8000.yaml \
    --csv repro/v1_todas_icp_sgbm_gt_A_hoy.csv
python3 analysis/calidad_sgbm.py --stereo-params configs/stereo_wls8000.yaml   # → analysis/calidad_sgbm.{csv,md} (entorno ROS cargado)
```

`repro/reproduccion.md` se generó con un bloque en línea (`stats.paired` y
`stats.summary` sobre el CSV antiguo y el de hoy).

`analysis/ic_wilson_vs_bootstrap.md` se generó con un bloque en línea que
llama a `stats.summary` sobre cada `test/*.csv`.

## Ficheros

| Carpeta | Contenido |
|---|---|
| `configs/` | YAML de ICP (A, B, `norej_*`) y estéreo (`wls_lambda` 500) congelados |
| `yolo/` | Cachés de detecciones: `v3_detnuevo.json` (la usada en todas las corridas con caja YOLO), `v3_detantiguo.json`, `v1_detnuevo.json`, `v1_detantiguo.json` |
| `train/` | 11 corridas sin rechazo y 2 de repetibilidad (`icp_depthgt_gt_rpT004_rep`, `repcheck_icp_depthgt_gt_rpF_60`) |
| `val/` | 10 corridas con su rechazo real |
| `test/` | 4 de ICP y 3 de D2CO (más `*.node.log` del nodo y *overlays*, no versionados) |
| `cross/` | 6 corridas cruzadas en train+val (v3 con detector antiguo; v1 con config B) |
| `analysis/` | Scripts (`stats.py` y los de cada fase) y las tablas `.md` que generan; `calidad_sgbm.{py,csv,md}`: error de SGBM en los píxeles del tubo |
| `anterior/` | Las 6 corridas de la evaluación anterior que usa la comparación, y sus horas de inicio/fin (`README.md` propio; el resto, archivado fuera del repositorio) |
| `repro/` | La corrida antigua `icp_sgbm_gtbbox` (v1, config A) repetida con el código de hoy y su comparación |
