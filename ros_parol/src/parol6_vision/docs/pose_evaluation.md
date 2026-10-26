# Evaluación de los estimadores de pose 6D (ICP estéreo vs D2CO monocular)

Protocolo para medir, de forma comparable y reproducible, los estimadores de
pose del tubo `mars2020tube` sobre el **dataset de evaluación de pose**: las
mismas imágenes, las mismas cajas 2D y las mismas métricas para todos.

## 1. Dónde tiene sentido estimar la pose: la huella agarrable

La NavCam va fija en el mástil y el brazo tiene un alcance limitado, así que
solo hay una región del suelo donde un tubo detectado puede además agarrarse.
`scripts/compute_grasp_footprint.py` la calcula por cinemática directa
muestreada del URDF (agarre top-down con la convención de
`tube_grasp_pose_node.py`: eje Z del `gripper_base_link` vertical y centro
del tubo a `finger_offset_z` por debajo) y la proyecta con la pose real del
mástil:

| Magnitud | Valor |
|---|---|
| Radio agarrable desde `base_link` | 0.09–0.37 m (excluida la placa de la base) |
| Distancia cámara→tubo | 1.12–1.34 m, mediana 1.22 m |
| Dentro del FOV / fuera de la zona muerta del estéreo | 100 % / 99 % |
| Región en la imagen izquierda | u 173–688, v 567–923 |
| Tamaño del tubo a 1024 px | 13–17 px de ancho, 114–150 px de largo |

El resultado está en `config/grasp_footprint_cells.json` (celdas XY de 2 cm en
`base_link`) y en `docs/grasp_footprint.png`. Más lejos de eso el trabajo es
del detector (YOLO), no del estimador de pose.

> El dataset de entrenamiento de YOLO original (`~/parol6_synth_dataset`) **no**
> contiene tubos en esta región: en `image_bins` el generador impone 0.55 m
> mínimos a la base para que el brazo aleatorio no atraviese el tubo. El
> regenerado (`~/parol6_synth_dataset_v3`) añade una tanda `footprint` con el
> 30 % de las muestras (README_SYNTH_DATASET_V2.md). Para evaluar la pose sigue
> haciendo falta un set aparte (par estéreo + depth GT).

## 2. Generar el dataset de evaluación de pose

Extensión del generador v2 (`synthetic_dataset_generator_v2.py`, mundo
`mars_optimized`):

```bash
ros2 launch parol6_gazebo dataset_generation.launch.py \
    num_samples:=600 output_dir:=$HOME/parol6_pose_eval_dataset_v3 seed:=42 \
    stereo_capture:=true placement_mode:=footprint \
    frac_grounded:=0.85 frac_buried:=0.15 frac_floating:=0.0 \
    negative_fraction:=0.0 split_ratios:=0.5,0.2,0.3 \
    randomize_lighting:=true save_hard_samples:=false
```

(La versión original, `~/parol6_pose_eval_dataset`, se generó sin
`randomize_lighting` y con `save_hard_samples` por defecto: 31 muestras
`hard` sin etiqueta en test y splits 284/115/201 en vez de 300/120/180.
Los scripts de evaluación apuntan por defecto a la v3; para la original,
pasa `--dataset ~/parol6_pose_eval_dataset`.)

Resultados de la evaluación sobre la v3 (y comparación con la original):
`results/pose_eval_v3/README.md`.

Qué cambia respecto al dataset YOLO:

- **`stereo_capture`**: la cámara de referencia es SIEMPRE la izquierda. Se
  guardan `images/` (izquierda, entrada común de todos los estimadores),
  `images_right/` (par estéreo para SGBM) y `depth/` (depth GT del ojo
  izquierdo, PNG uint16 en mm; `scene_camera.depth_scale = 0.001`).
  `scene_camera.json` añade `cam_K_right`, `cam_R_l2r`, `cam_t_l2r`
  (`p_right = R·p_left + t` entre frames ópticos; baseline 0.15 m).
- **`placement_mode:=footprint`**: los tubos se colocan en la huella
  agarrable, con la altura del terreno cacheado en cada celda
  (`meta.cells`, `meta.terrain_z`). Oclusión por rocas, albedo del tubo,
  enterrado parcial y multi-tubo como en el dataset original.
- **Brazo**: aleatorio con holgura mínima respecto a los tubos comprobada por
  FK antes de mover (`arm_clearance_m`), y un 30 % de muestras en la pose
  home (`arm_home_fraction`; `meta.arm_mode`). La oclusión del tubo por el
  brazo la mide igualmente el depth GT (`visib_fract`).
- **Splits**: 50/20/30. `train`+`val` para ajustar parámetros; `test` para
  la cifra final, una sola vez.

Validación antes de usarlo (igual que el dataset YOLO):

```bash
ros2 run parol6_vision visualize_dataset_samples.py --dataset_dir ~/parol6_pose_eval_dataset_v3 --n 20 --check
ros2 run parol6_vision dataset_stats.py --dataset_dir ~/parol6_pose_eval_dataset_v3
```

## 3. Harness común: `eval_pose_dataset.py`

Una instancia = un tubo en una imagen. Para cada una se fija la bbox 2D
(`--bbox`), se ejecuta el estimador y se compara la pose publicada con el GT
BOP (frame `navcam_left_optical_frame`, metros).

| Eje | Opciones |
|---|---|
| Estimador | `icp_lib` (icp_utils directo, sin ROS, mismo yaml que el nodo), `icp_node` (nodo real por servicio), `d2co_node` (nodo real por servicio) |
| Nube (ICP) | `depth_gt` (cota superior: aísla el registro), `sgbm` (par → `stereo_utils.StereoMatcher`, el mismo código que el nodo) |
| Bbox | `gt_visib` (+ `--bbox-margin-px`, `--bbox-jitter-px`), `gt_obj`, `yolo` (caché de `cache_yolo_detections.py`, asociada por IoU) |

Métricas por instancia (CSV) y resumen: cobertura, **recall ADD-S < 0.1·d
sobre el total** (una instancia sin pose es un fallo), ADD-S, error de
traslación total / radial / lateral, error de eje plegado a [0, 90°],
estratificado por visibilidad, modo, cabeza visible, brazo y oclusión.

Secuencia recomendada:

```bash
# detector real, una vez
ros2 run parol6_vision cache_yolo_detections.py --dataset ~/parol6_pose_eval_dataset_v3

# ICP: cota superior y realista, bbox perfecta y bbox YOLO (sin ROS)
ros2 run parol6_vision eval_pose_dataset.py --estimator icp_lib --cloud depth_gt --bbox gt_visib --csv results/icp_depthgt_gt.csv
ros2 run parol6_vision eval_pose_dataset.py --estimator icp_lib --cloud sgbm     --bbox gt_visib --csv results/icp_sgbm_gt.csv
ros2 run parol6_vision eval_pose_dataset.py --estimator icp_lib --cloud sgbm     --bbox yolo     --csv results/icp_sgbm_yolo.csv

# D2CO: nodo real (rama d2co) en otra terminal, luego el harness
ros2 run parol6_d2co d2co_pose_estimator_node --ros-args -p tracking_max_age_s:=0.0 \
    -p output_frame:="" -p broadcast_tf:=false -p publish_debug_overlay:=false \
    -p process_rate:=0.1 -p detection_max_age_s:=30.0
ros2 run parol6_vision eval_pose_dataset.py --estimator d2co_node --bbox gt_visib --csv results/d2co_gt.csv

# tabla comparativa emparejada
ros2 run parol6_vision compare_pose_results.py results/*.csv --md results/tabla.md
```

`icp_node` sirve para confirmar que el cableado del nodo coincide con
`icp_lib` (mismos números salvo el muestreo aleatorio del modelo):

```bash
ros2 run parol6_vision icp_pose_estimator_node.py --ros-args \
    --params-file $(ros2 pkg prefix parol6_vision)/share/parol6_vision/config/icp_pose_params.yaml \
    -p cloud_source:=depth -p depth_scale:=0.001 -p output_frame:="" \
    -p broadcast_tf:=false -p publish_debug_clouds:=false -p process_rate:=0.1 -p detection_max_age_s:=30.0
ros2 run parol6_vision eval_pose_dataset.py --estimator icp_node --cloud depth_gt --bbox gt_visib --limit 50
```

## 4. Lo que hay que saber de D2CO antes de leer sus números

Medido con la librería real sobre imágenes del dataset (`d2co_probe`, sonda
sin ROS): la distancia chamfer direccional es unilateral y, para un cilindro
liso, su coste a lo largo del rayo es un muro para profundidades menores que
la real y una meseta casi plana para mayores. Refinando desde la pose exacta
el optimizador se marcha en el 70–84 % de los casos y termina con un coste
menor que el de la verdad. Además el umbral de aceptación del nodo (4.5)
rechaza la propia pose GT en la mitad de las instancias porque el raster del
modelo está dominado por aristas subpíxel de la cabeza. Es decir: D2CO puede
anclar la posición lateral y el eje en el plano de imagen, no la profundidad.
Se mantiene en la comparación como referencia monocular.
