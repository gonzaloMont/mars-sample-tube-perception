# Generador de dataset sintético v2 (PAROL6 · NavCam estéreo · mars2020tube)

`nodos/synthetic_dataset_generator_v2.py` genera, **en una sola pasada**, un
dataset con doble propósito sobre la simulación marciana de Gazebo Classic:

1. **Detección 2D (YOLO)** para entrenar `best.pt`, el modelo que consume
   `yolo26n_camera_subscriber.py` desde `/navcam/left/navcam_left/image_raw`.
2. **Ground truth de pose 6D estilo BOP** para **evaluar** los estimadores
   existentes (`icp_pose_estimator_node.py`, `linemod_pose_estimator_node.py`)
   **estratificando por oclusión** (`visib_fract`), y para un posible futuro
   estimador aprendido (keypoints incluidos).

> El generador antiguo `nodos/synthetic_dataset_generator.py` está obsoleto y
> se conserva sin tocar. Este v2 no comparte código con él.

## Cómo lanzar

```bash
# prueba de humo (¡ejecuta después el visualizador antes de generar 1000!)
ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=20 \
    output_dir:=$HOME/parol6_synth_smoke

# generación completa
ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=1000 \
    output_dir:=$HOME/parol6_synth_dataset seed:=0

# QA
ros2 run parol6_vision visualize_dataset_samples.py --dataset_dir $HOME/parol6_synth_dataset --n 20 --check
ros2 run parol6_vision dataset_stats.py --dataset_dir $HOME/parol6_synth_dataset
```

El launch es ligero (sin MoveIt/RViz): Gazebo + robot + controladores +
NavCam + el nodo generador con `use_sim_time:=True` (obligatorio: las TF y
los stamps de imagen viven en tiempo de simulación). Mundo por defecto:
`mars_optimized` (terreno real recortado centrado en el origen). `mars.world`
original tiene el terreno desplazado ~(20,20,−2.4) y **no hay suelo bajo la
cámara**; si lo quieres usar, reubica robot/cámara con los argumentos.
Al terminar el generador, el launch se apaga solo (`shutdown_when_done:=false`
para dejar la simulación viva e inspeccionarla).

## Dataset repartido entre VARIOS mundos

Variar el fondo (relieve y textura del terreno) es de las pocas variaciones
de dominio útiles que quedan en monocromo. Como Gazebo Classic no puede
cambiar de mundo en caliente, `generate_dataset_multiworld.py` encadena una
ejecución del launch por mundo, todas **sobre el mismo `output_dir`** en modo
append (`start_index`): ids globales contiguos, JSON BOP acumulados y campo
`world` por muestra en `meta.jsonl`.

```bash
ros2 run parol6_vision generate_dataset_multiworld.py \
    --worlds mars_optimized=0.6,mars_bumpy=0.4 \
    --num_samples 1000 --output_dir $HOME/parol6_synth_dataset --seed 0

# interrumpido a medias → continúa lo que falte (misma spec de --worlds):
ros2 run parol6_vision generate_dataset_multiworld.py \
    --worlds mars_optimized=0.6,mars_bumpy=0.4 \
    --num_samples 1000 --output_dir $HOME/parol6_synth_dataset --resume
```

Detalles: las fracciones se normalizan (reparto por mayor resto, total
exacto); cada mundo usa `seed + 1000·k`; **cada tanda hace su propio split
70/20/10**, así que la proporción de mundos se conserva dentro de
train/val/test; `dataset_stats.py` imprime el recuento por mundo. Si una
tanda se queda sin progreso (`--stall_timeout`, 300 s) se aborta limpiamente
y puedes relanzar con `--resume`. También puedes usarlo manualmente: lanzar
`dataset_generation.launch.py` con `start_index:=N` añade muestras a un
dataset existente sin pisar nada.

⚠ **No puede haber otra simulación Gazebo abierta**: el orquestador lo
comprueba y aborta (con un gzserver ajeno vivo, el nuevo muere por conflicto
de puerto y el generador podría engancharse en silencio al mundo equivocado).
Cierra tu simulación o pasa `--kill_existing_gazebo`.

Todos los parámetros (fracciones de colocación, oclusión, nº de rocas,
umbrales, topics…) son parámetros ROS del nodo; ver el bloque
`_declare_params()` del fuente.

## Topics nuevos de profundidad GT (`gt_depth`)

`parol6_description/urdf/navcam_stereo.urdf.xacro` incorpora **dos sensores
depth adicionales**, co-localizados píxel a píxel con cada NavCam:

```
/navcam/left/navcam_left/gt_depth/depth/image_raw    (32FC1, metros)
/navcam/left/navcam_left/gt_depth/camera_info
/navcam/right/navcam_right/gt_depth/depth/image_raw  (32FC1, metros)
/navcam/right/navcam_right/gt_depth/camera_info
```

Son **solo para generación de dataset**. El pipeline de inferencia real no
los usa (su nube sale de `stereo_pointcloud_node.py`), y el topic que ICP
tiene configurado (`/navcam/left/navcam_left/depth/image_raw`) queda intacto.

## Cómo se construye cada muestra

1. **Pausa de física** y recolocación de la escena con `set_entity_state`
   sobre **pools de entidades estáticas reutilizables** (6 tubos con albedos
   distintos para simular polvo depositado + 14 rocas procedurales),
   aparcadas detrás de la cámara cuando no se usan.
2. **Colocación del tubo por retro-proyección**: al arrancar se cachea el
   depth GT de la escena vacía y se backproyecta a puntos 3D world por píxel.
   Elegir un píxel = elegir dónde aparece el tubo y a qué distancia → permite
   **estratificar** distancia (4 bins) y posición en imagen (rejilla 3×3).
   Modos: apoyado (~70 %), flotante estático (~15 %, refuerza invarianza de
   escala) y **semienterrado** (~15 %, hasta el 60 % del diámetro bajo la
   malla real del terreno → línea de corte irregular natural).
3. **Rocas**: icoesferas subdivididas con ruido radial (Open3D), material
   regolito gris apagado; incluye **rocas alargadas** (silueta de tubo) como
   *hard negatives*. Parte se colocan **cruzando la línea de visión** para
   una oclusión controlada (~50 % de muestras con oclusión, ~12 % fuerte).
4. **Brazo**: configuración articular aleatoria dentro de los límites reales
   del URDF, comandada vía `arm_group_controller` (JointTrajectory). El
   asentamiento se detecta por **estabilidad de posición** en `/joint_states`
   (las velocidades reportadas por la interfaz de posición de
   gazebo_ros2_control son ruido). **Nunca** `set_model_configuration`: sin
   par de sujeción el brazo colapsa por gravedad antes de capturar. Tras el
   asentamiento se verifica por TF que ningún link quede bajo el suelo, y con
   el depth se descarta el intento si el brazo tapa la lente.
5. **Captura**: se exige imagen RGB y depth GT con stamp **posterior** al
   asentamiento (la escena es estática: cualquier frame posterior es válido).
   Cámara left o right al azar, **una sola imagen por escena** (evita pares
   estéreo casi idénticos y fugas entre splits).
6. **GT consciente de oclusión**: la malla escalada del tubo se raycastea
   (Open3D/Embree, CPU) en el frame óptico → z-buffer “solo-objeto”. Un píxel
   es visible si `z_obj ≤ z_escena + ε` (ε = 5 mm, costura del enterrado).
   De ahí salen `bbox_visib` (el que usa YOLO), `bbox_obj` amodal,
   `visib_fract` y la visibilidad de cada keypoint. La pose 6D se compone
   como `T_optical_tube = inv(T_world_optical) · T_world_tube`, con la pose
   del tubo **releída** de Gazebo y la TF de cámara consultada por TF2.

## Formatos de salida

```
output_dir/
  images/{train,val,test}/img_XXXXXX.png   # gris replicado a 3 canales
  labels/{train,val,test}/img_XXXXXX.txt   # YOLO "0 cx cy w h"; vacío = negativo
  data.yaml                                # nc:1, names:['mars2020tube']
  bop/scene_camera.json                    # cam_K, depth_scale, cam_R_w2c, cam_t_w2c
  bop/scene_gt.json                        # {obj_id, cam_R_m2c, cam_t_m2c} POR INSTANCIA
  bop/scene_gt_info.json                   # bbox_obj, bbox_visib, visib_fract,
                                           # px_count_*, keypoints_2d, keypoints_visib
  bop/models_info.json                     # extents, diameter, symmetries_continuous
  meta.jsonl                               # 1 línea/muestra: cam, dist, modos, brazo…
  split_map.json                           # im_id → split
```

Decisiones que hay que conocer:

* **Unidades: METROS** en todo el GT (a diferencia del BOP oficial en mm),
  para comparar directamente con la salida de ICP/Open3D/ROS. `depth_scale=1`.
* **Imágenes mono guardadas como 3 canales replicados** (gris→BGR idéntico):
  Ultralytics y los pesos preentrenados funcionan sin cambios.
* **Simetría**: el tubo es ~de revolución; `models_info.json` declara
  `symmetries_continuous` sobre su eje Z. **Métrica recomendada: ADD-S.**
  Los errores de *flip* (180°) correlacionarán con muestras donde la cabeza
  (+Z) esté ocluida: compruébalo con `keypoints_visib[1]` (`head_+z`).
* **Keypoints** (frame `tube_link`): centro, extremo +Z (cabeza compleja),
  extremo −Z (tapa), y 3 puntos del borde de la cabeza a 0/120/240°. Con la
  simetría continua, el roll de los puntos del borde es arbitrario.
* **Política de visibilidad**: la etiqueta YOLO usa `bbox_visib` y solo se
  emite con `visib_fract ≥ 0.10`. Con `save_hard_samples:=true` (defecto) las
  muestras cuyo tubo principal queda por debajo se guardan marcadas `hard` y
  se enrutan **solo a test**, sin etiqueta: en la evaluación de YOLO cuentan
  como falsos positivos y en la de pose son fallos casi imposibles. Con
  `save_hard_samples:=false` (argumento del launch) se descartan y la muestra
  se regenera con el mismo split, así que los splits salen exactos.
* **Split 70/20/10 determinista** por semilla. Hay una imagen por escena;
  si en el futuro se capturasen ambos ojos, los dos deben ir al mismo split.
* **Evaluación de ICP/LINEMOD por oclusión**: agrupa las muestras por
  `visib_fract` (p. ej. bins [1.0–0.9], [0.9–0.7], [0.7–0.5], [0.5–0.3],
  [0.3–0.1]) y calcula ADD-S por bin: eso responde "¿hasta qué oclusión
  aguanta cada algoritmo?".

## Modo evaluación de pose (v2.1): par estéreo + depth GT + huella agarrable

Para evaluar ICP/D2CO hace falta lo que el dataset YOLO no tiene: el ojo
derecho, la profundidad GT y tubos donde el brazo llega. El generador lo
añade con dos parámetros (expuestos como argumentos del launch):

```bash
ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=600 \
    output_dir:=$HOME/parol6_pose_eval_dataset_v3 seed:=42 \
    stereo_capture:=true placement_mode:=footprint \
    frac_grounded:=0.85 frac_buried:=0.15 frac_floating:=0.0 \
    negative_fraction:=0.0 split_ratios:=0.5,0.2,0.3 \
    randomize_lighting:=true save_hard_samples:=false
```

* `stereo_capture:=true`: la cámara de referencia es SIEMPRE la izquierda
  (GT 6D en `navcam_left_optical_frame`); se guardan además
  `images_right/` (par estéreo) y `depth/` (depth GT izquierdo, PNG uint16
  en mm, `scene_camera.depth_scale = 0.001`), y `scene_camera.json` lleva
  `cam_K_right`, `cam_R_l2r`, `cam_t_l2r` (`p_right = R·p_left + t`).
* `placement_mode:=footprint`: los tubos se colocan en las celdas de
  `config/grasp_footprint_cells.json` (huella de agarre top-down calculada
  por `scripts/compute_grasp_footprint.py`, 1.12–1.34 m de la cámara), con
  la altura del terreno cacheado; se ignora `min_dist_from_base`. El brazo
  aleatorio se pre-filtra por cinemática directa para respetar una holgura
  con los tubos (`arm_clearance_m`), y una fracción de muestras usa la pose
  home (`arm_home_fraction`); `meta.jsonl` guarda `cells`, `terrain_z` y
  `arm_mode`.

Protocolo completo de evaluación: `docs/pose_evaluation.md`.

## Lo que NO hace (a propósito)

* No hornea polvo en suspensión, ruido de sensor ni *random erasing*: eso va
  en el **aumento de entrenamiento** (ver `train_yolo_navcam.md`), donde no
  invalida el GT 3D.
* No randomiza color (cámaras L8 monocromas): la variación fotométrica del
  tubo entra por el albedo del pool y, opcionalmente,
  `randomize_lighting:=true` (intensidad del sol; la dirección no se puede
  cambiar en Classic de forma fiable y se omite).

## Iluminación aleatoria (`randomize_lighting:=true`)

Por escena, la difusa del sol pasa a `[g, g·U(0.92,1), g·U(0.82,0.95)]` con
`g ~ U(light_intensity_min, light_intensity_max)` (defecto 0.55–1.0); la
atenuación se reenvía con los valores del mundo. El valor aplicado queda en
`meta.jsonl` (`light_diffuse`; `null` = luz fija). Requisitos y límites:

* El mundo debe cargar `libgazebo_ros_properties.so` (lo hace
  `mars_optimized.world`), que ofrece `/get_light_properties` y
  `/set_light_properties`. Si se pide luz aleatoria y el servicio no está, el
  generador aborta en vez de seguir con luz fija.
* El máximo es 1.0: `ignition::math::Color` interpreta componentes > 1 como
  escala 0–255 (1.5 → 0.006, escena casi negra). El generador lo rechaza.
* Efecto medido (misma escena, NavCam izquierda, media de la imagen):
  difusa 0 → 29, 0.55 → 91, 0.78 → 115, 1.0 → 138. La ambiental del mundo
  (0.5, `Ka` del terreno 0.4) solo aporta ~29 niveles.

## Dataset YOLO con zona de trabajo (dos tandas)

El generador admite un `placement_mode` por ejecución. Para que el detector
vea también tubos en la huella agarrable, se generan dos tandas sobre el
mismo `output_dir` (modo append): 70 % `image_bins` + 30 % `footprint`. Cada
tanda hace su propio split 70/20/10, así que la proporción se conserva en
train/val/test; semillas distintas para no repetir los pools de tubos/rocas.
En `footprint` sin `stereo_capture` la cámara es izquierda o derecha al azar.

```bash
ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=700 \
    output_dir:=$HOME/parol6_synth_dataset_v3 seed:=0 start_index:=0 \
    placement_mode:=image_bins randomize_lighting:=true save_hard_samples:=false
ros2 launch parol6_gazebo dataset_generation.launch.py num_samples:=300 \
    output_dir:=$HOME/parol6_synth_dataset_v3 seed:=1 start_index:=700 \
    placement_mode:=footprint randomize_lighting:=true save_hard_samples:=false
```

`meta.jsonl` guarda el modo de cada muestra en `placement`.
* No mueve la NavCam ni la base del brazo (geometría mástil↔brazo fija, como
  el sistema real).
