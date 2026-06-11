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
  emite con `visib_fract ≥ 0.10`. Muestras cuyo tubo principal queda por
  debajo se guardan marcadas `hard` y se enrutan **solo a test** (estrés de
  los estimadores); ponlas a discard con `save_hard_samples:=false`.
* **Split 70/20/10 determinista** por semilla. Hay una imagen por escena;
  si en el futuro se capturasen ambos ojos, los dos deben ir al mismo split.
* **Evaluación de ICP/LINEMOD por oclusión**: agrupa las muestras por
  `visib_fract` (p. ej. bins [1.0–0.9], [0.9–0.7], [0.7–0.5], [0.5–0.3],
  [0.3–0.1]) y calcula ADD-S por bin: eso responde "¿hasta qué oclusión
  aguanta cada algoritmo?".

## Lo que NO hace (a propósito)

* No hornea polvo en suspensión, ruido de sensor ni *random erasing*: eso va
  en el **aumento de entrenamiento** (ver `train_yolo_navcam.md`), donde no
  invalida el GT 3D.
* No randomiza color (cámaras L8 monocromas): la variación fotométrica del
  tubo entra por el albedo del pool y, opcionalmente,
  `randomize_lighting:=true` (intensidad del sol; la dirección no se puede
  cambiar en Classic de forma fiable y se omite).
* No mueve la NavCam ni la base del brazo (geometría mástil↔brazo fija, como
  el sistema real).
