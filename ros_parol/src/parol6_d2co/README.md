# parol6_d2co

Estimación de pose 6D del tubo mediante **D2CO** (*Direct Directional Chamfer
Optimization*, Imperoli & Pretto, ICVS 2015) integrada en el pipeline de
visión de PAROL6.

A diferencia del pipeline ICP (`parol6_vision/icp_pose_estimator_node.py`),
D2CO es **edge-based y monocular**: registra el modelo CAD del tubo contra los
bordes de UNA imagen (NavCam izquierda), sin necesidad de nube de puntos
estéreo. Publica el **mismo contrato** que el nodo ICP, por lo que el resto
del pipeline (`tube_grasp_pose_node`, `grasp_executor`, …) funciona sin
cambios.

```
 /navcam/left/.../image_raw ─┐
 /navcam/left/.../camera_info├─► d2co_pose_estimator_node ─► /vision/tube_pose (PoseStamped)
 /vision/detections (YOLO) ──┘            │                  /vision/d2co_overlay (debug)
                                          └─► TF: <output_frame> → tube_link_estimated
```

## Arquitectura del nodo

1. **YOLO** (`yolo26n_camera_subscriber.py`, reutilizado de `parol6_vision`)
   publica el bbox del tubo en `/vision/detections`.
2. El bbox (+margen) se convierte en **máscara del detector de bordes**
   (Canny/LSD): el tensor de distancias solo contiene bordes del entorno del
   tubo. Es el equivalente funcional del ROI, sin los problemas de
   reinicialización OpenGL (ver más abajo).
3. Se calcula el **tensor direccional de distancias** (DT3V del paper:
   `num_directions` canales, suavizado con peso `tensor_lambda`).
4. Se generan **hipótesis iniciales** desde el bbox: rayo de retroproyección
   del centro, profundidad estimada por el tamaño aparente
   (`z ≈ fx·L/px`, acotada a `[z_min, z_max]` y barrida con `z_offsets`) y
   orientaciones del eje del cilindro deducidas del aspecto del bbox
   (+inclinaciones `tilt_angles_deg`). Si hay una pose reciente aceptada
   (tracking), se usa como primera semilla.
5. Las hipótesis se puntúan con `getAvgDistance` (barato), las `refine_top_k`
   mejores se refinan con `DirectionalChamferMatching::refinePosition`
   (Levenberg-Marquardt de Ceres, `optimizer_iterations` iteraciones) y se
   acepta la mejor si su distancia media queda bajo `max_avg_dist`.
6. Publicación idéntica al nodo ICP: `PoseStamped` en el frame óptico o
   transformada a `output_frame` vía TF2, TF `tube_link_estimated`, overlay de
   depuración y `Pose` legacy opcional.

### Por qué no se usa el ROI nativo de D2CO

`cv_ext::PinholeCameraModel` soporta ROI (`setRegionOfInterest` +
`enableRegionOfInterest`: las proyecciones pasan a coordenadas del recorte y
la intrínseca se ajusta internamente — no hay que modificar K a mano). Pero
`RasterObjectModel3D::computeRaster()` crea su ventana OpenGL (GLFW) **una
única vez** y congela ahí el modelo de cámara: con un bbox que cambia cada
frame habría que recrear el contexto GL continuamente (fuga de ventanas).
Por eso este nodo procesa la imagen completa **reescalada**
(`image_scale`, def. 0.5 → 512×512, la intrínseca la escala
`setSizeScaleFactor` de forma consistente) y aplica el bbox como **máscara de
bordes**, que tiene el mismo efecto práctico sobre el coste y la robustez.

## Compilación

### 1. Librería externa libd2co (una sola vez)

La librería vive en `parol6_ws/src/alberto_pretto-d2co-75e4ad9dfe90` (con
`COLCON_IGNORE`; no es un paquete ROS). Ya incluye en `externals/`:

- `cv_ext` — dependencia de Bitbucket hoy desaparecida (¡no borrar!).
- `ceres_1_14/install` — Ceres 1.14 local (el 2.0 del sistema es incompatible).

y los parches de compatibilidad para Ubuntu 22.04 / OpenCV 4 / C++14
(`AutoDiff→NumericDiff` en `chamfer_matching.cpp`, `CV_DIST_L2→cv::DIST_L2`,
etc.).

```bash
cd ~/parol6_ws/src/alberto_pretto-d2co-75e4ad9dfe90

# cv_ext (estática)
cd externals/cv_ext && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 ..
make -j4

# libd2co.so
cd ../../.. && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
      -DCeres_DIR=$PWD/../externals/ceres_1_14/install/lib/cmake/Ceres ..
make -j4 d2co
```

> **Importante:** se fija `OpenCV_DIR` al OpenCV 4.5.4 del sistema (el mismo
> que usa `cv_bridge`). En esta máquina hay un OpenCV 4.14 en `/usr/local`
> que `find_package` encontraría primero; mezclar dos OpenCV en el mismo
> proceso corrompe el ABI.

### 2. Modelo CAD pre-escalado (una sola vez)

D2CO (OpenMesh) solo admite escala uniforme y el `mars2020tube.stl` original
necesita la escala anisótropa `(0.022939, 0.022939, 0.033212)`:

```bash
python3 scripts/make_scaled_tube_stl.py
# -> models/mars2020tube_scaled.stl (0.186 m de largo, Ø0.021 m, centrado)
```

### 3. Paquete ROS 2

```bash
cd ~/parol6_ws
colcon build --packages-select parol6_d2co
```

## Uso

```bash
# Gazebo + MoveIt con la NavCam:
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py camera:=navcam

# Pipeline D2CO (YOLO + estimador):
ros2 launch parol6_d2co d2co_pose.launch.py

# Activar YOLO la primera vez:
ros2 topic pub --once /orchestrator/vision_activate std_msgs/String '{data: activate}'

# Forzar una estimación puntual:
ros2 service call /d2co_pose_estimator_node/estimate_now std_srvs/srv/Trigger

# Pipeline completo de agarre con D2CO como estimador
# (sin el ICP: enable_vision:=false y D2CO aparte):
ros2 launch parol6_apps tube_grasp_pipeline.launch.py enable_vision:=false &
ros2 launch parol6_d2co d2co_pose.launch.py
```

El nodo necesita un servidor gráfico (la rasterización del modelo usa una
ventana GLFW oculta). En sesiones sin display: `xvfb-run -a ros2 launch …`.

## Parámetros principales (`config/d2co_pose_params.yaml`)

| Parámetro | Default | Significado |
|---|---|---|
| `image_scale` | 0.5 | Escala de la imagen de trabajo (512² desde 1024²). |
| `num_directions` | 60 | Canales de orientación del tensor (valor del paper). |
| `tensor_lambda` | 6.0 | Peso del término direccional λ (valor del paper). |
| `optimizer_iterations` | 50 | Iteraciones máx. de LM (Ceres). |
| `edge_detector` | CANNY | `CANNY` o `LSD`. |
| `bbox_margin_px` | 24 | Margen de la máscara alrededor del bbox YOLO. |
| `z_min/z_max/z_factors` | 0.3/2.0/×{0.8–1.8} | Barrido de profundidad (multiplicativo sobre z_est). |
| `tilt_angles_deg` | ±40° | Inclinaciones fuera de plano del eje del tubo. |
| `refine_top_k` | 3 | Hipótesis refinadas con LM por ciclo. |
| `max_avg_dist` | 4.5 | Umbral de aceptación (ver sección de validación). |
| `process_rate` | 1.0 Hz | Frecuencia del pipeline. |

## Verificación rápida en RViz

- `Image` → `/vision/d2co_overlay`: puntos amarillos = modelo proyectado en la
  pose estimada; rectángulo rojo = bbox YOLO. Deben solaparse con el tubo.
- `TF` → frame `tube_link_estimated`.
- `ros2 topic echo /vision/tube_pose` para la pose numérica.

## Validación en simulación (jun-2026, mundo mars_optimized)

Verdad-terreno vía `gz model`; tubo estático; bbox de YOLO real cacheada y
re-publicada (ver abajo). Dos escenas:

| Escena | Posición | Eje | Ciclo |
|---|---|---|---|
| Zona clara, tubo a 1.51 m | **53.5 mm** | ~17–22° | ~1.5 s (tracking) |
| Zona oscura (sin tapas visibles), 1.48 m | **79.9 mm** | ~21° | ~3.5 s (corrección) |

Mecanismos calibrados en esta validación (todos con parámetro):
- **Semillas por bordes**: centroide+PCA de la mayor componente conexa del
  Canny enmascarado (los bbox YOLO llegan holgados y descentrados; score 0.32–0.45).
- **Banda física de profundidad**: z ≤ z_est = fx·L/l_px (el escorzo solo acorta).
- **Corrección por anchura de raíles** (z = fx·Ø/w_px, inmune al tilt), aplicada
  SOLO si avg_dist > `width_correction_gate` (ajuste pobre = tapas sin contraste
  = deslizamiento por el valle de profundidad). Con buen ajuste el optimizador
  es más preciso que la medida (±0.5 px ≈ ±10 % z).
- `max_avg_dist 4.5`: la DT saturada a 30 px suma ~3 por rasgos ausentes sin
  implicar desalineación.

### Ejecutar la validación con pocos recursos
YOLO congela el equipo si corre continuo. Estrategia validada:
1. Arrancar YOLO, capturar UNA detección y matarlo:
   `python3 ~/parol6_ws/.d2co_validation/bbox_cache_replay.py capture`
2. Re-publicarla a 1 Hz (coste ~0): `... bbox_cache_replay.py replay`
3. Matar `gzclient` (las cámaras renderizan en gzserver).
4. Comparar con verdad-terreno: `... d2co_sim_validate.py 60`.

Límite conocido: la inclinación fuera de plano del eje queda en ~17–22° (poco
observable en monocular con tapas débiles); la posición es el dato robusto.
