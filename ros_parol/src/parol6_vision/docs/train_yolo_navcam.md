# Entrenamiento de YOLO para la NavCam (dataset sintético v2)

Receta de aumento para entrenar `best.pt` con el dataset de
`synthetic_dataset_generator_v2.py`. El generador produce imágenes **limpias
salvo oclusión geométrica real**; el "polvo" en espacio de imagen, el ruido
de sensor y el tapado sintético se aplican AQUÍ, en entrenamiento. Ninguna de
estas transformaciones altera la pose 3D, así que el GT BOP sigue siendo
válido para evaluar ICP/LINEMOD con las mismas imágenes.

## Claves del dataset

- Imágenes **monocromas** (NavCam L8) guardadas como PNG de **3 canales
  replicados**: Ultralytics y los pesos preentrenados (RGB) funcionan sin
  cambios.
- La etiqueta es el **bbox visible** (no amodal): con oclusión parcial el
  modelo aprende a detectar la parte que se ve, que es lo que verá en vuelo.
- `labels/*.txt` vacíos = imágenes de fondo (negativas): déjalas, reducen
  falsos positivos.

## Comando de entrenamiento recomendado (Ultralytics)

```bash
yolo detect train \
    model=yolo26n.pt \
    data=$HOME/parol6_synth_dataset/data.yaml \
    epochs=150 imgsz=1024 batch=8 \
    hsv_h=0.0 hsv_s=0.0 hsv_v=0.25 \
    degrees=10.0 translate=0.10 scale=0.6 shear=2.0 perspective=0.0005 \
    fliplr=0.5 flipud=0.2 \
    mosaic=1.0 mixup=0.10 \
    erasing=0.4 \
    seed=0
```

Racional, parámetro a parámetro:

| Parámetro | Valor | Por qué |
|---|---|---|
| `hsv_h`, `hsv_s` | **0.0** | Imágenes en grises: tono y saturación no existen. Activarlos solo quema presupuesto de aumento. **Nunca** apliques color/HSV a este dataset. |
| `hsv_v` | 0.25 | Sobre 3 canales replicados equivale a *brillo*: simula exposición/iluminancia variable del sol marciano. |
| `scale` | **0.6** (agresivo) | Invarianza de escala: complementa los bins de distancia del generador sin necesidad de abusar de tubos flotantes. |
| `degrees`, `shear`, `perspective` | pequeños | La NavCam está fija en el mástil: rotaciones grandes de imagen no ocurren; valores pequeños regularizan sin alejarse del dominio. |
| `translate` | 0.10 | Variación de encuadre. |
| `fliplr`/`flipud` | 0.5 / 0.2 | El tubo no tiene quiralidad en imagen; el terreno tampoco a estos efectos. |
| `mosaic` | 1.0 | Más contextos y escalas por batch. |
| `mixup` | 0.10 (ligero) | Robustez; más alto emborrona el dominio mono. |
| `erasing` | **0.4** | *Random erasing*: tapa parches manteniendo la caja → robustez a oclusión no vista (polvo pegado a la lente, gripper). La oclusión *real* (rocas/enterrado) ya viene horneada con bbox correcto; esto cubre el resto. |

## Polvo, ruido y velo: pipeline Albumentations (grises)

Para ruido de sensor y "velo de polvo" (bajada de contraste + bruma) usa
Albumentations en un dataloader propio o como preproceso offline de una
copia del split de train:

```python
import albumentations as A

# Apropiado para imágenes monocromas replicadas a 3 canales.
dust_and_sensor = A.Compose([
    # Velo de polvo: baja contraste y sube ligeramente el nivel de negro
    A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.25),
                               contrast_limit=(-0.35, 0.05), p=0.7),
    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.30,
                alpha_coef=0.06, p=0.25),          # bruma/scattering
    # Sensor
    A.GaussNoise(var_limit=(5.0, 40.0), per_channel=False, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.25),
    A.MotionBlur(blur_limit=3, p=0.10),
    # Oclusión sintética extra en imagen (la caja se mantiene a propósito)
    A.CoarseDropout(max_holes=6, max_height=48, max_width=48,
                    fill_value=110, p=0.30),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                            min_visibility=0.0))
```

Notas:
- `per_channel=False` en el ruido: los 3 canales deben seguir siendo
  idénticos (la cámara real es mono).
- Nada de `HueSaturationValue`, `RGBShift`, `ChannelShuffle`, etc.
- `CoarseDropout`/`erasing` mantienen el bbox completo aunque tapen parte
  del objeto: es deliberado (robustez a oclusión), igual que hace YOLO con
  `erasing`.

## Validación

- Entrena con `train`, ajusta con `val`. El split `test` incluye además las
  muestras **hard** (visib < 0.10, solo-test): úsalo para el informe final y
  para estresar también ICP/LINEMOD.
- Tras entrenar, prueba en el sim en vivo:
  `ros2 run parol6_vision yolo26n_camera_subscriber.py` con el nuevo
  `best.pt` apuntando a la NavCam izquierda.
