## Modelos necesarios

Descargar y colocar en esta carpeta:

- `yoloe-26l-seg.pt` — se descarga automáticamente con ultralytics la primera vez
  que se ejecuta el nodo, o manualmente:
  pip install ultralytics && yolo export model=yoloe-26l-seg.pt

- CLIP (`mobileclip2_b.ts`) — se descarga automáticamente cuando se llama a set_classes() por primera vez.
  Requiere: pip install git+https://github.com/openai/CLIP.git