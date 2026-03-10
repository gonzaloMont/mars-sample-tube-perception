#!/usr/bin/env bash
set -euo pipefail

# Script para spawnear objetos simples en Gazebo.
#
# Uso:
#   ./spawn_.sh <modelo> [x y z R P Y [entidad [r g b a]]]
#
# Ejemplos:
#   ./spawn_object.sh tube
#   ./spawn_object.sh box_small 0.32 0.00 0.02 0 0 0
#   ./spawn_object.sh tube_green 0.25 0.10 0.03 0 1.5708 0 mi_tubo_verde
#   ./spawn_object.sh box_rect 0.30 0.10 0.01 0 0 0 caja_verde 0.1 0.8 0.1 1.0

if [[ "${1:-}" == "" ]]; then
  echo "Uso: $0 <modelo> [x y z R P Y [entidad [r g b a]]]"
  echo
  echo "Modelos disponibles:"
  echo "  - tube"
  echo "  - tube_green"
  echo "  - tube_blue"
  echo "  - box_small"
  echo "  - box_rect"
  echo "  - sphere_small"
  echo "  - disk_flat"
  exit 1
fi

MODEL_NAME="$1"

# Pose por defecto (cerca del área de trabajo)
X="${2:-0.32}"
Y="${3:-0.00}"
Z="${4:-0.03}"
R="${5:-0.0}"
P="${6:-0.0}"
YAW="${7:-0.0}"
ENTITY_NAME="${8:-$MODEL_NAME}"

# Color opcional (RGBA). Si se pasa, se sobrescribe ambient/diffuse en un SDF temporal.
COLOR_R="${9:-}"
COLOR_G="${10:-}"
COLOR_B="${11:-}"
COLOR_A="${12:-}"

MODELS_DIR="$(ros2 pkg prefix --share parol6_gazebo)/models"
MODEL_FILE="$MODELS_DIR/${MODEL_NAME}.sdf"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: No existe el modelo: $MODEL_FILE"
  echo "Revisa el nombre del modelo o la instalación del paquete parol6_gazebo."
  exit 2
fi

SPAWN_FILE="$MODEL_FILE"
TMP_SDF=""

if [[ -n "$COLOR_R" || -n "$COLOR_G" || -n "$COLOR_B" || -n "$COLOR_A" ]]; then
  if [[ -z "$COLOR_R" || -z "$COLOR_G" || -z "$COLOR_B" || -z "$COLOR_A" ]]; then
    echo "ERROR: Para color debes pasar los 4 valores: r g b a"
    exit 3
  fi

  TMP_SDF="$(mktemp /tmp/${MODEL_NAME}_color_XXXXXX.sdf)"
  python3 - "$MODEL_FILE" "$TMP_SDF" "$COLOR_R" "$COLOR_G" "$COLOR_B" "$COLOR_A" <<'PY'
import re
import sys

src, dst, r, g, b, a = sys.argv[1:7]
rgba = f"{r} {g} {b} {a}"

with open(src, "r", encoding="utf-8") as f:
    data = f.read()

data = re.sub(r"<ambient>[^<]*</ambient>", f"<ambient>{rgba}</ambient>", data)
data = re.sub(r"<diffuse>[^<]*</diffuse>", f"<diffuse>{rgba}</diffuse>", data)

with open(dst, "w", encoding="utf-8") as f:
    f.write(data)
PY

  SPAWN_FILE="$TMP_SDF"
  trap '[[ -n "$TMP_SDF" && -f "$TMP_SDF" ]] && rm -f "$TMP_SDF"' EXIT
fi

echo "Spawneando '$MODEL_NAME' como entidad '$ENTITY_NAME'..."
echo "Pose: x=$X y=$Y z=$Z R=$R P=$P Y=$YAW"
if [[ -n "$COLOR_R" ]]; then
  echo "Color RGBA: $COLOR_R $COLOR_G $COLOR_B $COLOR_A"
fi

ros2 run gazebo_ros spawn_entity.py \
  -file "$SPAWN_FILE" \
  -entity "$ENTITY_NAME" \
  -x "$X" -y "$Y" -z "$Z" \
  -R "$R" -P "$P" -Y "$YAW"

echo "Listo."
