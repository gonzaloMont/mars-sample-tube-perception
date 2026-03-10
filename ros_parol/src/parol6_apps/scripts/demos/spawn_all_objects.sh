#!/usr/bin/env bash
set -euo pipefail

# Spawnea todos los objetos simples en posiciones distintas
# dentro del rango XY de [-0.4, 0.4].

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPAWN_SCRIPT="$SCRIPT_DIR/spawn_object.sh"

if [[ ! -x "$SPAWN_SCRIPT" ]]; then
  echo "ERROR: No se encuentra ejecutable $SPAWN_SCRIPT"
  echo "Haz: chmod +x $SPAWN_SCRIPT"
  exit 1
fi

echo "Spawneando objetos en Gazebo..."

# Borra entidades previas para permitir re-ejecutar el script sin errores
cleanup_entity() {
  local name="$1"
  ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{name: '$name'}" >/dev/null 2>&1 || true
}

cleanup_entity tube_red_1
cleanup_entity tube_green_1
cleanup_entity tube_blue_1
cleanup_entity box_small_1
cleanup_entity box_rect_1
cleanup_entity sphere_small_1
cleanup_entity disk_flat_1

# modelo         x      y      z      R   P       Y     entidad
"$SPAWN_SCRIPT" tube         0.28  -0.22  0.03   0   1.5708  0    tube_red_1 &
"$SPAWN_SCRIPT" tube_green   0.38   0.00  0.03   0   1.5708  0    tube_green_1 &
"$SPAWN_SCRIPT" tube_blue    0.28   0.22  0.03   0   1.5708  0    tube_blue_1 &

"$SPAWN_SCRIPT" box_small    0.00  -0.35  0.02   0   0       0    box_small_1 &
"$SPAWN_SCRIPT" box_rect     0.32  -0.30  0.01   0   0       0    box_rect_1 &
"$SPAWN_SCRIPT" sphere_small 0.00   0.35  0.02   0   0       0    sphere_small_1 &

"$SPAWN_SCRIPT" disk_flat    0.35   0.2  0.005  0   0       0    disk_flat_1 &

# esperar a que todos los procesos de fondo terminen
wait
echo "Listo: todos los objetos spawneados"
