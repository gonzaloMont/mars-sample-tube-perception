# Arquitectura

El workspace se divide en paquetes pequeños para que cada responsabilidad sea fácil de ubicar y mantener.

## Rol de cada paquete

### parol6_description
Contiene el modelo del robot, archivos Xacro/URDF, mallas y recursos geométricos. Es la fuente de verdad de la descripción del robot.

### parol6_gazebo
Contiene los mundos de Gazebo, modelos, parámetros de controladores y launch files de simulación.

### parol6_moveit
Contiene la configuración de MoveIt 2, ajustes de RViz y launch files de planificación.

### parol6_controllers
Contiene nodos de control, incluyendo el comportamiento del gripper virtual.

### parol6_apps
Contiene demos y scripts de orquestación para ejecutar flujos de prueba y pick-and-place.

### parol6_vision
Contiene el pipeline de cámara y detección para estimar poses objetivo.

### parol6_interfaces
Contiene mensajes ROS personalizados usados por el resto del workspace.

## Flujo principal

1. `parol6_description` define el robot.
2. `parol6_gazebo` o `parol6_moveit` levanta simulación o planificación.
3. `parol6_controllers` gestiona comportamiento de control específico.
4. `parol6_vision` estima poses desde cámara.
5. `parol6_apps` coordina demos y ciclos de pick-and-place.

## Convenciones de carpetas

- Los launch files viven en el directorio `launch/` de cada paquete.
- Los nodos ROS en Python viven en `scripts/` o `nodos/` según el paquete.
- Los recursos estáticos (mallas, mundos y modelos) se guardan junto al paquete que los utiliza.
