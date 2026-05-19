"""
parol6_vision — librería compartida del pipeline de visión del PAROL6.

Este paquete Python contiene la lógica REUTILIZABLE entre la fase offline
(generación de templates LINEMOD) y la fase online (nodo de inferencia ROS 2).

Mantener esta lógica en un único sitio es deliberado: si el escalado de la
malla, las intrínsecas de cámara o la configuración del detector divergieran
entre el render de entrenamiento y la inferencia, LINEMOD nunca produciría
un match válido.

Módulos:
    mesh_utils   — carga del .stl y escalado geométrico OBLIGATORIO del tubo.
    rendering    — muestreo de viewpoints + renderer offline (pyrender).
    linemod_db   — factory del detector cv2.linemod + I/O de la base de datos.
    pose_solver  — recuperación de la pose 6D a partir de un match LINEMOD.
"""

__all__ = ["mesh_utils", "rendering", "linemod_db", "pose_solver"]
