import cv2
import numpy as np

print("Cargando el mapa topográfico...")
# cv2.IMREAD_UNCHANGED es VITAL para mantener los 16-bits intactos
img = cv2.imread('terreno_gazebo_final.png', cv2.IMREAD_UNCHANGED)

if img is None:
    print("Error: No se encontró la imagen. Asegúrate de estar en la carpeta de texturas.")
else:
    print("Aplicando Filtro Gaussiano para eliminar el ruido estéreo de la NASA...")
    # Kernel de 15x15: lo suficiente para borrar los picos milimétricos
    # pero manteniendo intacta la forma de las rocas y dunas.
    img_suavizada = cv2.GaussianBlur(img, (15, 15), 0)

    cv2.imwrite('terreno_gazebo_suave.png', img_suavizada)
    print("¡Listo! Se ha guardado 'terreno_gazebo_suave.png'.")