import cv2
import numpy as np
import matplotlib.pyplot as plt

#Cargando imagen original en modo RGB
img = cv2.cvtColor(cv2.imread('Vacio-42000.png'),cv2.COLOR_BGR2RGB)

blur = cv2.medianBlur(img,9)

#Mostrando imagen original y filtrada
plt.subplot(121),plt.imshow(img),plt.title('original.png')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('filtrada.png')
plt.xticks([]),plt.yticks([])
plt.show()


############################################################################

#Cargando las imagenes que se van a restar
imagen1 = cv2.imread('Vacio-42000.png',0)
imagen2 = cv2.imread('Parado-10000.png',0)
imagen3 = cv2.imread('Agachado-10000.png',0)
imagen4 = cv2.imread('CaidaTipoFin-10467.png',0)

#Función de comparación
def compara(im1,im2):

    diferencia=cv2.subtract(im1,im2)

    if not np.any(diferencia):
        print("Las imagenes son iguales")
    else:
        print("Las imagenes son diferentes")

        # Creando la imagen de la diferencia
        #cv2.imwrite("diferencia_parado.png", diferencia)
        #cv2.imwrite("diferencia_agachado.png", diferencia)
        cv2.imwrite("diferencia_caida.png", diferencia)

        # Aplicamos un umbral
        umbral = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilatamos el umbral para tapar agujeros
        #umbral = cv2.dilate(umbral, None, iterations=2)

        # Creando la imagen del umbral
        #cv2.imwrite("umbral_parado.png", umbral)
        #cv2.imwrite("umbral_agachado.png", umbral)
        cv2.imwrite("umbral_caida.png", umbral)

#Realizo la comparación y debo poner la imagen de vacio que en este caso es imagen1 como segundo parametro de la resta
#para que la diferencia se muestre en color blanco
#compara(imagen2,imagen1)
#compara(imagen3,imagen1)
compara(imagen4,imagen1)

############################################################################
