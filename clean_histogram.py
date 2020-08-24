#Trazando Histogramas
#Hay dos formas para esto,
#Camino corto: use las funciones de trazado de Matplotlib
#Largo camino: use las funciones de dibujo de OpenCV

#Usando Matplotlib
#Matplotlib viene con una función de trazado de histograma: matplotlib.pyplot.hist ()

import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd

#Ejemplo 1
img = cv2.imread('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-100.png',0)
plt.hist(img.ravel(),256,[0,256])
plt.show()

#Ejemplo 2
#También se puede usar el diagrama normal de matplotlib, que sería bueno para el diagrama BGR.
img = cv2.imread('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-2.png')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


#Ejemplo 3
#Dir de la carpeta de imagenes
DIR = 'E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train'

#Llenando un data frame de histogramas
def upload_dfhistogram():
    for img1 in os.listdir(DIR):
        foto = cv2.imread('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/' + str(img1), 0)
        plt.hist(foto.ravel(), 256, [0, 256])
        plt.savefig('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/histogram/' + str(img1))

#Invoco a la rutina load_labels()
upload_dfhistogram()
