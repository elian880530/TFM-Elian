#Calcula la diferencia entre las imágenes en porcentaje, verifica la igualdad con la borrosidad opcional.

#Compare images
import os
import pandas as pd
import numpy as np
from imgcompare import is_equal, image_diff_percent

#Comparando las dos imagenes
#is_same = is_equal('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-2.png', 'E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-4.png')
#print(is_same)

#Usando el parámetro de tolerancia para permitir un cierto pase de diferencia igual
#is_same = is_equal('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-2.png', 'E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-4.png', tolerance=2.5)
#print(is_same)

#Obteniendo el porcentaje de diferencia
#percentage = image_diff_percent('E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-2.png', 'E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train/Agachado-4.png')
#print(percentage)

#Dir de la carpeta de imagenes
#DIR = 'E:/Elian/Asignaturas UIB/Trabajo de Fin de Master/cod proyecto/Ouch/train'
DIR = 'C:/Users/EGH/PycharmProjects/Ouch/train'


#Llenando un array de los nombres de imagenes a comparar
def load_labelsColumn():
    labels_column = [' ']
    for img in os.listdir(DIR):
        labels_column.append(img)
    return labels_column

#Invocando la rutina que llena el array de labels
arrayLabels = load_labelsColumn()
#print(arrayLabels)

#Construyo un data set con valores de los label en cada columna
data = pd.DataFrame(columns=arrayLabels)
print(data)

#Lleno cada fila del data frame con el valor de cada label
for k in range(len(arrayLabels)):
 data.loc[len(data)]=arrayLabels[k]
 print(k)

#Muestro las primeras 4 columnas
print(data[[' ','Agachado-100.png','Agachado-101.png','Agachado-102.png']])
print(data[:5])
data.iloc[1, 1] = 93
print(data.iloc[0:3, 0:3])

#Seleccionando el nombre de cada imagen para realizar la comparación
def load_labels():
    fila = 1
    for img1 in os.listdir(DIR):
        for img2 in os.listdir(DIR):
            #Obteniendo el porcentaje de diferencia
            url1 = 'C:/Users/EGH/PycharmProjects/Ouch/train/' + img1
            url2 = 'C:/Users/EGH/PycharmProjects/Ouch/train/' + img2
            percentage = image_diff_percent(url1,url2)
            data.loc[fila, [str(img2)]] = percentage
            print('Imprimiendo coordenadas de fila: ' + str(fila) + '   columna: ' + str(img2) + '   porcentage de diferencia: ' + str(data.loc[fila, [str(img2)]]))
        fila = fila + 1

#Invoco a la rutina load_labels()
load_labels()

#Guardando el data frame actualizado con los nuevos valores en un archivo csv
#Alt + Shift + E Run fragment code
data.to_csv('df_porcent_diferent.csv')

#Imprimo los primeros valore para comprobar si el data-set se lleno correctamente
print(data.head())
print(data.iloc[0:4, 0:4])
print(data.loc[0:4, ['Agachado-2.png','CaidaTipoFin-0.png','Parado-0.png','Vacio-1.png']])
#print(data.iloc[1, 400])

#Funcion que muestra todos los datos del dataframe que se le pase por parametros
def displaydf(dataframe, cols = None, rows = 20):
    with pd.option_context("display.max_columns", cols):
        with pd.option_context("display.max_rows", rows):
            print(dataframe)
    return True

