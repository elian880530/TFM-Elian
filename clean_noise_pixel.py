import numpy as np
from PIL import Image
import os


#----------------------------------------------------------------------------------------------------------------------

DIR = './Clean-Pixel-Vacio'

#Rutina que permite convertir todas las imagenes en matrices y guardarlas en un dataset
def load_pixel():
    data_pixel = []
    for img in os.listdir(DIR):
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path).convert('L')
            array_pixel = list(img.getdata())
            data_pixel.extend(array_pixel)
    return np.asarray(data_pixel)

#Invoco la rutina  que carga las imagenes en el dataset
data_pixel = load_pixel()

#----------------------------------------------------------------------------------------------------------------------

#Guardo la media en una variable y la imprimo
media_pixel = round(data_pixel.mean(),2)
print(media_pixel)

#Guardo la desviacion en una variable y la imprimo
desviacion_pixel = round(data_pixel.std(),2)
print(desviacion_pixel)

#Guardo la media - desviacion en una variable y la imprimo
media_menos_desviacion = round(media_pixel - desviacion_pixel,2)
print(media_menos_desviacion)

#Guardo la media + desviacion en una variable y la imprimo
media_mas_desviacion = round(media_pixel + desviacion_pixel,2)
print(media_mas_desviacion)

#----------------------------------------------------------------------------------------------------------------------

#DIR_imagen = './clean-noise'
#DIR_imagen = './Casa-Sa-Pobla-Agachado-Noise'
#DIR_imagen = './Casa-Sa-Pobla-CaidaTipoFin-Noise'
#DIR_imagen = './Casa-Sa-Pobla-Parado-Noise'
DIR_imagen = './Casa-Sa-Pobla-Vacio-Noise'

#Rutina que permite convertir todas las imagenes en array
def load_pixel_array():
    data_pixel_array = []
    for img in os.listdir(DIR_imagen):
        path = os.path.join(DIR_imagen, img)
        if "DS_Store" not in path:
            img = Image.open(path).convert('L')
            array_pixel = list(img.getdata())
            data_pixel_array.append(array_pixel)
    return data_pixel_array

#Invoco la rutina  que carga las imagenes en el dataset
data_pixel_array = load_pixel_array()
print("-----------------------")
print("Total de imagenes convertidas en array: " + str(len(data_pixel_array)))

#----------------------------------------------------------------------------------------------------------------------

total_pixel = 0
array_pixel_limpio = []

for i in range(len(data_pixel_array)):
    listado_pixel = data_pixel_array[i]
    #w = len(listado_pixel)
    #print("Imprimo ant de entrar: " + str(listado_pixel[0:300]))

    for j in range(len(listado_pixel)):
        if(listado_pixel[j] >= media_menos_desviacion and listado_pixel[j] <= media_mas_desviacion):
            listado_pixel[j] = 0
            total_pixel += 1
            #print("Pos j: " + str(j) + " y Pos i: " + str(i))
            #print(" Pos j: " + str(j))


    #print("Imprimo des de entrar: " + str(listado_pixel[0:300]))
    #print("-----------------------")
    #print(" Pos i: " + str(i))
    array_pixel_limpio.append(listado_pixel)


#----------------------------------------------------------------------------------------------------------------------

print("-----------------------")
print("Número total de pixeles cambiados: " + str(total_pixel))
print("Número total de imagenes limpiadas: " + str(len(array_pixel_limpio)))
print("-----------------------")

#for m in range(len(array_pixel_limpio)):
    #list_pixel = array_pixel_limpio[m]
    #print(list_pixel[0:300])
    #print("-----------------------")

#----------------------------------------------------------------------------------------------------------------------


for p in range(len(array_pixel_limpio)):
    pixels = array_pixel_limpio[p]
    #print("----------------------------")
    #print("Entrando al bucle")
    #print("Imprimiendo listado de pixel")
    #print(pixels[0:300])
    # Convert the pixels into an array using numpy
    array = np.array(pixels, dtype=np.uint8).reshape(270,480)
    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array, 'L')

    #new_image.save('./clean-noise/new-' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Agachado-Clean/Agachado-0' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Agachado-Clean/Agachado-00' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Agachado-Clean/Agachado-000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Agachado-Clean/Agachado-0000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Agachado-Clean/Agachado-00000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Agachado-Clean/Agachado-000000' + str(p) + '.png')

    #new_image.save('./Casa-Sa-Pobla-CaidaTipoFin-Clean/CaidaTipoFin-0' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-CaidaTipoFin-Clean/CaidaTipoFin-00' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-CaidaTipoFin-Clean/CaidaTipoFin-000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-CaidaTipoFin-Clean/CaidaTipoFin-0000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-CaidaTipoFin-Clean/CaidaTipoFin-00000' + str(p) + '.png')

    #new_image.save('./Casa-Sa-Pobla-Parado-Clean/Parado-0' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Parado-Clean/Parado-00' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Parado-Clean/Parado-000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Parado-Clean/Parado-0000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Parado-Clean/Parado-00000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Parado-Clean/Parado-000000' + str(p) + '.png')

    #new_image.save('./Casa-Sa-Pobla-Vacio-Clean/Vacio-0' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Vacio-Clean/Vacio-00' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Vacio-Clean/Vacio-000' + str(p) + '.png')
    #new_image.save('./Casa-Sa-Pobla-Vacio-Clean/Vacio-0000' + str(p) + '.png')
    new_image.save('./Casa-Sa-Pobla-Vacio-Clean/Vacio-00000' + str(p) + '.png')


