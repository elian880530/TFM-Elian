import numpy as np
import os
from PIL import Image
from random import shuffle
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt


#Configuro la ruta donde se encuenta el modelo entrenado
MODEL_DIR = './modelos/modelEpoch10.keras'

#En la siguiente linea cargamos el modelo
model = load_model(MODEL_DIR)

#----------------------------------------------------------------------------------------------------------------------

def label_img(name):

    #A continuacion describo los diferentes valores que tendrá que identificar el modelo
    #tomando el valor que se encuentre antes del - como la palabra que debe identificar
    word_label = name.split('-')[0]

    # 1- Declaro la clase Agachado
    if word_label == 'Agachado': return np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # 2- Declaro la clase Vacio
    if word_label == 'Vacio': return np.array([0, 1, 0, 0, 0, 0, 0, 0])

    # 3- Declaro la clase Parado
    if word_label == 'Parado': return np.array([0, 0, 1, 0, 0, 0, 0, 0])

    # 4- Declaro la clase CaidaTipoA
    if word_label == 'CaidaTipoA': return np.array([0, 0, 0, 1, 0, 0, 0, 0])

    # 5- Declaro la clase CaidaTipoB
    if word_label == 'CaidaTipoB': return np.array([0, 0, 0, 0, 1, 0, 0, 0])

    # 6- Declaro la clase CaidaTipoC
    if word_label == 'CaidaTipoC': return np.array([0, 0, 0, 0, 0, 1, 0, 0])

    # 7- Declaro la clase CaidaTipoD
    if word_label == 'CaidaTipoD': return np.array([0, 0, 0, 0, 0, 0, 1, 0])

    # 8- Declaro la clase CaidaTipoFin
    elif word_label == 'CaidaTipoFin': return np.array([0, 0, 0, 0, 0, 0, 0, 1])


#Declaramos el alto y el ancho de las imagenes
IMG_SIZE_Height = 710
IMG_SIZE_Width = 860

# Cargamos las imagenes del conjunto de Test
TEST_DIR = './test'

def load_test_data():
    test_data = []
    for img in os.listdir(TEST_DIR):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE_Height, IMG_SIZE_Width), Image.ANTIALIAS)
            test_data.append([np.array(img), label])
    shuffle(test_data)
    return test_data

# Cargamos todas las imagenes de Test en el set de datos test_data
test_data = load_test_data()

# Tupla con altura y anchura de imágenes utilizadas para reformar matrices.
# Esto se usa para trazar las imágenes.
testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE_Height, IMG_SIZE_Width, 1)
trainLabels = np.array([i[1] for i in test_data])

# Escogemos solo 10 elementos de las imagenes para predecir pq si los tomamos todos da error
images = testImages[0:10]
labels = trainLabels[0:10]

# Con esta funcion intentamos que el modelo clasifique la imagen sin que le pasemos el label identificando que classe de objeto es
# debido a que la evaluacion que hacemos anteriormente es que clasifique la imagen segun el label que le pasamos y no es esto lo que se desea ahora

predictModel = model.predict(x=images)
print('-------------------------------------------------------------')

print(predictModel)
print('-------------------------------------------------------------')

predictModelMatrix = np.argmax(predictModel, axis=1)
print(predictModelMatrix)
print('-------------------------------------------------------------')

print(labels)
print('-------------------------------------------------------------')



