from PIL import Image
import numpy as np
import os
from random import shuffle
from sklearn.metrics import confusion_matrix

#----------------------------------------------------------------------------------------------------------------------

DIR = './train'

def label_img(name):

    #A continuacion describo los diferentes valores que tendrá que identificar el modelo
    #tomando el valor que se encuentre antes del - como la palabra que debe identificar
    word_label = name.split('-')[0]

    # 0- Declaro la clase Vacio
    if word_label == 'Vacio': return np.array([1, 0, 0, 0])

    # 1- Declaro la clase Parado
    if word_label == 'Parado': return np.array([0, 1, 0, 0])

    # 2- Declaro la clase Agachado
    if word_label == 'Agachado': return np.array([0, 0, 1, 0])

    # 3- Declaro la clase CaidaTipoFin
    elif word_label == 'CaidaTipoFin': return np.array([0, 0, 0, 1])


    #Importante!!!!!!  Si quisiera agregar clases nuevas tendria que aumentar el numero de elementos binarios en el array
    #por ejemplo para un nuevo word_label == 'personasentado' : return np.array([0, 0, 0, 1])
    #ademas hay que cambiar el numero de clases que hay al final como por ejemplo las 3 que utilizo a continuacion
    #model.add(Dense(3, activation = 'softmax'))

#----------------------------------------------------------------------------------------------------------------------

IMG_SIZE_Height = 710
IMG_SIZE_Width = 860


def load_training_data():
    train_data = []
    for img in os.listdir(DIR):
        label = label_img(img)
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE_Height, IMG_SIZE_Width), Image.ANTIALIAS)
            train_data.append([np.array(img), label])

    shuffle(train_data)
    return train_data

#----------------------------------------------------------------------------------------------------------------------

train_data = load_training_data()
#print(train_data)
#plt.imshow(train_data[1][0], cmap = 'gist_gray')

#----------------------------------------------------------------------------------------------------------------------

trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE_Height, IMG_SIZE_Width, 1)
trainLabels = np.array([i[1] for i in train_data])

#----------------------------------------------------------------------------------------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization

# Comienza la construcción del modelo Keras Sequential.
model = Sequential()

# Agrega una capa de entrada que es similar a un feed_dict en TensorFlow.
# Tenga en cuenta que la forma de entrada debe ser una tupla que contenga el tamaño de la imagen.
# La entrada es una matriz aplanada con x elementos (img_size * img_size),
# pero las capas convolucionales esperan imágenes con forma (alto_definido, ancho_definido, 1), por tanto hacemos un input_shape=(IMG_SIZE_Height, IMG_SIZE_Width, 1)

# Primera capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE_Height, IMG_SIZE_Width, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Segunda capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Tercera capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Cuarta capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Quinta capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# La deserción (Dropout) se puede aplicar a las neuronas en cualquiera de las capas.#
# A continuación, agregamos una nueva capa de Dropout entre la capa oculta y la salida .
# La tasa de abandono se establece en 20% o 0.2, lo que significa que una de cada 5 entradas se excluirá aleatoriamente de cada ciclo de actualización.
# Además, como se recomienda en el documento original sobre Dropout, se impone una restricción en los pesos para cada capa oculta,
# asegurando que la norma máxima de los pesos no exceda un valor de 3. Esto se hace configurando el argumento kernel_constraint en Dense clase al construir las capas.
# La tasa de aprendizaje se elevó en un orden de magnitud y el impulso se incrementó a 0.9. Estos aumentos en la tasa de aprendizaje también se recomendaron en el documento original de abandono.

model.add(Dropout(0.2))

# Aplanar la salida de 4 niveles de las capas convolucionales
# a 2-rank que se puede ingresar a una capa totalmente conectada
model.add(Flatten())

# Primera capa completamente conectada  con ReLU-activation.
model.add(Dense(256, activation='relu'))

# La deserción (Dropout) se puede aplicar a las neuronas en cualquiera de las capas.#
# A continuación, agregamos una nueva capa de Dropout entre la capa oculta y la salida .
# La tasa de abandono se establece en 20% o 0.2, lo que significa que una de cada 5 entradas se excluirá aleatoriamente de cada ciclo de actualización.
# Además, como se recomienda en el documento original sobre Dropout, se impone una restricción en los pesos para cada capa oculta,
# asegurando que la norma máxima de los pesos no exceda un valor de 3. Esto se hace configurando el argumento kernel_constraint en Dense clase al construir las capas.
# La tasa de aprendizaje se elevó en un orden de magnitud y el impulso se incrementó a 0.9. Estos aumentos en la tasa de aprendizaje también se recomendaron en el documento original de abandono.

model.add(Dropout(0.2))

# Segunda capa completamente conectada  con ReLU-activation.
model.add(Dense(128, activation='relu'))

#En la siguiente línea tengo Dense = 4 lo que significa que utilizo 4 tipos de clases a identificar en el modelo
model.add(Dense(4, activation = 'softmax'))

#----------------------------------------------------------------------------------------------------------------------

#En la siguiente línea compilamos el modelo con todas sus métricas
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

#----------------------------------------------------------------------------------------------------------------------

#En la siguiente línea entrenamos el modelo y le pasamos en el batch_size = 8 la cantidad de imagenes que procesara por iteración
#pero como son demasiado grandes la memoria ram de la pc solo admite 8 si ponemos más peta el script.
#En la variable epochs = 3 le decimos cuantas veces iteramos sobre el conjunto de entrenamiento lo ideal sería 10000 pero se demora muchos días.

#En el ejemplo original se escogían lotes de 128 imagenes en cada iteración porque eran super pequeñas
#model.fit(trainImages, trainLabels, batch_size = 128, epochs = 1, verbose = 1)

#El modeo entrenado con 50 Epoch en Azure tiene un accuracy de 95%
#model.fit(trainImages, trainLabels, batch_size = 100, epochs = 50, verbose = 1)

#El modeo entrenado con 2 Epoch en mi pc tiene un accuracy de 81%
#model.fit(trainImages, trainLabels, batch_size = 100, epochs = 50, verbose = 1)

#El modeo entrenado con 5 Epoch en mi pc tiene un accuracy de 87.5%
model.fit(trainImages, trainLabels, batch_size = 20, epochs = 5, verbose = 1)



#En la siguiente línea configuro la carpeta y el nombre de como se guardara el modelo de las primeras 100 Epoch
MODEL_DIR = './modelos/modelEpoch5.keras'

#En la siguiente linea guardamos el primer modelo
model.save(MODEL_DIR)

#----------------------------------------------------------------------------------------------------------------------

# Cargamos las imagenes del conjunto de Test
# Test on Test Set
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
#plt.imshow(test_data[1][0], cmap = 'gist_gray')

#----------------------------------------------------------------------------------------------------------------------

print('----------------------------------------------------------------------------------------------------------------')

# Cargamos todas las metricas resultantes del modelo
testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE_Height, IMG_SIZE_Width, 1)
testLabels = np.array([i[1] for i in test_data])

print('----------------------------------------------------------------------------------------------------------------')

# Imprimimos las metricas que cargamos anteriormente
loss, acc = model.evaluate(testImages, testLabels, verbose = 1)

print('----------------------------------------------------------------------------------------------------------------')

# Imprimimos el porciento de acierto o accuracy del modelo
print("Accuracy del modelo")
print(acc * 100)

print('----------------------------------------------------------------------------------------------------------------')

# Con esta funcion intentamos que el modelo clasifique la imagen sin que le pasemos el label identificando que classe de objeto es
# debido a que la evaluacion que hacemos anteriormente es que clasifique la imagen segun el label que le pasamos y no es esto lo que se desea ahora

# Escogemos solo 10 elementos de las imagenes para predecir pq si los tomamos todos da error
images = testImages[0:8]
labels = testLabels[0:8]

# Con esta funcion intentamos que el modelo clasifique la imagen sin que le pasemos el label identificando que classe de objeto es
# debido a que la evaluacion que hacemos anteriormente es que clasifique la imagen segun el label que le pasamos y no es esto lo que se desea ahora


print('-------------------------------------------------------------')

predictModel = model.predict(x=images)
print("Predicción del modelo sobre el test en forma matricial")
print(predictModel)

print('-------------------------------------------------------------')

predictModelMatrix = np.argmax(predictModel, axis=1)
print("Predicción del modelo sobre el test en forma vectorial")
print(predictModelMatrix)

print('-------------------------------------------------------------')

print("Clase original del test en forma matricial")
print(labels)

print('-------------------------------------------------------------')

labelsMatrix = np.argmax(labels, axis=1)
print("Clase original del test en forma vectorial")
print(labelsMatrix)

print('-------------------------------------------------------------')

print('Matrix Confusion')
print(confusion_matrix(labelsMatrix,predictModelMatrix))
