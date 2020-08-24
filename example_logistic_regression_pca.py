from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

#IMG_SIZE_Height = 710
#IMG_SIZE_Width = 860

IMG_SIZE_Height = 270
IMG_SIZE_Width = 300

#Rutina que permite convertir todas las imagenes en matrices y guardarlas en un dataset
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

#Invoco la rutina  que carga las imagenes en el dataset
train_data = load_training_data()

#----------------------------------------------------------------------------------------------------------------------

#Realizo un reshape de las matrices de imagenes y labels
trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE_Height, IMG_SIZE_Width, 1)
trainLabels = np.array([i[1] for i in train_data])

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

#----------------------------------------------------------------------------------------------------------------------


# Cargamos todas las metricas resultantes del modelo
testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE_Height, IMG_SIZE_Width, 1)
testLabels = np.array([i[1] for i in test_data])

#----------------------------------------------------------------------------------------------------------------------

#Imprimo el valor shape que me indica que el dataset es una matriz de 4 dimensiones (cant imagenes,chanel,Height,Width)
testImages.shape
trainImages.shape

#Realizamos un nuevo reshape para reducir a solo 2 dimensiones la matriz (cant imagenes,chanel * Height * Width)
#trainImages = trainImages.reshape(400,1*710*860)
#testImages = testImages.reshape(200,1*710*860)

#trainImages = trainImages.reshape(20000,1*270*300)
#testImages = testImages.reshape(4000,1*270*300)

trainImages = trainImages.reshape(12000,1*270*300)
testImages = testImages.reshape(4000,1*270*300)

#Verificamos que el dataset de imagenes  sea de 2 dimensiones
testImages.shape
trainImages.shape

#Verificamos que el dataset de labels  sea de 2 dimensiones
testLabels.shape
trainLabels.shape

#----------------------------------------------------------------------------------------------------------------------

'''
import dask.array as da


rk = [1,2,3,4,5] 	#converts the list into dask array
d=da.asarray(rk)
print(d.compute()) 	#print type of d
print(type(d))


testImages = da.asarray(testImages)
trainImages = da.asarray(trainImages)



trainImages.nbytes

#a = np.zeros((156816, 36, 53806), dtype='uint8')

var_test = np.array(trainImages, dtype='uint8')
var_test = trainImages.reshape(20000,1*270*300)
var_test.nbytes

'''
#----------------------------------------------------------------------------------------------------------------------


#Transformo los datos con el método MinMaxScaler() a una escala particular
scaler = MinMaxScaler()
x_train = scaler.fit_transform(trainImages)
x_test = scaler.transform(testImages)

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(trainImages)

# Apply transform to both the training set and the test set.
x_train = scaler.transform(trainImages)
x_test = scaler.transform(testImages)
'''

from sklearn.decomposition import PCA
# Make an instance of the Model
#El código a continuación tiene .95 para el parámetro de número de componentes.
#Significa que scikit-learn elige el número mínimo de componentes principales de manera que se retenga el 95% de la varianza.
pca = PCA(.95)

#Montar PCA en el set de entrenamiento.
pca.fit(x_train)

#Apply the mapping (transform) to both the training set and the test set.
train_img = pca.transform(x_train)
test_img = pca.transform(x_test)

#----------------------------------------------------------------------------------------------------------------------
#Inicializo las variables que guardarán los nombres de las clases
y_train = trainLabels
y_test = testLabels

#Verificamos que el dataset de imagenes  sea de 2 dimensiones
x_train.shape
x_test.shape

#Verificamos que el dataset de labels  sea de 2 dimensiones
y_train.shape
y_test.shape

#----------------------------------------------------------------------------------------------------------------------

#Convertimos los valores binarios de los labels de los nombres de las clases a numeros enteros para que funcione correctamente
y_train = np.argmax(y_train, axis=1)
print("Clase original del test en forma vectorial")
print(y_train)

#Convertimos los valores binarios de los labels de los nombres de las clases a numeros enteros para que funcione correctamente
y_test = np.argmax(y_test, axis=1)
print("Clase original del test en forma vectorial")
print(y_test)

#----------------------------------------------------------------------------------------------------------------------

#All parameters not specified are set to their defaults
logisticRegr = LogisticRegression(max_iter=500)

#Model is learning the relationship between digits (x_train) and labels (y_train)
logisticRegr.fit(x_train, y_train)

#Make predictions on entire test data
predictions = logisticRegr.predict(x_test)

#Use score method to get accuracy of model
# 78% de accuracy
score = logisticRegr.score(x_test, y_test)
print(score * 100)

#----------------------------------------------------------------------------------------------------------------------

#Confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

#Imprimiendo la matriz de confusión en imagen
#sns.heatmap(cm, center=True)
#plt.show()

#----------------------------------------------------------------------------------------------------------------------

print("Classification Report")
print(classification_report(y_test, predictions))

#Method 1 (Seaborn)
#This method produces a more understandable and visually readable confusion matrix using seaborn.

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".4g", linewidths=.5, square = True, cmap = 'Blues_r', annot_kws={'size':15})
#plt.ylabel('Actual label')
plt.xlabel('Clase Vacio: 0             Clase Parado: 1            Clase Agachado: 2            Clase Caída: 3')
all_sample_title = 'Logistic Regression - Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()

# 0- Clase Vacio
# 1- Clase Parado
# 2- Clase Agachado
# 3- Clase Caída

#----------------------------------------------------------------------------------------------------------------------

import pickle

#En la siguiente línea configuro la carpeta y el nombre de como se guardara el modelo
MODEL_DIR = './modelos/model-LR'

#En la siguiente linea guardamos el primer modelo
pickle.dump(logisticRegr, open(MODEL_DIR, 'wb'))