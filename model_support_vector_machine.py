from PIL import Image
import numpy as np
import os
from random import shuffle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

trainImages = trainImages.reshape(20000,1*270*300)
testImages = testImages.reshape(4000,1*270*300)

#Verificamos que el dataset de imagenes  sea de 2 dimensiones
testImages.shape
trainImages.shape

#Verificamos que el dataset de labels  sea de 2 dimensiones
testLabels.shape
trainLabels.shape

#----------------------------------------------------------------------------------------------------------------------

#Transformo los datos con el método MinMaxScaler() a una escala particular
scaler = MinMaxScaler()
X_train = scaler.fit_transform(trainImages)
X_test = scaler.transform(testImages)

#Inicializo las variables que guardarán los nombres de las clases
y_train = trainLabels
y_test = testLabels

#Verificamos que el dataset de imagenes  sea de 2 dimensiones
X_train.shape
X_test.shape

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

#Create a svm Classifier whit Linear Kernel
clf = svm.SVC(kernel='linear')

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
# 86% de accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred) * 100)

#Imprimiendo la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(cm)

#----------------------------------------------------------------------------------------------------------------------

score = metrics.accuracy_score(y_test, y_pred)

print("Classification Report")
print(classification_report(y_test, y_pred))

#Method 1 (Seaborn)
#This method produces a more understandable and visually readable confusion matrix using seaborn.

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".4g", linewidths=.5, square = True, cmap = 'Blues_r', annot_kws={'size':15})
#plt.ylabel('Actual label')
plt.xlabel('Clase Vacio: 0             Clase Parado: 1            Clase Agachado: 2            Clase Caída: 3')
all_sample_title = 'Support Vector Machine - Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()

# 0- Clase Vacio
# 1- Clase Parado
# 2- Clase Agachado
# 3- Clase Caída

#----------------------------------------------------------------------------------------------------------------------

import pickle

#En la siguiente línea configuro la carpeta y el nombre de como se guardara el modelo de las primeras 100 Epoch
MODEL_DIR = './modelos/model-SVM'

#En la siguiente linea guardamos el primer modelo
pickle.dump(clf, open(MODEL_DIR, 'wb'))


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

'''

#Probando el tracking de un video de caida

DIR = './Tracking-Caida'

def label_img_tracking(name):

    #A continuacion describo los diferentes valores que tendrá que identificar el modelo
    #tomando el valor que se encuentre antes del - como la palabra que debe identificar
    word_label = name.split('-')[1]
    #print(word_label)

    # 0- Declaro la clase Vacio
    if word_label == 'Vacio'  :
        print(word_label)
        return np.array([1, 0, 0, 0])

    # 1- Declaro la clase Parado
    if word_label == 'Parado' :
        print(word_label)
        return np.array([0, 1, 0, 0])

    # 2- Declaro la clase Agachado
    if word_label == 'Agachado':
        print(word_label)
        return np.array([0, 0, 1, 0])

    # 3- Declaro la clase CaidaTipoFin
    elif word_label == 'CaidaTipoFin':
        print(word_label)
        return np.array([0, 0, 0, 1])


#Algoritmo de prueba para ver en que orden esta leyendo las imagenes
def label_img_tracking_split_numero(name):
    word_label = name.split('-')[0]
    return word_label

# Cargamos las imagenes del conjunto de Tracking
TRACKING_DIR = './Tracking-Caida'

def load_test_tracking():
    test_tracking = []
    for img in os.listdir(TRACKING_DIR):
        label = label_img_tracking(img)
        #Invocando el metodo que me dice en que orden lee las imagenes
        #label1 = label_img_tracking_split_numero(img)
        #print(label1)
        path = os.path.join(TRACKING_DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE_Height, IMG_SIZE_Width), Image.ANTIALIAS)
            test_tracking.append([np.array(img), label])
            #print(np.array(img), label)
            #print(test_tracking)
    #Este shuffle() randomizaba el orden de los valores del array
    #shuffle(test_tracking)
    return test_tracking

# Cargamos todas las imagenes del Tracking en el set de datos tracking_data
tracking_data = load_test_tracking()

# Cargamos todas las metricas resultantes del modelo
trackingImages = np.array([i[0] for i in tracking_data]).reshape(-1, IMG_SIZE_Height, IMG_SIZE_Width, 1)
trackingLabels = np.array([i[1] for i in tracking_data])
print(trackingLabels)

#Imprimo el valor shape que me indica que el dataset es una matriz de 4 dimensiones (cant imagenes,chanel,Height,Width)
trackingImages.shape
trackingLabels.shape

#Realizamos un nuevo reshape para reducir a solo 2 dimensiones la matriz (cant imagenes,chanel * Height * Width)
trackingImages = trackingImages.reshape(30,1*710*860)

#Verificamos que el dataset de imagenes  sea de 2 dimensiones
trackingImages.shape

#Transformo los datos con el método MinMaxScaler() a una escala particular
scaler = MinMaxScaler()
X_tracking = scaler.fit_transform(trackingImages)

#Inicializo las variables que guardarán los nombres de las clases
y_tracking = trackingLabels

#Verificamos que el dataset de imagenes  sea de 2 dimensiones
X_tracking.shape

#Verificamos que el dataset de labels  sea de 2 dimensiones
y_tracking.shape

#Imprimo y_tracking para verificar si esta ordenado
print(y_tracking)

#Convertimos los valores binarios de los labels de los nombres de las clases a numeros enteros para que funcione correctamente
y_tracking = np.argmax(y_tracking, axis=1)
print("Clase original del test en forma vectorial")
print(y_tracking)

#Predict the response for test dataset
y_predTracking = clf.predict(X_tracking)

# Model Accuracy: how often is the classifier correct?
# 86% de accuracy
print("Accuracy:",metrics.accuracy_score(y_tracking, y_predTracking) * 100)

#Imprimiendo la matriz de confusión
cm = confusion_matrix(y_tracking, y_predTracking)
print(cm)


# 0- Clase Vacio
# 1- Clase Parado
# 2- Clase Agachado
# 3- Clase Caída

#Variable que guarda el rango de valores
rango = np.arange(30)

#Inicializo los valores del data frame
df = pd.DataFrame({ 'Orden': rango, 'Original': y_tracking, 'Predicción': y_predTracking})

#Comienzo a dibujar las lineas
ax = plt.gca()
df.plot(kind='line',x='Orden',y='Original',ax=ax)
df.plot(kind='line',x='Orden',y='Predicción', color='red', ax=ax)
plt.xlabel('Clase Vacio: 0.0      Clase Parado: 1.0     Clase Agachado: 2.0     Clase Caída: 3.0',color='red')
plt.show()

'''

