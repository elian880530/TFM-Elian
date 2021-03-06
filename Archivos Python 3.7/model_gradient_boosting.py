from PIL import Image
import numpy as np
import os
from random import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score

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

#Train with Gradient Boosting algorithm
#Compute the accuracy scores on train and validation sets when training with different learning rates
#learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

#Implementando la validación cruzada en los parámetros

learning_rates= [0.001, 0.01, 0.05, 0.1, 0.75]
n_estimators = [10, 20, 30, 50]
max_features = [1, 2, 5]
max_depths = [2, 10, 20, 50]

for learning_rate in learning_rates:
	for n_estimator in n_estimators:
		for max_feature in max_features:
			for max_depth in max_depths:
				gb_cv = GradientBoostingClassifier(n_estimators = n_estimator, learning_rate = learning_rate, max_features = max_feature, max_depth = max_depth, random_state = 0)
				cv_scores = cross_val_score(gb_cv, X_train, y_train, cv=5)
				print("--------------")
				print("Learning rate:", learning_rate)
				print("N Estimators:", n_estimator)
				print("Max Feature:", max_feature)
				print("Max Depth:", max_depth)
				print("Accuracy score (CV): {0:3f}".format(np.mean(cv_scores)))
				print("--------------")
				print()

# UNA VEZ QUE SABEMOS CUÁL ES EL QUE MAYOR VALOR DE ACCURACY HA DADO, YA SE PUEDE HACER EL FIT EN EL TRAIN SIN EL CV:
#Mejores parametros obtenidos con 20000 imagenes de train:
# --------------
# Learning rate: 0.75
# N Estimators: 50
# Max Feature: 1
# Max Depth: 10
# Accuracy score (training): 1.000000
# Accuracy score (test): 0.936250
# --------------

#----------------------------------------------------------------------------------------------------------------------

#Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

#Mejores parametros con 150 imagenes de cada clase
#gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)

#Mejores parametros con 20000 imagenes del train
gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.75, max_features=1, max_depth = 10, random_state = 0)

#Mejores parametros con 40000 imagenes del train
#gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.75, max_features=2, max_depth = 10, random_state = 0)

gb.fit(X_train, y_train)
predictions = gb.predict(X_test)

# Model Accuracy, how often is the classifier correct?
# 83.5% de accuracy
print("Accuracy:",round(metrics.accuracy_score(y_test, predictions) * 100, 2))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

#Imprimiendo la matriz de confusión en imagen
#sns.heatmap(cm, center=True)
#plt.show()

print("Classification Report")
print(classification_report(y_test, predictions))

#----------------------------------------------------------------------------------------------------------------------

#Method 1 (Seaborn)
#This method produces a more understandable and visually readable confusion matrix using seaborn.

#Use score method to get accuracy of model
score = gb.score(X_test, y_test)
print(score * 100)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".4g", linewidths=.5, square = True, cmap = 'Blues_r', annot_kws={'size':15})
#plt.ylabel('Actual label')
plt.xlabel('Clase Vacio: 0             Clase Parado: 1            Clase Agachado: 2            Clase Caída: 3')
all_sample_title = 'Gradient Boosting - Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()

# 0- Clase Vacio
# 1- Clase Parado
# 2- Clase Agachado
# 3- Clase Caída

#----------------------------------------------------------------------------------------------------------------------

import pickle

#En la siguiente línea configuro la carpeta y el nombre de como se guardara el modelo
MODEL_DIR = './modelos/model-GB'

#En la siguiente linea guardamos el primer modelo
pickle.dump(gb, open(MODEL_DIR, 'wb'))


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

'''

'''

#Probando el tracking de un video de caida

#DIR = './Tracking-Caida'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/1_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/2_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/3_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/4_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/5_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/6_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/7_Caida_Tracking'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/26_Caida_Tracking_2_Personas'
#DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/27_Caida_Tracking_2_Personas'
DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/28_Caida_Tracking_Ruido'

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
#TRACKING_DIR = './Tracking-Caida'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/1_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/2_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/3_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/4_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/5_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/6_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/7_Caida_Tracking'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/26_Caida_Tracking_2_Personas'
#TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/27_Caida_Tracking_2_Personas'
TRACKING_DIR = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/28_Caida_Tracking_Ruido'


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
#trackingImages = trackingImages.reshape(724,1*270*300)
#trackingImages = trackingImages.reshape(564,1*270*300)
#trackingImages = trackingImages.reshape(501,1*270*300)
#trackingImages = trackingImages.reshape(569,1*270*300)
#trackingImages = trackingImages.reshape(551,1*270*300)
#trackingImages = trackingImages.reshape(531,1*270*300)
#trackingImages = trackingImages.reshape(324,1*270*300)
#trackingImages = trackingImages.reshape(6000,1*270*300)
trackingImages = trackingImages.reshape(12000,1*270*300)

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
y_predTracking = gb.predict(X_tracking)

# Model Accuracy: how often is the classifier correct?
# 86% de accuracy
print("Accuracy:",metrics.accuracy_score(y_tracking, y_predTracking) * 100)
score_test = metrics.accuracy_score(y_tracking, y_predTracking) * 100

#Imprimiendo la matriz de confusión
cm_test = confusion_matrix(y_tracking, y_predTracking)
print(cm_test)

plt.figure(figsize=(9,9))
sns.heatmap(cm_test, annot=True, fmt=".4g", linewidths=.5, square = True, cmap = 'Blues_r', annot_kws={'size':15})
#plt.ylabel('Actual label')
plt.xlabel('Clase Vacio: 0             Clase Parado: 1            Clase Agachado: 2            Clase Caída: 3')
#all_sample_title = 'Video 1 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 2 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 3 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 4 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 5 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 6 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 7 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 8 - Accuracy Score: {0}'.format(score_test)
#all_sample_title = 'Video 9 - Accuracy Score: {0}'.format(score_test)
all_sample_title = 'Video 10 - Accuracy Score: {0}'.format(score_test)
plt.title(all_sample_title, size = 15)
plt.show()


# 0- Clase Vacio
# 1- Clase Parado
# 2- Clase Agachado
# 3- Clase Caída

#Variable que guarda el rango de valores
#rango = np.arange(724)
#rango = np.arange(564)
#rango = np.arange(501)
#rango = np.arange(569)
#rango = np.arange(551)
#rango = np.arange(531)
#rango = np.arange(324)
#rango = np.arange(6000)
rango = np.arange(12000)

#Inicializo los valores del data frame
df = pd.DataFrame({ 'Orden': rango, 'Original': y_tracking, 'Predicción': y_predTracking})

#Comienzo a dibujar las lineas
ax = plt.gca()
df.plot(kind='line',x='Orden',y='Predicción', color='red', ax=ax)
df.plot(kind='line',x='Orden',y='Original',ax=ax)
plt.xlabel('Clase Vacio: 0.0      Clase Parado: 1.0     Clase Agachado: 2.0     Clase Caída: 3.0',color='red')
plt.show()





