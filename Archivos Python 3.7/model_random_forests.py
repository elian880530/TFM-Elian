from PIL import Image
import numpy as np
import os
from random import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
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

#Para los random forest poder clasificar correctamente un problem es necesario realizar un escalamiento de los atributos
sc = StandardScaler()

#Declaramos las variables de entrenamiento y prueba
X_train = sc.fit_transform(trainImages)
X_test = sc.transform(testImages)
y_train = trainLabels
y_test = testLabels

#----------------------------------------------------------------------------------------------------------------------

#Implementando la validación cruzada

#Declaramos los parametros del modelo
#Probamos con diferentes estimadores n_estimators = [10, 20, 30, 40, 50]
#Con n_estimators = 10 el accuracy = 83.5%
#Con n_estimators = 20 el accuracy = 75.5%
#Con n_estimators = 30 el accuracy = 81.5%
#Con n_estimators = 40 el accuracy = 77%
#Con n_estimators = 50 el accuracy = 79.5%

clf_cv = RandomForestRegressor(n_estimators=10, random_state=0)

# Entrenando el modelo
cv_scores = cross_val_score(clf_cv, X_train, y_train, cv=5)
print("Accuracy score (CV): {0:3f}".format(np.mean(cv_scores)))

# UNA VEZ QUE SABEMOS CUÁL ES EL QUE MAYOR VALOR DE ACCURACY HA DADO, YA SE PUEDE HACER EL FIT EN EL TRAIN SIN EL CV:

#----------------------------------------------------------------------------------------------------------------------

regressor = RandomForestRegressor(n_estimators=10, random_state=0)

#Entrenamos el modelo random forest
regressor.fit(X_train, y_train)

print('-------------------------------------------------------------')

#Realizamos la prediccion
y_pred = regressor.predict(X_test)

#Imprimimos los valores para ver si produjo un buen resultado
print(y_pred)
print(y_test)

print('-------------------------------------------------------------')

labelsMatrix = np.argmax(y_test, axis=1)
print("Clase original del test en forma vectorial")
print(labelsMatrix)

print('-------------------------------------------------------------')

predictModelMatrix = np.argmax(y_pred, axis=1)
print("Predicción del modelo sobre el test en forma vectorial")
print(predictModelMatrix)

print('-------------------------------------------------------------')

print('Matrix Confusion')
print(confusion_matrix(labelsMatrix,predictModelMatrix))

print('-------------------------------------------------------------')

# 75.5% de accuracy
print("Accuracy del modelo")
print(accuracy_score(labelsMatrix, predictModelMatrix) * 100)

print('-------------------------------------------------------------')

print('Reporte de Clasificación')
print(classification_report(labelsMatrix,predictModelMatrix))


#----------------------------------------------------------------------------------------------------------------------

cm = confusion_matrix(labelsMatrix,predictModelMatrix)
score = accuracy_score(labelsMatrix, predictModelMatrix)

#Method 1 (Seaborn)
#This method produces a more understandable and visually readable confusion matrix using seaborn.

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".4g", linewidths=.5, square = True, cmap = 'Blues_r', annot_kws={'size':15})
#plt.ylabel('Actual label')
plt.xlabel('Clase Vacio: 0             Clase Parado: 1            Clase Agachado: 2            Clase Caída: 3')
all_sample_title = 'Random Forests - Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()

# 0- Clase Vacio
# 1- Clase Parado
# 2- Clase Agachado
# 3- Clase Caída

#----------------------------------------------------------------------------------------------------------------------

import pickle

#En la siguiente línea configuro la carpeta y el nombre de como se guardara el modelo de las primeras 100 Epoch
MODEL_DIR = './modelos/model-RF'

#En la siguiente linea guardamos el primer modelo
pickle.dump(regressor, open(MODEL_DIR, 'wb'))