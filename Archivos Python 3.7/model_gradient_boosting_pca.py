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
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pickle
#----------------------------------------------------------------------------------------------------------------------

DIR = './train-pca'
#DIR = './train'

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

trainImages = trainImages.reshape(4000,1*270*300)
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

#----------------------------------------------------------------------------------------------------------------------

'''
 Los autovectores son las direcciones en las que la varianza de los datos es mayor.  
 Recordemos que, en teoría de probabilidad, la varianza  de una variable aleatoria es una medida de dispersión 
 (definida como la esperanza del cuadrado de la desviación de dicha variable respecto a su media). 
 Por tanto, las direcciones en las que la varianza es mayor,  representan la esencia principal de la información 
 contenida en el dataset, por eso se les llama componentes principales. Al igual que un autovector es una dirección, 
 el autovalor es un número, que representa el valor de la varianza sobre ese autovector. Por ello, para encontrar las 
 componentes principales que condensen esa esencia de la información del dataset, calcularemos primero la matriz de 
 covarianza, que nos da la medida de dispersión conjunta entre variables.

 Para ello, usaremos la función Covariance Matrix de la librería Numpy.
'''

# Calculamos la matriz de covarianza
print('NumPy covariance matrix: \n%s' %np.cov(X_train))


'''
Las direcciones sobre las que la varianza es mayor (= mayor dispersión de los datos = mayor información) se corresponden
con los autovectores, vectores propios o eigenvectors de esta matriz. Por tanto, el siguiente paso consiste en calcular 
estas direcciones.
'''

#Calculamos los autovalores y autovectores de la matriz y los mostramos
cov_mat = np.cov(X_train)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

'''
Si lo que queremos es reducir la dimensionalidad del dataset, perdiendo la menor información posible,
descartaremos los autovectores cuyos autovalores sean más bajos, ya que son aquellos que menos información aportan al 
conjunto global. Para ello, lo que se hace es ordenarlos por parejas de autovector, autovalor.
'''

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visualizamos la lista de autovalores en orden desdenciente
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])

#Verifico la cantidad de estos valores que existen
print(len(eig_pairs))

'''
El objetivo de este caso es proyectar este dataset 4D en un espacio de menor dimensionalidad, para mejorar la eficiencia 
de cálculo, al mismo tiempo que se retiene la mayor parte de la información. La pregunta clave será ¿cuál va ser este valor? 
¿3D?¿2D?¿1D?. Para ello seguiremos el siguiente proceso.

Una vez ordenados los autovalores, que recordamos son una medida de la varianza de los datos, la cuestión es decidir, 
cuál es el menor número de autovectores o componentes principales, con el que podemos expresar “la esencia principal” 
de la información contenida en ese dataset. Para ello, usaremos un métrica que se conoce como “varianza explicada”, 
que muestra cuánta varianza se puede atribuir a cada una de estas componentes principales.
'''

# A partir de los autovalores, calculamos la varianza explicada
tot = sum(eig_vals)
print("Total -----> " + str(tot))

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print("Var_Exp -----> " + str(var_exp))

cum_var_exp = np.cumsum(var_exp)
print("Cum_Var_Exp -----> " + str(cum_var_exp[162]))

# Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(6, 4))
    plt.step(range(4000), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------------------------------------------------

#El código a continuación tiene .85 para el parámetro de número de componentes que obtenemos de la grafica anterior debido
#a que el codo de la grafica se produce en el valor 85. Por tanto en estos componentes se muestra cuánta varianza se puede
#atribuir a cada una de estas componentes principales.

#Significa que scikit-learn elige el número mínimo de componentes principales de manera que se retenga el 85% de la varianza.
pca = PCA(.85)

#Montar PCA en el set de entrenamiento.
pca.fit(X_train)

#Apply the mapping (transform) to both the training set and the test set.
x_train_img = pca.transform(X_train)
x_test_img = pca.transform(X_test)

#----------------------------------------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------------------------------------

#Mejores parametros obtenidos con 20000 imagenes de train:
# --------------
# Learning rate: 0.75
# N Estimators: 50
# Max Feature: 1
# Max Depth: 10
# Accuracy score (training): 1.000000
# Accuracy score (test): 0.936250
# --------------

#Mejores parametros obtenidos con 4000 imagenes de train y con PCA y CV:
#--------------
#Learning rate: 0.75
#N Estimators: 50
#Max Feature: 5
#Max Depth: 50
#Accuracy score (CV): 1.000000
#--------------

#----------------------------------------------------------------------------------------------------------------------

#Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

#Mejores parametros con 150 imagenes de cada clase
#gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)

#Mejores parametros con 20000 imagenes del train
#gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.75, max_features=1, max_depth = 10, random_state = 0)

#Mejores parametros con 40000 imagenes del train
#gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.75, max_features=2, max_depth = 10, random_state = 0)

#Mejores parametros con 4000 imagenes del train
gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.75, max_features=5, max_depth = 50, random_state = 0)

gb.fit(x_train_img, y_train)
predictions = gb.predict(x_test_img)

# Model Accuracy, how often is the classifier correct?
# 83.5% de accuracy
print("Accuracy:",round(metrics.accuracy_score(y_test, predictions) * 100, 2))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

print("Classification Report")
print(classification_report(y_test, predictions))

#----------------------------------------------------------------------------------------------------------------------

#Method 1 (Seaborn)
#This method produces a more understandable and visually readable confusion matrix using seaborn.

#Use score method to get accuracy of model
score = gb.score(x_test_img, y_test)
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

#En la siguiente línea configuro la carpeta y el nombre de como se guardara el modelo
MODEL_DIR = './modelos/model-GB-PCA'

#En la siguiente linea guardamos el primer modelo
pickle.dump(gb, open(MODEL_DIR, 'wb'))
