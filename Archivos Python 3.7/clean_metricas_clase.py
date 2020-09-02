#Calculamos las metricas de varianza,deviacion y media.

#Metricas de las clases ['Agachado','CaidaTipoFin','Parado','Vacio']
import os
import pandas as pd
import numpy as np
from imgcompare import is_equal, image_diff_percent
from prettytable import PrettyTable

#Cargamos el archivo CSV que se guardaron los valores de diferencia de las imagenes
df = pd.read_csv('df_porcent_diferent.csv')

#Imprimo las cabeceras de este dataset y las 4 x 4 filas columnas
print(df.head())
print(df.iloc[0:4, 0:4])

#Imprimo la tupla que indica que este dataframe no es simetrico ya que tiene 401 filas y 402 columnas
df.shape

#Creamos un nuevo dataset donde quitaremos la fila 0 que está vacía y las columnas 0 y 1 que tienen el numero de indice y el nombre de las imagenes
df_clean = df.iloc[1:(df.shape[0]+1),2:(df.shape[1]+1)]

#Imprimimos la tupla que nos informa sobre la cantidad de filas x columnas del dataframe ya listo para ser utilizado porque tiene 400 x 400 de dimension
df_clean.shape

#Imprimimos los primeros 4 valores de la columna 0 del dataframe para comprobar que aunque se muestra la primera columna con el numero de indice de la fila, esta fue eliminada y el df ya es simetrico
print(df_clean.iloc[0:4, 0])

#Funcion que muestra todos los datos del dataframe que se le pase por parametros segun la cantidad de filas y columnas que se especifique
def displaydf(dataframe, cols = 10, rows = 10):
    with pd.option_context("display.max_columns", cols):
        with pd.option_context("display.max_rows", rows):
            print(dataframe)
    return True


#####################################################################################################################################################################
#Clase Agachado
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Agachado por lo tanto el rango es hasta 100 ya que la ultima fila o columna no se tiene en cuenta cuando se utiliza iloc[] y se cuenta a partir de 0 y son 100 imagenes de cada clase
df_agachado = df_clean.iloc[0:100,0:100]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Agachado
df_agachado.shape

#Imprimo el ultimo valor de la coordenada 99 x 99 que tiene que dar 0 para que coincida con la diagonal
df_agachado.iloc[99,99]

#Invoco la rutina para que muestre dataframe completo
displaydf(df_agachado)

#Convierto el dataframe en una matriz de 100 x 100
matriz_agachados = df_agachado.to_numpy()

#Imprimo los primeros 4 valores de esta matriz para comprobar que la diagonal esta bien y todos los valores de la misma son 0
print(matriz_agachados[0:4, 0:4])

#Obtengo los indices de abajo de la diagonal para calcular la media de la clase agachado
diagonal_inferior_indices_agachado = np.tril_indices_from(matriz_agachados, -1)

#Imprimo los indices para comprobar que son los valores que me interesan y que no incluyeron los valores 0 de la diagonal
print(diagonal_inferior_indices_agachado)

#Creo un array donde le declaro que los valores que tendra seran los valores de los indices obtenidos anteriormente
array_agachado_indices = matriz_agachados[diagonal_inferior_indices_agachado]

#Imprimimos el array y comprobamos que tiene los mismos valores que la diagonal inferior
print(array_agachado_indices[0:4])

#Creo un nuevo array pasandole como los valores anteriores pero convirtiendolos a float para poder calcular las metricas
array_agachado = np.array(array_agachado_indices, float)

#Compruebo que desaparecieron las comillas simple por tanto los valores son float
print(array_agachado[0:4])

#Guardo la media en una variable y la imprimo
media_agachados = round(array_agachado.mean(),2)
print(media_agachados)

#Guardo la varianza en una variable y la imprimo
varianza_agachados = round(array_agachado.var(),2)
print(varianza_agachados)

#Guardo la desviacion en una variable y la imprimo
desviacion_agachados = round(array_agachado.std(),2)
print(desviacion_agachados)


#####################################################################################################################################################################
#Clase CaidaTipoFin
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase CaidaTipoFin por lo tanto el rango es hasta 100-200 ya que la ultima fila o columna no se tiene en cuenta cuando se utiliza iloc[]
df_CaidaTipoFin = df_clean.iloc[100:200,100:200]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase CaidaTipoFin
df_CaidaTipoFin.shape

#Imprimo el ultimo valor de la coordenada 99 x 99 que tiene que dar 0 para que coincida con la diagonal
df_CaidaTipoFin.iloc[99,99]

#Invoco la rutina para que muestre dataframe completo
displaydf(df_CaidaTipoFin)

#Convierto el dataframe en una matriz de 100 x 100
matriz_CaidaTipoFin = df_CaidaTipoFin.to_numpy()

#Imprimo los primeros 4 valores de esta matriz para comprobar que la diagonal esta bien y todos los valores de la misma son 0
print(matriz_CaidaTipoFin[0:4, 0:4])

#Obtengo los indices de abajo de la diagonal para calcular la media de la clase CaidaTipoFin
diagonal_inferior_indices_CaidaTipoFin = np.tril_indices_from(matriz_CaidaTipoFin, -1)

#Imprimo los indices para comprobar que son los valores que me interesan y que no incluyeron los valores 0 de la diagonal
print(diagonal_inferior_indices_CaidaTipoFin)

#Creo un array donde le declaro que los valores que tendra seran los valores de los indices obtenidos anteriormente
array_CaidaTipoFin_indices = matriz_CaidaTipoFin[diagonal_inferior_indices_CaidaTipoFin]

#Imprimimos el array y comprobamos que tiene los mismos valores que la diagonal inferior
print(array_CaidaTipoFin_indices[0:4])

#Creo un nuevo array pasandole como los valores anteriores pero convirtiendolos a float para poder calcular las metricas
array_CaidaTipoFin = np.array(array_CaidaTipoFin_indices, float)

#Compruebo que desaparecieron las comillas simple por tanto los valores son float
print(array_CaidaTipoFin[0:4])

#Guardo la media en una variable y la imprimo
media_CaidaTipoFin = round(array_CaidaTipoFin.mean(),2)
print(media_CaidaTipoFin)

#Guardo la varianza en una variable y la imprimo
varianza_CaidaTipoFin = round(array_CaidaTipoFin.var(),2)
print(varianza_CaidaTipoFin)

#Guardo la desviacion en una variable y la imprimo
desviacion_CaidaTipoFin = round(array_CaidaTipoFin.std(),2)
print(desviacion_CaidaTipoFin)


#####################################################################################################################################################################
#Clase Parado
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Parado por lo tanto el rango es hasta 200-300 ya que la ultima fila o columna no se tiene en cuenta cuando se utiliza iloc[]
df_Parado = df_clean.iloc[200:300,200:300]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Parado
df_Parado.shape

#Imprimo el ultimo valor de la coordenada 99 x 99 que tiene que dar 0 para que coincida con la diagonal
df_Parado.iloc[99,99]

#Invoco la rutina para que muestre dataframe completo
displaydf(df_Parado)

#Convierto el dataframe en una matriz de 100 x 100
matriz_Parado = df_Parado.to_numpy()

#Imprimo los primeros 4 valores de esta matriz para comprobar que la diagonal esta bien y todos los valores de la misma son 0
print(matriz_Parado[0:4, 0:4])

#Obtengo los indices de abajo de la diagonal para calcular la media de la clase Parado
diagonal_inferior_indices_Parado = np.tril_indices_from(matriz_Parado, -1)

#Imprimo los indices para comprobar que son los valores que me interesan y que no incluyeron los valores 0 de la diagonal
print(diagonal_inferior_indices_Parado)

#Creo un array donde le declaro que los valores que tendra seran los valores de los indices obtenidos anteriormente
array_Parado_indices = matriz_Parado[diagonal_inferior_indices_Parado]

#Imprimimos el array y comprobamos que tiene los mismos valores que la diagonal inferior
print(array_Parado_indices[0:4])

#Creo un nuevo array pasandole como los valores anteriores pero convirtiendolos a float para poder calcular las metricas
array_Parado = np.array(array_Parado_indices, float)

#Compruebo que desaparecieron las comillas simple por tanto los valores son float
print(array_Parado[0:4])

#Guardo la media en una variable y la imprimo
media_Parado = round(array_Parado.mean(),2)
print(media_Parado)

#Guardo la varianza en una variable y la imprimo
varianza_Parado = round(array_Parado.var(),2)
print(varianza_Parado)

#Guardo la desviacion en una variable y la imprimo
desviacion_Parado = round(array_Parado.std(),2)
print(desviacion_Parado)


#####################################################################################################################################################################
#Clase Vacio
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Vacio por lo tanto el rango es hasta 300-400 ya que la ultima fila o columna no se tiene en cuenta cuando se utiliza iloc[]
df_Vacio = df_clean.iloc[300:400,300:400]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Vacio
df_Vacio.shape

#Imprimo el ultimo valor de la coordenada 99 x 99 que tiene que dar 0 para que coincida con la diagonal
df_Vacio.iloc[99,99]

#Invoco la rutina para que muestre dataframe completo
displaydf(df_Vacio)

#Convierto el dataframe en una matriz de 100 x 100
matriz_Vacio = df_Vacio.to_numpy()

#Imprimo los primeros 4 valores de esta matriz para comprobar que la diagonal esta bien y todos los valores de la misma son 0
print(matriz_Vacio[0:4, 0:4])

#Obtengo los indices de abajo de la diagonal para calcular la media de la clase Vacio
diagonal_inferior_indices_Vacio = np.tril_indices_from(matriz_Vacio, -1)

#Imprimo los indices para comprobar que son los valores que me interesan y que no incluyeron los valores 0 de la diagonal
print(diagonal_inferior_indices_Vacio)

#Creo un array donde le declaro que los valores que tendra seran los valores de los indices obtenidos anteriormente
array_Vacio_indices = matriz_Vacio[diagonal_inferior_indices_Vacio]

#Imprimimos el array y comprobamos que tiene los mismos valores que la diagonal inferior
print(array_Vacio_indices[0:4])

#Creo un nuevo array pasandole como los valores anteriores pero convirtiendolos a float para poder calcular las metricas
array_Vacio = np.array(array_Vacio_indices, float)

#Compruebo que desaparecieron las comillas simple por tanto los valores son float
print(array_Vacio[0:4])

#Guardo la media en una variable y la imprimo
media_Vacio = round(array_Vacio.mean(),2)
print(media_Vacio)

#Guardo la varianza en una variable y la imprimo
varianza_Vacio = round(array_Vacio.var(),2)
print(varianza_Vacio)

#Guardo la desviacion en una variable y la imprimo
desviacion_Vacio = round(array_Vacio.std(),2)
print(desviacion_Vacio)



#####################################################################################################################################################################
#Clase Agachado-CaidaTipoFin
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Agachado-CaidaTipoFin
df_agachado_CaidaTipoFin = df_clean.iloc[0:100,100:200]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Agachado-CaidaTipoFin
df_agachado_CaidaTipoFin.shape

#Realizo diferentes comprobaciones ya que aqui el elemento en la posicion 99 x 99 no tiene que dar 0
#Compruebo los valores directamente con los del CSV
df_agachado_CaidaTipoFin.iloc[0,0]
df_agachado_CaidaTipoFin.iloc[99,99]
df_agachado_CaidaTipoFin.iloc[0,99]

#Convierto el dataframe en una matriz de 100 x 100
matriz_agachado_CaidaTipoFin = df_agachado_CaidaTipoFin.to_numpy()

#Creo un nuevo array pasandole los valores anteriores de la matriz matriz_agachado_CaidaTipoFin pero pasandole el metodo flatten() a esta matriz para convertirla en array
#Tambien indico que son valores float para poder calcular las metricas
#Aqui no hay que trabajar solo con los elementos de un lado de la diagonal ya que las coordenadas escogidas al principio df_clean.iloc[0:100,100:200] provoca que los 100 x 100 valores sean relevantes en el dataframe
#En el ppt se explica detalladamente
array_agachado_CaidaTipoFin = np.array(matriz_agachado_CaidaTipoFin.flatten(),float)

#Se imprime la longitud de la matriz para comprobar que se tomaron lo 100 * 100 = 10000 valores de la matriz
print(len(array_agachado_CaidaTipoFin))

#Guardo la media en una variable y la imprimo
media_agachado_CaidaTipoFin = round(array_agachado_CaidaTipoFin.mean(),2)
print(media_agachado_CaidaTipoFin)

#Guardo la varianza en una variable y la imprimo
varianza_agachado_CaidaTipoFin = round(array_agachado_CaidaTipoFin.var(),2)
print(varianza_agachado_CaidaTipoFin)

#Guardo la desviacion en una variable y la imprimo
desviacion_agachado_CaidaTipoFin = round(array_agachado_CaidaTipoFin.std(),2)
print(desviacion_agachado_CaidaTipoFin)

#####################################################################################################################################################################
#Clase Agachado-Parado
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Agachado-Parado
df_agachado_Parado = df_clean.iloc[0:100,200:300]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Agachado-Parado
df_agachado_Parado.shape

#Realizo diferentes comprobaciones ya que aqui el elemento en la posicion 99 x 99 no tiene que dar 0
#Compruebo los valores directamente con los del CSV
df_agachado_Parado.iloc[0,0]
df_agachado_Parado.iloc[99,99]
df_agachado_Parado.iloc[0,99]

#Convierto el dataframe en una matriz de 100 x 100
matriz_agachado_Parado = df_agachado_Parado.to_numpy()

#Creo un nuevo array pasandole los valores anteriores de la matriz matriz_agachado_Parado pero pasandole el metodo flatten() a esta matriz para convertirla en array
#Tambien indico que son valores float para poder calcular las metricas
#Aqui no hay que trabajar solo con los elementos de un lado de la diagonal ya que las coordenadas escogidas al principio df_clean.iloc[0:100,100:200] provoca que los 100 x 100 valores sean relevantes en el dataframe
#En el ppt se explica detalladamente
array_agachado_Parado = np.array(matriz_agachado_Parado.flatten(),float)

#Se imprime la longitud de la matriz para comprobar que se tomaron lo 100 * 100 = 10000 valores de la matriz
print(len(array_agachado_Parado))

#Guardo la media en una variable y la imprimo
media_agachado_Parado = round(array_agachado_Parado.mean(),2)
print(media_agachado_Parado)

#Guardo la varianza en una variable y la imprimo
varianza_agachado_Parado = round(array_agachado_Parado.var(),2)
print(varianza_agachado_Parado)

#Guardo la desviacion en una variable y la imprimo
desviacion_agachado_Parado = round(array_agachado_Parado.std(),2)
print(desviacion_agachado_Parado)

#####################################################################################################################################################################
#Clase Agachado-Vacio
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Agachado-Vacio
df_agachado_Vacio = df_clean.iloc[0:100,300:400]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Agachado-Vacio
df_agachado_Vacio.shape

#Realizo diferentes comprobaciones ya que aqui el elemento en la posicion 99 x 99 no tiene que dar 0
#Compruebo los valores directamente con los del CSV
df_agachado_Vacio.iloc[0,0]
df_agachado_Vacio.iloc[99,99]
df_agachado_Vacio.iloc[0,99]

#Convierto el dataframe en una matriz de 100 x 100
matriz_agachado_Vacio = df_agachado_Vacio.to_numpy()

#Creo un nuevo array pasandole los valores anteriores de la matriz matriz_agachado_Vacio pero pasandole el metodo flatten() a esta matriz para convertirla en array
#Tambien indico que son valores float para poder calcular las metricas
#Aqui no hay que trabajar solo con los elementos de un lado de la diagonal ya que las coordenadas escogidas al principio df_clean.iloc[0:100,100:200] provoca que los 100 x 100 valores sean relevantes en el dataframe
#En el ppt se explica detalladamente
array_agachado_Vacio = np.array(matriz_agachado_Vacio.flatten(),float)

#Se imprime la longitud de la matriz para comprobar que se tomaron lo 100 * 100 = 10000 valores de la matriz
print(len(array_agachado_Vacio))

#Guardo la media en una variable y la imprimo
media_agachado_Vacio = round(array_agachado_Vacio.mean(),2)
print(media_agachado_Vacio)

#Guardo la varianza en una variable y la imprimo
varianza_agachado_Vacio = round(array_agachado_Vacio.var(),2)
print(varianza_agachado_Vacio)

#Guardo la desviacion en una variable y la imprimo
desviacion_agachado_Vacio = round(array_agachado_Vacio.std(),2)
print(desviacion_agachado_Vacio)

#####################################################################################################################################################################
#Clase Parado-CaidaTipoFin
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase parado-CaidaTipoFin
df_parado_CaidaTipoFin = df_clean.iloc[100:200,200:300]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase parado-CaidaTipoFin
df_parado_CaidaTipoFin.shape

#Realizo diferentes comprobaciones ya que aqui el elemento en la posicion 99 x 99 no tiene que dar 0
#Compruebo los valores directamente con los del CSV
df_parado_CaidaTipoFin.iloc[0,0]
df_parado_CaidaTipoFin.iloc[99,99]
df_parado_CaidaTipoFin.iloc[0,99]

#Convierto el dataframe en una matriz de 100 x 100
matriz_parado_CaidaTipoFin = df_parado_CaidaTipoFin.to_numpy()

#Creo un nuevo array pasandole los valores anteriores de la matriz matriz_parado_CaidaTipoFin pero pasandole el metodo flatten() a esta matriz para convertirla en array
#Tambien indico que son valores float para poder calcular las metricas
#Aqui no hay que trabajar solo con los elementos de un lado de la diagonal ya que las coordenadas escogidas al principio df_clean.iloc[0:100,100:200] provoca que los 100 x 100 valores sean relevantes en el dataframe
#En el ppt se explica detalladamente
array_parado_CaidaTipoFin = np.array(matriz_parado_CaidaTipoFin.flatten(),float)

#Se imprime la longitud de la matriz para comprobar que se tomaron lo 100 * 100 = 10000 valores de la matriz
print(len(array_parado_CaidaTipoFin))

#Guardo la media en una variable y la imprimo
media_parado_CaidaTipoFin = round(array_parado_CaidaTipoFin.mean(),2)
print(media_parado_CaidaTipoFin)

#Guardo la varianza en una variable y la imprimo
varianza_parado_CaidaTipoFin = round(array_parado_CaidaTipoFin.var(),2)
print(varianza_parado_CaidaTipoFin)

#Guardo la desviacion en una variable y la imprimo
desviacion_parado_CaidaTipoFin = round(array_parado_CaidaTipoFin.std(),2)
print(desviacion_parado_CaidaTipoFin)

#####################################################################################################################################################################
#Clase Vacio-CaidaTipoFin
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Vacio-CaidaTipoFin
df_Vacio_CaidaTipoFin = df_clean.iloc[100:200,300:400]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Vacio-CaidaTipoFin
df_Vacio_CaidaTipoFin.shape

#Realizo diferentes comprobaciones ya que aqui el elemento en la posicion 99 x 99 no tiene que dar 0
#Compruebo los valores directamente con los del CSV
df_Vacio_CaidaTipoFin.iloc[0,0]
df_Vacio_CaidaTipoFin.iloc[99,99]
df_Vacio_CaidaTipoFin.iloc[0,99]

#Convierto el dataframe en una matriz de 100 x 100
matriz_Vacio_CaidaTipoFin = df_Vacio_CaidaTipoFin.to_numpy()

#Creo un nuevo array pasandole los valores anteriores de la matriz matriz_Vacio_CaidaTipoFin pero pasandole el metodo flatten() a esta matriz para convertirla en array
#Tambien indico que son valores float para poder calcular las metricas
#Aqui no hay que trabajar solo con los elementos de un lado de la diagonal ya que las coordenadas escogidas al principio df_clean.iloc[0:100,100:200] provoca que los 100 x 100 valores sean relevantes en el dataframe
#En el ppt se explica detalladamente
array_Vacio_CaidaTipoFin = np.array(matriz_Vacio_CaidaTipoFin.flatten(),float)

#Se imprime la longitud de la matriz para comprobar que se tomaron lo 100 * 100 = 10000 valores de la matriz
print(len(array_Vacio_CaidaTipoFin))

#Guardo la media en una variable y la imprimo
media_Vacio_CaidaTipoFin = round(array_Vacio_CaidaTipoFin.mean(),2)
print(media_Vacio_CaidaTipoFin)

#Guardo la varianza en una variable y la imprimo
varianza_Vacio_CaidaTipoFin = round(array_Vacio_CaidaTipoFin.var(),2)
print(varianza_Vacio_CaidaTipoFin)

#Guardo la desviacion en una variable y la imprimo
desviacion_Vacio_CaidaTipoFin = round(array_Vacio_CaidaTipoFin.std(),2)
print(desviacion_Vacio_CaidaTipoFin)

#####################################################################################################################################################################
#Clase Vacio-Parado
#####################################################################################################################################################################

#Escojo solamente las columnas de la clase Vacio-Parado
df_Vacio_Parado = df_clean.iloc[100:200,300:400]

#Imprimo la tupla de filas x columnas que me indica que tengo los valores de las 100 imagenes de la clase Vacio-Parado
df_Vacio_Parado.shape

#Realizo diferentes comprobaciones ya que aqui el elemento en la posicion 99 x 99 no tiene que dar 0
#Compruebo los valores directamente con los del CSV
df_Vacio_Parado.iloc[0,0]
df_Vacio_Parado.iloc[99,99]
df_Vacio_Parado.iloc[0,99]

#Convierto el dataframe en una matriz de 100 x 100
matriz_Vacio_Parado = df_Vacio_Parado.to_numpy()

#Creo un nuevo array pasandole los valores anteriores de la matriz matriz_Vacio_Parado pero pasandole el metodo flatten() a esta matriz para convertirla en array
#Tambien indico que son valores float para poder calcular las metricas
#Aqui no hay que trabajar solo con los elementos de un lado de la diagonal ya que las coordenadas escogidas al principio df_clean.iloc[0:100,100:200] provoca que los 100 x 100 valores sean relevantes en el dataframe
#En el ppt se explica detalladamente
array_Vacio_Parado = np.array(matriz_Vacio_Parado.flatten(),float)

#Se imprime la longitud de la matriz para comprobar que se tomaron lo 100 * 100 = 10000 valores de la matriz
print(len(array_Vacio_Parado))

#Guardo la media en una variable y la imprimo
media_Vacio_Parado = round(array_Vacio_Parado.mean(),2)
print(media_Vacio_Parado)

#Guardo la varianza en una variable y la imprimo
varianza_Vacio_Parado = round(array_Vacio_Parado.var(),2)
print(varianza_Vacio_Parado)

#Guardo la desviacion en una variable y la imprimo
desviacion_Vacio_Parado = round(array_Vacio_Parado.std(),2)
print(desviacion_Vacio_Parado)

#####################################################################################################################################################################
#Tabla de metricas
#####################################################################################################################################################################

#Clase Agachado-Agachado
#Clase Agachado-Parado
#Clase Agachado-Caída
#Clase Agachado-Vacío

#Clase Parado-Parado
#Clase Parado-Caída
#Clase Parado-Vacío

#Clase Caída-Caída
#Clase Caída-Vacío

#Clase Vacío-Vacío

#Declaro la variable donde invocare la rutina pasandole parametros
x = PrettyTable()

#Imprimo todas las metricas organizadas en una tabla
x.field_names = ["Clase"              , "Media"                     , "Varianza"                        , "Desviación"]
x.add_row(      ["Vacío-Vacío"        , media_Vacio                 , varianza_Vacio                    , desviacion_Vacio                  ])
x.add_row(      ["Caída-Vacío"        , media_Vacio_CaidaTipoFin    , varianza_Vacio_CaidaTipoFin       , desviacion_Vacio_CaidaTipoFin     ])
x.add_row(      ["Caída-Caída"        , media_CaidaTipoFin          , varianza_CaidaTipoFin             , desviacion_CaidaTipoFin           ])
x.add_row(      ["Parado-Vacío"       , media_Vacio_Parado          , varianza_Vacio_Parado             , desviacion_Vacio_Parado           ])
x.add_row(      ["Parado-Caída"       , media_parado_CaidaTipoFin   , varianza_parado_CaidaTipoFin      , desviacion_parado_CaidaTipoFin    ])
x.add_row(      ["Parado-Parado"      , media_Parado                , varianza_Parado                   , desviacion_Parado                 ])
x.add_row(      ["Agachado-Vacío"     , media_agachado_Vacio        , varianza_agachado_Vacio           , desviacion_agachado_Vacio         ])
x.add_row(      ["Agachado-Caída"     , media_agachado_CaidaTipoFin , varianza_agachado_CaidaTipoFin    , desviacion_agachado_CaidaTipoFin  ])
x.add_row(      ["Agachado-Parado"    , media_agachado_Parado       , varianza_agachado_Parado          , desviacion_agachado_Parado        ])
x.add_row(      ["Agachado-Agachado"  , media_agachados             , varianza_agachados                , desviacion_agachados              ])

#Imprimo
print(x)

#Declaro la variable donde invocare la rutina pasandole parametros
y = PrettyTable()

#Como la desviación es la raiz cuadrada de la varianza solamente imprimire una matriz de confucion por cada metrica
#Una matriz de confusión por la media
y.field_names = ["Media"            , "Agachado"                    , "Parado"                      , "Caído"                           ,"Vacío"                     ]
y.add_row(      ["Agachado"         , media_agachados               , media_agachado_Parado         , media_agachado_CaidaTipoFin       , media_agachado_Vacio       ])
y.add_row(      ["Parado"           , media_agachado_Parado         , media_Parado                  , media_parado_CaidaTipoFin         , media_Vacio_Parado         ])
y.add_row(      ["Caído"            , media_agachado_CaidaTipoFin   , media_parado_CaidaTipoFin     , media_CaidaTipoFin                , media_Vacio_CaidaTipoFin   ])
y.add_row(      ["Vacío"            , media_agachado_Vacio          , media_Vacio_Parado            , media_Vacio_CaidaTipoFin          , media_Vacio                ])

#Imprimo
print(y)

#Declaro la variable donde invocare la rutina pasandole parametros
w = PrettyTable()

#Como la desviación es la raiz cuadrada de la varianza solamente imprimire una matriz de confucion por cada metrica
#Una matriz de confusión por la varianza
w.field_names = ["Varianza"         , "Agachado"                       , "Parado"                         , "Caído"                              ,"Vacío"                     ]
w.add_row(      ["Agachado"         , varianza_agachados               , varianza_agachado_Parado         , varianza_agachado_CaidaTipoFin       , varianza_agachado_Vacio       ])
w.add_row(      ["Parado"           , varianza_agachado_Parado         , varianza_Parado                  , varianza_parado_CaidaTipoFin         , varianza_Vacio_Parado         ])
w.add_row(      ["Caído"            , varianza_agachado_CaidaTipoFin   , varianza_parado_CaidaTipoFin     , varianza_CaidaTipoFin                , varianza_Vacio_CaidaTipoFin   ])
w.add_row(      ["Vacío"            , varianza_agachado_Vacio          , varianza_Vacio_Parado            , varianza_Vacio_CaidaTipoFin          , varianza_Vacio                ])

#Imprimo
print(w)