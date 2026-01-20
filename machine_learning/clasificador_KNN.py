
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd

from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors     import KNeighborsClassifier

# Módulos Propios:
from base_de_datos.lectura      import leer_archivos_MAG, leer_archivo_Fruchtman
from base_de_datos.conversiones import segundos_a_día

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Entrenamiento:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def entrenar(directorio: str, años: list[str], ventana: int = 600, vecinos: int = 15, promedio: int = 1) -> Pipeline:
  """
  La funcion entrenar recibe un directorio en formato string que contiene tanto las mediciones finales MAG recortadas como los bow shocks
  detectados por Fruchtman, y una lista de strings en la variable 'años' que representa los posibles valores de entre '2014' y '2019' (que 
  corresponden a los bow shocks del catálogo de Fruchtman) con los que se desea entrenar al KNN. El entero 'ventana' representa el ancho (en
  valores de tiempo en segundos) de mediciones NO-Bow Shock (NBS) que se desean contemplar y que se encuentran inmediatamente junto a éstos
  (los ya detectados de Fruchtman), y el entero vecinos, representa el número de k-vecinos que se desea utilizar para el KNN. La función
  devuelve el Pipeline del entrenamiento.
  Los conjuntos BS y NBS se balancean para evitar sesgos en la clasificación, y el pipeline entrenado = StandardScaler + KNeighborsClassifier.
  """
  X_BS,  y_BS  =    bow_shocks_fruchtman(directorio, años)                    # Construcción del dataset
  X_NBS, y_NBS = no_bow_shocks_fruchtman(directorio, años, ventana, promedio) # 
  N: int = len(X_BS)                                                          # Balanceo de clases
  X_entrenamiento: np.ndarray = np.vstack(     [X_BS, X_NBS[:N]])             # Dataset final
  y_entrenamiento: np.ndarray = np.concatenate([y_BS, y_NBS[:N]])             # 
  secuencia_KNN: Pipeline = Pipeline([                                        # Pipeline KNN
    ('scaler', StandardScaler()),                                             # 
    ('knn', KNeighborsClassifier(                                             # 
      n_neighbors=vecinos, weights='distance', metric='euclidean'             # 
    ))                                                                        # 
  ])                                                                          # 
  secuencia_KNN.fit(X_entrenamiento, y_entrenamiento)                         # 
  return secuencia_KNN                                                        # 

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def vector_característico(data: pd.DataFrame) -> np.ndarray:
  """
  La función vector_característico recibe un parámetro 'data' que representa un dataframe del tipo:
  
  Día_decimal | Bx | By | Bz | Xpc | Ypc | Zpc | Xss | Yss | Zss
  ....          ..   ..   ..   ...   ...   ...   ...   ...   ...,
  
  ya sea por un archivo de fruchtman o de varios archivos del recorte final (Vignes) de los datos, y construye el vector de características
  (una matriz de N-filas por 4 columnas) a utilizar por el algoritmo KNN. Se eligió tomar como este vector a aquél que contenga el módulo de B
  y las componentes cartesianas del sistema de coordenadas Sun-State. Éstos vectores se colocan en el vector resultante en forma vertical:
  X = [|B|, Xss, Yss, Zss], es decir:

  |B| | Xss | Yss | Zss
  ..    ...   ...   ...
  """
  Bx,By,Bz,Xss,Yss,Zss = [data[j].to_numpy() for j in [1,2,3,7,8,9]] # Extraigo las componentes de B (que son en sistema PC), y coordenadas SS.
  B_modulo             = np.sqrt(Bx**2 + By**2 + Bz**2)              # Calculo el módulo de B (coincide con el del sistema SS).
  X                    = np.column_stack([B_modulo, Xss, Yss, Zss])  # Apilo verticalmente el módulo de B y las posiciones en la matriz X.
  return X                                                           # Devuelvo la matriz X (col_0 = |B|, col_1 = Xss, etc.).

#———————————————————————————————————————————————————————————————————————————————————————
def bow_shocks_fruchtman(directorio: str, años: list[str]) -> tuple[np.ndarray, np.ndarray]:
  """
  La función bow_shocks_fruchtman recibe en formato string un 'directorio' que contiene una carpeta (y subcarpetas) con los archivos .txt de
  los bow shocks detectados por Fruchtman entre los años 2014 y 2019, y una lista de strings 'años' que representa la cantidad de años que
  se desean cargar. La función calcula el vector característico de los archivos de Fruchtman de los años pasados por parámetro y devuelve una
  tupla cuyo primer elemento es una matriz con esos vectores característicos apilados, y el segundo es un vector de unos (1's) de longitud
  igual a esa matriz de vectores, que representa las etiquetas "Bow Shock" para esos valores de |B| y posiciones SS.
  """
  lista: list[np.ndarray] = []                                      # Inicializo una lista de np.ndarrays vacía.
  for año in años:                                                  # Para cada año de los pasados por parámetro,
    data_BS: pd.DataFrame = leer_archivo_Fruchtman(directorio, año) # leo todo el contenido del archivo en la variable 'data_BS',
    X_BS_año: np.ndarray  = vector_característico(data_BS)          # guardo el vector característico del archivo en la variable 'X_BS_año',
    lista.append(X_BS_año)                                          # y agrego este np.ndarray a la lista.
  X_BS: np.ndarray = np.vstack(lista)                               # En 'X_BS', apilo elementos de lista => matriz de dimensión (N_total, 4).
  y_BS: np.ndarray = np.ones(len(X_BS), dtype=int)                  # En 'y_BS' genero un vector de etiquetas con unos (1's) de longitud X_BS.
  return X_BS, y_BS                                                 # Devuelvo la matriz y el vector de etiquetas.

#———————————————————————————————————————————————————————————————————————————————————————
def no_bow_shocks_fruchtman(directorio: str, años: list[str], ventana: int, promedio: int = 1) -> tuple[np.ndarray, np.ndarray]:
  """
  La función no_bow_shocks_fruchtman recibe en formato string un 'directorio' que contiene una carpeta (y subcarpetas) con los archivos .txt
  de los bow shocks detectados por Fruchtman entre los años 2014 y 2019, y que a su vez contiene las subcarpetas correspondientes que contienen
  los archivos MAG finales, recortados por hemisferio norte y región de Vignes. La función recibe una lista de strings 'años' que representa
  la cantidad de años que se desean cargar y calcula el vector característico para las regiones 'no-Bow Shock' al igual que la función
  bow_shocks_fruchtman y devuelve una tupla de la matriz de no bow shocks, y un vector de ceros (etiqueta no-BS) de dicha longitud.
  """
  lista: list[np.ndarray] = []                                              # Inicializo una lista de np.ndarrays vacía.
  ruta_final: str = os.path.join(directorio, 'recorte_Vignes')              # Obtengo la ruta final de los archivos MAG con todos los recortes.
  for año in años:                                                          # Para cada año de los pasados por parámetro,
    t_0, t_f = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'               # obtengo el tiempo inicial y final (todo el año) para la función
    data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_final,t_0,t_f,promedio) # leer_archivos_MAG => obtengo las mediciones del año en data_MAG,
    data_BS: pd.DataFrame  = leer_archivo_Fruchtman(directorio, año)        # leo todo el contenido del archivo Fruchtman en 'data_BS', y
    data_NBS: pd.DataFrame = obtener_bordes(data_MAG,                       # obtengo los bordes de los BS en 'data_NBS' con los datos de MAG.
                                            data_BS[0].to_numpy(),ventana)  # y ajusto la ventana en segundos que quiero utilizar.
    X_NBS: np.ndarray      = vector_característico(data_NBS)                # Creo y guardo el vector característico de los NO-BS en 'X_NBS'
    lista.append(X_NBS)                                                     # y agrego éstos a la lista.
  X_NBS: np.ndarray = np.vstack(lista)                                      # En 'X_NBS', apilo elementos de lista => matriz de (años, N).
  y_NBS: np.ndarray = np.zeros(len(X_NBS), dtype=int)                       # En 'y_BS' genero un vector de etiquetas cero de longitud X_NBS.
  return X_NBS, y_NBS                                                       # Devuelvo la matriz y el vector de etiquetas.

#———————————————————————————————————————————————————————————————————————————————————————
def obtener_bordes(data_MAG: pd.DataFrame, tiempos_BS: np.ndarray, ventana: int = 600) -> pd.DataFrame:
  """
  La función obtener_bordes recibe un dataframe con datos de la sonda MAG y un np.ndarray de tiempos que contiene el tiempo de los bow shocks
  cuyas muestras vecinas se desea extraer. Además, recibe un parámetro float ('ventana') que permite ajustar la ventana temporal (en segundos)
  de mediciones no bow shock que se desea obtener. La función devuelve un dataframe que representa los datos vecinos (obtenidos de 'data_MAG')
  a los tiempos de bow shocks ingresados, exceptuando a los bow shocks en sí mismos.
  """
  t: np.ndarray = data_MAG.iloc[:,0].to_numpy(dtype=float) # Obtengo el vector de tiempos del archivo MAG.
  máscara: np.ndarray = np.zeros(len(t), dtype=bool)       # Creo una máscara booleana de long t (de MAG) llena de False ([False, ..., False]).
  t_día: float = segundos_a_día(ventana)                   # Obtengo el tiempo en formato día (entre 0 y 1).
  for t_BS in tiempos_BS:                                  # Para cada tiempo de bow shock de 'tiempos_BS',
    cercano = np.abs(t-t_BS) < t_día                       # creo una máscara booleana. Si |t-t_BS| < t_día => es True.
    máscara |= cercano                                     # Máscara (OR=) Cercano (sobreescribo máscara con cada elemento True en 'cercano').
  for t_BS in tiempos_BS:                                  # Para cada tiempo de bow shock de 'tiempos_BS'
    máscara &= (t_BS != t)                                 # Máscara (AND=) no-BS (elimino BS, pues los lejanos no fueron True al pasar por el
  data_NBS: pd.DataFrame = data_MAG[máscara]               # primer for). En data_NBS, obtengo los datos MAG con máscara de elementos cercanos.
  return data_NBS                                          # Devuelvo dicha data de no-bow shocks.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Predicción de Nuevos Bow Shocks: Devuelve los BS que se han identificado en un año dado.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def predecir_bow_shocks(knn: Pipeline, directorio: str, año: str, promedio: int = 1) -> pd.DataFrame:
  """
  La función predecir_bow_shocks recibe el algoritmo KNN, y un directorio y un año en formatos string, que representan el directorio donde se
  encuentran los archivos con las mediciones de MAG, y el año cuyos bow shocks debido a transición de la posición de MAVEN de la región
  upstream a downstream, o viceversa, desean predecirse. Devuelve un dataframe con aquellas mediciones que han sido etiquetadas como 1 (bow
  shock) que el algoritmo KNN ha aprendido a etiquetar luego de su entrenamiento.
  """
  t_0, t_f = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'               # Obtengo el tiempo inicial y final (todo el año) para la función
  ruta_final: str = os.path.join(directorio, 'recorte_Vignes')            # Obtengo la ruta final de los archivos MAG con todos los recortes.
  data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_final,t_0,t_f,promedio) # leer_archivos_MAG => obtengo las mediciones del año en 'data_MAG',
  X: np.ndarray          = vector_característico(data_MAG)                # Calculo el vector característico de todos esos datos,
  y_pred: np.ndarray     = knn.predict(X)                                 # y obtengo la predicción del algoritmo KNN.
  return data_MAG[y_pred == 1]                                            # Devuelvo los datos que hayan sido etiquetados como 1 (bow shock).

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————