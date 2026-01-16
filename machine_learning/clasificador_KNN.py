
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import numpy  as np
import pandas as pd

from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors     import KNeighborsClassifier

# Módulos Propios:
from base_de_datos.lectura import leer_archivos_MAG

# 3766 CHOQUES FRUCHTMAN
#data = leer_archivos_MAG(directorio, tiempo_inicial, tiempo_final)
#tiempo, Bx,By,Bz, X,Y,Z = [data[i].to_numpy() for i in range(7)]            #   Extraigo toda la información del .sts de ese intervalo

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# PASO 1: CONSTRUIR EL VECTOR CARACTERÍSTICO.
def construir_features(data: pd.DataFrame) -> np.ndarray:
  """
  Construye el vector de características: [|B|, Xss, Yss, Zss]
  """
  Bx,By,Bz,Xss,Yss,Zss = [data[j].to_numpy() for j in [1,2,3,7,8,9]]
  B_modulo = np.sqrt(Bx**2 + By**2 + Bz**2)
  X = np.column_stack([B_modulo, Xss, Yss, Zss])
  return X

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————

# PASO 2: CARGAR LOS CHOQUES DE FRUCHTMAN.
def cargar_bow_shocks_fruchtman(directorio: str, años: list[str]):
  X_list = []
  for año in años:
    path = f'{directorio}/fruchtman/merge/fruchtman_{año}_merge.txt'
    data = pd.read_csv(path, sep=' ', header=None)
    X_bs = construir_features(data)
    X_list.append(X_bs)
  X_bs = np.vstack(X_list)
  y_bs = np.ones(len(X_bs), dtype=int)
  return X_bs, y_bs

""" USO:
años_train = [2014, 2015, 2016, 2017, 2018, 2019]
X_bs, y_bs = cargar_bow_shocks_fruchtman(directorio, años_train)
"""

# PASO 3: CONSTRUIR LAS MUESTRAS "NO-BOW SHOCK".
def extraer_no_bow_shocks(data_maven: pd.DataFrame, tiempos_bs: np.ndarray, ventana: float = 0.001): # .001 ~ 86 seconds
  """
  Extrae puntos cercanos temporalmente a los bow shocks, excluyendo el evento central.
  """
  tiempos = data_maven[0].to_numpy()
  mask = np.zeros(len(tiempos), dtype=bool)
  for t_bs in tiempos_bs:
    cercano = np.abs(tiempos - t_bs) < ventana
    mask |= cercano
  for t_bs in tiempos_bs: # eliminar exactamente el punto BS
    mask &= (tiempos != t_bs)
  data_nbs = data_maven[mask]
  return data_nbs

def construir_clase_negativa(directorio: str, años: list[str]):
  X_list = []
  for año in años:
    # cargar MAVEN continuo del año
    data_maven = leer_archivos_MAG(directorio, tiempo_inicial=f'1/1/{año}-00:00:00', tiempo_final=f'31/12/{año}-23:59:59')
    # cargar tiempos de BS
    path_bs = f'{directorio}/fruchtman/merge/fruchtman_{año}_merge.txt'
    data_bs = pd.read_csv(path_bs, sep=' ', header=None)
    tiempos_bs = data_bs[0].to_numpy()
    data_nbs = extraer_no_bow_shocks(data_maven, tiempos_bs)
    X_nbs = construir_features(data_nbs)
    X_list.append(X_nbs)
  X_nbs = np.vstack(X_list)
  y_nbs = np.zeros(len(X_nbs), dtype=int)
  return X_nbs, y_nbs

# PASO 4: CONSTRUIR EL DATASET DE ENTRENAMIENTO
X_nbs, y_nbs = construir_clase_negativa(directorio, años_train)

# balance (important)
n_bs = len(X_bs)
X_nbs = X_nbs[:n_bs]
y_nbs = y_nbs[:n_bs]

X_train = np.vstack([X_bs, X_nbs])
y_train = np.concatenate([y_bs, y_nbs])

# PASO 5: ENTRENAR EL KNN
knn_pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean'))
])

knn_pipeline.fit(X_train, y_train)

# PASO 6: PROBAR
def detectar_bow_shocks_año(knn, directorio, año):
  data = leer_archivos_MAG(directorio, tiempo_inicial=f'1/1/{año}-00:00:00', tiempo_final=f'31/12/{año}-23:59:59')
  X = construir_features(data)
  y_pred = knn.predict(X)
  return data[y_pred == 1]

""" USO
bs_2021 = detectar_bow_shocks_año(knn_pipeline, directorio, 2021)
print(f"Bow shocks detectados en 2021: {len(bs_2021)}")
"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————