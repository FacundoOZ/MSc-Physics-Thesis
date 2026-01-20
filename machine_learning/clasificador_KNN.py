
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors     import KNeighborsClassifier

# Módulos Propios:
from base_de_datos.lectura      import leer_archivos_MAG, leer_archivo_Fruchtman
from base_de_datos.conversiones import segundos_a_día

def atributos(B, Xss, Yss, Zss) -> np.ndarray:
  return np.array([B.mean(), B.std(), np.max(np.abs(np.diff(B))), Xss.mean(), Yss.mean(), Zss.mean()]) # Vector característico de una ventana.

def construir_dataset(mag: pd.DataFrame, fruchtman: pd.DataFrame, ventana: int) -> tuple[np.ndarray, np.ndarray]:
  """
  Construye X, y a partir de MAG y Fruchtman.
  mag columns:
    ['t', '|B|', 'Xss', 'Yss', 'Zss']
  fruchtman column:
    ['t']
  """
  t_mag = mag['t'].to_numpy()
  t_bs  = fruchtman['t'].to_numpy()
  X, y = [], []
  i = 0
  N = len(mag)
  while i + ventana <= N:
    V = mag.iloc[i:i+ventana] # V = VENTANA MAG
    t0 = V['t'].iloc[0]
    tf = V['t'].iloc[-1]
    es_BS = np.any((t_bs >= t0) & (t_bs <= tf))    # --- label ---
    y.append(int(es_BS))
    X.append(atributos(V['|B|'].to_numpy(), V['Xss'].to_numpy(), V['Yss'].to_numpy(), V['Zss'].to_numpy())) # --- features ---
    i += ventana  # sin overlap
  return np.array(X), np.array(y)

def entrenar_KNN(mag: pd.DataFrame, fruchtman: pd.DataFrame, ventana: int, K: int = 5):
  X, y = construir_dataset(mag, fruchtman, ventana)
  if len(np.unique(y)) < 2:
    raise ValueError(f"El dataset tiene una sola clase. BS={y.sum()}, NBS={len(y)-y.sum()}")
  if K >= len(X):
    raise ValueError(f"K={K} es demasiado grande para {len(X)} muestras.")
  scaler = StandardScaler()
  Xs = scaler.fit_transform(X)
  knn = KNeighborsClassifier(n_neighbors=K)
  knn.fit(Xs, y)
  return knn, scaler





#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————