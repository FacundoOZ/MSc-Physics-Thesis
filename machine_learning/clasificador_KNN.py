
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import os
import numpy as np
import pandas as pd
import joblib

from typing                import Union
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Módulos Propios:
from base_de_datos.conversiones   import módulo, R_m
from base_de_datos.lectura        import leer_archivos_MAG
from machine_learning.estadística import estadística, estadística_módulos

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Clasificador_KNN_Binario : para la detección de Bow Shocks mediante mediciones de campo magnético del instrumento MAG de la sonda MAVEN.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class Clasificador_KNN_Binario:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Inicializador (Constructor):
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def __init__(
      self,                                                                                # Estado de la variable actual.
      K: int,                                                                              # Metaparámetro K para las ventanas vecinas.
      ventana: int = 300,                                                                  # Ancho de ventana (en segundos)
      variables: Union[list[str], None] = None,                                            # Variables a utilizar del vector característico.
      ventanas_NBS: list[int] = [-1,1,2]                                                   # Posición de ventanas_NBS respecto a ventana BS.
  ) -> None:
    """Inicializador de atributos de la clase Clasificador_KNN_Binario para la detección de Bow Shocks. El metaparámetro 'K' es un entero que
    determina la cantidad de ventanas vecinas que se utilizarán para entrenar al algoritmo KNN (recomendado entre 1 y 30). El parámetro
    'ventana' es un entero que permite ajustar la cantidad de tiempo en segundos que tendrá el ancho de las ventanas para entrenar al
    algoritmo KNN (recomendado entre 60 y 600), y cuya finalidad es representar de forma adecuada el ancho de duración de los bow shocks.
    El parámetro 'variables' es una lista de strings que permite elegir qué magnitudes físicas medidas por el instrumento MAG de la sonda
    MAVEN se desean utilizar para el entrenamiento del algoritmo KNN. Si su valor es None, utiliza en forma predeterminada las variables
    ['B','Xss','Yss','Zss']. Los valores posibles son: ['B','R','Bx','By','Bz','Xpc','Ypc','Zpc','Xss','Yss','Zss']."""
    if K <= 0:                                                                             # Si K<=0, no es entero válido,
      raise ValueError("'K' debe ser un entero positivo (recomendado 1 ≤ K ≤ 30).")        # => devuelvo un mensaje.
    if ventana <= 0:                                                                       # Si la ventana es <= 0 no es un tiempo válido,
      raise ValueError("'ventana' debe ser un entero positivo (recomendado 60 ≤ v ≤ 600).")# => devuelvo un mensaje.
    if variables is None:                                                                  # Si no se definen las variables,
      variables: list[str] = ['B','Xss','Yss','Zss']                                       # utilizo las predeterminadas [|B|,Xss,Yss,Zss]
    if not all(isinstance(v, str) for v in variables):                                     # Si las variables pasadas por parámetro no son
      raise TypeError("'variables' debe ser una lista de strings.")                        # strings => devuelvo un mensaje.
    self.K: int                  = K                                                       # Metaparámetro K.
    self.ventana: int            = ventana                                                 # Ventana de puntos.
    self.variables: list[str]    = list(variables)                                         # Variables para entrenamiento.
    self.ventanas_NBS: list[int] = list(ventanas_NBS)                                      # Ventanas para el entrenamiento de zona NBS.
    self.entrenado: bool = False                                                           # Booleano del estado del KNN.
    self.scaler = StandardScaler()                                                         # Re-escaleo de variables.
    self.knn    = KNeighborsClassifier(                                                    # Clasificador KNN.
      n_neighbors=self.K,                                                                  # Número de vecinos.
      weights='distance',                                                                  # Pesos (distancia).
      metric='euclidean',                                                                  # Métrica a utilizar (euclídea predeterminada).
      n_jobs=-1                                                                            # Permite utilizar todo el CPU disponible.
    )

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Vector Característico:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def vector_característico(self, data_MAG: pd.DataFrame) -> np.ndarray:
    """La función vector característico recibe un dataframe 'data_MAG' de longitud menor o igual a 'ventana' del KNN, que es del tipo de
    archivo MAG recortado por recorte de Vignes, cuyas columnas son de la forma:
    día_decimal Bx By Bz Xpc Ypc Zpc Xss Yss Zss
    ....        .. .. .. ... ... ... ... ... ...
    y extrae el vector característico por ventana para alimentar al KNN, con las variables pasadas por parámetro al Clasificador_KNN_Binario.
    Devuelve una lista de valores float convertida a array (np.ndarray[list[float]]) donde cada valor representa las magnitudes estadísticas
    (del archivo estadística.py) correspondientes a las 'variables' pasadas por parámetro al KNN."""
    vector: list[float] = []                                                     # Inicializo el vector característico vacío como list[float]
    Bx,By,Bz,Xpc,Ypc,Zpc,Xss,Yss,Zss = [data_MAG[j].to_numpy() for j in range(1,10)]# Extraigo las magnitudes físicas del archivo tipo MAG. 
    B, R = módulo(Bx,By,Bz), módulo(Xss,Yss,Zss, norm=R_m)                       # Calculo módulos: NORMALIZO R! => B,R son del mismo orden. 
    dicc: dict[str,np.ndarray] = {                                               # Creo diccionario con todas las variables del archivo MAG.
      'Bx' : Bx , 'By' : By , 'Bz' : Bz ,                                        # Componentes de campo magnético (en sistema PC)
      'Xpc': Xpc, 'Ypc': Ypc, 'Zpc': Zpc, 'Xss': Xss, 'Yss': Yss, 'Zss': Zss}    # Componentes de posición de la sonda (en sistemas PC y SS).
    for var in self.variables:                                                   # Para cada elemento de la lista de variables de la clase KNN,
      if var in ('B','R'):                                                       # si las variables son B ó R (no pertenecen al diccionario),
        vector.extend(estadística_módulos(B if var=='B' else R))                 # calculo sus estadísticas (módulos) y las agrego a vector.
      elif var in dicc:                                                          # si no, si pertenecen al dicc,
        vector.extend(estadística(dicc[var]))                                    # calculo sus estadísticas (para componentes) y las agrego.
      else:                                                                      # Si no,
        raise ValueError(f"Variable desconocida: {var}")                         # la variables es inválida => devuelvo un mensaje.
    return np.array(vector)                                                      # Devuelvo el vector convertido a np.ndarray.

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Muestras de Entrenamiento BS y NBS:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def muestras_entrenamiento(self, data_MAG: pd.DataFrame, data_Fruchtman: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    La función muestras_entrenamiento recibe un dataframe 'data_MAG' que contiene los datos de las mediciones recortadas por Vignes del
    instrumento MAG en el intervalo temporal que se haya indicado (1 ó más años) y recibe un np.ndarray 'data_Fruchtman' que contiene los
    tiempos en los que ocurrieron los choques (en día decimal) que Fruchtman ha registrado, pertenecientes al mismo intervalo de tiempo
    indicado en MAG. La función calcula la ventana del BS Fruchtman y las ventanas NBS vecinas a éste para cada día decimal del archivo 
    Fruchtman, y luego calcula los vectores característicos de cada una. Devuelve una tupla cuya primera componente son los vectores 
    característicos BS y NBS, y cuya segunda componente son las etiquetas (1 para los BS y 0 para los NBS).
    """
    X: list[np.ndarray]            = []                                             # Inicializo una lista de vectores característicos 'X'.
    y: list[int]                   = []                                             # Inicializo una lista de etiquetas (enteros 0 ó 1) 'y'.
    media_ventana: int             = (self.ventana) // 2                            # Calculo el ancho de ventana/2 => división entera! (//).
    días_decimales_MAG: np.ndarray = data_MAG[0].to_numpy()                         # Obtengo días decimales del archivo MAG y convierto a np.
    for dia_decimal in data_Fruchtman:                                              # Para cada BS (día decimal) del archivo Fruchtman:
      j_BS: int        = np.searchsorted(días_decimales_MAG, dia_decimal)           # Busco el j de data_MAG más cercano a t_BS en O(log n).
      t0_BS: int       = max(0, j_BS - media_ventana)                               # El inicio de la ventana será dicho j - (ventana/2),
      tf_BS: int       = min(   j_BS + media_ventana, len(data_MAG))                # y el final será dicho j + (ventana/2).
      v_BS: np.ndarray = self.vector_característico(data_MAG.iloc[t0_BS : tf_BS])   # Calculo el vector característico para la ventana BS,
      X.append(v_BS)                                                                # lo agrego a la lista de vectores característicos X,
      y.append(1)                                                                   # y su etiqueta será 1 (bow shock).
      for desplazamiento in (self.ventanas_NBS):                                    # Para cada desplazamiento de la lista ventanas_NBS,
        j_NBS: int        = j_BS  + (desplazamiento*(self.ventana))                 # el j_NBS (centrado), será el j_BS + dicho desp*ventana.
        t0_NBS: int       = max(0, j_NBS - media_ventana)                           # Calculo el tiempo inicial,
        tf_NBS: int       = min(   j_NBS + media_ventana, len(data_MAG))            # y final de la ventana para el nuevo j_NBS.
        if t0_NBS < 0 or tf_NBS > len(data_MAG):                                    # Si los tiempos se salen de los límites del archivo,
          continue                                                                  # omito esta ventana.
        v_NBS: np.ndarray = self.vector_característico(data_MAG.iloc[t0_NBS:tf_NBS])# Si no, calculo el vector de la ventana NBS,
        X.append(v_NBS)                                                             # lo agrego a la lista de vectores característicos X,
        y.append(0)                                                                 # y su etiqueta será 0 (no bow shock).
    return np.array(X), np.array(y)                                                 # Convierto las listas a np.array para el KNN.

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Predecir:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def predict(self, data_MAG: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Docstring
    """
    etiqueta:     list[int]         = []                                   # 
    probabilidad: list[list[float]] = []                                   # 
    j_ventana:    list[int]         = []                                   # 
    for i in range(0, len(data_MAG), self.ventana // 2):                   # 
      j_0: int = i                                                         # 
      j_f: int = min(i + self.ventana, len(data_MAG))                      # 
      if j_f-j_0 >= ((self.ventana) // 2):                                 # 
        ventana: pd.DataFrame = data_MAG[j_0 : j_f]                        # 
        v: np.ndarray = self.vector_característico(ventana)                # 
        if v is not None:                                                  # 
          v_escalado: np.ndarray = self.scaler.transform(v.reshape(1,-1))  # 
          pred: int              = self.knn.predict(v_escalado)[0]         # 
          prob: np.ndarray       = self.knn.predict_proba(v_escalado)[0]   # 
          etiqueta.append(pred)                                            # 
          probabilidad.append(prob)                                        # 
        else:                                                              # 
          etiqueta.append(None)                                            # 
          probabilidad.append([None, None])                                # 
          continue                                                         # 
        j_ventana.append(j_0 + ((j_f-j_0) // 2))                           # 
    return np.array(etiqueta), np.array(probabilidad), np.array(j_ventana) # 

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # :
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def train(self, X, y):
    self.knn.fit(self.scaler.fit_transform(X), y)
    self.entrenado = True
    print(f"Trained Clasificador_KNN_Binario: {len(X)} samples, BS={sum(y)}, NBS={len(y)-sum(y)}")

  def save(self, filename):
    joblib.dump({'knn': self.knn, 'scaler': self.scaler, 'ventana': self.ventana, 'K': self.K, 'entrenado': self.entrenado}, filename)

  def load(self, filename):
    data = joblib.load(filename)
    self.knn, self.scaler, self.ventana, self.K, self.entrenado = data['knn'], data['scaler'], data['ventana'], data['K'], data['entrenado']
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def example_usage(directorio, años: list[str], test: list[str], ventana, K, promedio):
  if años == ['2014']:
    raise ValueError('No hay suficientes muestras para el entrenamiento solo con el año 2014, combine con otros o elija otro.')
  classifier = Clasificador_KNN_Binario(ventana, K)
  all_X, all_y = [], []
  for año in años:
    data_MAG = leer_archivos_MAG(os.path.join(directorio, 'recorte_Vignes'), f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59', promedio)
    bs_data = pd.read_csv(f'{directorio}/fruchtman/hemisferio_N/fruchtman_{año}_merge_hemisferio_N.sts', sep=' ', header=None)
    year = int(año)
    decimal_year = bs_data.iloc[:, 0].astype(float)
    start = pd.Timestamp(f"{year}-01-01")
    bs_times = start + pd.to_timedelta(decimal_year - 1, unit='D')
    X, y = classifier.muestras_entrenamiento(data_MAG, bs_times.to_numpy())
    all_X.append(X); all_y.append(y)

  X_train, y_train = np.vstack(all_X), np.concatenate(all_y)
  classifier.train(X_train, y_train)
  #classifier.save('bowshock_knn_model.pkl')
  for año in test:
    test_data = leer_archivos_MAG(os.path.join(directorio, 'recorte_Vignes'), f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59', promedio)
    print(f"\nAnalyzing {año}...")
    # ---------------------------
    predictions, probabilities, window_centers = classifier.predict(test_data)
    bs_centers = window_centers[predictions == 1]    # Step 3: Pick only BS predictions
    maven_times = test_data.iloc[:, 0].to_numpy()  # Step 4: Recover actual datetime. first column assumed datetime
    bs_times = maven_times[bs_centers]             # use bs_centers indices
    # Step 4: Recover actual datetime
    bs_times = pd.to_datetime(bs_times)            # Convert NumPy datetime64 array to pandas Timestamp array
    year_start = pd.Timestamp(f"{bs_times[0].year}-01-01")    # Now you can access the year
    decimal_days = (bs_times - year_start).total_seconds() / 86400 + 1    # Decimal day-of-year
    bs_count = len(bs_times)                        # Step 6: Save results
    total = len(predictions)
    print(f'Total windows: {total}')
    print(f'Bow Shock detections: {bs_count}')
    print(f'Percentage: {bs_count/total*100:.2f}%')
    results = pd.DataFrame({
        'prediction': predictions,
        'prob_BS': probabilities[:,1],
        'prob_NBS': probabilities[:,0]
    })
    # Add predicted BS datetime and decimal day
    bs_results = pd.DataFrame({
        'datetime': bs_times,
        'decimal_day': decimal_days
    })
    results.to_csv(f'bowshock_predictions_{año}.csv', index=False)
    bs_results.to_csv(f'bowshock_times_{año}.csv', index=False)
    # ---------------------------


#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————