
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
# ClasificadorKNN : para la detección de Bow Shocks mediante mediciones de campo magnético del instrumento MAG de la sonda MAVEN.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class ClasificadorKNN:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Inicializador (Constructor):
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def __init__(
      self,                                                                                # Estado de la variable actual.
      K: int,                                                                              # Metaparámetro K para las ventanas vecinas.
      ventana: int = 300,                                                                  # Ancho de ventana (en segundos)
      variables: Union[list[str], None] = None                                             # Variables a utilizar del vector característico.
  ) -> None:
    """Inicializador de los atributos de la clase ClasificadorKNN para la detección de Bow Shocks. El metaparámetro 'K' es un entero que
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
    self.K: int               = K                                                          # Metaparámetro K.
    self.ventana: int         = ventana                                                    # Ventana de puntos.
    self.variables: list[str] = list(variables)                                            # Variables para entrenamiento.
    self.entrenado: bool      = False                                                      # Booleano del estado del KNN.
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
  def vector_característico(self, data: pd.DataFrame) -> np.ndarray:
    """La función vector característico recibe un dataframe 'data' del tipo archivo MAG recortado, cuyas columnas son de la forma:
    día_decimal Bx By Bz Xpc Ypc Zpc Xss Yss Zss
    ....        .. .. .. ... ... ... ... ... ...
    y extrae el vector característico por ventana para alimentar al KNN, con las variables pasadas por parámetro a la clase ClasificadorKNN.
    Devuelve una lista de valores float convertida a array (np.ndarray[list[float]]) donde cada valor representa las magnitudes estadísticas
    (del archivo estadística.py) correspondientes a las 'variables' pasadas por parámetro al KNN."""
    Bx,By,Bz,Xpc,Ypc,Zpc,Xss,Yss,Zss = [data[j].to_numpy() for j in range(1,10)] # Extraigo todas las magnitudes físicas del archivo tipo MAG. 
    B, R = módulo(Bx,By,Bz), módulo(Xss,Yss,Zss, norm=R_m)                                 # Calculo los módulos 
    dicc: dict[str,np.ndarray] = {                                               # 
      'Bx' : Bx , 'By' : By , 'Bz' : Bz ,                                        # 
      'Xpc': Xpc, 'Ypc': Ypc, 'Zpc': Zpc, 'Xss': Xss, 'Yss': Yss, 'Zss': Zss}    # 
    vector: list[float] = []                                                     # 
    for var in self.variables:                                                   # 
      if var in ('B','R'):                                                       # 
        vector.extend(estadística_módulos(B if var=='B' else R))                 # 
      elif var in dicc:                                                          # 
        vector.extend(estadística(dicc[var]))                                    # 
      else:                                                                      # 
        raise ValueError(f"Variable desconocida: {var}")                         # 
    res: np.ndarray = np.array(vector)                                           # 
    return res                                                                   # 

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Muestras BS y NBS:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def prepare_training_samples(self, maven_data, bowshock_times, año):
    X, y = [], []
    # convert MAVEN time column to datetime
    col0 = maven_data.iloc[:, 0]
    if pd.api.types.is_datetime64_any_dtype(col0):
      maven_times = col0.to_numpy()
    elif 'datetime' in maven_data.columns:
      maven_times = maven_data['datetime'].to_numpy()
    else:
      base = pd.Timestamp(f"{año}-01-01")
      maven_times = (base + pd.to_timedelta(col0.astype(float) - 1, unit='D')).to_numpy()
    half_win = self.ventana // 2
    for t_bs in bowshock_times:
      idx_center = np.searchsorted(maven_times, t_bs)
      # BS window
      start, end = max(0, idx_center - half_win), min(len(maven_data), idx_center + half_win)
      feat = self.vector_característico(maven_data.iloc[start:end])
      if feat is not None:
        X.append(feat); y.append(1)
      # 3 NBS windows before/after
      for shift in [-1, 1, 2]:
        idx_nbs = idx_center + shift * self.ventana
        if idx_nbs - half_win < 0 or idx_nbs + half_win > len(maven_data):
          continue
        feat = self.vector_característico(maven_data.iloc[idx_nbs - half_win : idx_nbs + half_win])
        if feat is not None:
          X.append(feat); y.append(0)
    return np.array(X), np.array(y)

  def train(self, X, y):
    self.knn.fit(self.scaler.fit_transform(X), y)
    self.entrenado = True
    print(f"Trained ClasificadorKNN: {len(X)} samples, BS={sum(y)}, NBS={len(y)-sum(y)}")

  def predict(self, maven_data):
    predictions = []
    probabilities = []
    window_centers = []
    for i in range(0, len(maven_data), self.ventana//2):
      start_idx = i
      end_idx = min(len(maven_data), i + self.ventana)
      if end_idx - start_idx >= 10:
        window = maven_data[start_idx:end_idx]
        features = self.vector_característico(window)
        if features is not None:
          features_scaled = self.scaler.transform(features.reshape(1,-1))
          pred = self.knn.predict(features_scaled)[0]
          prob = self.knn.predict_proba(features_scaled)[0]
          predictions.append(pred)
          probabilities.append(prob)
          # Append actual center index of this window
          window_centers.append(start_idx + (end_idx - start_idx)//2)
        else:
          predictions.append(0)
          probabilities.append([1.0,0.0])
          window_centers.append(start_idx + (end_idx - start_idx)//2)
    return np.array(predictions), np.array(probabilities), np.array(window_centers)

  def save(self, filename):
    joblib.dump({'knn': self.knn, 'scaler': self.scaler, 'ventana': self.ventana, 'K': self.K, 'entrenado': self.entrenado}, filename)

  def load(self, filename):
    data = joblib.load(filename)
    self.knn, self.scaler, self.ventana, self.K, self.entrenado = data['knn'], data['scaler'], data['ventana'], data['K'], data['entrenado']


def example_usage(directorio, años: list[str], test: list[str], ventana, K, promedio):
  if años == ['2014']:
    raise ValueError('No hay suficientes muestras para el entrenamiento solo con el año 2014, combine con otros o elija otro.')
  classifier = ClasificadorKNN(ventana, K)
  all_X, all_y = [], []
  for año in años:
    maven_data = leer_archivos_MAG(os.path.join(directorio, 'recorte_Vignes'), f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59', promedio)
    bs_data = pd.read_csv(f'{directorio}/fruchtman/hemisferio_N/fruchtman_{año}_merge_hemisferio_N.sts', sep=' ', header=None)
    year = int(año)
    decimal_year = bs_data.iloc[:, 0].astype(float)
    start = pd.Timestamp(f"{year}-01-01")
    bs_times = start + pd.to_timedelta(decimal_year - 1, unit='D')
    X, y = classifier.prepare_training_samples(maven_data, bs_times.to_numpy(), año)
    all_X.append(X); all_y.append(y)

  X_train, y_train = np.vstack(all_X), np.concatenate(all_y)
  classifier.train(X_train, y_train)
  #classifier.save('bowshock_knn_model.pkl')
  for año in test:
    print(f"\nAnalyzing {año}...")
    test_data = leer_archivos_MAG(os.path.join(directorio, 'recorte_Vignes'), f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59', promedio)
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