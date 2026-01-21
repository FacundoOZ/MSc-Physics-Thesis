import os
import numpy as np
import pandas as pd
from numpy import sqrt, mean, std, median, percentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from base_de_datos.lectura import leer_archivos_MAG
import joblib
import warnings
warnings.filterwarnings('ignore')


class KNN:
  """KNN classifier for Bow Shock (BS=1) vs Non-Bow Shock (NBS=0) windows."""

  def __init__(self, ventana: int, K: int):
    self.ventana = ventana
    self.K = K
    self.knn = KNeighborsClassifier(n_neighbors=K, weights='distance', n_jobs=-1)
    self.scaler = StandardScaler()
    self.entrenado = False

  def extract_window_features(self, df: pd.DataFrame):
    if len(df) < 10: 
      return None
    Bx,By,Bz,Xss,Yss,Zss = [df.iloc[:,j].astype(float).T for j in [1,2,3,7,8,9]]

    B_módulo = sqrt(Bx**2 + By**2 + Bz**2)
    features = []
    for x in [Xss, Yss, Zss]:
      features.extend([mean(x), std(x), x.max(), x.min(), median(x), percentile(x, 25), percentile(x, 75)])
    #R = sqrt(Xss**2 + Yss**2 + Zss**2)
    #features.extend([mean(R), std(R), R.min(), R.max()])
    B_mag = sqrt(Bx**2 + By**2 + Bz**2)
    features.extend([mean(B_mag), std(B_mag), B_mag.max() / mean(B_mag) if mean(B_mag) > 0 else 0])
    return np.array(features)

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
      feat = self.extract_window_features(maven_data.iloc[start:end])
      if feat is not None:
        X.append(feat); y.append(1)
      # 3 NBS windows before/after
      for shift in [-1, 1, 2]:
        idx_nbs = idx_center + shift * self.ventana
        if idx_nbs - half_win < 0 or idx_nbs + half_win > len(maven_data):
          continue
        feat = self.extract_window_features(maven_data.iloc[idx_nbs - half_win : idx_nbs + half_win])
        if feat is not None:
          X.append(feat); y.append(0)
    return np.array(X), np.array(y)

  def train(self, X, y):
    self.knn.fit(self.scaler.fit_transform(X), y)
    self.entrenado = True
    print(f"Trained KNN: {len(X)} samples, BS={sum(y)}, NBS={len(y)-sum(y)}")

  def predict(self, maven_data):
    predictions = []
    probabilities = []
    window_centers = []
    for i in range(0, len(maven_data), self.ventana//2):
      start_idx = i
      end_idx = min(len(maven_data), i + self.ventana)
      if end_idx - start_idx >= 10:
        window = maven_data[start_idx:end_idx]
        features = self.extract_window_features(window)
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
  classifier = KNN(ventana, K)
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