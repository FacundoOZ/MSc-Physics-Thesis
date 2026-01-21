
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import numpy as np
import pandas as pd
from numpy import sqrt, mean, std, median, percentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from base_de_datos.lectura import leer_archivos_MAG
import joblib
import warnings
warnings.filterwarnings('ignore')

class KNN:
  """
  Clase del algoritmo de Machine Learning supervisado KNN (k-vecinos más cercanos) binario para datos Bow Shock (BS=1) y No-Bow Shock (NBS=0).
  """
  def __init__(self, ventana: int = 600, K: int = 5): #, promedio: int = 1):
    """
    Constructor (inicializador) de los objetos que contiene la clase KNN.
    """
    self.ventana: int    = ventana
    self.K: int          = K
    self.knn             = KNeighborsClassifier(K=K, weights='distance', n_jobs=-1)
    self.scaler          = StandardScaler()
    self.entrenado: bool = False
  
  def extract_window_features(self, datos_ventana):
    """Extract features from a 600-second window of data."""
    features = []
    if len(datos_ventana) < 10:                                                      # Ensure we have enough data points
      return None
    Bx,By,Bz,Xss,Yss,Zss = [datos_ventana[:,j].astype(float) for j in [1,2,3,7,8,9]] # Extract Bx, By, Bz columns (assuming indices 1, 2, 3)
    for B_i in [Bx,By,Bz]:                                                         # Magnetic field statistics
      features.extend([mean(B_i),std(B_i),np.max(B_i),np.min(B_i),median(B_i),percentile(B_i,25),percentile(B_i,75)])
    r = sqrt(Xss**2 + Yss**2 + Zss**2)                                          # Distance from Mars
    features.extend([mean(r), std(r), np.min(r), np.max(r)])
    B_mag = sqrt(Bx**2 + By**2 + Bz**2)                                         # Magnetic field magnitude
    features.extend([mean(B_mag), std(B_mag), np.max(B_mag)/mean(B_mag) if mean(B_mag) > 0 else 0])
    return np.array(features)
  
  def prepare_training_samples(self, maven_data, bowshock_times):
    """
    Prepare training samples from MAVEN data and bowshock times.
    maven_data : numpy array (Full MAVEN data with columns: [day, Bx, By, Bz, Xpc, Ypc, Zpc, Xss, Yss, Zss])
    bowshock_times : list of datetime (List of bowshock occurrence times)
    Returns:
    --------
    X : numpy array (Feature matrix)
    y : numpy array (Labels 1 for BS, 0 for NBS)
    """
    X = []
    y = []
    # Convert maven_data days to datetime (assuming same año)
    # You'll need to adjust this based on your time handling
    maven_times = [datetime.fromtimestamp(t) for t in maven_data[:, 0]]  # Adjust as needed
    for bs_time in bowshock_times:
      idx = np.argmin(np.abs([(t - bs_time).total_seconds() for t in maven_times])) # Find index closest to bowshock time
      start_idx = max(0, idx - self.ventana//2)                                     # Create BS sample (label = 1)
      end_idx = min(len(maven_data), idx + self.ventana//2)
      if end_idx - start_idx >= 10:                                                 # Minimum points
        window = maven_data[start_idx:end_idx]
        features = self.extract_window_features(window)
        if features is not None:
          X.append(features)
          y.append(1)
      for offset in [-3, -2, -1, 1, 2, 3]:                                          # Create NBS samples (label=0)-3 windows on left and right
        nbs_idx = idx + offset * self.ventana
        if 0 <= nbs_idx < len(maven_data):
          start_idx = max(0, nbs_idx - self.ventana//2)
          end_idx = min(len(maven_data), nbs_idx + self.ventana//2)
          if end_idx - start_idx >= 10:
            window = maven_data[start_idx:end_idx]
            features = self.extract_window_features(window)
            if features is not None:
              X.append(features)
              y.append(0)
    return np.array(X), np.array(y)
  
  def train(self, X, y):
    """Train the KNN classifier."""
    X_scaled = self.scaler.fit_transform(X)     # Scale features
    self.knn.fit(X_scaled, y)                   # Train classifier
    self.entrenado = True
    print(f"Trained KNN with {len(X)} samples")
    print(f"BS samples: {sum(y)}")
    print(f"NBS samples: {len(y) - sum(y)}")
  
  def predict(self, maven_data):
    """Predict bowshocks in MAVEN data."""
    if not self.entrenado:
      raise ValueError("Classifier not trained. Call train() first.")
    predictions = []
    probabilities = []
    for i in range(0, len(maven_data), self.ventana//2):                  # Slide window through data (50% overlap)
      start_idx = i
      end_idx = min(len(maven_data), i + self.ventana)
      if end_idx - start_idx >= 10:
        window = maven_data[start_idx:end_idx]
        features = self.extract_window_features(window)
        if features is not None:
          features_scaled = self.scaler.transform(features.reshape(1,-1)) # Scale and predict
          pred = self.knn.predict(features_scaled)[0]
          prob = self.knn.predict_proba(features_scaled)[0]
          predictions.append(pred)
          probabilities.append(prob)
        else:
          predictions.append(0)
          probabilities.append([1.0,0.0])                                 # Default to NBS
    return np.array(predictions), np.array(probabilities)

def save(self, filename):
  """Save the trained model."""
  model_data = {'knn': self.knn, 'scaler': self.scaler, 'ventana': self.ventana, 'K': self.K, 'entrenado': self.entrenado}
  joblib.dump(model_data, filename)

def load(self, filename):
  """Load a trained model."""
  model_data      = joblib.load(filename)
  self.knn        = model_data['knn']
  self.scaler     = model_data['scaler']
  self.ventana    = model_data['ventana']
  self.K          = model_data['K']
  self.entrenado = model_data['entrenado']

# Usage example
def example_usage(directorio: str, promedio: int):
  classifier = KNN(ventana=600, K=5)                              # Initialize classifier
  años = ['2014','2015','2016','2017','2018','2019']  # Load your data (adjust based on your actual data loading)
  all_X, all_y = [], []
  for año in años:
    t0, tf = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'
    maven_data = leer_archivos_MAG(directorio, t0, tf, promedio) # Load MAVEN data for the año
    bs_data = pd.read_csv(f'directorio/fruchtman/merge/fruchtman_{año}_recortado.txt', sep='\s+', header=None) # Load BS times for the año
    bs_times = [datetime.strptime(f"{año}-{row[0]:.6f}", "%Y-%j.%f") for row in bs_data.values] # Convert to datetime (adjust based on your format)
    X_year, y_year = classifier.prepare_training_samples(maven_data.values, bs_times) # Prepare training samples
    all_X.append(X_year)
    all_y.append(y_year)
  X_train, y_train = np.vstack(all_X), np.concatenate(all_y) # Combine all years
  classifier.train(X_train, y_train)                         # Train classifier
  classifier.save('bowshock_knn_model.pkl')                  # Save model
  for año in ['2020','2021','2022','2023','2024','2025']:   # Test on 2020-2025
    print(f'\nAnalyzing {año}...')
    test_data = leer_archivos_MAG(directorio, f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59', promedio=None) # Load test data
    predictions, probabilities = classifier.predict(test_data.values) # Predict
    bs_count = sum(predictions == 1)                         # Count bowshocks
    total = len(predictions)
    print(f'Total windows: {total}')
    print(f'Bow Shock detections: {bs_count}')
    print(f'Percentage: {bs_count/total*100:.2f}%')
    results = pd.DataFrame({'prediction': predictions, 'prob_BS': probabilities[:,1], 'prob_NBS': probabilities[:,0]}) # Save results
    results.to_csv(f'bowshock_predictions_{año}.csv', index=False)



#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————