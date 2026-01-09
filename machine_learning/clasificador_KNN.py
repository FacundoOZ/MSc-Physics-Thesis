
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
from numpy                   import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import classification_report, confusion_matrix

# Módulos Propios:
from plots.MAG import leer_archivos_MAG

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'

# Extraigo los datos en el intervalo de tiempo deseado, con la función reciclada de plots:
data = leer_archivos_MAG(directorio=ruta+'datos_recortados', año=2016, tiempo_inicial='1/3-00:00:00', tiempo_final='30/3-23:59:00')
tiempo, Bx,By,Bz, X,Y,Z = [data[i].to_numpy() for i in range(7)]            #   Extraigo toda la información del .sts de ese intervalo

# Creo un DataFrame con los datos del intervalo (t0,tf) deseado, para poder calcular cosas:
df = pd.DataFrame({
  't': tiempo,
  'B_x': Bx,
  'B_y': By,
  'B_z': Bz,
  'x': X,
  'y': Y,
  'z': Z
})

# Calculo el módulo de B, la diferencia temporal (dt) y la derivada temporal de B.
df['B'] = sqrt(df['B_x']**2 + df['B_y']**2 + df['B_z']**2)
df['dt'] = (df['t'] - df['t'].iloc[0])*86400
df['dB/dt'] = np.gradient(df['B'], df['dt'])

tamaño: int = 300 # 5 minutos, 8 minutos y 10 minutos.

def generar_ventanas(df: pd.DataFrame, tamaño: int):
  # Divide los datos en ventanas consecutivas de longitud 'tamaño' (en segundos) y calcula estadísticas relevantes del campo magnético y su derivada temporal.
  ventanas = []
  features = []

  n = len(df)
  fs = 1  # frecuencia de muestreo (1 Hz)
  step = tamaño  # tamaño de ventana en segundos

  for i in range(0, n - step, step):
    sub = df.iloc[i:i+step]
    ventanas.append((sub['t'].iloc[0], sub['t'].iloc[-1]))  # rango temporal de cada ventana

    feat = {
      'mean_B': sub['B'].mean(),
      'std_B': sub['B'].std(),
      'max_B': sub['B'].max(),
      'min_B': sub['B'].min(),
      'mean_dBdt': sub['dB/dt'].mean(),
      'std_dBdt': sub['dB/dt'].std(),
      'range_B': sub['B'].max() - sub['B'].min(),
      'mean_x': sub['x'].mean(),
      'mean_y': sub['y'].mean(),
      'mean_z': sub['z'].mean()
    }
    features.append(feat)

  return pd.DataFrame(features), ventanas

# ELIJO UNA VENTANA DE 5 MINUTOS:
dfs_features = {}
df_feat, ventanas = generar_ventanas(df, tamaño)
dfs_features[tamaño] = (df_feat, ventanas)
print(f"Ventanas de {tamaño}s: {len(df_feat)} muestras generadas")

mean_times = [ (v[0] + v[1])/2 for v in ventanas ]  # midpoint of each window

# Create figure with 4 subplots
fig, axs = p.subplots(4, 1, figsize=(14, 10), sharex=True)

# 1️⃣ Original |B| vs time
axs[0].plot(df['t'], df['B'], label='|B| original', color='blue')
axs[0].set_ylabel('|B| [nT]')
axs[0].set_title('Original |B| vs time')
axs[0].grid(True)
axs[0].legend()

# 2️⃣ Mean |B| per window
axs[1].plot(mean_times, df_feat['mean_B'], label='mean_B', color='red')
axs[1].set_ylabel('mean_B')
axs[1].set_title('Mean |B| per window')
axs[1].grid(True)
axs[1].legend()

# 3️⃣ Std |B| per window
axs[2].plot(mean_times, df_feat['std_B'], label='std_B', color='orange')
axs[2].set_ylabel('std_B')
axs[2].set_title('Std deviation of |B| per window')
axs[2].grid(True)
axs[2].legend()

# 4️⃣ Std of dB/dt per window
axs[3].plot(mean_times, df_feat['std_dBdt'], label='std_dB/dt', color='green')
axs[3].set_ylabel('std_dB/dt')
axs[3].set_title('Std deviation of d|B|/dt per window')
axs[3].set_xlabel('Decimal day of year')
axs[3].grid(True)
axs[3].legend()

p.tight_layout()
p.show()

# Elijo 20 nanoTeslas como límite para distinguir la región de viento solar de la región de MPB
threshold_B = 20.0  # nT, adjust based on your plots
df_feat['region'] = (df_feat['mean_B'] > threshold_B).astype(int)
#print(df_feat['region'].value_counts())

# ENTRENAMIENTO KNN___________________________________________
#=============================================================

X = df_feat[['mean_B', 'std_B', 'std_dBdt']].to_numpy()
y = df_feat['region'].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
  X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform','distance']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='f1')
grid.fit(X_train, y_train)

best_knn = grid.best_estimator_
print("Best KNN parameters:", grid.best_params_)

y_pred = best_knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

df_feat['predicted'] = best_knn.predict(scaler.transform(X))

mean_times = [ (v[0] + v[1])/2 for v in ventanas ]  # midpoint of each 5-min window

p.figure(figsize=(14,5))
p.plot(df['t'], df['B'], label='|B| original', color='blue')
for i, t_center in enumerate(mean_times):
  if df_feat['predicted'].iloc[i] == 1:
    p.axvspan(t_center-2.5/1440, t_center+2.5/1440, color='red', alpha=0.2)  # 2.5 min = half window in days
p.xlabel('Decimal day of year')
p.ylabel('|B| [nT]')
p.title('Predicted MPB regions over original |B|')
p.grid(True)
p.legend()
p.show()


#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————