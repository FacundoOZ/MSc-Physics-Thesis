
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo que contiene las estadísticas a utilizar para el vector característico por ventana en el algoritmo KNN.
#============================================================================================================================================

import numpy as np

ε: float = 1e-6

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Estadísticas para el campo magnético (B) en unidades de nT:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def estadística_B(B: np.ndarray) -> list[float]:
  """
  La función estadística_B utiliza una estadística correspondiente a una ventana con mediciones del módulo del campo magnético, es decir,
  mediciones de |B| = sqrt(Bx**2 + By**2 + Bz**2). Esta estadística contempla la media y el gradiente del campo magnético, enfatizando en
  discontinuidades y coherencia del salto, entre otras.
  """
  if len(B) < 3:                                 # Si len(B) es al menos 3, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*9                               #=> si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  dB = np.gradient(B)                            # Obtengo la derivada temporal de B (dB/dt).
  j_max: int = np.argmax(np.abs(dB))             # Obtengo el índice correspondiente al máximo valor de dB/dt.
  res: list[float] = [                           # En la variable res, voy a agregar (appendear) toda la estadística de |B|:
    B.mean(),                                    # la media,
    B.std(),                                     # las fluctuaciones,
    np.max(np.abs(dB)),                          # el salto máximo,
    dB.std(),                                    # la variabilidad del gradiente,
    np.percentile(np.abs(dB), 95),               # el gradiente fuerte típico,
    j_max/len(dB),                               # la localización del shock,
    np.sum(np.abs(dB[:j_max])),                  # la actividad upstream,
    np.sum(np.abs(dB[j_max:])),                  # la actividad downstream,
    np.max(np.abs(dB))/(np.mean(np.abs(dB)) + ε),# y el filo.
  ]                                              #
  return res                                     # Devuelvo res.

def estadística_componentes_B(B_i: np.ndarray) -> list[float]:
  """
  La función estadística_componentes_B utiliza una estadística especial para las componentes del campo magnético, haciendo énfasis en la
  captura de la coherencia direccional del salto, calculando el máximo del valor absoluto de la componente y el signo del gradiente, entre
  otras.
  """
  if len(B_i) < 3:                                   # Si len(x) >= 3, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*8                                   #=> si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  dB_i = np.gradient(B_i)                            # Obtengo la derivada temporal de la componente i-ésima de B (dB_i/dt).
  j_max: int = np.argmax(np.abs(dB_i))               # Obtengo el índice correspondiente al máximo de la derivada.
  res: list[float] = [                               # En la variable res, agrego toda la estadística de la componente B_i:
    B_i.std(),                                       # la variabilidad,
    np.max(np.abs(B_i)),                             # la amplitud,
    dB_i.std(),                                      # el gradiente RMS,
    np.max(np.abs(dB_i)),                            # el salto máximo,
    np.percentile(np.abs(dB_i), 95),                 # el gradiente extremo típico,
    j_max/len(dB_i),                                 # dónde ocurre,
    np.mean(np.sign(dB_i)),                          # la coherencia direccional,
    np.max(np.abs(dB_i))/(np.mean(np.abs(dB_i)) + ε),# y el filo.
  ]                                                  #
  return res                                         # Devuelvo res.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Estadística para la posición de la sonda (X,Y,Z) en sistemas de coordenadas PC ó SS en unidades de km (normalizadas por R_m=3396.3):
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def estadística_R(R: np.ndarray) -> list[float]:
  """
  La función estadística_R utiliza una estadística correspondiente a una ventana con mediciones del módulo de la posición de MAVEN (la
  distancia de MAVEN al planeta, que utilizará las mediciones normalizadas por R_m), es decir, mediciones de |R| = sqrt(X**2 + Y**2 + Z**2).
  Esta estadística contempla la asimetría temporal de la posición, y saltos, entre otras magnitudes adecuadas para detectar transiciones tipo
  bow shock.
  """
  if len(R) < 3:                                # Mientras haya al menos 3 puntos, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*10                             # => si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  x_izq, _, x_der = dividir_ventana(R)          # Obtengo el primer y el último tercio de ventana, respectivamente (la dividí en 3).
  j_max: int = np.argmax(np.abs(np.gradient(R)))# Obtengo el índice correspondiente al máximo de la derivada dR/dt.
  res = [                                       # En la variable res, agrego toda la estadística |R|:
    R.mean(),                                   # la media global,
    R.std(),                                    # la variabilidad global,
    x_der.mean() - x_izq.mean(),                # el salto medio (clave),
    x_der.std()/(x_izq.std() + ε),              # la asimetría de ruido,
    np.median(R),                               # la mediana,
    np.percentile(R, 75) - np.percentile(R, 25),# Cuán ancho es 1/4 de los datos de la segunda mitad: Interquartile range (IQR),
    R.max() - R.min(),                          # la amplitud total,
    j_max/len(np.gradient(R)),                  # la posición del salto (0–1),
    np.sum(np.abs(np.gradient(R)[:j_max])),     # la actividad upstream,
    np.sum(np.abs(np.gradient(R)[j_max:])),     # y la actividad downstream.
  ]                                             #
  return res                                    # Devuelvo res.

def estadística_componentes_R(x_i: np.ndarray) -> list[float]:
  """
  La función estadística_componentes_R utiliza una estadística especial para las componentes de la posición de MAVEN, que si bien son
  mediciones poco bruscas (suaves) son útiles para aportar contexto al algoritmo KNN.
  """
  if len(x_i) < 3:                      # Mientras haya al menos 3 puntos, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*7                      # => si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  x_izq, _, x_der = dividir_ventana(x_i)# Obtengo el primer y el último tercio de ventana, respectivamente (la dividí en 3).
  res: list[float] = [                  # En la variable res, agrego toda la estadística de la componente x_i:
    x_i.mean(),                         # la media,
    x_i.std(),                          # la desviación estándar,
    x_der.mean() - x_izq.mean(),        # la deriva espacial,
    np.median(x_i),                     # la mediana,
    np.percentile(x_i, 25),             # el porcentaje de datos del primer cuarto,
    np.percentile(x_i, 75),             # el porcentaje de datos del último cuarto,
    x_i.max() - x_i.min(),              # y el salto medio del primer (1/3) y el último tramo (3/3).
  ]                                     #
  return res                            # Devuelvo res.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def dividir_ventana(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  La función dividir_ventana, permite dividir en 3 partes iguales la ventana a la cual se le realizará la estadística, lo que permite
  contemplar diferencias entre la zona inicial y la zona terminal de la ventana, para notar cambios en sus magnitudes.
  Las mediciones de campo magnético son ruidosas y ya contienen la discontinuidad del choque, por lo que una división de ventanas es
  irrelevante. En contraste, la posición varía monotonamente, y la división de ventanas permite distinguir zonas upstream y downstream.
  """
  n: int = len(x)                                   # En n, obtengo la cantidad total de puntos.
  if n < 3:                                         # Si no hay al menos 3 puntos, para poder calcular magnitudes adecuadamente:
    return x, x, x                                  # devuelvo una tripla con 3 referencias al mismo objeto (no lo pude dividir).
  return x[ : n//3], x[n//3 : 2*n//3], x[2*n//3 : ] # Si no, divido en 1/3, 2/3 y 3/3 de mediciones (divido los datos en 3 subventanas).
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————