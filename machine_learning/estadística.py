
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
  discontinuidades y coherencia del salto.
  """
  if len(B) < 3:                                            # Si len(x) es al menos 3, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*9                                          #=> si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  j_max: int = np.argmax(np.abs(np.gradient(B)))            # Obtengo el índice correspondiente al máximo valor de B.
  res: list[float] = (stats_B(B, np.gradient(B), j_max) + [ # En la variable res, uso la estadística básica de B, y agrego (appendeo):
    B.mean(),                                               # la media,
    np.sum(np.abs(np.gradient(B)[ : j_max])),               # la actividad pre máximo, y
    np.sum(np.abs(np.gradient(B)[j_max : ]))                # la actividad post máximo.
  ])                                                        # 
  return res                                                # Devuelvo res.

def estadística_componentes_B(Bx: np.ndarray) -> list[float]:
  """
  La función estadística_componentes_B utiliza una estadística especial para las componentes del campo magnético, haciendo énfasis en la
  captura de la coherencia direccional del salto, calculando el máximo del valor absoluto de la componente y el signo del gradiente.
  """
  if len(Bx) < 3:                                             # Si len(x) >= 3, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*8                                            #=> si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  j_max: int = np.argmax(np.abs(np.gradient(Bx)))             # Obtengo el índice correspondiente al máximo valor de Bx_j.
  res: list[float] = (stats_B(Bx, np.gradient(Bx), j_max) + [ # En la variable res, uso la estadística básica de B, y agrego:
    np.max(np.abs(Bx)),                                       # la amplitud máxima de dicha componente,
    np.mean(np.sign(np.gradient(Bx)))                         # y la media del signo del gradiente (la coherencia direccional).
  ])                                                          # 
  return res                                                  # Devuelvo res.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def stats_B(B: np.ndarray, dB: np.ndarray, j_max: int) -> list[float]:
  """
  La función stats_B contiene la estadística básica que contienen tanto la función estadística_B como estadística_componentes_B (el módulo
  de B y sus componentes). Contempla fluctuaciones, variabilidad del gradiente, filo del choque, entre otras magnitudes.
  """
  res: list[float] = [                            # En la variable res, calculo todas las magnitudes que considero como estadística básica:
    B.std(),                                      # Las fluctuaciones de campo magnético (desviación estándar de B).
    dB.std(),                                     # La variabilidad del gradiente (desviación estándar de dB/dt).
    np.max(np.abs(dB)),                           # El salto máximo en valor absoluto.
    np.percentile(np.abs(dB), 95),                # El porcentaje típico del gradiente fuerte.
    j_max/len(dB),                                # La localización del choque.
    np.max(np.abs(dB))/(np.mean(np.abs(dB)) + ε), # El filo del choque, utilizando la perturbación épsilon.
  ]                                               # 
  return res                                      # Devuelvo res.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Estadística para la posición de la sonda (X,Y,Z) en sistemas de coordenadas PC ó SS en unidades de km (normalizadas por R_m=3396.3):
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def estadística_R(R: np.ndarray) -> list[float]:
  """
  La función estadística_R utiliza una estadística correspondiente a una ventana con mediciones del módulo de la posición de MAVEN (la
  distancia de MAVEN al planeta, que utilizará las mediciones normalizadas por R_m), es decir, mediciones de |R| = sqrt(X**2 + Y**2 + Z**2).
  Esta estadística contempla la asimetría temporal de la posición, y saltos, magnitudes adecuadas para detectar transiciones tipo bow shock.
  """
  if len(R) < 3:                                    # Mientras haya al menos 3 puntos, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*10                                 # => si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  j_max: int    = np.argmax(np.abs(np.gradient(R))) # Obtengo el índice correspondiente al máximo valor de R.
  x_izq,_,x_der = dividir_ventana(R)                # Obtengo el primer y el último tercio de ventana, respectivamente (la dividí en 3).
  res: list[float] = (stats_R(R, x_izq, x_der) + [  # En la variable res, uso la estadística básica de R, y agrego:
    np.percentile(R, 75) - np.percentile(R, 25),    # Cuán ancho es 1/4 de los datos de la segunda mitad: Interquartile range (IQR).
    x_der.std()/(x_izq.std() + ε),                  # La asimetría de ruido.
    j_max/len(np.gradient(R)),                      # La posición del salto (0–1).
    np.sum(np.abs(np.gradient(R)[ : j_max])),       # La actividad pre máxmio.
    np.sum(np.abs(np.gradient(R)[j_max : ])),       # La actividad post máximo.
  ])                                                # 
  return res                                        # Devuelvo res.

def estadística_componentes_R(x: np.ndarray) -> list[float]:
  """
  La función estadística_componentes_R utiliza una estadística especial para las componentes de la posición de MAVEN, que si bien son
  mediciones poco bruscas (suaves) son útiles para aportar contexto al algoritmo KNN.
  """
  if len(x) < 3:                                   # Mientras haya al menos 3 puntos, puedo obtener coherentes gradientes, percentiles, etc.
    return [0.0]*7                                 # => si hay 2 puntos o menos no aporta valor físico => devuelvo 0.
  x_izq, _, x_der = dividir_ventana(x)             # Obtengo el primer y el último tercio de ventana, respectivamente (la dividí en 3).
  res: list[float] = (stats_R(x, x_izq, x_der) + [ # En la variable res, uso la estadística básica de Bx, y agrego:
    np.percentile(x, 25),                          # Calculo el porcentaje de datos del primer cuarto,
    np.percentile(x, 75),                          # y del tercer cuarto.
  ])                                               # 
  return res                                       # Devuelvo res.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def stats_R(x: np.ndarray, x_izq: np.ndarray, x_der: np.ndarray) -> list[float]:
  """
  La función stats_R contiene la estadística básica que contienen tanto la función estadística_R como estadística_componentes_R (la
  distancia de MAVEN al planeta, y sus componentes). Contempla media global, desviación estandar, salto medio, entre otras magnitudes.
  """
  res: list[float] = [                           # En la variable res, calculo todas las magnitudes que considero como estadística básica:
    x.mean(),                                    # La media global de la posición.
    x.std(),                                     # La variabilidad global (desviación estándar).
    np.median(x),                                # La robustez (mediana).
    x.max() - x.min(),                           # La amplitud total.
    x_der.mean() - x_izq.mean(),                 # El salto medio del primer (1/3) y el último tramo (3/3)
  ]                                              # 
  return res                                     # Devuelvo res.

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