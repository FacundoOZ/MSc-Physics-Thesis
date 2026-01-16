
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo que contiene los parámetros y ajustes utilizados por Vignes (2000) para el año 1997.
#============================================================================================================================================

import numpy as np
from typing import Callable
from numpy  import sqrt, cos, sin

# Módulos Propios
from base_de_datos.conversiones import R_m

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# hipérbola_Vignes: función que contiene los parámetros Vignes (1997) para poder graficar las componentes cartesianas de la hipérbola en xy.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def hipérbola_Vignes(
    *,
    θ0: float = -0.12,         # Punto inicial de la malla de puntos para θ (seteado en -0.12 por mí, no por Vignes).
    θf: float = 2.3,           # Punto terminal de la malla de puntos para θ (seteado en 2.3 por mí, no por Vignes).
    X0: float = 0.64,          # ± 0.02 [R_m]        # Offset (desplazamiento) en el eje Xss (por Vignes).
    ε: float  = 1.03,          # ± 0.01              # Excentricidad (por Vignes)
    L: float  = 2.04,          # ± 0.02 [R_m]        # Semi-latus rectum (por Vignes)
    α: float  = np.deg2rad(4), # 4°=0.0698132 rad    # Ángulo de aberración en SS (X' opuesto a flujo medio de viento solar) (por Vignes).
    cant_puntos: int = 450     # Vignes lo llama N_b # Representa la suavidad (definición) de la hipérbola (por Vignes).
    #R_SD = 1.64 # ± 0.08 [R_m]# Éstos dos se pueden obtener (distancia sub-solar: desde el planeta al punto de la cónica en el eje X'_ss).
    #R_TD = 2.62 # ± 0.09 [R_m]# distancia sub-solar: desde el planeta al punto de la cónica en el eje Y'_ss.
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  La función hipérbola_Vignes representa la curva cónica que ajustó Vignes (1997) en el paper del año 2000, con los datos de un año de 
  medicionesde la sonda Mars Global Surveyor (MGS). La función recibe los parámetros de semi-lado recto (L) y excentricidad (ε) de la cónica, 
  y un punto inicial (θ0) y terminal (θf) y una cantidad de puntos, con los cuales realiza una malla para el parámetro θ de la función polar
  R(θ), e introduciendo el offset X0, calcula y devuelve las componentes cartesianas de la curva polar, los np.ndarray x e y. Además, devuelve 
  su versión aberrada, que consiste en una matriz de rotación con respecto al ángulo α seteado por Vignes en 4°, con los np.ndarrays x_a e y_a.
  """
  θ: np.ndarray = np.linspace(θ0, θf, cant_puntos) # Malla de puntos para trazar la curva.
  R: np.ndarray = L/(1 + ε*cos(θ))                 # Función R(θ) = L/(1+εcos(θ)): ecuación de la cónica en coordenadas polares (r,θ)
  x: np.ndarray = R*cos(θ) + X0                    # Componente x de la ecuación de la cónica en coordenadas cartesianas (SS) con offset X0
  y: np.ndarray = R*sin(θ)                         # Componente y de la ecuación de la cónica en coordenadas cartesianas (SS)
  x_a: np.ndarray = x*cos(α) - y*sin(α)            # Componentes aberradas => aplico la matriz de rotación 2x2 a las coordenadas (x,y) del
  y_a: np.ndarray = x*sin(α) + y*cos(α)            # sistema de coordenadas SS (x,y) => obtengo (x',y').
  return x, y, x_a, y_a                            # Devuelvo los np.ndarray (componentes cartesianas) de la cónica (x,y), y (x',y').

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# función_hipérbola_Vignes: dadas las mediciones x_datos pasadas por parámetro construye la cónica Vignes correspondiente que pasa por X0
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def función_hipérbola_Vignes(
    x_datos: np.ndarray,                                          # Mediciones de la componente X en coordenadas SS (Xss) de un bow shock.
    X0: float,                                                    # Único parámetro offset libre que se desea ajustar (desplazamiento en Xss).
    *,
    cant_puntos: int = 450                                        # Cantidad de puntos con los que se desea ajustar (precisión del modelo)
) -> np.ndarray:
    """
    La función función_hipérbola_Vignes recibe un np.ndarray de datos que representa mediciones de las componentes X en el sistema de
    referencia SS (Xss) donde ocurrió un bow shock, y recibe un parámetro float X0. Utilizando todos los parámetros standard dados por
    Vignes (2000), excepto X0, traza la hipérbola única para el intervalo dado por las mediciones x_datos cuyo offset es X0, y la devuelve
    en formato y=f(x) solo para y>0 (devuelve y). El parámetro 'cant_puntos' permite ajustar la definición de la hipérbola creada.
    """
    _,_,x_a,y_a = hipérbola_Vignes(X0=X0,cant_puntos=cant_puntos) # Construyo la cónica de Vignes con el X0 y la cantidad de puntos propuestos.
    y_modelo: np.ndarray = np.empty_like(x_datos)                 # Creo un ndarray de longitud x_datos sin nada, el cual voy a llenar.
    for i, Xss_i in enumerate(x_datos):                           # Para cada valor de mis datos (medición Fruchtman en Xss) i, de 0 a len(x),
      j: int = np.argmin(np.abs(x_a - Xss_i))                     # busco el j de la cónica construida más cercano a la medición actual Xss_i,
      y_modelo[i] = np.abs(y_a[j])                                # y coloco en mi y_modelo, el valor abs (solo y>0) del y_a[j] hallado.
    return y_modelo                                               # Devuelvo la hipérbola Vignes de forma y=f(x) que ajustó mejor X0 y x_datos.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# FUNCIONES AUXILIARES PARA LOS RESULTADOS OBTENIDOS
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def hipérbola_máxima() -> tuple[np.ndarray, np.ndarray]:
  """
  Hipérbola del tipo Vignes construida como una hipérbola de la forma y tamaño adecuados que pasa por el punto máximo del 2015.
  """
  _, _, x_a_max, y_a_max = hipérbola_Vignes(θ0= 0.41, θf=2.0, X0=1.40, α=-np.deg2rad(10)) # Obtengo solo las coordenadas aberradas
  return x_a_max, y_a_max                                                                 # y las devuelvo.

def hipérbola_mínima() -> tuple[np.ndarray, np.ndarray]:
  """
  Hipérbola del tipo Vignes construida como una hipérbola de la forma y tamaño adecuados que pasa por el punto mínimo del 2019.
  """
  _, _, x_a_min, y_a_min = hipérbola_Vignes(θ0=-0.39, θf=1.9, X0=0.13, α= np.deg2rad(20)) # Obtengo solo las coordenadas aberradas
  return x_a_min, y_a_min                                                                 # y las devuelvo.

def segmento_izquierdo() -> Callable[[np.ndarray], np.ndarray]:
  """
  Segmento que conecta el punto terminal de la hipérbola mínima con el punto terminal de la hipérbola máxima.
  """
  Xmin, Ymin = hipérbola_mínima()                             # Obtengo las coordenadas aberradas de las hipérbolas mínima
  Xmax, Ymax = hipérbola_máxima()                             # y máxima.
  x_A, y_A, x_B, y_B = Xmin[-1], Ymin[-1], Xmax[-1], Ymax[-1] # Obtengo las coordenadas (x,y) del último punto de la hipérbola mín y máx.
  def recta(y: np.ndarray) -> np.ndarray:                     # Utilizo la ecuación de la recta que pasa por dos puntos.
    m = (y-y_A)/(y_B-y_A)                                     # Definición de la pendiente.
    return m*(x_B-x_A) + x_A                                  # Devuelvo la recta que pasa por (x_A,y_A) (hipérbola mín) y (x_B,y_B) (máx.)
  return recta                                                # Devuelvo la recta, que es un np.ndarray.

def mínimo_2019() -> tuple[float, float]:
  """
  Mínimo bow shock del 2019 (17/01/2019-19:30:29):
  17.812841 0.880000 -2.100000 3.780000 1432.477000 765.354000 -5156.698000 266.488000 -706.178000 -5353.465000
  """
  Xss = 266.488000/R_m                                 # Obtengo la coordenada Xss normalizada por el radio marciano.
  cil = sqrt((-706.178000)**2 + (-5353.465000)**2)/R_m # Obtengo la proyección de las componentes yz, en el plano vertical y también normalizo.
  return (Xss, cil)                                    # Devuelvo una tupla que contiene el punto en el plano (X,sqrt(Y**2 + Z**2)).

def máximo_2015() -> tuple[float, float]:
  """
  Máximo bow shock del 2015 (10/01/2015-19:22:35):
  10.807355 -9.730000 -6.030000 5.380000 -2175.504000 -1628.225000 -8501.208000 4701.380000 2458.306000 -7176.923000
  """
  Xss = 4701.380000/R_m                                # Obtengo la coordenada Xss normalizada por el radio marciano.
  cil = sqrt((2458.306000)**2 + (-7176.923000)**2)/R_m # Obtengo la proyección de las componentes yz, en el plano vertical y también normalizo.
  return (Xss, cil)                                    # Devuelvo una tupla que contiene el punto en el plano (X,sqrt(Y**2 + Z**2)).

def mínimo_2015() -> tuple[float, float]:
  """
  Mínimo bow shock del 2015 (23/07/2015-04:41:22):
  204.195399 0.570000 -2.980000 -0.580000 4935.614000 -1546.648000 2916.324000 -328.066000 -3450.585000 4821.119000
  """
  Xss = -328.066000/R_m                                # Obtengo la coordenada Xss normalizada por el radio marciano.
  cil = sqrt((-3450.585000)**2 + (4821.119000)**2)/R_m # Obtengo la proyección de las componentes yz, en el plano vertical y también normalizo.
  return (Xss, cil)                                    # Devuelvo una tupla que contiene el punto en el plano (X,sqrt(Y**2 + Z**2)).
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————