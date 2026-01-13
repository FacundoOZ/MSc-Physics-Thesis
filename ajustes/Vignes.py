
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo que contiene los parámetros y ajustes utilizados por Vignes (2000) para el año 1997.
#============================================================================================================================================

import numpy as np
from numpy   import cos, sin

#X0 = 0.78*R_m # DE DÓNDE SALE ESTO QUE USA CAMILA? NO LO ENCUENTRO EN EL PAPER DE VIGNES.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# cónica_Vignes: función que contiene los parámetros de Vignes (1997) para poder graficar las componentes cartesianas de la hipérbola en xy.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def cónica_Vignes(
    *,
    θ0: float = -0.05,         # Punto inicial de la malla de puntos para θ (seteado en -0.05 por mí, no por Vignes).
    θf: float = 2.3,           # Punto terminal de la malla de puntos para θ (seteado en 2.3 por mí, no por Vignes).
    X0: float = 0.64,          # ± 0.02 [R_m]        # Offset (desplazamiento) en el eje Xss.
    ε: float  = 1.03,          # ± 0.01              # Excentricidad
    L: float  = 2.04,          # ± 0.02 [R_m]        # Semi-latus rectum
    α: float  = np.deg2rad(4), # 4°=0.0698132 rad    # Ángulo de aberración (4 grados) respecto a SS (X' opuesto a flujo medio de viento solar)
    cant_puntos: int = 450     # Vignes lo llama N_b # Representa la suavidad (definición) de la hipérbola.
    #R_SD = 1.64 # ± 0.08 [R_m]# Éstos dos se pueden obtener (distancia sub-solar: desde el planeta al punto de la cónica en el eje X'_ss).
    #R_TD = 2.62 # ± 0.09 [R_m]# distancia sub-solar: desde el planeta al punto de la cónica en el eje Y'_ss.
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  La función cónica_Vignes representa la curva cónica que ajustó Vignes (1997) en el paper del año 2000, con los datos de un año de mediciones
  de la sonda Mars Global Surveyor (MGS). La función recibe los parámetros de semi-lado recto (L) y excentricidad (ε) de la cónica, y un punto
  inicial (θ0) y terminal (θf) y una cantidad de puntos, con los cuales realiza una malla para el parámetro θ de la función polar R(θ), e
  introduciendo el offset X0, calcula y devuelve las componentes cartesianas de la curva polar, los np.ndarray x e y. Además, devuelve su
  versión aberrada, que consiste en una matriz de rotación con respecto al ángulo α seteado por Vignes en 4°, con los np.ndarrays x_a e y_a.
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
    x_datos: np.ndarray,                                      # Mediciones de la componente X en coordenadas SS (Xss) de un bow shock.
    X0: float,                                                # El único parámetro offset libre que se desea ajustar (desplazamiento en Xss).
    *,
    cant_puntos: int = 450                                    # Cantidad de puntos con los que se desea ajustar (precisión del modelo)
) -> np.ndarray:
    """
    La función función_hipérbola_Vignes recibe un np.ndarray de datos que representa mediciones de las componentes X en el sistema de
    referencia SS (Xss) donde ocurrió un bow shock, y recibe un parámetro float X0. Utilizando todos los parámetros standard dados por
    Vignes (2000), excepto X0, traza la hipérbola única para el intervalo dado por las mediciones x_datos cuyo offset es X0, y la devuelve
    en formato y=f(x) solo para y>0 (devuelve y). El parámetro 'cant_puntos' permite ajustar la definición de la hipérbola creada.
    """
    _,_,x_a,y_a = cónica_Vignes(X0=X0, cant_puntos=cant_puntos) # Construyo la cónica de Vignes con el X0 y la cantidad de puntos propuestos.
    y_modelo: np.ndarray = np.empty_like(x_datos)               # Creo un ndarray de longitud x_datos sin nada, el cual voy a llenar.
    for i, Xss_i in enumerate(x_datos):                         # Para cada valor de mis datos (medición Fruchtman en Xss) i, de 0 a len(x),
      j: int = np.argmin(np.abs(x_a - Xss_i))                   # busco el j de la cónica construida más cercano a la medición actual Xss_i,
      y_modelo[i] = np.abs(y_a[j])                              # y coloco en mi y_modelo, el valor abs (solo y>0) del y_a[j] hallado.
    return y_modelo                                             # Devuelvo la hipérbola Vignes de forma y=f(x) que ajustó mejor X0 y x_datos.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————