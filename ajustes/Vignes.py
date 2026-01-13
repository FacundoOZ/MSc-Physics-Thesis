
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | 
#============================================================================================================================================

import numpy as np
from numpy   import cos, sin

#X0 = 0.78*R_m # DE DÓNDE SALE ESTO QUE USA CAMILA? NO LO ENCUENTRO EN EL PAPER DE VIGNES.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# cónica_Vignes: función que contiene los parámetros de Vignes (1997) para poder graficar las componentes cartesianas de la hipérbola en xy.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def cónica_Vignes(
    θ0: float = -0.05,          # Punto inicial de la malla de puntos para θ (seteado en -0.05 por mí, no por Vignes).
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

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def ajuste_hipérbola_Vignes(
    x_data: np.ndarray,
    X0: float,
    *,
    cant_puntos: int
) -> np.ndarray:
    """
    Modelo de ajuste basado en la cónica de Vignes (1997, 2000), con un único parámetro libre: el desplazamiento X0.
    Este modelo NO es una función analítica y(x), sino un modelo paramétrico (x(θ), y(θ)). Para cada valor observado de x_data, se busca
    el punto de la cónica cuya coordenada x_a sea la más cercana, y se toma el valor correspondiente de y_a.

    x_data : np.ndarray      Coordenadas X observadas (eje X'_SS aberrado).
    X0 : float               Offset en el eje X_SS (parámetro libre del ajuste).
    cant_puntos : int        Número de puntos usados para discretizar la cónica (controla la suavidad y precisión del modelo).
    y_model : np.ndarray     Valores modelados de |Y'_SS| correspondientes a x_data.
    """
    _,_,x_a,y_a = cónica_Vignes(X0=X0, cant_puntos=cant_puntos) # Construyo la cónica completa con el X0 propuesto
    y_model: np.ndarray = np.empty_like(x_data)                 # Para cada x observado, busco el punto más cercano sobre la cónica
    for i, xi in enumerate(x_data):
      j: int = np.argmin(np.abs(x_a - xi))
      y_model[i] = np.abs(y_a[j])  # modelo simétrico respecto al eje X
    return y_model

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————