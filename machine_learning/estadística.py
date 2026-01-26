
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | 
#============================================================================================================================================

import numpy as np

def estadística_componentes_R(x: np.ndarray) -> list[float]:
  """
  Desviación estándar captura regiones calmas vs regiones de choque.
  El máximo detecta saltos en los choques.
  Percentiles detecta distribuciones asimétricas.
  """
  res: list[float] = [   #
    x.mean(),            # Media.
    x.std(),             # Desviación estándar.
    x.max(),             # Máximo.
    x.min(),             # Mínimo.
    np.median(x),        # Mediana.
    np.percentile(x,25), # Porcentaje del 25%.
    np.percentile(x,75)  # Porcentaje del 75%.
  ]                      #
  return res             #

def estadística_componentes_B(x: np.ndarray) -> list[float]:
  """
  Documentación
  """
  if len(x) < 2:                  #
    return [0.0]*5                #
  dx = np.gradient(x)             #
  res: list[float] = [            #
    x.std(),                      # Desviación estándar.
    np.max(np.abs(x)),            #
    dx.std(),                     #
    np.max(np.abs(dx)),           #
    np.percentile(np.abs(dx), 95) #
  ]                               #
  return  res                     #

def estadística_R(x: np.ndarray) -> list[float]:
  """
  Documentación
  """
  media = x.mean()                       # Media
  res: list[float] = [                   #
    media,                               #
    x.std(),                             # Desviación estandar
    x.max()/media if media > 0 else 0.0, # ?
    np.median(x),                        # Mediana.
    np.percentile(x,25),                 # Porcentaje del 25%.
    np.percentile(x,75),                 # Porcentaje del 75%.
    x.max() - x.min()                    # Máximo.
  ]                                      #
  return res                             #

def estadística_B(x: np.ndarray) -> list[float]:
  """
  Documentación
  """
  if len(x) < 2:                         #
    return [0.0]*7                       #
  dx    = np.gradient(x)                 #
  media = x.mean()                       # Media
  res: list[float] = [                   #
    media,                               #
    x.std(),                             # Desviación estandar
    x.max()/media if media > 0 else 0.0, # ?
    dx.mean(),                           #
    dx.std(),                            #
    np.max(np.abs(dx)),                  #
    np.percentile(np.abs(dx), 95)        #
  ]                                      #
  return res                             #
