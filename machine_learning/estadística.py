
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | 
#============================================================================================================================================

import numpy as np

def estadística(x: np.ndarray) -> list[float]:
  """
  Desviación estándar captura regiones calmas vs regiones de choque.
  El máximo detecta saltos en los choques.
  Percentiles detecta distribuciones asimétricas.
  """
  res: list[float] = [
    x.mean(),            # Media.
    x.std(),             # Desviación estándar.
    x.max(),             # Máximo.
    x.min(),             # Mínimo.
    np.median(x),        # Mediana.
    np.percentile(x,25), # Porcentaje del 25%.
    np.percentile(x,75)  # Porcentaje del 75%.
  ]
  return res

def estadística_módulos(x: np.ndarray) -> list[float]:
  """
  Documentación
  """
  media = x.mean()                      # Media
  res: list[float] = [                  #
    media,                              #
    x.std(),                            # Desviación estandar
    x.max()/media if media > 0 else 0.0 # ?
  ]
  return res
