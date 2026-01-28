
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para convertir magnitudes físicas entre sí
#============================================================================================================================================

import numpy  as np
import pandas as pd
from datetime import datetime, timedelta

R_m: float = 3396.3 # Radio marciano máximo (km)

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# función módulo:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def módulo(*componentes: np.ndarray, norm: float = 1.0) -> np.ndarray:
  """La función módulo calcula el módulo de un vector de componentes 2D ó 3D utilizando la distancia euclídea. El parámetro norm permite
  normalizar el módulo utilizando que sqrt(sum( (x_i/norm)**2 )) = sqrt(sum(x_i**2)) / norm."""
  if len(componentes) < 2:
    raise ValueError('Ingrese dos ó más componentes.')
  suma_cuadrados = np.zeros_like(componentes[0], dtype=float)
  for c in componentes:
    suma_cuadrados += c**2
  res: np.ndarray = np.sqrt(suma_cuadrados)/norm
  return res

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# fecha_UTC_a_DOY: 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def fecha_UTC_a_DOY(dia: str, mes: str, año: str) -> str:
  """
  La función fecha_UTC_a_DOY recibe en formato string un día, un mes y un año, y lo convierte a formato DOY (day of year), es decir, 
  devuelve un string que representa un número entero entre 001 y 365 (ó 366 para los años bisiestos).
  """
  fecha = datetime(int(año), int(mes), int(dia)) # Convierto los strings a enteros y creo una variable llamada fecha con la fecha asociada.
  return f'{fecha.timetuple().tm_yday:03d}'      # Devuelvo dicha fecha en formato día del año, y con formato de dos ceros (por ej.: 005).

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# dias_decimales_a_datetime: 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def dias_decimales_a_datetime(dia_decimal: np.ndarray, año: int) -> pd.DatetimeIndex:
  """
  Recibe una lista de días decimales en formato float (por ejemplo [123.75, 361.98]) y un año (por ejemplo 2019), y devuelve un DatetimeIndex
  que contiene una lista de objetos datetime con los días decimales del año correspondiente convertidos a formato 'AÑO-MES-DÍA HH:MM:SS'.
  """
  base = datetime(año, 1, 1)                                               # Agrego la variable tiempo (de tipo datetime) a res.
  return pd.to_datetime([base + timedelta(days=d-1) for d in dia_decimal]) # Devuelvo res en formato datetime.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# tiempo_UTC_en_segundos:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def tiempo_UTC_en_segundos(t: str) -> int:   # Función auxiliar para convertir tiempo UTC en segundos
  """
  Recibe un tiempo en formato string 'HH:MM:SS' y devuelve un entero que representa la cantidad de segundos que transcurrieron en ese día
  desde las '00:00:00' hasta el tiempo pasado por parámetro.
  """
  h,m,s = map(int, t.split(':')) # Separa la cadena por ':' y convierte h, m, s a enteros.
  res: int = 3600*h + 60*m + s   # Convierte todos los enteros en segundos y los guarda en la variable res.
  return res                     # Devuelvo res.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# dia_decimal_a_fecha_UTC / fecha_UTC_a_dia_decimal: permiten convertir una fecha específica a día decimal o viceversa
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def dia_decimal_a_fecha_UTC(dia_decimal: float, año: int) -> str:
  """
  Recibe un día decimal en formato float y un año en formato int, y devuelve un string en formato 'DD/MM/YYYY-HH:MM:SS' que representa la
  fecha UTC correspondiente al día, mes, hora, minutos y segundos del año que ha sido pasado por parámetro.
  """
  res = (                                             # En la variable tiempo,
    datetime(año, 1, 1) +                             # toma como referencia el día 1 de enero del año dado.
    timedelta(days = int(dia_decimal) - 1) +          # Sumo los días completos transcurridos desde el 1 de enero (menos el 1 de enero)
    timedelta(days = dia_decimal - int(dia_decimal))  # Sumo la parte fraccionaria, que corresponde a las horas, minutos y segundos.
  )
  return res.strftime('%d/%m/%Y-%H:%M:%S')            # Devuelvo res en formato datetime.
#———————————————————————————————————————————————————————————————————————————————————————
def fecha_UTC_a_dia_decimal(fecha_UTC: str, formato: str = '%d/%m/%Y-%H:%M:%S') -> float:
  """
  Recibe un string en formato de fecha string 'DD/MM/YYYY-HH:MM:SS' (predeterminado) y devuelve el día decimal del año correspondiente.
  """
  fecha      = datetime.strptime(fecha_UTC, formato)              # Convierto el string a objeto datetime con el formato pasado por parámetro
  inicio_año = datetime(fecha.year, 1, 1)                         # Inicio el contador datetime mediante 'inicio_año' (el 1 de enero) 
  dias       = (fecha - inicio_año).days + 1                      # Días completos transcurridos desde el 1 de enero
  segundos   = (3600*fecha.hour + 60*fecha.minute + fecha.second) # Parte fraccionaria del día.
  fraccion   = segundos/86400                                     # En fracción, guardo la parte no entera del día,
  res: float = dias + fraccion                                    # res, será el entero 'días' + la parte no entera fraccionaria.
  return res                                                      # Devuelvo res.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# segundos_a_día:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def segundos_a_día(s: int) -> float:
  """
  Recibe un entero que representa una cantidad de segundos y lo convierte a formato día, por ejemplo s=600 => devuelve 0.00694....
  """
  res: float = s/86400.0 # En la variable res, calculo y guardo los segundos ingresados en formato día.
  return res             # Devuelvo res.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# minutos_a_día:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def minutos_a_día(m: int) -> float:
  """
  Recibe un entero que representa una cantidad de minutos y lo convierte a formato día, por ejemplo m=30 => devuelve 0.0208....
  """
  res: float = m/1440.0 # En la variable res, calculo y guardo los minutos ingresados en formato día.
  return res            # Devuelvo res.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————