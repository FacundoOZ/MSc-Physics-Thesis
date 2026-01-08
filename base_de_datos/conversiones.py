
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para convertir magnitudes físicas entre sí
#============================================================================================================================================

from datetime import datetime, timedelta

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# dia_decimal_a_fecha_UTC / fecha_UTC_a_dia_decimal : permiten convertir una fecha específica a día decimal o viceversa
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def dia_decimal_a_fecha_UTC(
    dia_decimal: float,
    año: int
) -> str:
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

def fecha_UTC_a_dia_decimal(
    fecha_UTC: str,
    formato: str = '%d/%m/%Y-%H:%M:%S'
) -> float:
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
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————