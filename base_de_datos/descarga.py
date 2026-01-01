
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para descargar los datos de MAVEN MAG de la base de datos de Colorado (en 1 Hz)
#============================================================================================================================================

import os                                # Para archivos
import time                              # Para setear intervalos de tiempo intencionados
import requests                          # Para acceder al server de LASP
import random
from tqdm     import tqdm                # Para mostrar barras de progreso de descarga
from datetime import datetime, timedelta # Contiene correctamente los días de cada año

link = 'https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2/'                                 # Link de la base de datos de MAVEN
ID   = {'User-Agent': 'Mozilla/5.0 (compatible; FacundoDownloader/1.0; +https://lasp.colorado.edu)'} # Identificación como usuario para LASP

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# descargar_archivo_MAG: función para descargar 1 único archivo
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def descargar_archivo_MAG(
    directorio: str,                                             # Ubicación de la carpeta en donde se descargará el archivo
    dia: str,                                                    # Recibe el dia en cualquier formato (por ej.: 2 ó 02)
    mes : str,                                                   # Recibe el mes en cualquier formato (por ej.: 3 ó 03)
    año: str,                                                    # Recibe el año
    coord: str = 'pc'                                            # Tipo de coordenadas que se desean descargar
) -> str:
  """
  La función descrgar_archivo_MAG recibe en formato string un día, un mes y un año, y accede directamente a la base de datos de colorado de
  la página con enlace público lasp.colorado.edu. Una vez allí descarga el archivo .sts del instrumento MAG (magnetómetro) de MAVEN, 
  correspondiente a las mediciones de campo magnético en coordenadas centradas en Marte (PC: PlanetoCéntricas) con resolución de 1 Hz (los
  de baja resolución: aprox. 86000 datos por día) para utilizar posteriormente en ML.
  """
  dia_con_cero: str = f'{int(dia):02d}'                          # PRECAUCIÓN: los links de LASP llevan 0 delante en los días
  mes_con_cero: str = f'{int(mes):02d}'                          # y en los meses
  DOY: str           = dia_del_año(dia, mes, año)                # Obtengo el DOY, que determina la fecha en formato "día del año"
  carpeta_destino = os.path.join(directorio, año, str(int(mes))) # Creo las carpetas (path) de la forma: directorio/año/mes
  os.makedirs(carpeta_destino, exist_ok=True)                    # Creo el path establecido si es que no existe aún (si existe -> OK)
  for j in ('01', '02'):
    nombre = f'mvn_mag_l2_{año}{DOY}{coord}1s_{año}{mes_con_cero}{dia_con_cero}_v01_r{j}.sts' # Para el 6-8/2015; 6/2019; 11/2020 -> '02.sts'
    ruta   = os.path.join(carpeta_destino, nombre)               # Establezco el path + el nombre del archivo
    if os.path.exists(ruta) and os.path.getsize(ruta) > 0:
      return 'Preexistente'
    url = link + f'{año}/{mes_con_cero}/' + nombre               # Accedo a la dirección que contiene el día '.sts' a descargar
    try:
      r = requests.get(url, headers=ID, timeout=30, stream=True) # Envío petición para descargar archivo (corta si pasan 60 segundos)
      if r.status_code == 200:                                   # Si la petición es aceptada, (200 = 'OK'),
        with open(ruta, 'wb') as f:                              # crea el archivo en 'ruta' en writing binary mode=wb (escritura binaria)
          for chunk in r.iter_content(chunk_size=8192):          # Descargo 8 KB en cada iteración (más eficiente y ocupa menos RAM)
            if chunk:                                            # que es recomendado para grandes bases de datos.
              f.write(chunk)                                     # Escribe el contenido de los 8 KB.
          return 'Descargado'
      elif r.status_code == 404:                                 # Si la version 01 no se encontró -> voy a la siguiente iteración del for
        continue                                                 # -> entro a '02'.
      else:
        print(f'[{r.status_code}] {url}')
        return 'Error'
    except requests.RequestException as e:
      print(f'[error] {url} -> {e}')
  return 'No encontrado'                                         # Si no se encontró ni 01 ni 02 -> return 'No encontrado'.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# descargar_paquete_MAG: función para descargar todos los archivos de un intervalo de tiempo (hasta 11 años)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def descargar_paquete_MAG(
    directorio: str,                                                         # Carpeta donde se guardarán los archivos
    fecha_inicio: str,                                                       # Recibe la fecha inicio en tipo dia/mes/año en cualquier formato
    fecha_final: str,                                                        # Recibe la fecha final  en tipo dia/mes/año en cualquier formato
    coord: str = 'pc'                                                        # Tipo de coordenadas del paquete que se desean descargar
) -> None:
  """
  Descarga todo un paquete de archivos del MAG en un intervalo de tiempo especificado. Recibe una fecha de inicio, y una fecha final en
  formatos dia/mes/año, y descarga todo en el argumento directorio pasado por parámetro.
  """
  fecha_inicio = datetime.strptime(fecha_inicio, '%d/%m/%Y')                 # Fecha de inicio (omito el primer mes, que fue de prueba)
  fecha_final  = datetime.strptime(fecha_final,  '%d/%m/%Y')                 # Fecha de final
  cant_dias    = (fecha_final - fecha_inicio).days + 1
  contador: dict[str,int] = {
    'Descargado': 0, 'Preexistente': 0, 'No encontrado': 0, 'Error': 0       # Creo un diccionario para el recuento de archivos
  }
  j = fecha_inicio
  with tqdm(total=cant_dias, desc='Descargando', unit='día') as pbar:        # Uso barra de progreso tqdm
    while j <= fecha_final:                                                  # Mientras la fecha actual esté en el intervalo deseado,
      dia, mes, año = f'{j.day:02d}', f'{j.month:02d}', str(j.year)          # Obtengo los strings del día, mes y año
      contador[descargar_archivo_MAG(directorio, dia, mes, año, coord)] += 1 # Agrego 1 a la clave del diccionario que corresponda
      pbar.set_postfix(contador)                                             # Actualizo el diccionario,
      pbar.update(1)                                                         # y la barra de progreso
      j += timedelta(days=1)                                                 # El iterador suma 1 día al loop
      #time.sleep(random.uniform(1, 2))                                      # Intervalo de tiempo de espera para no sobrecargar el server.
  print("Resultado final:", contador)                                        # Resultado de la descarga

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def dia_del_año(
    dia: str,
    mes: str,
    año: str
) -> str:
  """
  La función dia_del_año recibe en formato string un día, un mes y un año, y lo convierte a formato DOY (day of year), es decir, 
  devuelve un string que representa un número entero entre 001 y 365 (ó 366 para los años bisiestos).
  """
  fecha = datetime(int(año), int(mes), int(dia)) # Convierto los strings a enteros y creo una variable llamada fecha con la fecha asociada.
  return f'{fecha.timetuple().tm_yday:03d}'      # Devuelvo dicha fecha en formato día del año, y con formato de dos ceros (por ej.: 005).
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————