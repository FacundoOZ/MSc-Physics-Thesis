
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para unir los archivos de MAVEN MAG PlanetoCentric y Sun-State Coordinates recortados con recorte.py
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd
from tqdm import tqdm

from base_de_datos.conversiones import dias_decimales_a_datetime

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# unir_archivo_MAG: función para unir 2 archivos en 1 (que contenga las coordenadas PC y SS)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def unir_archivo_MAG(
    directorio: str,                                                                # Carpeta base donde se encuentran los archivos a unir.
    archivo_pc: str                                                                 # Nombre del archivo_pc en formato str que se desea unir. 
) -> None:
  """
  La función unir_archivo_MAG recibe 'archivo_pc', el nombre de un archivo en coordenadas planetocéntricas (PC) en formato string, y un
  directorio (también en formato string) donde se encuentran las carpetas 'datos_recortados_pc' y 'datos_recortados_ss', que dentro contienen
  subcarpetas año, mes y sus archivos .sts correspondientes. La función crea una nueva carpeta (si aún no existe) 'datos_recortados_merge'
  que contiene el archivo PC original y 3 columnas adicionales extraídas de el archivo SS correspondiente a la misma fecha. Para que los
  datos se correspondan al mismo tiempo en día decimal, se busca la fila en SS que coincida con el primer elemento del día decimal PC, y se
  toman los datos SS desde ahí en adelante.

  Archivo_PC:
    día_decimal    Bx    By    Bz    x_pc    y_pc    z_pc
    ....           ...   ...   ...   ...     ...     ...
  Archivo_SS:
    día_decimal    Bx    By    Bz    x_ss    y_ss    z_ss
    ....           ...   ...   ...   ...     ...     ...
  Archivo_merge:
    día_decimal    Bx    By    Bz    x_pc    y_pc    z_pc    x_ss    y_ss    z_ss
    ....           ...   ...   ...   ...     ...     ...     ...     ...     ...

    Observación:
  Las componentes Bx,By,Bz dependen del sistema de referencia por lo que las componentes de B en PC y SS no coinciden. Sin embargo, el
  módulo de B (sqrt(Bx**2+By**2+Bz**2)) sí coincide, por lo que descarto las SS.
  """
  archivo_ss = archivo_pc.replace('pc1s', 'ss1s')                                   # Con el nombre archivo_pc, obtengo el nombre en SS.
  año: str   = archivo_pc[11:15]                                                    # Mediante el str archivo_pc obtengo el año
  mes: str   = str(int(archivo_pc[27:29]))                                          # y el mes (en formato str), removiendo ceros (si hay).
  ruta_pc    = os.path.join(directorio,'datos_recortados_pc',   año,mes,archivo_pc) # Con año/mes/nombre, obtengo la ruta del archivo PC,
  ruta_ss    = os.path.join(directorio,'datos_recortados_ss',   año,mes,archivo_ss) # del archivo SS,
  ruta_merge = os.path.join(directorio,'datos_recortados_merge',año,mes)            # y del archivo de salida que será la unión de ambos.
  os.makedirs(ruta_merge, exist_ok=True)                                            # Creo las carpetas de destino correspondientes,
  ruta_final = os.path.join(ruta_merge, archivo_pc.replace('pc1s', 'merge1s'))      # y la ruta final del archivo unión (con nombre incluído)
  if os.path.exists(ruta_final):                                                    # Si dicha ruta (con el nombre) ya existe,
    print(f"El archivo {archivo_pc.replace('pc1s', 'merge1s')} ya existe.")         # => el archivo ya fue unido.
    return                                                                          # => no hago nada.
  try:                                                                              # Si no, tratamos de unir los archivos PC y SS
    PC = pd.read_csv(ruta_pc, sep=r'\s+', header=None, engine='python')             # Leo el contenido de las rutas correspondientes de los
    SS = pd.read_csv(ruta_ss, sep=r'\s+', header=None, engine='python')             # archivos PC y SS, respectivamente, sin título.
    SS = SS[SS[0] >= PC[0].iloc[0]].reset_index(drop=True)                          # Reescribo al SS cuando su día decimal coincida con PC
    if len(SS) != len(PC):                                                          # Si las longitudes son distintas,
      raise ValueError(f'Longitudes distintas: PC={len(PC)}, SS={len(SS)}')         # algo anda mal => chequear.
    if not np.allclose(PC[0], SS[0]):                                               # Si los días decimales punto a punto no son similares
      raise ValueError('Desalineación en día decimal.')                             # entre sí => hay algo mal con los tiempos.
    res = pd.concat([PC, SS[[4,5,6]]], axis=1)                                      # Concateno el archivo PC con las últimas 3 cols de SS
    res.to_csv(ruta_final, sep=' ', index=False, header=False)                      # (las posiciones), y convierto a csv con separación TAB.
  except FileNotFoundError as e:                                                    # Si no se encontró el archivo,
    print('Archivo no encontrado:', e.filename)                                     # arrojo un error,
  except Exception as e:                                                            # y si ocurre algún otro tipo de error,
    print('Error al mergear', archivo_pc, '->', e)                                  # devuelvo un aviso.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# unir_paquete_MAG: función para unir todos los archivos de un año entero
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def unir_paquete_MAG(
    directorio: str,                                                                             # Carpeta de archivos que se desea unir
    año: str                                                                                     # Año en formato string que se desea unir
) -> None:
  """
  La función unir_paquete_MAG recibe un directorio en formato string que contiene las carpetas 'datos_recortados_pc' y 'datos_recortados_ss'
  y recibe un año en formato string que representa el año cuyas mediciones se unirán. Une todos los archivos de datos recortados PC y SS en
  una subcarpeta dentro de 'datos_recortados_merge' cuyo nombre es 'año'.
  """
  lista: list[str] = []                                                                          # Creo lista vacía -> irán nombres de archivo
  for ruta_actual, _, archivos in os.walk(os.path.join(directorio, 'datos_recortados_pc', año)): # Recorro todos los .sts de la carpeta.
    for archivo in archivos:                                                                     # Para cada archivo en ella,
      if archivo.endswith('.sts') and 'pc1s' in archivo:                                         # si termina en .sts y tiene 'pc1s' en nombre
        lista.append(os.path.join(ruta_actual, archivo))                                         # -> lo agrego a la lista
  for elem in tqdm(lista, desc=f'Uniendo año {año}', unit='archivo'):                            # -> tqdm usará la longitud de la lista
    unir_archivo_MAG(directorio, os.path.basename(elem))                                         # -> obtengo el nombre y uno los archivos

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# unir_datos_fruchtman_MAG: 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def unir_datos_fruchtman_MAG(
    directorio: str,                                                                             # Carpeta de archivos que se desea unir
    año: str                                                                                     # Año en formato string que se desea unir
) -> None:
  """
  La función unir_datos_fruchtman_MAG recibe 
  """
  fruchtman_path = os.path.join(directorio, f"fruchtman_{año}_recortado.txt")
  fruchtman_days = np.loadtxt(fruchtman_path)
  output_rows = []
  current_doy = None
  mag_data = None
  for dec_day in fruchtman_days:
    dt = dias_decimales_a_datetime(np.array([dec_day]), año)[0]
    doy   = dt.timetuple().tm_yday
    month = dt.month
    day   = dt.day
    if doy != current_doy:
      mag_filename = (f"mvn_mag_l2_{año}{doy:03d}merge1s_{año}{month:02d}{day:02d}_v01_r01_recortado.sts")
      mag_path = os.path.join(directorio, "datos_recortados_merge", str(año), str(month), mag_filename)
      mag_data = np.loadtxt(mag_path)
      mag_decimal_days = mag_data[:, 0]
      current_doy = doy
    idx = find_nearest_index(mag_decimal_days, dec_day)
    merged_row = np.concatenate(([dec_day], mag_data[idx, 1:]))
    output_rows.append(merged_row)
  output = np.array(output_rows)
  output_path = os.path.join(directorio, f"fruchtman_{año}_recortado_merge.txt")
  np.savetxt(output_path, output, fmt="%.6f")

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def find_nearest_index(arr: np.ndarray, value: float) -> int:
  """
  Documentación
  """
  idx = np.searchsorted(arr, value)
  if idx == 0:
    return 0
  if idx == len(arr):
    return len(arr) - 1
  before = arr[idx - 1]
  after  = arr[idx]
  return idx if abs(after - value) < abs(value - before) else idx - 1



#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————