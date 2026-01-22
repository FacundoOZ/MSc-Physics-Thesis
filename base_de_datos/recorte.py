
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para recortar los archivos de MAVEN MAG Sun-State y PlanetoCentric Coordinates descargados por descarga.py
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd
from tqdm              import tqdm
from typing            import Any
from scipy.interpolate import interp1d

# Módulos Propios:
from base_de_datos.conversiones import R_m, módulo, fecha_UTC_a_dia_decimal
from ajustes.Vignes             import hipérbola_mínima, hipérbola_máxima, segmento_izquierdo

columnas = [6,7,8,9,11,12,13] # 6 => día_decimal #          # [7,8,9] => B_x,B_y,B_z [nT] #          # [11,12,13] => x,y,z [km] #

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_archivo_MAG: función para recortar 1 único archivo
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_archivo_MAG(
    directorio: str,                                                                   # Carpeta donde se encuentra el archivo a recortar.
    archivo: str,                                                                      # Nombre del archivo a recortar en formato string.
    coord: str = 'pc'                                                                  # Tipo de coordenadas del archivo a recortar.
) -> None:
  """
  La función recortar_archivo_MAG recibe en formato string el nombre de un archivo y lo recorta y agrega a una carpeta 'datos_recortados_coord'
  en las subcarpetas correspondientes al año y al mes del archivo que se ha seleccionado (extrae el año y el mes del nombre del archivo).
  Si ya fue recortado, no lo recorta.
  """
  año: str  = archivo[11:15]                                                           # Extraigo el año del string archivo,
  mes: str  = str(int(archivo[27:29]))                                                 # y el mes (remuevo los ceros delante si hubiera)
  ruta_i    = os.path.join(directorio, año, mes, archivo)                              # Obtengo la ruta inicial (ruta origen) del archivo
  ruta_f    = os.path.join(directorio, f'datos_recortados_{coord}', año, mes)          # Obtengo la ruta final donde se guardará el recortado
  os.makedirs(ruta_f, exist_ok=True)                                                   # Creo la carpeta de destino (si es que no existe aún)
  recortado = os.path.join(ruta_f, archivo.replace('.sts', '_recortado.sts'))          # Establezco la ruta_completa + nombre_nuevo_archivo
  if os.path.exists(recortado):                                                        # Si el archivo recortado ya existe,
    print(f'El archivo "{os.path.basename(archivo)}" ya ha sido recortado.')           # imprimo un mensaje de aviso
    return                                                                             # => salgo
  try:                                                                                 # Si no,
    data = pd.read_csv(ruta_i, sep=r'\s+', header=None, skiprows=160, engine='python') # Extraigo los datos de ruta_i, y omito 160 líneas
    datos_recortados = data.iloc[:, columnas]                                          # En 'datos_recortados' extraigo las columnas deseadas
    datos_recortados.to_csv(recortado, sep=' ', index=False, header=False)             # Convierto ruta final (con nombre) a CSV y sep. TAB
    #print(f'El archivo "{os.path.basename(archivo)}" se ha recortado.')               # Omitir esto para usar todo el paquete.
  except FileNotFoundError:                                                            # Si el archivo de origen no está,
    print(f'El archivo "{os.path.basename(archivo)}" no se ha encontrado.')            # devuelve error.
  except Exception as e:                                                               # Si hay algún otro tipo de error,
    print('El archivo', os.path.basename(archivo), '->', e)                            # creo un aviso.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_paquete_MAG: función para recortar todos los archivos de un año entero
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_paquete_MAG(
    directorio: str,                                                       # Carpeta donde se encuentran los archivos que se desean recortar.
    año: str,                                                              # Año en formato string cuyas mediciones se desean recortar.
    coord: str = 'pc'                                                      # Tipo de coordenadas del paquete a recortar.
) -> None:
  """
  Recorre todos los archivos .sts en la carpeta MAG/año y llama a recortar_archivo_MAG para cada uno. Los archivos recortados se guardan en
  datos_recortados_merge/año/mes.
  """
  lista: list[str] = []                                                    # Creo una lista vacía donde irán los nombres de archivo (str)
  for ruta_actual, _, archivos in os.walk(os.path.join(directorio, año)):  # Recorro todos los archivos .sts dentro de MAG
    for archivo in archivos:                                               # Para cada archivo de todos los archivos que hay,
      if archivo.endswith('.sts'):                                         # si termina en formato '.sts',
        lista.append(os.path.join(ruta_actual, archivo))                   # lo agrego a la lista de archivos.
  for elem in tqdm(lista, desc=f'Recortando año {año}', unit='archivo'):   # La barra de progreso dependerá de los archivos de 'lista'
    recortar_archivo_MAG(directorio, os.path.basename(elem), coord)        # Extraigo solo el nombre del archivo, para pasarle a la funcion.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_hemisferios_MAG: función para recortar, en principio, los datos del hemisferio norte de las mediciones MAG.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_hemisferios_MAG(
    directorio: str,                                                                        # Carpeta donde se encuentra el archivo a recortar.
    archivo: str,                                                                           # Nombre del archivo a recortar en formato string.
    hemisferio: str = 'norte'                                                               # Hemisferio que se desea recortar.
) -> None:
  """
  La función recortar_hemisferios_MAG recibe en formato string el nombre de un archivo de la carpeta datos_recortados_merge y lo recorta aún
  más, conservando únicamente las mediciones correspondientes al hemisferio norte, o bien las del hemisferio norte y lado diurno del planeta.
  Estos datos de los primeros 4 ó 2 octantes de las coordenadas PC (z_pc > 0) y (z_pc > 0 ; x_ss > 0) se guardan en un archivo .sts en un
  directorio denominado "hemisferio_N" y "hemisferio_ND" (hemisferio norte y hemisferio norte diurno), respectivamente, y en las subcarpetas
  del año y mes asociadas al nombre del archivo original. Si el archivo ya fue recortado, no lo recorta.
  """
  año: str    = archivo[11:15]                                                              # Extraigo del string archivo el año
  mes: str    = str(int(archivo[30:32]))                                                    # y el mes (remuevo los ceros delante del mes)
  ruta_origen = os.path.join(directorio, 'datos_recortados_merge', año, mes, archivo)       # Obtengo la ruta de origen del archivo
  if hemisferio == 'norte':                                                                 # Si solo quiero el hemisferio norte:
    ruta_final    = os.path.join(directorio, 'hemisferio_N', año, mes)                      # Obtengo la ruta del archivo recortado.
    os.makedirs(ruta_final, exist_ok=True)                                                  # Creo la carpeta de destino (si aún no existe).
    archivo_final = os.path.join(ruta_final, archivo.replace('.sts', '_hemisferio_N.sts'))  # Establezco ruta completa + nombre del archivo.
  elif hemisferio == 'norte_diurno':                                                        # Si quiero el hemisferio norte y lado diurno:
    ruta_final    = os.path.join(directorio, 'hemisferio_ND', año, mes)                     # Ídem.
    os.makedirs(ruta_final, exist_ok=True)                                                  # Creo la carpeta de destino (si aún no existe).
    archivo_final = os.path.join(ruta_final, archivo.replace('.sts', '_hemisferio_ND.sts')) # Establezco ruta completa + nombre del archivo.
  if os.path.exists(archivo_final):                                                         # Si el archivo ya existe, entonces ya fue leido
    print(f'El archivo "{os.path.basename(archivo)}" ya ha sido leido.')
    return
  try:                                                                                      # Si no,
    data = pd.read_csv(ruta_origen, sep=' ', header=None, engine='python')                  # Extraigo los datos de ruta_origen, con sep TAB
    z_pc = data.iloc[:,6]                                                                   # Defino la columna z del archivo 'data'
    if hemisferio == 'norte':                                                               # Si solo quiero el hemisferio norte ->
      datos_hemisferio = data[z_pc > 0]                                                     # En datos_hemisferio extraigo toda fila con z>0
    elif hemisferio == 'norte_diurno':                                                      # Si quiero el hemisferio norte y lado diurno ->
      x_ss = data.iloc[:,7]                                                                 # Defino la columnas x del archivo 'data'
      datos_hemisferio = data[(x_ss > 0) & (z_pc > 0)]                                      # y extraigo toda fila que cumpla x>0 y z>0
    datos_hemisferio.to_csv(archivo_final, sep=' ', index=False, header=False)              # Convierto ruta final a CSV y separación TAB
    #print(f'El archivo "{os.path.basename(archivo)}" se ha leído correctamente.')          # Omito esto para recortar todo el paquete.
  except FileNotFoundError:                                                                 # Si el archivo origen no está,
    print(f'El archivo "{os.path.basename(archivo)}" no se ha encontrado.')                 # aviso que no se ha encontrado.
  except Exception as e:                                                                    # Si hay algún otro tipo de error,
    print('El archivo', os.path.basename(archivo), '->', e)                                 # lo aviso.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_hemisferios_paquete_MAG: función para recortar todos los archivos de un año entero
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_hemisferios_paquete_MAG(
    directorio: str,                                                                            # Carpeta origen del archivo a recortar.
    año: str,                                                                                   # Año cuyas mediciones se desean recortar.
    hemisferio: str = 'norte'                                                                   # Hemisferio que se desea recortar.
) -> None:
  """
  Recorre todos los archivos .sts en la carpeta datos_recortados_merge/año y llama a recortar_hemisferios_MAG para cada uno. Los archivos
  recortados se guardan en la carpeta hemisferio_N/año/mes correspondiente, ó el hemisferio que corresponda.
  """
  nombres: list[str] = []                                                                       # Creo lista donde irán los nombres (str)
  for ruta_actual,_,archivos in os.walk(os.path.join(directorio,'datos_recortados_merge',año)): # Recorro datos_recortados_merge
    for archivo in archivos:                                                                    # Para cada archivo de todos los que hay,
      if archivo.endswith('.sts'):                                                              # si termina en formato '.sts',
        nombres.append(os.path.join(ruta_actual, archivo))                                      # lo agrego a mi lista 'nombres'
  print(f'Se encontraron {len(nombres)} archivos .sts para recortar (año {año}).')              # Muestro la cant de archivos encontrados.
  for j in tqdm(nombres, desc=f'Recortando año {año}', unit='archivo'):                         # Para barra de progreso uso => len(nombres)
    if hemisferio == 'norte':                                                                   # Si quiero solo el hemisferio norte,
      recortar_hemisferios_MAG(directorio, os.path.basename(j), hemisferio='norte')             # le paso hemisferio='norte' a la función.
    elif hemisferio == 'norte_diurno':                                                          # Si no,
      recortar_hemisferios_MAG(directorio, os.path.basename(j), hemisferio='norte_diurno')      # le paso hemisferio='norte_diurno'

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_datos_fruchtman_MAG: función para recortar el catálogo de bow shocks de fruchtman y quedarme solo con los días decimales (col[0])
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_datos_fruchtman_MAG(
    directorio: str,                                                            # Directorio del archivo, y donde se guardarán los recortes.
    archivo: str,                                                               # Nombre del archivo.
    año: int                                                                    # Año que se desea recortar.
) -> None:
  """
  La función recortar_datos_fruchtman_MAG lee el catálogo Fruchtman, filtra por año, conserva solo la columna temporal, convierte a día
  decimal y guarda un único archivo por año.
  """
  ruta_i = os.path.join(directorio, archivo)                                    # Obtengo la ruta inicial como directorio + archivo.
  data   = pd.read_csv(ruta_i, comment=';', header=None, skipinitialspace=True) # Leo y guardo el contenido omitiendo la primera línea.
  años   = data[0].str.slice(0,4).astype(int)                                   # Obtengo los años del archivo .txt.
  data_años = data[años == año]                                                 # Obtengo los datos del año pasado por parámetro.
  if data_años.empty:                                                           # Si no hay nada en el año ingresado,
    print(f'No hay datos para el año {año}')                                    # devuelvo un mensaje,
    return                                                                      # y salgo.
  días_decimales = data_años[0].apply(                                          # En días_decimales guardo dichos datos de la columna 0:
    fecha_UTC_a_dia_decimal,                                                    # convierto los valores de fecha y hora a día decimal usando
    formato='%Y-%m-%d/%H:%M:%S'                                                 # la función de conversiones, pero con el formato fruchtman.
  )
  nombre_f = f'fruchtman_{año}_recortado.txt'                                   # Creo el nombre del archivo final (de salida).
  ruta_f   = os.path.join(directorio, nombre_f)                                 # Creo su ruta como directorio + nombre,
  días_decimales.to_csv(ruta_f, index=False, header=False, float_format='%.6f') # y lo guardo (exporto) como csv.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_Vignes_MAG: función para recortar, en principio, los datos del hemisferio norte de las mediciones MAG.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_Vignes_MAG(
    directorio: str,                                                             # Carpeta donde se encuentra el archivo a recortar.
    archivo: str,                                                                # Nombre del archivo a recortar en formato string.
    región: dict[str, Any],                                                      # Dicc que contiene las variables de la geometría a recortar.
) -> None:
  """
  La función recortar_Vignes_MAG recibe en formato string un directorio y un archivo que representan la carpeta donde se encuentra el archivo
  a recortar, y el nombre del archivo, respectivamente. Extrae el año y mes del nombre del archivo y lo busca en las carpetas correspondientes
  y recibe un diccionario que conteine los parámetros necesarios para la región que se desea recortar. Utiliza como máscara para los datos
  dicha región y devuelve el archivo recortado en el mismo directorio, pero en una carpeta llamada 'recorte_Vignes'.
  """
  año: str = archivo[11:15]                                                      # Extraigo del string archivo el año
  mes: str = str(int(archivo[30:32]))                                            # y el mes (remuevo los ceros delante del mes)
  ruta_i   = os.path.join(directorio, 'hemisferio_N', año, mes, archivo)         # Obtengo la ruta de origen del archivo
  ruta_f   = os.path.join(directorio, 'recorte_Vignes', año, mes)                # Obtengo la ruta del archivo recortado
  os.makedirs(ruta_f, exist_ok=True)                                             # Creo la carpeta de destino (si aún no existe)
  archivo_final = os.path.join(ruta_f, archivo.replace(                          # Establezco la ruta completa + nombre del archivo del
    '_recortado_hemisferio_N.sts', '_final.sts'))                                # archivo final, reemplazando _recortado... por _final.
  if os.path.exists(archivo_final):                                              # Si el archivo ya existe, entonces ya fue leido,
    print(f'El archivo "{os.path.basename(archivo)}" ya ha sido recortado.')     # => aviso en un mensaje
    return                                                                       # y salgo.
  try:                                                                           # Si no, intento leerlo.
    data        = pd.read_csv(ruta_i, sep=' ', header=None, engine='python')     # Extraigo los datos de ruta_i, con sep TAB
    Xss,Yss,Zss = [data.iloc[:,k].to_numpy() for k in [7,8,9]]                   # Extraigo las componentes x,y,z en coordenadas SS.
    máscara     = región_Vignes(Xss,Yss,Zss, región['y_A'], región['y_B'],       # En máscara, defino la región de Vignes para las coordenadas
                                región['f_min'],región['f_max'],región['recta']) # SS extraídas (región que se desea preservar).
    data[máscara].to_csv(archivo_final, sep=' ', index=False, header=False)      # Aplico la máscara a los datos del archivo, y convierto a csv
    #print(f'"{os.path.basename(archivo)}" se ha leído correctamente.')          # Omito esto para recortar todo el paquete.
  except FileNotFoundError:                                                      # Si el archivo origen no está,
    print(f'El archivo "{os.path.basename(archivo)}" no se ha encontrado.')      # aviso que no se ha encontrado.
  except Exception as e:                                                         # Si hay algún otro tipo de error,
    print('El archivo', os.path.basename(archivo), '->', e)                      # lo aviso.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def preparar_región_Vignes() -> dict[str, Any]:
  """
  La función preparar_región_Vignes no recibe ningún parámetro. Obtiene los np.ndarrays (todos los puntos x,y) de las hipérbolas mínima y 
  máxima y del segmento superior izquierdo que encierran la región cuyas mediciones se desean tener en cuenta, acotada por las hipérbolas del
  tipo Vignes. En un diccionario del tipo [string, Any] devuelve todo lo necesario para aplicar el recorte sin tener que redefinirlo cada vez.
  """
  recta     = segmento_izquierdo()                                                     # Obtengo el segmento que une las hipérbolas.
  Xmax,Ymax = hipérbola_máxima()                                                       # Obtengo los np.ndarrays de la hipérbola derecha 
  Xmin,Ymin = hipérbola_mínima()                                                       # (máxima) y la izquierda (mínima).
  j_min,j_max = np.argsort(Ymin), np.argsort(Ymax)                                     # Obtengo los índices que ordenan Y de forma creciente.
  f_min = interp1d(Ymin[j_min],Xmin[j_min], bounds_error=False, fill_value=np.nan)     # f_min(Y) y f_max(Y) representan hipérbolas mín/máx
  f_max = interp1d(Ymax[j_max],Xmax[j_max], bounds_error=False, fill_value=np.nan)     # bien parametrizadas en Y de abajo hacia arriba.
  return {'y_A':Ymin[-1], 'y_B':Ymax[-1], 'f_min':f_min, 'f_max':f_max, 'recta':recta} # Devuelvo el dicc con los parámetros de la región.

#———————————————————————————————————————————————————————————————————————————————————————
def región_Vignes(
    Xss: np.ndarray, Yss: np.ndarray, Zss: np.ndarray,              # Componentes x,y,z en coordenadas SS
    y_A: float, y_B: float,                                         # Último punto de la coordenada y de hipérbola mín y máx, respectivamente.
    f_min, f_max, recta                                             # Funciones interpoladas para hipérbola mín, máx y el segmento izquierdo.
) -> np.ndarray:
  """
  La función región_Vignes devuelve una máscara booleana que preservará únicamente los puntos que cumplen con la condición de pertenecer a la
  región de Vignes tomada como la región encerrada entre las hipérbolas mínima y máxima, acotada inferiormente por la recta y=0 y superiormente
  por el segmento superior izquierdo llamado 'recta' en el plano 2D (Xss, sqrt(Yss**2 + Zss**2)) normalizado por el radio marciano R_m.
  Devuelve un np.ndarray que representa dicha máscara booleana.
  """
  X = Xss/R_m                                                       # X es la componente x en el sistema SS normalizada por R_m de archivo MAG.
  Y = módulo(Yss, Zss, norm=R_m)                                    # Y es la proyección vertical del plano yz normalizada por R_m.
  x_min_h = f_min(Y)                                                # Las variables x_min_h y x_max_h representan las hipérbolas mín y máx,
  x_max_h = f_max(Y)                                                # respectivamente, evaluadas en el eje vertical (Y).
  máscara_inf = (          (Y<=y_A) & (X>=x_min_h)  & (X<=x_max_h)) # Máscara inferior =      hipérbola_mín/hipérbola_max  para   0 < Y < y_A.
  máscara_sup = ((Y>y_A) & (Y<=y_B) & (X>=recta(Y)) & (X<=x_max_h)) # Máscara superior = segmento_izquierdo/hipérbola_máx  para y_A < Y < y_B.
  return máscara_inf | máscara_sup                                  # Devuelvo la máscara completa de la región de Vignes.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_Vignes_paquete_MAG: función para recortar todos los archivos MAG de un año entero usando Vignes
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def recortar_Vignes_paquete_MAG(
    directorio: str,                                                                  # Carpeta base donde están los datos MAG.
    año: str                                                                          # Año en formato string a recortar.
) -> None:
  """
  Recorre todos los archivos .sts que se encuentran dentro de la carpeta 'año' (string pasado por parámetro) dentro de la carpeta 'directorio'
  pasada por parámetro. Llama a la función recortar_Vignes_MAG y utiliza preparar_región_Vignes() para recortar únicamente la región de
  Vignes en cada uno. La región de Vignes se calcula una sola vez (al principio) y los archivos recortados se guardan en el directorio en la
  carpeta llamada 'recorte_Vignes' y en las subcarpetas correspondientes al año/mes.
  """
  lista: list[str] = []                                                               # Creo una lista con los archivos a recortar.
  for ruta_actual,_,archivos in os.walk(os.path.join(directorio,'hemisferio_N',año)): # Recorro todos los archivos del directorio de ese año.
    for archivo in archivos:                                                          # Para cada archivo,
      if archivo.endswith('.sts'):                                                    # si termina en formato '.sts'
        lista.append(os.path.join(ruta_actual, archivo))                              # entonces lo agrego a la lista.
  for elem in tqdm(lista, desc=f'Recortando año {año}', unit='archivo'):              # Barra de progreso respecto a la longitud de 'lista'.
    recortar_Vignes_MAG(directorio, os.path.basename(elem), preparar_región_Vignes()) # Recorto cada archivo mediante la región de Vignes.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————