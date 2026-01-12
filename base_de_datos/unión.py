
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para unir los archivos de MAVEN MAG PlanetoCentric y Sun-State Coordinates recortados con recorte.py
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd
from tqdm import tqdm

# Módulos Propios:
from base_de_datos.conversiones import dias_decimales_a_datetime, fecha_UTC_a_DOY

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
    directorio: str,                                                                            # Carpeta de archivos que se desea unir.
    año: str                                                                                    # Año en formato string que se desea unir.
) -> None:
  """
  La función unir_datos_fruchtman_MAG recibe en formato string un directorio, que representa la ruta donde se encuentran los archivos
  'fruchtman_{año}_recortado.txt', y un año, que representa el archivo que se desea unir del año correspondiente.
  Dado el archivo fruchtman, la función hace lo siguiente: para cada fila (cada día decimal), convierte el día a fecha UTC => va a buscar el
  archivo 'mvn_mag_l2_{año}{DOY}merge1s_{año}{mes}{dia}_v01_r01_recortado.sts' a las carpetas correspondientes que representa el archivo de
  la misma fecha, y selecciona el día decimal del MAG que más se acerque al de Fruchtman (no coinciden). Finalmente, guarda en el archivo
  'fruchtman_{año}_merge.txt' toda la fila del archivo MAG correspondiente (usa el día decimal de MAG) que contiene las componentes del campo
  B (en PC) y las posiciones de la sonda en los sistemas de coordenadas PC y SS.
  
  Archivo 'fruchtman_{año}_recortado.txt':
    dia_decimal
    ....
  Archivo_merge:
    día_decimal    Bx    By    Bz    x_pc    y_pc    z_pc    x_ss    y_ss    z_ss               # REEMPLAZA DÍA FRUCHTMAN => DÍA MERGE.
    ....           ...   ...   ...   ...     ...     ...     ...     ...     ...
  """
  ruta_fruch: str = os.path.join(directorio, 'fruchtman', f'fruchtman_{año}_recortado.txt')     # Creo la ruta Fruchtman del año solicitado.
  archivo_fruch: np.ndarray = np.loadtxt(ruta_fruch)                                            # Cargo el archivo Fruchtman.
  filas: list[np.ndarray] = []                                                                  # En la lista filas, colocaré el resultado.
  DOY_actual: str | None            = None                                                      # No recargo el archivo varias veces => uso
  data_mag: np.ndarray | None       = None                                                      # variables aux para el DOY, el archivo actual,
  dias_decimales: np.ndarray | None = None                                                      # y la columna días decimales del archivo MAG.
  for elem in archivo_fruch:                                                                    # Recorro cada día decimal del Fruchtman.
    dt       = dias_decimales_a_datetime(np.array([elem]), int(año))[0]                         # Día decimal a objeto datetime del año.
    dia: str = dt.strftime('%d')                                                                # Extraigo en formato str de dos dígitos el día
    mes: str = dt.strftime('%m')                                                                # y el mes del archivo Fruchtman.
    DOY: str = fecha_UTC_a_DOY(dia, mes, año)                                                   # Calculo el DOY de la fecha actual.
    if DOY != DOY_actual:                                                                       # Si éste cambió => cargo nuevo archivo MAG.
      nombre   = (f'mvn_mag_l2_{año}{DOY}merge1s_{año}{mes}{dia}_v01_r01_recortado.sts')        # Construyo el nombre MAG correspondiente
      ruta_mag = os.path.join(directorio, 'datos_recortados_merge', año, str(int(mes)), nombre) # y su ruta completa.
      if not os.path.exists(ruta_mag):                                                          # Si el archivo no existe,
        nombre   = nombre.replace('_r01_', '_r02_')                                             # me fijo si se trataba de uno de versión 2
        ruta_mag = os.path.join(directorio,'datos_recortados_merge',año, str(int(mes)), nombre) # y construyo su ruta correspondiente.
        if not os.path.exists(ruta_mag):                                                        # Si no existe ninguno de los dos,
          print(f'No se encontró el archivo MAG del {dia}/{mes}/{año}.')                        # => no se encontró => devuelvo un aviso,
          data_mag       = None                                                                 # Como el archivo no existe => evito chequear
          dias_decimales = None                                                                 # bowshocks con ese mismo DOY.
          continue                                                                              # Finalmente, continuo el for (sig. iteración).
      data_mag       = np.loadtxt(ruta_mag)                                                     # Cargo el archivo MAG del día actual.
      dias_decimales = data_mag[:,0]                                                            # Extraigo la col días decimales del MAG.
      DOY_actual     = DOY                                                                      # Actualizo el DOY actual.
    j: int = hallar_índice_más_cercano(dias_decimales, elem)                                    # Busco el j del día decimal más cercano a MAG.
    filas.append(data_mag[j])                                                                   # REEMPLAZO EL DÍA FRUCHTMAN POR EL DIA MAG.
  archivo_merge: np.ndarray = np.array(filas)                                                   # Convierto la lista de filas a np.array.
  ruta_merge: str = os.path.join(directorio, f'fruchtman_{año}_merge.sts')                      # Construyo la ruta del archivo de salida.
  np.savetxt(ruta_merge, archivo_merge, fmt='%.6f')                                             # Guardo el archivo final en formato texto.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares # O(log n)
#———————————————————————————————————————————————————————————————————————————————————————
def hallar_índice_más_cercano(
    dias_MAG: np.ndarray,                                    # Array ordenado de valores numéricos (creciente)
    dia_fruch: float                                         # Valor para el cual se desea encontrar el elemento más cercano
) -> int:
  """
  La función hallar_índice_más_cercano recibe una lista 'dias_MAG' en formato np.ndarray (array de numpy) que corresponde a los días
  decimales del archivo MAG (que se encuentran en orden estrictamente creciente), y un float 'dia_fruch' que corresponde a un día decimal del
  archivo Fruchtman. Mediante el algoritmo de búsqueda binaria (search sort), encuentra en tiempo O(log n) el índice del array 'dias_MAG' en
  el que debería ir el 'día_fruch' para preservar el orden creciente. Finalmente chequea que la distancia con los elementos más cercanos (el
  de arriba y el de abajo) sea la mínima, y acomoda el índice si es necesario. Además, contempla casos borde j=0 ó j=len(dias_MAG)-1. Devuelve
  un entero que representa el índice del elemento del array 'dias_MAG' que es el más cercano a 'dia_fruch'.
  """
  j: int = np.searchsorted(dias_MAG, dia_fruch)              # j es el índice donde debe estar dia_fruch en dias_MAG para mantener el orden.
  if j == 0:                                                 # Si 'dia_fruch' es menor o igual que todos los elementos de dias_MAG,
    return 0                                                 # => el índice más cercano es el primer elemento (el cero).
  if j == len(dias_MAG):                                     # Si no, si 'dia_fruch' es mayor que todos los elementos de dias_MAG,
    return len(dias_MAG) - 1                                 # => el índice más cercano es el último.
  anterior: float   = dias_MAG[j-1]                          # Si no, en la var float 'anterior' me guardo el dia_fruch inmediatamente menor,
  posterior:  float = dias_MAG[j]                            # y en la variable float 'posterior' el dia_fruch inmediatamente mayor.
  if abs(posterior - dia_fruch) < abs(dia_fruch - anterior): # Comparo las distancias absolutas del dia_fruch con dichos valores,
    return j                                                 # y si el más cercano era el que venía después => el j es correcto.
  else:                                                      # En cambio, si el más cercano es el que estaba antes,
    return j - 1                                             # => debo devolver el j menos una posición.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————