
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para leer los datos de MAVEN MAG y los archivos de Fruchtman unidos por unión.py
#============================================================================================================================================

import os                                # Para archivos
import numpy  as np
import pandas as pd
from tqdm     import tqdm                # Para mostrar barras de progreso de descarga
from datetime import datetime, timedelta # Contiene correctamente los días de cada año

# Módulos Propios:
from base_de_datos.conversiones import fecha_UTC_a_DOY, dias_decimales_a_datetime
from base_de_datos.unión        import hallar_índice_más_cercano

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# leer_archivos_MAG: lee y concatena los archivos MAG unidos desde un tiempo inicial a un tiempo final.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def leer_archivos_MAG(
    directorio: str,                                                             # Carpeta de la base de datos de los archivos a leer
    tiempo_inicial: str, tiempo_final: str,                                      # tiempo inicial y final en formato str 'DD/MM/YYYY-HH:MM:SS'
    promedio: int = 1                                                            # Promedio en segundos de la cantidad de muestras a leer.
) -> pd.DataFrame:
  """
  Lee y concatena ordenadamente todos los archivos .sts del año y en el directorio pasados por parámetro (tanto los de terminación 'r01' como
  'r02') que se encuentren entre el intervalo [t0, tf] (inclusive) determinado por las variables tiempo_inicial y final, en formato string
  'DD/MM/YYYY-HH:MM:SS'. Mediante el entero positivo 'promedio' se realiza un promedio de la cantidad de mediciones pasadas por parámetro
  (por ejemplo: si promedio = 20, se leen 20 mediciones y se las promedia por un solo valor, luego se leen otras 20 y se las promedia, etc.),
  y como las muestras son de 1 Hz, el promedio representa la cantidad de segundos de mediciones a promediar.
  """
  t0 = pd.to_datetime(tiempo_inicial, format='%d/%m/%Y-%H:%M:%S')                # Convierto strings tiempo_inicial/final a objetos datetime
  tf = pd.to_datetime(tiempo_final,   format='%d/%m/%Y-%H:%M:%S')                # (le indico a pandas como extraer DD/MM/AA, HH:MM:SS).
  if tf < t0:                                                                    # Si el tiempo_incial es posterior al inicial,
    raise ValueError('El str tiempo_final debe ser posterior a tiempo_inicial')  # arrojo un error de valores.
  cant_dias: int = (tf - t0).days + 1                                            # Cantidad de días a recorrer (inclusive).
  lista_sts: list[pd.DataFrame] = []                                             # Listas donde se acumularán los dataframes leídos.
  with tqdm(total=cant_dias, desc='Leyendo archivos MAG', unit='día') as pbar:   # Con tqdm, leo la cantidad de días con 1 día por unidad.
    for año in range(t0.year, tf.year+1):                                        # Para el iterador año entre año_0 y año_f (+1 por el range)
      inicio = max(t0, datetime(año, 1, 1))                                      # Calculo la fecha exacta de inicio usando max,
      fin    = min(tf, datetime(año, 12, 31, 23, 59, 59))                        # y min, mediante tiempos datetime
      j = inicio                                                                 # Creo otro iterador j que representará la fecha.
      while j <= fin:                                                            # El j irá desde fecha_inicial a fecha_final.
        dia, mes = j.strftime('%d'), j.strftime('%m')                            # Extraigo día y mes del iterador (strings) con 2 dígitos.
        DOY: str = fecha_UTC_a_DOY(dia, mes, str(año))                           # Calculo el Day Of Year (DOY) de la fecha actual.
        ruta_base: str = os.path.join(directorio, str(año), str(int(mes)))       # Ruta base donde deberían estar los archivos de ese día.
        nombres: list[str] = [                                                   # Creo una lista de dos strings que contiene los posibles
          f'mvn_mag_l2_{año}{DOY}merge1s_{año}{mes}{dia}_v01_r01_recortado.sts', # nombres que puede tener el archivo correspondiente a ese
          f'mvn_mag_l2_{año}{DOY}merge1s_{año}{mes}{dia}_v01_r02_recortado.sts'] # dia. Con terminación 'r01' ó 'r02' (r=revisión).
        if 'hemisferio_N' in directorio:                                         # Si se desea graficar el hemisferio norte,
          nombres = [x.replace('.sts', '_hemisferio_N.sts') for x in nombres]    # reemplazo el nombre por la terminación correpondiente.
        elif 'recorte_Vignes' in directorio:                                     # Si se desea graficar el recorte de Vignes,
          nombres = [x.replace('_recortado.sts', '_final.sts') for x in nombres] # reemplazo '_recortado' por '_final'.
        elif 'hemisferio_ND' in directorio:                                      # Y si se desea graficar solo el hemisferio norte diurno,
          nombres = [x.replace('.sts', '_hemisferio_ND.sts') for x in nombres]   # también.
        encontrado: bool = False                                                 # Creo una variable que representa el estado del archivo
        for archivo in nombres:                                                  # Para cada archivo de la lista 'nombres',
          ruta_archivo: str = os.path.join(ruta_base, archivo)                   # guardo su ubicación en la variable ruta_archivo.
          if os.path.exists(ruta_archivo):                                       # Si el archivo existe, hago lo siguiente:
            if os.path.getsize(ruta_archivo) == 0:                               # Si no contiene nada, (puede pasar debido a algún recorte)
              continue                                                           # paso a la siguiente iteración del for.
            df = pd.read_csv(ruta_archivo, sep=' ', header=None)                 # lo leo completamente,
            if promedio > 1:                                                     # Si el promedio es mayor a 1,
              df = df.groupby(df.index // promedio).mean(numeric_only=True)      # => tomo la media cada 'promedio' cantidad de muestras,
            df[0] = dias_decimales_a_datetime(df[0].to_numpy(), año)             # convierto la col 0 a datetime CON EL AÑO CORRESPONDIENTE,
            lista_sts.append(df)                                                 # EN lista_sts CREO UN EJE t ABSOLUTO con todos los años
            encontrado = True                                                    # y actualizo la variable encontrado (se encontró)
            break                                                                # Ya no itero más el for.
        #if not encontrado:                                                      # Si no se encontró ni el 'r01' ni el 'r02', entonces
        #  print(f'No se encontraron archivos del {dia}/{mes}/{año}')            # aviso la fecha del archivo que no se encontró.
        j += timedelta(days=1)                                                   # Avanzo al iterador al día siguiente,
        pbar.update(1)                                                           # y actualizo la barra de progreso.
  if not lista_sts:                                                              # Si no se encontró absolutamente ningún archivo, 
    raise FileNotFoundError('No se encontraron archivos en el rango dado.')      # devuelvo un mensaje.
  datos: pd.DataFrame = pd.concat(lista_sts, ignore_index=True)                  # Concateno todos los dataframes de lista_sts en uno solo.
  datos = datos[(datos[0] >= t0) & (datos[0] <= tf)]                             # Recorto los datos exactos del intervalo (t_i,t_f) ingresado,
  return datos.reset_index(drop=True)                                            # y devuelvo el dataframe final con índices limpios.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# leer_archivo_Fruchtman: lee el archivo Fruchtman de un año correspondiente.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def leer_archivo_Fruchtman(
    directorio: str,                                                          # Directorio donde se encuentra la carpeta 'merge'.
    año: str,                                                                 # Año del archivo a leer.
    hemisferio_N: bool = True                                                 # Booleano que permite leer datos completos, ó el recorte Zpc>0
) -> pd.DataFrame:
  """
  La función leer_archivo_Fruchtman recibe en formato string un directorio y un año, que representan la carpeta donde se encuentra el archivo
  a leer de tipo 'fruchtman_{año}_merge_hemisferio_N.sts', ubicado dentro de la carpeta 'merge' correspondiente, y el año cuyo archivo se
  desea leer, respectivamente. Si el booleano 'hemisferio_N' = True, lee los datos tras efectuar el recorte Zpc > 0, sino, lee los datos
  originales. Devuelve un pd.DataFrame con los datos cargados.
  """
  if hemisferio_N:                                                            # Si hemisferio_N=True,
    nombre: str = f'fruchtman_{año}_merge_hemisferio_N.sts'                   # reconstruyo el nombre del archivo del año correspondiente,
    ruta: str = os.path.join(directorio, 'fruchtman', 'hemisferio_N', nombre) # y la ruta completa + nombre.
  else:                                                                       # Si hemisferio_N=False,
    nombre: str = f'fruchtman_{año}_merge.sts'                                # reconstruyo el nombre del archivo original (merge),
    ruta: str = os.path.join(directorio, 'fruchtman', nombre)                 # y su ruta completa + nombre.
  return pd.DataFrame(np.loadtxt(ruta))                                       # Devuelvo los datos cargados en formato dataframe.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# leer_bow_shocks_KNN: busca las componentes de campo magnético y posición en los archivos MAG asociadas a los t_BS predichos por KNN.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def leer_bow_shocks_KNN(
    directorio: str,                                                                               # Directorio donde están los archivos.
    modelo: str,                                                                                   # Modelo KNN cuyos bow shocks quiero leer.
    post_procesamiento: bool,                                                                      # Booleano para leer post-procesados.
    año: str                                                                                       # Año cuyos datos del bow shocks se buscan.
) -> pd.DataFrame:
    """
    La función leer_bow_shocks_KNN recibe en formato str un 'directorio' y un 'año', que representan la ruta donde se encuentra el archivo
    'tiempos_BS_{año}.txt' que contiene los tiempos en día decimal predichos para los bow shocks del año correspondiente por el KNN (en las
    subcarpetas 'KNN', 'predicción'), y donde también se encuentran todos los archivos MAG que se buscarán en la carpeta 'recorte_Vignes' del
    año correspondiente, y que se utilizarán para obtener los valores de campo magnético y posiciones de la sonda en el tiempo BS del archivo
    MAG más cercano al tiempo determinado por el KNN. Para encontrar el índice más cercano a cada tiempo BS en los archivos MAG
    correspondientes, se utiliza la función hallar_índice_más_cercano que lo busca en tiempo O(log n). El string 'modelo' y el booleano
    'post_procesamiento' representan el tipo de modelo KNN a utilizar, y si desean leerse las mediciones originales, o post-procesadas.
    La función devuelve un dataframe de igual longitud que el archivo '.txt' original, pero con el contenido de las posiciones y campos.
    """
    if post_procesamiento:                                                                         # Si post_procesamiento=True,
      archivo_BS: str = f'tiempos_BS_{año}_promedio.txt'                                           # obtengo el nombre del archivo,
      ruta_BS: str = os.path.join(directorio,'KNN','predicción',modelo,'post_procesamiento',archivo_BS)# y su ruta completa + nombre.
    else:                                                                                          # Si no,
      ruta_BS: str = os.path.join(directorio,'KNN','predicción',modelo, f'tiempos_BS_{año}.txt')   # Obtengo la ruta original.
    t_BS: np.ndarray            = np.loadtxt(ruta_BS)                                              # Obtengo el contenido t_BS (día decimal).
    fechas_BS: pd.DatetimeIndex = dias_decimales_a_datetime(t_BS, int(año))                        # Convierto días a fecha datetime.
    t0: str = fechas_BS.min().strftime('%d/%m/%Y-%H:%M:%S')                                        # Obtengo el tiempo inicial,
    tf: str = fechas_BS.max().strftime('%d/%m/%Y-%H:%M:%S')                                        # y el tiempo final de los bow shocks.
    data_MAG: pd.DataFrame  = leer_archivos_MAG(os.path.join(directorio,'recorte_Vignes'), t0, tf) # Obtengo data_MAG en el intervalo (t0,tf),
    tiempos_MAG: np.ndarray = pd.to_datetime(data_MAG[0].to_numpy())                               # y los tiempos en día decimal (columna 0).
    filas: list[np.ndarray] = []                                                                   # Creo lista donde irán las nuevas filas. 
    for t in fechas_BS:                                                                            # Para cada elem de las fechas,
      j: int = hallar_índice_más_cercano(tiempos_MAG, t)                                           # busco el j más cercano en MAG (O(log n)),
      filas.append(data_MAG.iloc[j].to_numpy())                                                    # y appendeo toda la fila MAG de ese j.
    return pd.DataFrame(filas)                                                                     # Devuelvo un dataframe de longitud t_BS.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————