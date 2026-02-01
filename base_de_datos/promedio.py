
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para promediar los tiempos_BS consecutivos detectados (en día decimal) por un modelo de clasificador_KNN.py
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd

# Módulos Propios:
from base_de_datos.conversiones import dias_decimales_a_datetime

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# promediar_archivo_temporal_KNN: función para realizar un promedio en segundos entre tiempos_BS cercanos de un único archivo.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def promediar_archivo_temporal_KNN(
    directorio: str,                                                                # Carpeta donde se encuentra el archivo a recortar.
    año: str,                                                                       # Año de los tiempos bow shock correspondientes.
    modelo: str,                                                                    # Carpeta del modelo KNN donde están las predicciones.
    promedio: int = 40                                                              # Umbral en segundos para promediar t_BS consecutivos.
) -> None:
  """
  La función promediar_archivo_temporal_KNN recibe en formato string un 'directorio', un 'año' y un 'modelo' que representan la carpeta donde
  se encuentran las subcarpetas correspondientes del modelo KNN elegido, y el archivo 'tiempos_BS_{año}.txt' que contiene los tiempos de bow
  shock detectados por el algoritmo KNN en formato día decimal, y un entero positivo 'promedio' que representa el intervalo de tiempo máximo
  en segundos que desea considerarse, para el cual todos los tiempos cuya diferencia con el siguiente sea menor a 'promedio', serán
  promediados (incluyendo varios consecutivos).
  La función devuelve un archivo promediado en la ubicación directorio + 'modelo' + 'post_procesamiento' + 'tiempos_BS_{año}_promedio.txt'.
  """
  ruta_base: str = os.path.join(directorio,'KNN','predicción')                      # Obtengo ruta base donde se encontrarán los archivos.
  archivo_KNN: str = f'tiempos_BS_{año}.txt'                                        # Construyo el nombre del archivo del año a promediar.
  ruta_KNN: str = os.path.join(ruta_base, modelo, archivo_KNN)                      # Obtengo ruta_completa + modelo + nombre_archivo.
  if not os.path.exists(ruta_KNN):                                                  # Si el archivo no existe, no puedo promediar nada,
    raise FileNotFoundError(f"No se encontró el archivo {ruta_KNN}")                # => devuelvo un mensaje de error.
  ruta_f: str = os.path.join(ruta_base, modelo, 'post_procesamiento')               # Construyo la ruta final del archivo.
  os.makedirs(ruta_f, exist_ok=True)                                                # Si la carpeta no existe, la creo.
  archivo_f: str = os.path.join(ruta_f, archivo_KNN.replace('.txt','_promedio.txt'))# Creo la ruta completa + nombre de archivo final.
  if os.path.exists(archivo_f):                                                     # Si el archivo existe, ya fue promediado,
    print(f"El archivo '{os.path.basename(archivo_f)}' ya ha sido promediado.")     # => devuevlo un mensaje print,
    return                                                                          # y salgo de la función.
  contenido: np.ndarray = np.loadtxt(ruta_KNN, skiprows=1)                          # En 'contenido' leo todo el archivo (omito primer fila).
  if contenido.size == 0:                                                           # Si el archivo está vacío,
    raise ValueError("El archivo de entrada está vacío.")                           # devuelvo un mensaje de error.
  inicio_año: pd.Timestamp  = pd.Timestamp(int(año), 1, 1)                          # Obtengo el 1 de enero del año que corresponda,
  días_BS: pd.DatetimeIndex = dias_decimales_a_datetime(contenido, int(año))        # Obtengo días_BS en formato objeto datetime del año.
  res: list[float] = []                                                             # Inicializo lista floats vacía donde irá el resultado.
  j: int = 0                                                                        # Inicializo un entero iterador j.
  N: int = len(días_BS)                                                             # Obtengo la longitud del archivo (para no recomputar).
  while j < N:                                                                      # Mientras j se encuentre en rango del archivo,
    grupo: list[pd.Timestamp] = [días_BS[j]]                                        # Inicializo lista 'grupo' con el día j-ésimo: [t_j]
    t_ref: pd.Timestamp       = días_BS[j]                                          # El día decimal de referencia actual será el j-ésimo.
    j_sig: int = j + 1                                                              # El entero j_siguiente será el j-ésimo + 1.
    while j_sig < N and abs((días_BS[j_sig]-t_ref).total_seconds()) < promedio:     # Mientras j_sig esté en rango y t_{j_sig}-t_j < prom,
      grupo.append(días_BS[j_sig])                                                  # agrego j_sig al grupo,
      promedio_grupo: float = np.mean([t.value for t in grupo])                     # y calculo el promedio del grupo (de j y de j+1).
      t_ref  = pd.to_datetime(int(promedio_grupo))                                  # actualizo t_referencia con el promedio de j y j+1.
      j_sig += 1                                                                    # Avanzo al siguiente de j_sig (j+2) y repito el while.
    día_BS_promediado: float = (t_ref - inicio_año).total_seconds()/86400 + 1       # Si ya no puedo promediar, obtengo el día_promediado,
    res.append(día_BS_promediado)                                                   # y lo agrego a la lista res en formato float.
    j = j_sig                                                                       # Actualizo j, al último j al que haya llegado en j_sig.
  np.savetxt(archivo_f, np.array(res), fmt='%.10f')                                 # Cuando llego al fin del archivo, lo guardo
  print(f"El archivo '{archivo_f}' se ha promediado correctamente.")                # y devuelvo un mensaje.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————