
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de Validación Cruzada (Cross-Validation) usando métrica TPR (true positives rate).
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd

from typing import Any

# Módulos Propios:
from base_de_datos.lectura             import leer_archivos_MAG, leer_archivo_Fruchtman
from machine_learning.clasificador_KNN import entrenar

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# ejecutar_validación_cruzada: función para realizar Cross-Validation sobre años_entrenamiento con métrica TPR (tasa_verdaderos_positivos). 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def ejecutar_validación_cruzada(
    directorio: str,                                                            # Carpeta donde se encuentran las mediciones Fruchtman y MAG.
    años_entrenamiento: list[str] = ['2014','2015','2016','2017','2018','2019'],# Años que se desean entrenar.
    K: int = 1,                                                                 # Cantidad de vecinos más cercanos a utilizar por el KNN.
    variables: list[str] = ['B','Xss','Yss','Zss'],                             # Variables a utilizar para el vector característico del KNN.
    promedio: int = 1,                                                          # Promedio para suavizar las muestras de MAVEN MAG.
    ventana: int = 300,                                                         # Ancho de ventana en segundos a utilizar (representa el BS).
    ventanas_NBS: list[int] = [-1,1,2],                                         # Posiciones de ventanas vecinas al BS para entrenar zona NBS.
    superposición_ventana: int = 50,                                            # Superposición entre ventanas (en %) para la predicción. 
    tolerancia: int = 300                                                       # Tolerancia en segundos entre el BS real y predicho por KNN.
) -> pd.DataFrame:

  """
  La función ejecutar_validación_cruzada realiza el algoritmo de Cross-Validation sobre un knn con todos los parametros que se han ingresado
  por parámetro: 'K', 'variables', 'promedio', 'ventana', 'ventanas_NBS', 'superposición_ventana'. En 'años_entrenamiento' debe recibir
  todos los años de BS previamente detectados (para la supervisión del modelo), en nuestro caso los años 2014-2019 de Fruchtman; y posee
  un parametro 'tolerancia' que representa el tiempo en segundos que se considera aceptable para la detección del BS por el KNN, respecto
  del t_BS real predicho por Fruchtman.
  La función entrena el KNN con los parámetros ingresados con todos los años de 'años_entrenamiento' excepto uno, y lo prueba para dicho año,
  calculando la tasa de verdaderos positivos (TPR), y luego repite el proceso para cada uno de los otros años. Devuelve un dataframe que
  contiene las tasas TP, la cantidad de BS que se poseían y la cantidad detectados para cada año.
  """
  res: list[dict[str, Any]] = []                                                          # Inicializo variable res (lista de dicc) a llenar.
  ruta_MAG: str = os.path.join(directorio, 'recorte_Vignes')                              # Obtengo ruta MAG de archivos con recorte Vignes.
  for año in años_entrenamiento:                                                          # Para cada año de todos los años de Fruchtman:
    print(f'\nValidación cruzada año {año}')                                              # Escribo un pequeño mensaje,
    knn = entrenar(                                                                       # En la variable 'knn' entreno el KNN,
      directorio            = directorio,                                                 # con todos los valores que han sido pasados por
      años_entrenamiento    = [x for x in años_entrenamiento if x != año],                # parámetro a la función ejecutar_validación....
      K                     = K,
      variables             = variables,
      promedio              = promedio,
      ventana               = ventana,
      ventanas_NBS          = ventanas_NBS,
      superposición_ventana = superposición_ventana
    )
    t0, tf                 = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'               # Obtengo intervalo de tiempo de todo el año de MAG.
    data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_MAG, t0, tf, promedio)                # Leo archivos MAG del año con el promedio indicado.
    data_Fru: pd.DataFrame = leer_archivo_Fruchtman(directorio, año)                      # Leo el archivo Fruchtman del año correspondiente.
    dias_Fru: pd.Series    = data_Fru.iloc[:,0].astype(float)                             # Extraigo días decimales Fruchtman y paso a float.
    pred, _, j_ventana          = knn.predecir_ventana(data_MAG)                          # Obtengo sólo etiquetas y j con predecir_ventana.
    j_BS_pred: np.ndarray       = j_ventana[pred == 1]                                    # Obtengo solo los índices de BS.
    t_BS_pred: pd.DatetimeIndex = pd.to_datetime(data_MAG.iloc[:,0].to_numpy()[j_BS_pred])# Obtengo los t_BS de los j_BS predichos.
    t_BS: pd.Series = pd.Timestamp(f'{año}-01-01') + pd.to_timedelta(dias_Fru-1, unit='D')# Convierto t_BS a objetos datetime adecuadamente.
    TP: int = 0                                                                           # Inicializo variable int TP (verdaderos positivos).
    for t_Fru in t_BS:                                                                    # Para cada t (día decimal) en los BS de Fruchtman, 
      if np.any(np.abs((t_BS_pred - t_Fru).total_seconds()) <= tolerancia):               # si hay t_BS_predicho contemplado en la tolerancia,
        TP += 1                                                                           # sumo a TP (una detección correcta!).
    tasa_TP: float = TP/len(t_BS) if len(t_BS) > 0 else np.nan                            # Calculo tasa_TP = TP_totales / cant_t_BS_Fru.
    res.append({                                                                          # En la variable res (diccionario),
      'Año': año,                                                                         # agrego el año de la validación cruzada,
      'BS_Fruchtman': len(t_BS),                                                          # la cantidad de BS que detectó Fruchtman,
      'BS_detectados': TP,                                                                # la cantidad de BS que detectó el KNN,
      'TPR': tasa_TP,                                                                     # y el resultado de la TPR.
      'K': K,                                                                             # Además, agrego todos los parámetros del KNN
      'Variables': variables,                                                             # utilizados.
      'Promedio': promedio,
      'Ventana': ventana,
      'Ventanas_NBS': ventanas_NBS,
      'Superposición_ventana': superposición_ventana,
      'Tolerancia': tolerancia                                                            # y la tolerancia que utilizó la validación_cruzada.
    })
  return pd.DataFrame(res)                                                                # Devuelvo res en formato dataframe.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————