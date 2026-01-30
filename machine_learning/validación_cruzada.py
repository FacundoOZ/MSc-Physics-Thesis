
# Modularizar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de Validación Cruzada (Cross-Validation) usando métrica TPR (true positives rate).
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd

from typing   import Any
from datetime import datetime

# Módulos Propios:
from base_de_datos.lectura             import leer_archivos_MAG, leer_archivo_Fruchtman
from machine_learning.clasificador_KNN import entrenar

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# ejecutar_validación_cruzada: función para realizar Cross-Validation sobre años_entrenamiento con métrica TPR (tasa_verdaderos_positivos). 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def ejecutar_validación_cruzada(
    directorio: str,                                                                      # Carpeta donde se encuentran las mediciones Fruchtman y MAG.
    años_entrenamiento: list[str] = ['2015','2016','2017','2018','2019'],                 # Años que se desean entrenar.
    K: int = 1,                                                                           # Cantidad de vecinos más cercanos a utilizar por el KNN.
    variables: list[str] = ['B','R','Bx','By','Bz','Xss','Yss','Zss'],                    # Variables a utilizar para el vector característico del KNN.
    promedio: int = 5,                                                                    # Promedio para suavizar las muestras de MAVEN MAG.
    ventana: int = 60,                                                                    # Ancho de ventana en segundos a utilizar (representa el BS).
    ventanas_NBS: list[int] = [2],                                                        # Posiciones de ventanas vecinas al BS para entrenar zona NBS.
    tolerancia: int = 120,                                                                # Tolerancia en segundos entre el BS real y predicho por KNN.
    post_procesamiento: bool = False,                                                     # Post-procesamiento de mediciones BS consecutivas (promedio).
    umbral: int = 30                                                                      # Umbral en minutos a utilizar por el post-procesamiento BS.
) -> None:
  """
  La función ejecutar_validación_cruzada realiza el algoritmo de Cross-Validation sobre un knn con todos los parametros que se han ingresado
  por parámetro: 'K', 'variables', 'promedio', 'ventana', 'ventanas_NBS', 'post_procesamiento' y 'umbral'. En 'años_entrenamiento' debe
  recibir todos los años de BS previamente detectados (para la supervisión del modelo), en nuestro caso los años 2014-2019 de Fruchtman;
  y posee un parametro 'tolerancia' que representa el tiempo en segundos que se considera aceptable para la detección del BS por el KNN,
  respecto del t_BS real predicho por Fruchtman. La función entrena el KNN con los parámetros ingresados con todos los años de
  'años_entrenamiento' excepto uno, y lo prueba para dicho año, calculando la tasa de verdaderos positivos (TPR), y luego repite el proceso
  para cada uno de los otros años. Devuelve un archivo en la carpeta destino 'directorio'+'KNN'+'validación_cruzada'+'CV_modelo_K{K}.txt' que
  contiene todos los parametros que se utilizó en el KNN, y las tasas TP, la cantidad de BS que se poseían y la cantidad detectados para cada
  año. Se realiza la lectura de archivos MAG previamente, para ahorrar mucho tiempo.
  """
  t_inicio: datetime = datetime.now()                                                     # Obtengo el t_inicial a la hora de la ejecución.
  print(f"Tiempo de inicio del algoritmo: {t_inicio.strftime('%H:%M:%S')}\n")             # del algoritmo, y lo enseño en un mensaje.
  ruta_MAG: str = os.path.join(directorio, 'recorte_Vignes')                              # Obtengo ruta MAG de archivos con recorte Vignes.
  lista:     list[dict[str, Any]]    = []                                                 # Inicializo variable lista (de dicc) a llenar.
  MAG_cache: dict[str, pd.DataFrame] = {}                                                 # Inicializo variable MAG_cache => leo 1 sola vez.
  print('Leyendo todos los archivos MAG:')                                                # Aviso que ésta es la lectura previa de archivos.
  for año in años_entrenamiento:                                                          # Para cada año de todos los años de Fruchtman:
    t0, tf         = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'                       # Obtengo intervalo de tiempo de todo el año de MAG.
    MAG_cache[año] = leer_archivos_MAG(ruta_MAG, t0, tf, promedio)                        # Leo archivos MAG 1 sola vez con el promedio dado.
  for año in años_entrenamiento:                                                          # Para todos los años de los años de Fruchtman:
    print(f'\nValidación cruzada año {año}')                                              # Escribo un pequeño mensaje,
    knn = entrenar(                                                                       # En la variable 'knn' entreno el KNN,
      directorio         = directorio,                                                    # con todos los valores que han sido pasados por
      años_entrenamiento = [x for x in años_entrenamiento if x != año],                   # parámetro a la función ejecutar_validación....
      K                  = K,
      variables          = variables,
      promedio           = promedio,
      ventana            = ventana,
      ventanas_NBS       = ventanas_NBS,
      MAG_cache          = MAG_cache
    )
    data_MAG: pd.DataFrame = MAG_cache[año]                                               # los guardé en el dicc MAG_cache => los obtengo.
    data_Fru: pd.DataFrame = leer_archivo_Fruchtman(directorio, año)                      # Leo el archivo Fruchtman del año correspondiente.
    dias_Fru: pd.Series    = data_Fru.iloc[:,0].astype(float)                             # Extraigo días decimales Fruchtman y paso a float.
    t0_año: pd.Timestamp   = pd.Timestamp(f'{año}-01-01')                                 # En t0_año, guardo 1/enero del año en formato str.
    t_BS: pd.Series        = t0_año + pd.to_timedelta(dias_Fru-1, unit='D')               # Convierto t_BS a objetos datetime adecuadamente.
    pred, prob, j_v  = knn.predecir_ventana(data_MAG)                                     # Obtengo sólo etiquetas y j con predecir_ventana.
    print(f"Initial: len(pred)={len(pred)}, BS count={sum(pred == 1)}")
    if post_procesamiento:                                                                # Si el booleano post_procesamiento=True, entonces
      pred, prob, j_v   = knn.post_procesar_BS(data_MAG, pred, prob, j_v, umbral)         # EJECUTO POSTPROCESAMIENTO VENTANAS_BS VECINAS.
      print(f"After post-proc: len(pred)={len(pred)}, BS count={sum(pred == 1)}")
    j_BS_pred: int = np.round(j_v[pred == 1]).astype(int)                                 # Obtengo solo los índices de BS,
    print(f"j_BS_pred length: {len(j_BS_pred)}")
    t_BS_pred: pd.DatetimeIndex = pd.to_datetime(data_MAG.iloc[:,0].to_numpy()[j_BS_pred])# Obtengo los t_BS de los j_BS predichos.
    TP: int = 0; FP: int = 0                                                              # Inicializo TP y FP (verdaderos/falsos positivos).
    if len(t_BS_pred) > 0:                                                                # Si hay tiempos BS predichos,
      diferencia         = np.abs((t_BS_pred.values[:,None] - t_BS.values[None,:])        # Calculo la diferencia entre el t_BS_predicho,
                                  .astype('timedelta64[s]').astype(int))                  # y el t_BS de Fruchtman y convierto a int.
      diferencia_inversa = np.abs((t_BS.values[:,None] - t_BS_pred.values[None,:])        # Calculo la diferencia entre los t_BS de Fruchtman,
                                  .astype('timedelta64[s]').astype(int))                  # y el t_BS_predicho (la inversa).
      TP = np.sum(np.any(diferencia <= tolerancia, axis=0))                               # TP es la suma de los encontrados en la tolerancia.
      FP = np.sum(~np.any(diferencia_inversa <= tolerancia, axis=0))                      # FP es la suma de los encontrados en la tolerancia.
    TPR: float = round(TP/len(t_BS), 3) if len(t_BS) > 0 else np.nan                      # Calculo TPR=TP_totales/cant_t_BS_Fru (3 dígitos).
    Pre: float = round(TP/(TP + FP), 3) if (TP + FP) > 0 else np.nan                      # Calculo la precisión del modelo con los FP.
    F1: float = round(2*Pre*TPR/(Pre + TPR), 3) if Pre > 0 and TPR > 0 else np.nan        # Calculo la métrica F1
    lista.append({                                                                        # En la variable lista (lista de diccionarios),
      'Año': año,                                                                         # agrego el año de la validación cruzada,
      'K': K,                                                                             # y todos los parámetros del KNN que se utilizaron.
      'Variables': variables,
      'Promedio': promedio,
      'Ventana': ventana,
      'Ventanas_NBS': ventanas_NBS,
      'Tolerancia': tolerancia,                                                           # Agrego la tolerancia que usó validación_cruzada,
      'BS_Fruchtman': len(t_BS),                                                          # la cantidad de BS originales de Fruchtman,
      'BS_detectados': len(t_BS_pred),                                                    # la cantidad de BS que detectó el KNN,
      'TP': TP,                                                                           # los verdaderos positivos,
      'FP': FP,                                                                           # los falsos positivos,
      'TPR': TPR,                                                                         # el resultado de la tasa de verdaderos positivos,
      'Precision': Pre,                                                                   # la precisión,
      'F1': F1                                                                            # y la métrica F1.
    })
  res: pd.DataFrame = pd.DataFrame(lista)                                                 # Convierto la lista completa a formato dataframe.
  ruta_validación: str = os.path.join(directorio, 'KNN', 'validación_cruzada')            # Obtengo la ruta de la carpeta destino,
  os.makedirs(ruta_validación, exist_ok=True)                                             # si no existe la creo,
  ruta_res: str        = os.path.join(ruta_validación, f'CV_modelo_K{K}.txt')             # Creo la ruta + nombre del archivo a guardar.
  res.to_csv(ruta_res, sep=' ', index=False)                                              # Convierto los resultados a .txt y los guardo.
  t_fin: datetime = datetime.now()                                                        # Obtengo el tiempo actual al final del proceso,
  print(f"\nTiempo del fin del algoritmo: {t_fin.strftime('%H:%M:%S')}\n")                # para imprimirlo,
  print(f'El algoritmo se ha ejecutado correctamente en un tiempo de {t_fin-t_inicio}.\n')# y devolver un aviso del fin del algoritmo.
  print(res)                                                                              # Muestro el resultado (el diccionario).

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————