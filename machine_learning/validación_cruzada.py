
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | 
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd

# Módulos Propios:
from base_de_datos.lectura             import leer_archivos_MAG
from machine_learning.clasificador_KNN import entrenar

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# : 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def ejecutar_validación_cruzada(
    directorio: str,                                                                      #
    años_entrenamiento: list[str],                                                        #
    K: int,                                                                               #
    variables: list[str],                                                                 #
    ventana: int,                                                                         #
    ventanas_NBS: list[int],                                                              #
    superposición_ventana: int,                                                           #
    promedio: int,                                                                        #
    tolerancia: int = 300                                                                 #
) -> pd.DataFrame:
  """
  Ejecuta validación cruzada leave-one-year-out para el clasificador KNN.
  Para cada año:
    - Entrena el KNN con todos los otros años_entrenamiento
    - Predice ventanas del año dejado fuera
    - Calcula la TPR (True Positive Rate)

  Devuelve un DataFrame con la TPR por año.
  """
  resultados = []                                                                         # 
  ruta_MAG: str = os.path.join(directorio, 'recorte_Vignes')                              # 
  for año in años_entrenamiento:                                                          # 
    print(f'\n===== Validación cruzada: año {año} =====')                                 # 
    lista_años = [a for a in años_entrenamiento if a != año]                              # 
    knn = entrenar(                                                                       # 
      directorio=directorio,                                                              # 
      años_entrenamiento=lista_años,                                                      # 
      K=K,                                                                                # 
      variables=variables,                                                                # 
      ventana=ventana,                                                                    # 
      ventanas_NBS=ventanas_NBS,                                                          # 
      superposición_ventana=superposición_ventana,                                        # 
      promedio=promedio                                                                   # 
    )                                                                                     # 
    pred, _, j_ventana = knn.predecir_ventana(data_MAG)                                   # 
    j_BS_pred: np.ndarray = j_ventana[pred == 1]                                          # 
    t_BS_pred: pd.DatetimeIndex = pd.to_datetime(data_MAG.iloc[:,0].to_numpy()[j_BS_pred])# 
    t0, tf         = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'                       # obtengo intervalo de tiempo de todo el año de MAG,
    archivo_F: str = f'hemisferio_N/fruchtman_{año}_merge_hemisferio_N.sts'               # obtengo el nombre de Fruchtman correspondiente,
    ruta_Fru: str  = os.path.join(directorio, 'fruchtman', archivo_F)                     # y obtengo la ruta completa del archivo Fruchtman.
    data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_MAG, t0, tf, promedio)                # Leo archivos MAG del año con el promedio indicado.
    data_Fru: pd.DataFrame = pd.read_csv(ruta_Fru, sep=' ', header=None)                  # Leo todo el archivo Fruchtman del año.
    dias_Fru: pd.Series    = data_Fru.iloc[:,0].astype(float)                             # Extraigo días decimales Fruchtman y paso a float.
    t_BS = pd.Timestamp(f'{año}-01-01') + pd.to_timedelta(dias_Fru-1, unit='D')           # Convierto t_BS a objetos datetime adecuadamente.
    TP = 0                                                                                # 
    for t_Fru in t_BS:                                                                    # 
      if np.any(np.abs((t_BS_pred - t_Fru).total_seconds()) <= tolerancia):               # 
        TP += 1                                                                           # 
    tasa_TP = TP / len(t_BS) if len(t_BS) > 0 else np.nan                                 # 
    print(f'TPR {año}: {tasa_TP:.3f}')                                                    # 
    resultados.append({'Año': año, 'TPR': tasa_TP, 'BS': len(t_BS), 'BS_detectados': TP}) # 
  return pd.DataFrame(resultados)                                                         # 




#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————