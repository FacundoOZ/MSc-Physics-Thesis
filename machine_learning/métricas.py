
# Comentar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para calcular el rendimiento y la precisión de un modelo KNN respecto a datos Fruchtman supervisados.
#============================================================================================================================================

import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
from tqdm   import tqdm

# Módulos Propios:
from base_de_datos.conversiones import dias_decimales_a_datetime
from base_de_datos.lectura      import leer_métricas_KNN
from base_de_datos.unión        import hallar_índice_más_cercano
from plots.estilo_plots         import guardar_figura

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# calcular_métricas_KNN_con_Fruchtman: función para calcular las métricas Recall, Precision y F1 de un modelo KNN contra los BS de Fruchtman.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def calcular_métricas_KNN_con_Fruchtman(
    directorio: str,                                                                   # Carpeta donde se encuentran los archivos Fru y KNN.
    años: list[str],                                                                   # Lista de años cuyas métricas deseo calcular.
    modelo_KNN: str,                                                                   # Tipo de modelo KNN cuyas métricas deseo calcular.
    post_procesamiento: bool = False,                                                  # Tipo de post-procesado del modelo KNN.
    hemisferio_N: bool = True,                                                         # Hemisferios Fruchtman considerados.
    tolerancia: int = 300                                                              # Tolerancia entre datos Fruchtman y KNN (en segundos).
) -> None:
  """
  La función calcular_métricas_KNN_con_Fruchtman recibe en formato string un 'directorio', un 'modelo_KNN' y una lista de strings 'años', que
  representan la carpeta de origen donde se encuentran las subcarpetas correspondientes con los tiempos bow shock en día decimal detectados
  por Fruchtman (para los años 2014-2019), como por los detectados por el modelo KNN ingresado con el 'post_procesamiento' correspondiente,
  y calcula las métricas Recall ó TPR (tasa de verdaderos positivos), Precision y F1 para los años ingresados por parámetro. Si el booleano
  'hemisferio_N'=False, compara con la cantidad de bow shocks Fruchtman originales, pero si es True, solo los del recorte Zpc > 0.
  La función calcula las métricas en base al entero 'tolerancia', que representa la diferencia de tiempo en segundos que se considerará entre
  los BS detectados por Fruchtman y el modelo KNN, y devuelve un dataframe con los resultados en la carpeta de predicción correspondiente.
    """
  lista: list[dict[str,float]] = []                                                    # Inicio la variable final lista (lista[diccionarios]).
  for año in tqdm(años, desc='Calculando métricas del año', unit='año'):               # Para cada año entre los años seleccionados,
    ruta_base: str = os.path.join(directorio,'KNN','predicción', modelo_KNN)           # Obtengo la ruta base del KNN.
    if post_procesamiento:                                                             # Si hay post-procesamiento,
      archivo_KNN: str = f'tiempos_BS_{año}_promedio.txt'                              # construyo el nombre del archivo del año promediado,
      ruta_KNN: str = os.path.join(ruta_base,'post_procesamiento',archivo_KNN)         # y la ruta completa + nombre.
    else:                                                                              # Si no,
      ruta_KNN: str = os.path.join(ruta_base, f'tiempos_BS_{año}.txt')                 # construyo la ruta completa + nombre del año (original).
    días_KNN: np.ndarray = np.loadtxt(ruta_KNN)                                        # Cargo los datos KNN de dicho año en 'días_KNN'.
    if hemisferio_N:                                                                   # Luego, si hemisferio_N=True,
      archivo_Fru: str = f'fruchtman_{año}_merge_hemisferio_N.sts'                     # construyo el nombre del archivo fruchtman del año,
      ruta_Fru: str = os.path.join(directorio,'fruchtman','hemisferio_N', archivo_Fru) # y obtengo la ruta completa + nombre (hemisferio z>0).
    else:                                                                              # Si no,
      ruta_Fru: str = os.path.join(directorio,'fruchtman',f'fruchtman_{año}_merge.sts')# obtengo la ruta completa + nombre original.
    días_Fru: np.ndarray = np.loadtxt(ruta_Fru, usecols=0)                             # Cargo los datos Fruchtman de dicho año en 'días_Fru'.
    t_KNN: pd.DatetimeIndex = dias_decimales_a_datetime(días_KNN, int(año))            # Convierto a objetos datetime los tiempos KNN,
    t_Fru: pd.DatetimeIndex = dias_decimales_a_datetime(días_Fru, int(año))            # y los tiempos Fruchtman.
    TP: int = 0; FN: int = 0; FP: int = 0                                              # Inicio true positives, y false positives/negatives.
    for día_F, t_f in zip(días_Fru, t_Fru):                                            # Para cada día/tiempo de Fruchtman, busco el j del KNN
      j: int = hallar_índice_más_cercano(días_KNN, día_F)                              # donde debe ir día_F para preservar el orden creciente.
      if abs((t_KNN[j] - t_f).total_seconds()) <= tolerancia:                          # Si la diferencia absoluta entre tiempos <= tolerancia,
        TP += 1                                                                        # entonces el KNN predijo el BS detectado por Fruchtman.
      else:                                                                            # Si no,
        FN += 1                                                                        # el KNN se perdió un BS real, que Fruchtman detecto.
    for día_K, t_k in zip(días_KNN, t_KNN):                                            # Para cada día/tiempo de KNN, busco el j de Fruchtman
      j: int = hallar_índice_más_cercano(días_Fru, día_K)                              # donde debe ir día_K para preservar el orden creciente.
      if abs((t_Fru[j] - t_k).total_seconds()) > tolerancia:                           # Si la diferencia es mayor a la tolerancia,
        FP += 1                                                                        # el KNN predijo un BS que en principio, Fruchtman no.
    lista.append({                                                                     # Agrego a la lista todos los valores del dicc:
      'Año': int(año),                                                                 # -El año correspondiente,
      'BS_Fru': len(días_Fru),                                                         # -La cantidad total de BS detectados por Fruchtman,
      'BS_KNN': len(días_KNN),                                                         # -La cantidad total de BS detectados por el KNN,
      'TP': TP,                                                                        # -Los verdaderos positivos del KNN,
      'FP': FP,                                                                        # -Los falsos positivos del KNN,
      'FN': FN,                                                                        # -Los falsos negativos del KNN,
      'Recall': métrica_TPR(TP, FN),                                                   # -El resultado de la métrica Recall (TPR),
      'Precision': métrica_PPV(TP, FP),                                                # -El resultado de la métrica Precision,
      'F1': métrica_F1(TP, FP, FN)                                                     # -El resultado de la métrica F1.
    })                                                                                 # Repito todo el proceso para cada año.
  res: pd.DataFrame = pd.DataFrame(lista)                                              # Convierto la lista a formato dataframe en res,
  print(res)                                                                           # la enseño en un print,
  if post_procesamiento:                                                               # y si hubo post-procesamiento,
    nombre_final: str = f'métricas_promedio_{tolerancia}.txt'                          # creo el nombre final con la tolerancia incluída,
    ruta_final: str = os.path.join(ruta_base, 'post_procesamiento', nombre_final)      # creo la ruta_final + nombre post procesado,
  else:                                                                                # y si no,
    ruta_final: str = os.path.join(ruta_base, f'métricas_{tolerancia}.txt')            # la ruta_final + nombre con tolerancia,
  res.to_csv(ruta_final, sep=' ', index=False)                                         # y creo un archivo .txt en la ruta final correspondiente.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def métrica_TPR(TP: int, FN: int) -> float:
  """
  Cálculo de la métrica Recall ó TPR (tasa de verdaderos positivos), que se define como x = TP/(TP + FN).
  """
  if (TP + FN) == 0:            # Si el divisor es nulo,
    return 0.0                  # no devuelvo nada.
  return round(TP/(TP + FN), 3) # Si no, calculo la métrica con los verdaderos positivos y los falsos negativos.

#———————————————————————————————————————————————————————————————————————————————————————
def métrica_PPV(TP: int, FP: int) -> float:
  """
  Cálculo de la métrica Precision, que se define como x = TP/(TP + FP).
  """
  if (TP + FP) == 0:            # Si el divisor es nulo,
    return 0.0                  # no devuelvo nada.
  return round(TP/(TP + FP), 3) # Si no, calculo la métrica con los verdaderos y falsos positivos.

#———————————————————————————————————————————————————————————————————————————————————————
def métrica_F1(TP: int, FP: int, FN: int) -> float:
  """
  Cálculo de la métrica F1, que se define como x = 2*Precision*Recall/(Precision + Recall).
  """
  TPR: float = métrica_TPR(TP, FN)       # Calculo la métrica TPR.
  PPV: float = métrica_PPV(TP, FP)       # Calculo la métrica PPV.
  if PPV + TPR == 0:                     # Si el divisor es nulo,
    return 0.0                           # no devuelvo nada.
  return round(2*PPV*TPR/(PPV + TPR), 3) # Si no, calculo la métrica F1 con las métricas TPR y PPV.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_parámetros_KNN: 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
promedios: list[int]    = [1,2,3,4,5,6,7,8,9,10]
ventana: list[int]      = [20,40,60,80,100,120]
ventanas_NBS: list[str] = ['[-4]','[-3]','[-2]','[-1]','[1]','[2]','[3]','[4]','[-2,2]','[-4,-3,3,4]','[-3,-2,-1,2]','[-3,-2,2,3]','[-4,-3,-2,2]']
K: list[int]            = [1,2,3,4,5,6,7,8,9,10,11,12]
tolerancia: list[int]   = [20,40,60,80,100,120,140,160,180,200,240,300,360,420,480,540,600]

def graficador_parámetros_KNN(
    directorio: str,                                                                      #
    parámetro: str,                                                                       #
    post_procesamiento: bool = True,                                                      #
    métricas: list[str] = ['TPR','PPV','F1'],                                             #
    errores: bool = True,                                                                 #
    guardar: bool = False                                                                 #
) -> None:
  """
  Docstring
  """
  p.figure()
  if parámetro=='promedio':                                                               # CASO 1: PARÁMETRO='promedio'
    res:     dict[str, list[float]] = {m: [] for m in métricas}                           #
    res_std: dict[str, list[float]] = {m: [] for m in métricas}                           #
    for j in promedios:                                                                   # del 1 al 10 inclusives
      modelo: str = f'Eclipse_promedio{j}'                                                #
      data_p: pd.DataFrame = leer_métricas_KNN(directorio, modelo, post_procesamiento)    #
      for m in métricas:                                                                  #
        media, std = calcular_métrica_global(data_p, métrica=m)                           #
        res[m].append(media)                                                              #
        res_std[m].append(std)                                                            #
    for m in métricas:                                                                    #
      if errores:                                                                         #
        p.errorbar(promedios, res[m], yerr=res_std[m],                                    #
                   marker='o', capsize=4, label=f'{m} global')                            #
      else:                                                                               #
        p.plot(promedios, res[m], marker='o', label=f'{m} global')                        #
    p.xlabel('Promedio [s]')                                                              #
  elif parámetro=='ventanas_NBS':                                                         # CASO 2: PARÁMETRO='ventanas_NBS'
    for j in ventana:                                                                     #
      res:     dict[str, list[float]] = {m: [] for m in métricas}                         #
      res_std: dict[str, list[float]] = {m: [] for m in métricas}                         #
      pos_ventanas_NBS = []                                                               #
      for k, pos in enumerate(ventanas_NBS):                                              #
        modelo: str = f'Eclipse_ventana{j}_NBS{k}'                                        #
        try:                                                                              #
          data_v: pd.DataFrame = leer_métricas_KNN(directorio, modelo, post_procesamiento)#
        except FileNotFoundError:                                                         #
          continue                                                                        #
        pos_ventanas_NBS.append(pos)                                                      #
        for m in métricas:                                                                #
          media, std = calcular_métrica_global(data_v, métrica=m)                         #
          res[m].append(media)                                                            #
          res_std[m].append(std)                                                          #
      for m in métricas:                                                                  #
        if errores:                                                                       #
          p.errorbar(pos_ventanas_NBS, res[m], yerr=res_std[m],                           #
                     marker='o', capsize=4, label=f'{m} global ventana={j} s')            #
        else:                                                                             #
          p.plot(pos_ventanas_NBS, res[m], marker='o', label=f'{m} global ventana={j} s') #
    p.xlabel('Posición de ventanas_NBS')                                                  #
  elif parámetro=='K':                                                                    # CASO 3: PARÁMETRO='K'
    res:     dict[str, list[float]] = {m: [] for m in métricas}                           #
    res_std: dict[str, list[float]] = {m: [] for m in métricas}                           #
    for k in K:                                                                           #
      modelo: str = f'Eclipse_k{k}'                                                       #
      data_k: pd.DataFrame = leer_métricas_KNN(directorio, modelo, post_procesamiento)    #
      for m in métricas:                                                                  #
        media, std = calcular_métrica_global(data_k, métrica=m)                           #
        res[m].append(media)                                                              #
        res_std[m].append(std)                                                            #
    for m in métricas:                                                                    #
      if errores:                                                                         #
        p.errorbar(K, res[m], yerr=res_std[m], marker='o', capsize=4, label=f'{m} global')#
      else:                                                                               #
        p.plot(K, res[m], marker='o', label=f'{m} global')                                #
    p.xlabel(r'$k$ (número de vecinos)')                                                  #
  elif parámetro=='tolerancia':                                                           #
    res:     dict[str, list[float]] = {m: [] for m in métricas}                           #
    res_std: dict[str, list[float]] = {m: [] for m in métricas}                           #
    for j in tolerancia:                                                                  #
      modelo: str = 'Eclipse_k12'                                                         #
      data_t: pd.DataFrame = leer_métricas_KNN(directorio, modelo, post_procesamiento, j) #
      for m in métricas:                                                                  #
        media, std = calcular_métrica_global(data_t, métrica=m)                           #
        res[m].append(media)                                                              #
        res_std[m].append(std)                                                            #
    for m in métricas:                                                                    #
      if errores:                                                                         #
        p.errorbar(tolerancia, res[m], yerr=res_std[m],                                   #
                   marker='o', capsize=4, label=f'{m} global')                            #
      else:                                                                               #
        p.plot(tolerancia, res[m], marker='o', label=f'{m} global')                       #
    p.xlabel('Tolerancia [s]')                                                            #
  p.ylabel('Métricas globales')                                                           #
  p.title('Promedio de métricas de CV (2014-2019) respecto del metaparámetro')            #
  p.grid(which='major', alpha=.2,  linestyle='-')                                         #
  p.grid(which='minor', alpha=.15, linestyle=':')                                         #
  p.legend()                                                                              #
  if guardar:                                                                             #
    guardar_figura()                                                                      #
  p.show()                                                                                #

#———————————————————————————————————————————————————————————————————————————————————————
# Función Auxiliar
#———————————————————————————————————————————————————————————————————————————————————————
def calcular_métrica_global(
    archivo_métricas: pd.DataFrame,                           #
    métrica: str                                              #
) -> tuple[float, float]:
  """
  Docstring
  """
  if métrica=='TPR':                                          #
    res: list[float] = archivo_métricas['Recall']             #[1:]
  elif métrica=='PPV':                                        #
    res: list[float] = archivo_métricas['Precision']          #[1:]
  elif métrica=='F1':                                         #
    res: list[float] = archivo_métricas['F1']                 #[1:]
  else:                                                       #
    raise ValueError(f'No se encuentra la métrica: {métrica}')#
  media = np.mean(res)                                        #
  std = np.std(res, ddof=1)                                   #
  return (media, std)                                         #

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————