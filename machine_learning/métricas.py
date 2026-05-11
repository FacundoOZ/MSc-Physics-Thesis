
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
# El barrido de parámetros se realizó en intervalos coherentes. Estos fueron los siguientes:
promedios: list[int]    = [1,2,3,4,5,6,7,8,9,10]
ventana: list[int]      = [20,40,60,80,100,120]
ventanas_NBS: list[str] = ['[-4]','[-3]','[-2]','[-1]','[1]','[2]','[3]','[4]','[-2,2]','[-4,-3,3,4]','[-3,-2,-1,2]','[-3,-2,2,3]','[-4,-3,-2,2]']
K: list[int]            = [1,2,3,4,5,6,7,8,9,10,11,12]
tolerancia: list[int]   = [20,40,60,80,100,120,140,160,180,200,240,300,360,420,480,540,600]

def graficador_parámetros_KNN(
  directorio: str,                                                                              # Directorio donde se encuentran los archivos.
  parámetro: str,                                                                               # Metaparámetro que deseo graficar.
  post_procesamiento: bool = True,                                                              # Booleano para graficar mediciones procesadas.
  métricas: list[str] = ['TPR','PPV','F1'],                                                     # Métricas que se desean graficar.
  errores: bool = True,                                                                         # Booleano para graficar con/sin barras de error.
  guardar: bool = False                                                                         # Booleano para guardar la figura en formato .pdf.
) -> None:
  """
  La función graficador_parámetros_KNN permite graficar todas las optimizaciones de los metaparámetros 'promedio', 'ventana' y 'ventanas_NBS'
  simultáneamente, 'K' y 'tolerancia'. Para ello, recibe los strings 'directorio', donde se encuentran todos los archivos y accede a las
  subcarpetas correspondientes, y 'parámetro', que determina el metaparámetro cuyas métricas se desea graficar. El parámetro booleano
  'post_procesamiento' determina si los archivos de métricas que se leerán serán los post-procesados o no, y la lista de strings 'métricas'
  determina qué métricas globales se graficarán: 'TPR', 'PPV' y/o 'F1'. Si los booleanos 'errores' o 'guardar' son verdaderos grafica con o
  sin barras de error, respectivamente, y permite guardar la figura en formato .pdf de alta calidad. La función no devuelve nada.
  """
  p.figure()                                                                                    # Creo la figura.
  if parámetro == 'promedio':                                                                   # Si quiero graficar el metaparámetro promedio,
    res, res_std, x = obtener_resultados(                                                       # obtengo todos los resultados de la lista del
      promedios, métricas,                                                                      # barrido ('promedios') de la 'métricas' corresp.
      lambda j: leer_métricas_KNN(directorio, f'Eclipse_promedio{j}', post_procesamiento))      # y leo las métricas globales.
    graficar_métricas(x, res, res_std, métricas, errores)                                       # Grafico las métricas con o sin errores,
    configurar_gráfico('Promedio [s]')                                                          # y coloco el promedio en segundos como etiqueta en x.
  elif parámetro == 'ventanas_NBS':                                                             # Si no, si el parámetro es ventanas_NBS,
    for j in ventana:                                                                           # debo hacer el plot múltiple barriendo cada ventana.
      res, res_std, x = obtener_resultados(                                                     # Para cada ventana, obtengo los resultados,
        list(range(len(ventanas_NBS))), métricas,                                               # de las métricas deseadas,
        lambda k: leer_métricas_KNN(directorio,f'Eclipse_ventana{j}_NBS{k}',post_procesamiento))# con o sin post-procesado, etc..
      posiciones = [ventanas_NBS[k] for k in x]                                                 # Obtengo las posiciones del eje x para plotear.
      graficar_métricas(posiciones, res, res_std, métricas, errores, f'ventana={j} s')          # Grafico todos los resultados (uso label_extra),
    configurar_gráfico('Posición de ventanas_NBS')                                              # y coloco como eje x la posición de ventanas_NBS.
  elif parámetro == 'K':                                                                        # Si no, si el parámetro es K,
    res, res_std, x = obtener_resultados(                                                       # hago lo mismo que con promedio,
      K, métricas,                                                                              # colocando el barrido de K para el eje x.
      lambda k: leer_métricas_KNN(directorio, f'Eclipse_k{k}', post_procesamiento))             # Leo las métricas,
    graficar_métricas(x, res, res_std, métricas, errores)                                       # las grafico,
    configurar_gráfico(r'$k$ (número de vecinos)')                                              # y coloco la etiqueta en x.
  elif parámetro == 'tolerancia':                                                               # Si no, por último, si el parámetro es 'tolerancia',
    res, res_std, x = obtener_resultados(                                                       # obtengo los resultados,
      tolerancia, métricas,                                                                     # obtengo los valores en x y las métricas,
      lambda t: leer_métricas_KNN(directorio, 'Eclipse_k12', post_procesamiento, t))            # con o sin post-procesado, ...
    graficar_métricas(x, res, res_std, métricas, errores)                                       # Grafico,
    configurar_gráfico('Tolerancia [s]')                                                        # y coloco etiqueta en eje x.
  if guardar:                                                                                   # Si guardar=True,
    guardar_figura()                                                                            # guardo la figura en formato .pdf.
  p.show()                                                                                      # Muestro el plot.

#———————————————————————————————————————————————————————————————————————————————————————
# Función Auxiliar
#———————————————————————————————————————————————————————————————————————————————————————
def calcular_métrica_global(
    archivo_métricas: pd.DataFrame,                           # Dataframe que contiene todo el archivo de métricas.
    métrica: str                                              # Métrica cuyo valor medio y error deseo calcular.
) -> tuple[float, float]:
  """
  La función calcular_métrica_global recibe un dataframe 'archivo_métricas' que contiene toda la información del archivo de métricas asociado
  a un metaparámetro que se calculó mediante cross-validation. El parámetro string 'métrica' determina qué métrica deseo leer de dicho archivo.
  La función devuelve una tupla cuyo primer elemento es la media de todas las métricas 'métrica' de los años 2014 al 2019, y cuyo segundo
  elemento es la desviación estándar de dicha media.
  """
  if métrica=='TPR':                                          # Si quiero la métrica 'TPR' global,
    res: list[float] = archivo_métricas['Recall']             # obtengo sus valores (que están en columna 'Recall') en la lista de floats res.
  elif métrica=='PPV':                                        # Si no, si quiero la métrica 'PPV' global,
    res: list[float] = archivo_métricas['Precision']          # hago lo mismo pero para la columna 'Precision'
  elif métrica=='F1':                                         # Si no, si quiero la métrica 'F1' global,
    res: list[float] = archivo_métricas['F1']                 # repito para la columna de nombre 'F1'.
  else:                                                       # Si no era ninguna de las anteriores,
    raise ValueError(f'No se encuentra la métrica: {métrica}')# => no se encuentra la métrica que se ingresó.
  media: float = np.mean(res)                                 # Calculo la media con numpy,
  std:   float = np.std(res, ddof=1)                          # y la desviación estándar, ambas en formato float.
  return (media, std)                                         # Devuelvo la tupla con la métrica global media y su error.

#———————————————————————————————————————————————————————————————————————————————————————
def obtener_resultados(
    x_values: list,                                              # Valores del metaparámetro del eje x.
    métricas: list[str],                                         # Métricas 'TPR' 'PPV' o 'F1' cuyos valores medios/errores deseo obtener.
    loader                                                       # Función a llamar para la lectura de métricas KNN.
) -> tuple[dict[str, list[float]], dict[str, list[float]], list]:
  """
  La función obtener_resultados calcula la media y la desviación estándar de todos los archivos asociados a un metaparámetro. Para ello, recorre
  todos los valores de la lista 'x_values' que contiene el barrido de un metaparámetro, y recibe una lista de strings 'métricas' que determina
  de cuáles métricas desean calcularse los valores medios y los errores. La función cargará todos los dataframes cuando la función 'loader' los
  llame, y calculará las métricas globales deseadas del archivo mediante la función auxiliar calcular_métrica_global. Devuelve una tripla cuyos
  valores son diccionarios con claves métricas, y cuyos valores son los promedios calculados y sus errores, y el último elemento de la tripla
  es la lista de valores del metaparámetro que pudieron calcularse.
  """
  res:     dict[str, list[float]] = {m: [] for m in métricas}    # Inicializo listas vacías para cada valor de métrica m deseado,
  res_std: dict[str, list[float]] = {m: [] for m in métricas}    # y su respectivo diccionario con sus errores (desviaciones estándar de res)
  x_validos: list = []                                           # Inicializo una lista vacía de valores del metaparámetro válidos.
  for x in x_values:                                             # Para cada valor de los de la lista del metaparámetro,
    try:                                                         # Intento extraer la info del archivo.
      data: pd.DataFrame = loader(x)                             # Para ello llamo a la función loader (que será leer_métricas_KNN).
    except FileNotFoundError:                                    # Si no se encuentra el archivo,
      continue                                                   # paso a la siguiente iteración del for.
    x_validos.append(x)                                          # Si no pasé a la siguiente iteración => pude leerlo => lo agrego a mi lista.
    for m in métricas:                                           # Para cada valor 'TPR', 'PPV' o 'F1' de métricas,
      media, std = calcular_métrica_global(data, métrica=m)      # calculo la media y la std del archivo,
      res[m].append(media)                                       # agrego el resultado de la media a res,
      res_std[m].append(std)                                     # y del error a la lista de res_std de desviaciones.
  return res, res_std, x_validos                                 # Devuelvo una tripla con todos los elementos luego cuando terminó el for.

#———————————————————————————————————————————————————————————————————————————————————————
def graficar_métricas(
  x: list,                                                                      # Lista de valores del eje x (por ejemplo valores promedio).
  res: dict[str, list[float]],                                                  # Diccionario de métricas TPR/PPV/F1 de valores respecto a x.
  res_std: dict[str, list[float]],                                              # Diccionario de las desviaciones estándar de dichos valores.
  métricas: list[str],                                                          # Métricas que se desean graficar: 'TPR', 'PPV' o 'F1'
  errores: bool,                                                                # Booleano que representa si se graficará con o sin errores.
  label_extra: str=''                                                           # Label extra para el plot múltiple de ventanas_NBS por ventana.
) -> None:
  """
  La función graficar_métricas recibe una lista 'x' que representa los valores del eje x. Recibe los diccionarios 'res' y 'res_std' cuyas
  claves son strings que representan las métricas TPR, PPV o F1, y sus valores son listas de floats que representan los valores de dichas
  métricas para cada valor de eje x. El parámetro 'métricas' es una lista de strings que determina las métricas que se graficarán, y el
  booleano 'errores' determina si los datos se graficarán con barras de error o no. El parámetro 'label_extra' permite distinguir el uso
  de distintas ventanas en segundos para el gráfico múltiple de ventanas_NBS. La función no devuelve nada.
  """
  for m in métricas:                                                            # Para cada métrica que se desea graficar,
    label: str = f'{m} global {label_extra}'.strip()                            # construyo el string de su etiqueta con o sin extras.
    if errores:                                                                 # Si quiero graficar las barras de error,
      p.errorbar(x, res[m], yerr=res_std[m], marker='o', capsize=4, label=label)# grafico cada valor de la clave m de res contra x, colocando
    else:                                                                       # con sus errores, techito y pisito y sus etiquetas. Si no,
      p.plot(x, res[m], marker='o', label=label)                                # grafico cada valor de clave m de res contra x interpolado.

#———————————————————————————————————————————————————————————————————————————————————————
def configurar_gráfico(
    xlabel: str                                                               # Etiqueta del eje x
) -> None:
  """
  Configura los elementos estéticos del gráfico y coloca la etiqueta en x mediante el string 'xlabel' correspondiente. No devuelve nada.
  """
  p.xlabel(xlabel)                                                            # Coloco el eje x ingresado.
  p.ylabel('Métricas globales')                                               # Coloco el mismo eje y para todos los casos,
  p.title('Promedio de métricas de CV (2014-2019) respecto del metaparámetro')# y el mismo título,
  p.grid(which='major', alpha=.2,  linestyle='-')                             # con ejes principales transparentes y con estilo '-',
  p.grid(which='minor', alpha=.15, linestyle=':')                             # y con ejes secundarios aún menos visibles con estilo ':'.
  p.legend()                                                                  # Muestro los labels.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————